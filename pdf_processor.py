"""
PDF 처리 및 벡터화 모듈
Docling을 사용한 PDF 텍스트 추출 및 FAISS 벡터 DB 구축
"""

import os
import tempfile
from typing import List, Dict, Any, Optional
import numpy as np
from pathlib import Path
import hashlib
from datetime import datetime
import json
import re
import logging

# 로거 설정
logger = logging.getLogger(__name__)

# Docling 관련 import
try:
    from docling.document_converter import DocumentConverter
except ImportError:
    logger.error("Docling이 설치되지 않았습니다. pip install docling을 실행해주세요.")
    DocumentConverter = None

# FAISS 관련 import
try:
    import faiss
    from sentence_transformers import SentenceTransformer
except ImportError:
    logger.error("FAISS 또는 sentence-transformers가 설치되지 않았습니다.")
    logger.error("pip install faiss-cpu sentence-transformers를 실행해주세요.")
    faiss = None
    SentenceTransformer = None

class PDFProcessor:
    """PDF 처리 및 벡터화 클래스"""
    
    def __init__(self, vector_db_path: str = "faiss_vector_db"):
        self.vector_db_path = Path(vector_db_path)
        self.vector_db_path.mkdir(exist_ok=True)
        
        # 문제 저장 디렉토리 생성
        self.questions_dir = Path("extracted_questions")
        self.questions_dir.mkdir(exist_ok=True)
        
        # 벡터 모델 초기화
        self.embedding_model = None
        self.index = None
        self.documents = []
        self.metadata = []
        
        self._initialize_models()
    
    def _initialize_models(self):
        """벡터 모델 및 FAISS 인덱스 초기화"""
        if SentenceTransformer is None or faiss is None:
            logger.error("필요한 라이브러리가 설치되지 않았습니다.")
            return
        
        try:
            # GPU 사용 가능 여부 확인 및 설정
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            logger.info(f"🔧 [INFO] 사용할 디바이스: {device}")
            if device == 'cuda':
                logger.info(f"🔧 [INFO] GPU 모델: {torch.cuda.get_device_name(0)}")
            
            # 한국어에 특화된 임베딩 모델 사용
            self.embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', device=device)
            
            # FAISS 인덱스 초기화 (L2 거리 기반)
            dimension = self.embedding_model.get_sentence_embedding_dimension()
            
            # FAISS는 CPU 사용 (Python 3.12에서 GPU FAISS 미지원)
            self.index = faiss.IndexFlatL2(dimension)
            logger.info(f"🔧 [INFO] FAISS CPU 인덱스 사용 (Python 3.12)")
            
            logger.info(f"✅ 벡터 모델 초기화 완료 (차원: {dimension})")
        except Exception as e:
            logger.error(f"❌ 벡터 모델 초기화 실패: {e}")
    
    def process_pdf(self, pdf_file_path: str, subject: str = "정보시스템감리사", original_filename: str = None) -> Dict[str, Any]:
        """PDF 파일 처리 및 벡터화"""
        logger.info(f"\n📄 [PDF 처리] 파일: {pdf_file_path}")
        
        if DocumentConverter is None:
            return {"success": False, "error": "Docling 라이브러리가 설치되지 않았습니다."}
        
        try:
            # Docling GPU 설정
            import os
            import torch
            if torch.cuda.is_available():
                os.environ['DOCLING_ACCELERATOR'] = 'cuda'
                logger.info(f"🔧 [INFO] Docling GPU 설정: cuda")
            else:
                os.environ['DOCLING_ACCELERATOR'] = 'cpu'
                logger.info(f"🔧 [INFO] Docling GPU 설정: cpu")
            
            converter = DocumentConverter()
            result = converter.convert(pdf_file_path)
            # 전체 텍스트 추출 (Markdown 기준)
            full_text = result.document.export_to_markdown()
            
            # 문제 추출 및 저장
            extracted_questions = self._extract_questions_from_text(full_text, subject, original_filename)
            logger.info(f"📝 [문제 추출] {len(extracted_questions)}개의 문제 추출 완료 (최대한 많이)")
            
            # 추출된 문제를 TXT 파일로 저장
            if extracted_questions:
                self._save_questions(extracted_questions, subject, original_filename)
            else:
                logger.warning("⚠️ 추출된 문제가 없어서 TXT 파일을 생성하지 않습니다.")
            
            # 텍스트 청크 분할 및 벡터화는 기존 로직 재사용
            text_chunks = self._extract_and_chunk_text_from_text(full_text, subject)
            if not text_chunks:
                return {"success": False, "error": "PDF에서 텍스트를 추출할 수 없습니다."}
            
            # 실제 파일명 사용 (없으면 임시 파일명 사용)
            filename_to_use = original_filename if original_filename is not None else str(Path(pdf_file_path).name)
            self._vectorize_and_store(text_chunks, subject, filename_to_use)
            self._save_metadata()
            logger.info(f"✅ PDF 처리 완료 - {len(text_chunks)}개 청크 생성")
            return {
                "success": True,
                "chunks_count": len(text_chunks),
                "questions_count": len(extracted_questions),
                "subject": subject,
                "filename": filename_to_use
            }
        except Exception as e:
            error_msg = f"PDF 처리 중 오류 발생: {e}"
            logger.error(f"❌ {error_msg}")
            return {"success": False, "error": error_msg}
    
    def _extract_questions_from_text(self, full_text: str, subject: str, original_filename: str = None) -> List[Dict[str, Any]]:
        """텍스트에서 문제만 최대한 많이 추출 (슬라이딩 윈도우 방식으로 연속성 검증)"""
        import re
        lines = full_text.split('\n')
        
        # 3자리까지 숫자 + 다양한 구분자 패턴 (줄 앞 공백 허용)
        question_patterns = [
            # 가장 일반적인 패턴들 (우선순위 높음)
            r'^\s*(\d{1,3})\.\s+',  # 108. (기본형, 공백 필수)
            r'^\s*-\s*(\d{1,3})\.\s+',  # - 108. (대시 접두사, 공백 필수)
            r'^\s*(\d{1,3})\)\s+',  # 108) (공백 필수)
            r'^\s*-\s*(\d{1,3})\)\s+',  # - 108) (대시 접두사, 공백 필수)
            
            # 문제 번호만 있는 경우
            r'^\s*(\d{1,3})\.\s*$',  # 108. (줄 끝)
            r'^\s*-\s*(\d{1,3})\.\s*$',  # - 108. (줄 끝)
            
            # 괄호형
            r'^\s*\((\d{1,3})\)\s*',  # (108)
            r'^\s*\[(\d{1,3})\]\s*',  # [108]
            r'^\s*【(\d{1,3})】\s*',  # 【108】
            r'^\s*〈(\d{1,3})〉\s*',  # 〈108〉
            
            # 키워드형
            r'^\s*문제\s*(\d{1,3})\s*[\.\)]?\s*',  # 문제 108. 
            r'^\s*(\d{1,3})\s*번\s*[\.\)]?\s*',  # 108번.
            r'^\s*문항\s*(\d{1,3})\s*[\.\)]?\s*',  # 문항 108.
            
            # 특정 키워드와 함께 시작하는 패턴들
            r'^\s*(\d{1,3})\.\s*다음\s*중',
            r'^\s*(\d{1,3})\.\s*올바른\s*것은',
            r'^\s*(\d{1,3})\.\s*틀린\s*것은',
            r'^\s*(\d{1,3})\.\s*적절한\s*것은',
            r'^\s*(\d{1,3})\.\s*가장\s*적절한',
            
            # 대시 접두사가 있는 키워드 패턴들
            r'^\s*-\s*(\d{1,3})\.\s*다음\s*중',
            r'^\s*-\s*(\d{1,3})\.\s*올바른\s*것은',
            r'^\s*-\s*(\d{1,3})\.\s*틀린\s*것은',
            r'^\s*-\s*(\d{1,3})\.\s*적절한\s*것은',
            r'^\s*-\s*(\d{1,3})\.\s*가장\s*적절한',
        ]
        
        # 1단계: 모든 가능한 문제 번호 위치 찾기
        potential_questions = []
        logger.info(f"🔍 [문제 추출] 전체 라인 수: {len(lines)}개")
        logger.info(f"🔍 [문제 추출] 처음 10라인 미리보기:")
        for i, line in enumerate(lines[:10]):
            logger.info(f"   라인 {i}: {line[:100]}")
        
        for line_idx, line in enumerate(lines):
            line = line.strip('\r')
            if not line:
                continue
            
            # 디버깅: 특정 문제 번호가 포함된 라인들을 집중 검사
            contains_target_numbers = any(f" {i}." in line or f"- {i}." in line or f"{i}." in line[:10] for i in range(50, 80))
            contains_any_number = any(str(i) in line for i in range(1, 1000))
            
            if line_idx < 20 or contains_target_numbers:
                logger.info(f"🔍 라인 {line_idx} 검사: {line}")
                
                # 각 패턴을 개별적으로 테스트
                for pattern_idx, pattern in enumerate(question_patterns):
                    match = re.match(pattern, line)
                    if match:
                        detected_number = match.group(1)
                        logger.info(f"   ✅ 패턴 {pattern_idx} 매칭 성공: {pattern}")
                        logger.info(f"   📍 추출된 번호: {detected_number}")
                    elif contains_target_numbers:
                        logger.info(f"   ❌ 패턴 {pattern_idx} 매칭 실패: {pattern}")
                
            for pattern_idx, pattern in enumerate(question_patterns):
                match = re.match(pattern, line)
                if match:
                    detected_number = match.group(1)
                    if detected_number.isdigit() and 1 <= int(detected_number) <= 999:
                        potential_questions.append({
                            "line_idx": line_idx,
                            "number": int(detected_number),
                            "line": line
                        })
                        logger.info(f"✅ 문제 발견: {detected_number}번 (라인 {line_idx}, 패턴 {pattern_idx}: {pattern})")
                        logger.info(f"   전체 라인: {line}")
                        break
        
        logger.info(f"🔍 [1단계] 잠재 문제 {len(potential_questions)}개 발견")
        if potential_questions:
            logger.info(f"📋 발견된 문제 번호들: {[q['number'] for q in potential_questions]}")
        
        # 2단계: 중복 제거 및 번호순 정렬
        # 같은 번호의 중복 문제 제거 (가장 먼저 발견된 것만 유지)
        seen_numbers = set()
        verified_questions = []
        
        for potential_q in potential_questions:
            if potential_q["number"] not in seen_numbers:
                verified_questions.append(potential_q)
                seen_numbers.add(potential_q["number"])
                logger.debug(f"✅ 문제 {potential_q['number']}번 추가")
            else:
                logger.debug(f"❌ 문제 {potential_q['number']}번 중복 제외")
        
        # 번호순으로 정렬
        verified_questions.sort(key=lambda x: x["number"])
        
        logger.info(f"🔍 [2단계] 연속성 검증 후 문제 {len(verified_questions)}개 확정")
        
        # 3단계: 검증된 문제들 사이의 텍스트 추출
        questions = []
        for i, verified_q in enumerate(verified_questions):
            start_line_idx = verified_q["line_idx"]
            
            # 다음 문제까지의 라인 범위 결정
            if i + 1 < len(verified_questions):
                end_line_idx = verified_questions[i + 1]["line_idx"] - 1
            else:
                end_line_idx = len(lines) - 1
            
            # 문제 텍스트 추출
            question_lines = []
            for line_idx in range(start_line_idx, min(end_line_idx + 1, len(lines))):
                line = lines[line_idx].strip('\r')
                if line:  # 빈 줄이 아닌 경우만 추가
                    question_lines.append(line)
            
            if question_lines:
                question_text = '\n'.join(question_lines)
                questions.append({
                    "number": str(verified_q["number"]),
                    "text": question_text,
                    "start_line": start_line_idx,
                    "end_line": end_line_idx
                })
                logger.debug(f"📝 문제 {verified_q['number']}번 추출: {len(question_text)} 문자")
        
        logger.info(f"📝 [3단계] 최종 문제 추출 완료: {len(questions)}개")
        logger.info(f"📋 [추출된 문제 번호]: {[q['number'] for q in questions]}")
        
        return questions
    
    def _extract_questions_with_ai(self, full_text: str, subject: str) -> List[Dict[str, Any]]:
        """AI를 사용하여 문제 추출 (정규표현식 실패 시 대체 방법)"""
        try:
            # 텍스트가 너무 길면 앞부분만 사용
            text_sample = full_text[:10000] if len(full_text) > 10000 else full_text
            
            # 문제 추출을 위한 프롬프트
            prompt = f"""
다음 {subject} 기출문제 텍스트에서 문제들을 추출해주세요.

텍스트:
{text_sample}

다음 형식으로 응답해주세요:

=== 문제 1 ===
[문제 내용]

=== 문제 2 ===
[문제 내용]

...

문제 번호는 텍스트에서 찾을 수 있는 번호를 사용하거나, 순서대로 1, 2, 3...으로 부여해주세요.
문제는 객관식, 주관식, 서술형 등 모든 유형을 포함합니다.
문제의 시작과 끝을 명확히 구분해주세요.
"""

            # Azure OpenAI API 호출
            import openai
            from config import Config
            
            # OpenAI 설정
            openai.api_key = Config.OPENAI_API_KEY
            openai.azure_endpoint = Config.AZURE_ENDPOINT
            openai.api_type = Config.OPENAI_API_TYPE
            openai.api_version = Config.OPENAI_API_VERSION
            
            response = openai.chat.completions.create(
                model=Config.DEPLOYMENT_NAME,
                messages=[
                    {"role": "system", "content": "당신은 기출문제 텍스트에서 문제를 정확히 추출하는 전문가입니다."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
            )
            
            result = response.choices[0].message.content
            if not result:
                return []
            
            # AI 응답에서 문제 파싱
            questions = []
            current_question = None
            current_lines = []
            
            lines = result.split('\n')
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # 문제 시작 감지
                if line.startswith('=== 문제 ') and line.endswith(' ==='):
                    # 이전 문제 저장
                    if current_question and current_lines:
                        question_text = '\n'.join(current_lines).strip()
                        if len(question_text) > 30:
                            questions.append({
                                "number": current_question,
                                "text": question_text,
                                "start_line": 0,
                                "end_line": 0
                            })
                    
                    # 새 문제 시작
                    current_question = line.replace('=== 문제 ', '').replace(' ===', '')
                    current_lines = []
                else:
                    # 현재 문제에 라인 추가
                    if current_question is not None:
                        current_lines.append(line)
            
            # 마지막 문제 저장
            if current_question and current_lines:
                question_text = '\n'.join(current_lines).strip()
                if len(question_text) > 30:
                    questions.append({
                        "number": current_question,
                        "text": question_text,
                        "start_line": 0,
                        "end_line": 0
                    })
            
            logger.info(f"🤖 [AI 문제 추출] {len(questions)}개 문제 추출 완료")
            return questions
            
        except Exception as e:
            logger.error(f"❌ [AI 문제 추출] 오류: {e}")
            return []
    
    def _save_questions(self, questions: List[Dict[str, Any]], subject: str, original_filename: str = None):
        """추출된 문제를 txt 파일로만 저장"""
        try:
            logger.info(f"💾 [문제 저장] {len(questions)}개 문제 저장 시작...")
            
            # 파일명 생성
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_filename = (original_filename or "").replace('.pdf', '') if original_filename else f"{subject}_{timestamp}"
            
            # TXT 파일만 저장
            txt_file = self.questions_dir / f"{base_filename}_questions.txt"
            logger.info(f"📄 [문제 저장] 저장할 파일: {txt_file}")
            
            with open(txt_file, 'w', encoding='utf-8') as f:
                f.write(f"# {subject} 기출문제\n")
                f.write(f"# 출처: {original_filename}\n")
                f.write(f"# 추출일: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"# 총 문제 수: {len(questions)}개\n\n")
                
                for i, question in enumerate(questions, 1):
                    f.write(f"=== 문제 {question['number']} ===\n")
                    f.write(f"{question['text']}\n\n")
            
            logger.info(f"✅ 문제 저장 완료:")
            logger.info(f"   📄 TXT: {txt_file}")
            logger.info(f"   📊 저장된 문제 수: {len(questions)}개")
            
        except Exception as e:
            logger.error(f"❌ 문제 저장 중 오류: {e}")
            import traceback
            logger.error(f"❌ 상세 오류: {traceback.format_exc()}")
    
    def _extract_and_chunk_text_from_text(self, full_text: str, subject: str) -> List[Dict[str, Any]]:
        """텍스트(문자열)에서 청크 분할"""
        chunks = []
        try:
            # 표 감지 및 특별 처리
            lines = full_text.split('\n')
            current_chunk_lines = []
            current_length = 0
            in_table = False
            table_start_line = -1
            
            for i, line in enumerate(lines):
                # 표 시작/끝 감지
                is_table_line = line.strip().startswith('|') or ('|' in line.strip() and len(line.strip()) > 10)
                
                if is_table_line and not in_table:
                    # 표 시작
                    in_table = True
                    table_start_line = i
                    # 표 시작 전까지의 청크 저장
                    if current_chunk_lines and len('\n'.join(current_chunk_lines).strip()) >= 100:
                        chunk_text = '\n'.join(current_chunk_lines)
                        chunk_id = hashlib.md5(f"{subject}_{len(chunks)}_{chunk_text[:50]}".encode()).hexdigest()
                        chunks.append({
                            "id": chunk_id,
                            "text": chunk_text.strip(),
                            "start_pos": len(chunks) * 1000,
                            "end_pos": len(chunks) * 1000 + len(chunk_text),
                            "subject": subject,
                            "created_at": datetime.now().isoformat()
                        })
                    current_chunk_lines = []
                    current_length = 0
                
                elif not is_table_line and in_table:
                    # 표 끝
                    in_table = False
                    # 표 전체를 하나의 청크로 저장
                    table_lines = lines[table_start_line:i]
                    table_text = '\n'.join(table_lines)
                    if len(table_text.strip()) >= 50:
                        chunk_id = hashlib.md5(f"{subject}_{len(chunks)}_table_{table_start_line}".encode()).hexdigest()
                        chunks.append({
                            "id": chunk_id,
                            "text": table_text.strip(),
                            "start_pos": len(chunks) * 1000,
                            "end_pos": len(chunks) * 1000 + len(table_text),
                            "subject": subject,
                            "created_at": datetime.now().isoformat(),
                            "is_table": True
                        })
                    current_chunk_lines = []
                    current_length = 0
                
                # 현재 라인 추가
                if not in_table:
                    current_chunk_lines.append(line)
                    current_length += len(line) + 1
                    
                    # 일반 텍스트 청크 크기 체크
                    if current_length >= 1000:
                        chunk_text = '\n'.join(current_chunk_lines)
                        if len(chunk_text.strip()) >= 100:
                            chunk_id = hashlib.md5(f"{subject}_{len(chunks)}_{chunk_text[:50]}".encode()).hexdigest()
                            chunks.append({
                                "id": chunk_id,
                                "text": chunk_text.strip(),
                                "start_pos": len(chunks) * 1000,
                                "end_pos": len(chunks) * 1000 + len(chunk_text),
                                "subject": subject,
                                "created_at": datetime.now().isoformat()
                            })
                        current_chunk_lines = []
                        current_length = 0
            
            # 마지막 청크 처리
            if current_chunk_lines:
                chunk_text = '\n'.join(current_chunk_lines)
                if len(chunk_text.strip()) >= 100:
                    chunk_id = hashlib.md5(f"{subject}_{len(chunks)}_{chunk_text[:50]}".encode()).hexdigest()
                    chunks.append({
                        "id": chunk_id,
                        "text": chunk_text.strip(),
                        "start_pos": len(chunks) * 1000,
                        "end_pos": len(chunks) * 1000 + len(chunk_text),
                        "subject": subject,
                        "created_at": datetime.now().isoformat()
                    })
                
        except Exception as e:
            logger.error(f"텍스트 추출 중 오류: {e}")
        return chunks
    
    def _vectorize_and_store(self, chunks: List[Dict[str, Any]], subject: str, pdf_file_path: str):
        """청크를 벡터화하고 FAISS에 저장"""
        if self.embedding_model is None or self.index is None:
            logger.error("벡터 모델이 초기화되지 않았습니다.")
            return
        
        try:
            # 텍스트 추출
            texts = [chunk["text"] for chunk in chunks]
            
            # 벡터화
            logger.info("🔄 텍스트 벡터화 중...")
            embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
            
            # FAISS 인덱스에 추가
            self.index.add(embeddings.astype('float32'))
            
            # 메타데이터 저장
            for i, chunk in enumerate(chunks):
                chunk["embedding_id"] = len(self.documents) + i
                chunk["pdf_source"] = str(Path(pdf_file_path).name)
                self.documents.append(chunk["text"])
                self.metadata.append(chunk)
            
            logger.info(f"✅ {len(chunks)}개 청크 벡터화 완료")
            
        except Exception as e:
            logger.error(f"벡터화 중 오류: {e}")
    
    def search_similar_chunks(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """유사한 청크 검색"""
        if self.embedding_model is None or self.index is None:
            return []
        
        try:
            # 쿼리 벡터화
            query_embedding = self.embedding_model.encode([query])
            
            # FAISS 검색
            distances, indices = self.index.search(query_embedding.astype('float32'), n_results)
            
            # 결과 포맷팅
            results = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx < len(self.metadata):
                    result = {
                        "rank": i + 1,
                        "distance": float(distance),
                        "text": self.documents[idx],
                        "metadata": self.metadata[idx]
                    }
                    results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"검색 중 오류: {e}")
            return []
    
    def get_chunks_by_subject(self, subject: str, n_results: int = 10) -> List[Dict[str, Any]]:
        """과목별 청크 조회"""
        results = []
        
        for metadata in self.metadata:
            if metadata.get("subject") == subject:
                results.append({
                    "text": metadata.get("text", ""),
                    "metadata": metadata
                })
                if len(results) >= n_results:
                    break
        
        return results
    
    def _save_metadata(self):
        """메타데이터를 파일로 저장"""
        try:
            metadata_file = self.vector_db_path / "metadata.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "total_chunks": len(self.documents),
                    "metadata": self.metadata,
                    "last_updated": datetime.now().isoformat()
                }, f, ensure_ascii=False, indent=2)
            
            # FAISS 인덱스 저장
            if self.index:
                index_file = self.vector_db_path / "faiss_index.bin"
                faiss.write_index(self.index, str(index_file))
            
            logger.info(f"✅ 메타데이터 및 인덱스 저장 완료")
            
        except Exception as e:
            logger.error(f"메타데이터 저장 중 오류: {e}")
    
    def load_existing_data(self):
        """기존 데이터 로드"""
        try:
            metadata_file = self.vector_db_path / "metadata.json"
            index_file = self.vector_db_path / "faiss_index.bin"
            
            if metadata_file.exists() and index_file.exists():
                # 메타데이터 로드
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.metadata = data.get("metadata", [])
                    self.documents = [meta.get("text", "") for meta in self.metadata]
                
                # FAISS 인덱스 로드
                if faiss:
                    self.index = faiss.read_index(str(index_file))
                
                logger.info(f"✅ 기존 데이터 로드 완료 - {len(self.documents)}개 청크")
                return True
            
        except Exception as e:
            logger.error(f"기존 데이터 로드 중 오류: {e}")
        
        return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """벡터 DB 통계 정보"""
        stats = {
            "total_chunks": len(self.documents),
            "total_metadata": len(self.metadata),
            "index_size": self.index.ntotal if self.index else 0,
            "subjects": list(set([meta.get("subject", "") for meta in self.metadata if meta.get("subject")])),
            "pdf_sources": list(set([meta.get("pdf_source", "") for meta in self.metadata if meta.get("pdf_source")]))
        }
        return stats
    
    def clear_all_data(self):
        """모든 데이터 삭제"""
        try:
            self.documents = []
            self.metadata = []
            if self.index:
                self.index.reset()
            
            # 파일 삭제
            metadata_file = self.vector_db_path / "metadata.json"
            index_file = self.vector_db_path / "faiss_index.bin"
            
            if metadata_file.exists():
                metadata_file.unlink()
            if index_file.exists():
                index_file.unlink()
            
            logger.info("✅ 모든 데이터 삭제 완료")
            
        except Exception as e:
            logger.error(f"데이터 삭제 중 오류: {e}")
    
    def get_extracted_questions(self, subject: str) -> List[Dict[str, Any]]:
        """특정 시험의 추출된 문제 목록 조회 (TXT 파일만 사용)"""
        questions = []
        
        try:
            # 모든 TXT 파일 검색 (더 유연한 매칭)
            all_txt_files = list(self.questions_dir.glob("*_questions.txt"))
            logger.info(f"🔍 [파일 검색] 전체 TXT 파일: {[f.name for f in all_txt_files]}")
            
            # 과목명 매칭 (공백 제거하여 비교)
            subject_clean = subject.replace(" ", "").replace("　", "")  # 공백과 전각공백 제거
            matching_files = []
            
            for txt_file in all_txt_files:
                filename_clean = txt_file.name.replace(" ", "").replace("　", "")  # 공백 제거
                if subject_clean in filename_clean or subject in txt_file.name:
                    matching_files.append(txt_file)
                    logger.info(f"✅ [파일 매칭] {txt_file.name} - 매칭됨")
                else:
                    logger.debug(f"❌ [파일 매칭] {txt_file.name} - 매칭 안됨")
            
            logger.info(f"🔍 [파일 검색] {subject}와 매칭된 파일: {len(matching_files)}개")
            
            for txt_file in matching_files:
                try:
                    txt_questions = self._parse_questions_from_txt(txt_file, subject)
                    questions.extend(txt_questions)
                    logger.info(f"📄 [파일 로드] {txt_file.name}에서 {len(txt_questions)}개 문제 로드")
                except Exception as e:
                    logger.warning(f"⚠️ TXT 파일 읽기 오류 ({txt_file}): {e}")
                    continue
            
            # 문제 번호 순으로 정렬
            questions.sort(key=lambda x: int(x.get("number", 0)) if x.get("number", "0").isdigit() else 0)
            
            logger.info(f"✅ {subject} 시험의 추출된 문제 {len(questions)}개 로드 완료 (TXT 파일만 사용)")
            return questions
            
        except Exception as e:
            logger.error(f"❌ 추출된 문제 조회 중 오류: {e}")
            return []
    
    def _parse_questions_from_txt(self, txt_file: Path, subject: str) -> List[Dict[str, Any]]:
        """TXT 파일에서 문제들을 파싱 (개별 문제 분리)"""
        questions = []
        
        try:
            logger.info(f"🔍 [TXT 파싱] 파일 읽기 시작: {txt_file.name}")
            with open(txt_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            logger.info(f"🔍 [TXT 파싱] 파일 크기: {len(content)} 문자")
            logger.info(f"🔍 [TXT 파싱] 파일 내용 미리보기 (처음 500자):")
            logger.info(content[:500])
            
            # 파일에서 출처 정보 추출
            source_file = "unknown"
            extraction_date = ""
            
            lines = content.split('\n')
            logger.info(f"🔍 [TXT 파싱] 전체 라인 수: {len(lines)}개")
            
            for line in lines:
                if line.startswith('# 출처:'):
                    source_file = line.replace('# 출처:', '').strip()
                    logger.info(f"📄 [TXT 파싱] 출처 정보: {source_file}")
                elif line.startswith('# 추출일:'):
                    extraction_date = line.replace('# 추출일:', '').strip()
                    logger.info(f"📅 [TXT 파싱] 추출일: {extraction_date}")
            
            # 문제 섹션 분리
            question_sections = content.split('=== 문제 ')
            logger.info(f"🔍 [TXT 파싱] '=== 문제 '로 분리한 섹션 수: {len(question_sections)}개")
            
            for i, section in enumerate(question_sections):
                logger.info(f"🔍 [TXT 파싱] 섹션 {i}: {len(section)} 문자")
                if i > 0:  # 첫 번째 섹션은 헤더
                    logger.info(f"   섹션 {i} 내용 미리보기: {section[:100]}...")
            
            for section_idx, section in enumerate(question_sections[1:], 1):  # 첫 번째는 헤더이므로 제외
                logger.info(f"🔍 [TXT 파싱] 섹션 {section_idx} 처리 중...")
                lines = section.split('\n')
                if not lines:
                    logger.warning(f"⚠️ [TXT 파싱] 섹션 {section_idx}: 빈 섹션")
                    continue
                
                # 문제 번호 추출
                first_line = lines[0].strip()
                logger.info(f"🔍 [TXT 파싱] 섹션 {section_idx} 첫 라인: '{first_line}'")
                
                if not first_line:
                    logger.warning(f"⚠️ [TXT 파싱] 섹션 {section_idx}: 첫 라인이 비어있음")
                    continue
                    
                # === 제거 후 숫자 확인
                clean_line = first_line.replace('=', '').strip()
                logger.info(f"🔍 [TXT 파싱] 섹션 {section_idx} === 제거 후: '{clean_line}'")
                
                if not clean_line.isdigit():
                    logger.warning(f"⚠️ [TXT 파싱] 섹션 {section_idx}: 문제 번호가 숫자가 아님: '{clean_line}'")
                    continue
                
                section_number = clean_line
                logger.info(f"✅ [TXT 파싱] 섹션 {section_idx}: 문제 번호 '{section_number}' 추출")
                
                # 문제 내용 추출 (문제 번호 라인 제외)
                section_content = '\n'.join(lines[1:]).strip()
                logger.info(f"🔍 [TXT 파싱] 섹션 {section_idx}: 내용 길이 {len(section_content)} 문자")
                
                # TXT 파일에서는 각 섹션이 이미 완전한 문제이므로 그대로 사용
                if section_content:
                    questions.append({
                        "number": section_number,
                        "text": section_content,
                        "source_file": source_file,
                        "extraction_date": extraction_date,
                        "start_line": 0,
                        "end_line": 0
                    })
                    logger.info(f"✅ [TXT 파싱] 문제 {section_number}번 성공적으로 로드: {len(section_content)} 문자")
                else:
                    logger.warning(f"⚠️ [TXT 파싱] 섹션 {section_idx}: 내용이 비어있음")
            
            logger.info(f"📄 TXT 파일에서 {len(questions)}개 문제 파싱 완료: {txt_file.name}")
            return questions
            
        except Exception as e:
            logger.error(f"❌ TXT 파일 파싱 실패 {txt_file}: {e}")
            import traceback
            logger.error(f"❌ 상세 오류: {traceback.format_exc()}")
            return []
    
    def _extract_individual_questions_from_section(self, section_content: str, section_number: str, 
                                                 source_file: str, extraction_date: str) -> List[Dict[str, Any]]:
        """섹션 내용에서 개별 문제들을 추출 (단순화된 버전)"""
        questions = []
        
        # 문제 번호 패턴들 (더 정확한 패턴)
        question_patterns = [
            r'^(\d+)\.\s*',  # 38. (공백 포함)
        ]
        
        lines = section_content.split('\n')
        current_question = None
        current_question_lines = []
        current_question_number = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # 문제 시작 감지
            is_question_start = False
            detected_number = None
            
            for pattern in question_patterns:
                import re
                match = re.match(pattern, line)
                if match:
                    detected_number = match.group(1)
                    # 연도가 아닌 문제 번호인지 확인 (1-1000 범위로 확장)
                    if detected_number.isdigit() and 1 <= int(detected_number) <= 1000:
                        is_question_start = True
                        logger.debug(f"✅ 문제 시작 감지: {line[:50]}... (패턴: {pattern})")
                        break
            
            if not is_question_start and line.startswith(('38.', '39.', '40.', '41.', '42.')):
                logger.debug(f"🔍 패턴 미매칭 라인: {line[:50]}...")
            
            if is_question_start:
                # 이전 문제 저장
                if current_question and current_question_lines:
                    question_text = '\n'.join(current_question_lines).strip()
                    if len(question_text) > 20:  # 최소 길이 체크 (완화)
                        questions.append({
                            "number": current_question_number,
                            "text": question_text,
                            "source_file": source_file,
                            "extraction_date": extraction_date,
                            "start_line": 0,
                            "end_line": 0
                        })
                        logger.debug(f"✅ 문제 {current_question_number} 추출: {len(question_text)} 문자")
                
                # 새 문제 시작
                current_question = True
                current_question_lines = [line]
                current_question_number = detected_number
            else:
                # 현재 문제에 라인 추가
                if current_question is not None:
                    current_question_lines.append(line)
        
        # 마지막 문제 저장
        if current_question and current_question_lines:
            question_text = '\n'.join(current_question_lines).strip()
            if len(question_text) > 20:
                questions.append({
                    "number": current_question_number,
                    "text": question_text,
                    "source_file": source_file,
                    "extraction_date": extraction_date,
                    "start_line": 0,
                    "end_line": 0
                })
                logger.debug(f"✅ 마지막 문제 {current_question_number} 추출: {len(question_text)} 문자")
        
        return questions
    
    def search_extracted_questions(self, query: str, subject: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """추출된 문제에서 검색"""
        questions = self.get_extracted_questions(subject)
        if not questions:
            return []
        
        # 간단한 키워드 기반 검색
        query_lower = query.lower()
        matched_questions = []
        
        for question in questions:
            question_text = question.get("text", "").lower()
            if query_lower in question_text:
                matched_questions.append(question)
        
        # 상위 n_results개 반환
        return matched_questions[:n_results]
    
    def get_random_extracted_question(self, subject: str) -> Optional[Dict[str, Any]]:
        """추출된 문제에서 랜덤 선택"""
        questions = self.get_extracted_questions(subject)
        if not questions:
            return None
        
        import random
        return random.choice(questions)
    
    def get_extracted_question_by_number(self, subject: str, question_number: str) -> Optional[Dict[str, Any]]:
        """문제 번호로 특정 문제 조회"""
        questions = self.get_extracted_questions(subject)
        
        for question in questions:
            if question.get("number") == question_number:
                return question
        
        return None
    
    def search_extracted_questions_semantic(self, query: str, subject: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """추출된 문제에서 semantic 검색"""
        questions = self.get_extracted_questions(subject)
        if not questions:
            return []
        
        if self.embedding_model is None:
            logger.warning("⚠️ 벡터 모델이 초기화되지 않아 키워드 검색으로 대체합니다.")
            return self.search_extracted_questions(query, subject, n_results)
        
        try:
            # 쿼리 벡터화
            query_embedding = self.embedding_model.encode([query])
            
            # 모든 문제 텍스트 벡터화
            question_texts = [q["text"] for q in questions]
            question_embeddings = self.embedding_model.encode(question_texts)
            
            # 코사인 유사도 계산
            from sklearn.metrics.pairwise import cosine_similarity
            similarities = cosine_similarity(query_embedding, question_embeddings)[0]
            
            # 유사도와 함께 결과 구성
            results_with_scores = []
            for i, (question, similarity) in enumerate(zip(questions, similarities)):
                results_with_scores.append({
                    "question": question,
                    "score": float(similarity),
                    "rank": i + 1
                })
            
            # 유사도 기준으로 정렬
            results_with_scores.sort(key=lambda x: x["score"], reverse=True)
            
            # 상위 n_results개 반환
            top_results = results_with_scores[:n_results]
            
            # 결과 포맷팅
            results = []
            for result in top_results:
                question_data = result["question"]
                results.append({
                    "content": question_data["text"],
                    "metadata": {
                        "type": "extracted_question",
                        "subject": subject,
                        "question_number": question_data["number"],
                        "pdf_source": "추출된 기출문제",
                        "score": result["score"],
                        "rank": result["rank"]
                    },
                    "score": result["score"]
                })
            
            logger.info(f"✅ 추출된 문제 semantic 검색 완료: {len(results)}개 결과")
            return results
            
        except Exception as e:
            logger.error(f"❌ 추출된 문제 semantic 검색 중 오류: {e}")
            # 오류 발생 시 키워드 검색으로 대체
            return self.search_extracted_questions(query, subject, n_results)

# 전역 PDF 프로세서 인스턴스
pdf_processor = PDFProcessor() 