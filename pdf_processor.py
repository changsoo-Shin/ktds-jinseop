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
            logger.info(f"📝 [문제 추출] {len(extracted_questions)}개의 문제 추출 완료")
            
            # 텍스트 청크 분할 및 벡터화는 기존 로직 재사용
            text_chunks = self._extract_and_chunk_text_from_text(full_text, subject)
            if not text_chunks:
                return {"success": False, "error": "PDF에서 텍스트를 추출할 수 없습니다."}
            
            # 실제 파일명 사용 (없으면 임시 파일명 사용)
            filename_to_use = original_filename if original_filename is not None else Path(pdf_file_path).name
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
        """텍스트에서 문제만 추출"""
        questions = []
        lines = full_text.split('\n')
        
        # 문제 시작 패턴들
        question_patterns = [
            r'^(\d+)\s*[\.\)]\s*',  # 1. 또는 1)
            r'^문제\s*(\d+)\s*[\.\)]?\s*',  # 문제 1. 또는 문제 1)
            r'^문항\s*(\d+)\s*[\.\)]?\s*',  # 문항 1. 또는 문항 1)
            r'^(\d+)\s*번\s*',  # 1번
            r'^문제\s*(\d+)\s*번\s*',  # 문제 1번
            r'^문항\s*(\d+)\s*번\s*',  # 문항 1번
            r'^(\d+)\s*[\.\)]\s*[가-힣]',  # 1. 다음 중 또는 1) 다음 중
        ]
        
        current_question = None
        current_question_lines = []
        question_number = None
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            # 문제 시작 감지
            is_question_start = False
            detected_number = None
            
            for pattern in question_patterns:
                match = re.match(pattern, line)
                if match:
                    detected_number = match.group(1)
                    # 연도가 아닌 문제 번호인지 확인 (1-200 범위)
                    if detected_number.isdigit() and 1 <= int(detected_number) <= 200:
                        is_question_start = True
                        break
            
            if is_question_start:
                # 이전 문제 저장
                if current_question and current_question_lines:
                    question_text = '\n'.join(current_question_lines).strip()
                    if len(question_text) > 50:  # 최소 길이 체크
                        questions.append({
                            "number": question_number,
                            "text": question_text,
                            "start_line": current_question,
                            "end_line": i - 1
                        })
                
                # 새 문제 시작
                current_question = i
                current_question_lines = [line]
                question_number = detected_number
            else:
                # 현재 문제에 라인 추가
                if current_question is not None:
                    current_question_lines.append(line)
        
        # 마지막 문제 저장
        if current_question and current_question_lines:
            question_text = '\n'.join(current_question_lines).strip()
            if len(question_text) > 50:
                questions.append({
                    "number": question_number,
                    "text": question_text,
                    "start_line": current_question,
                    "end_line": len(lines) - 1
                })
        
        # 문제 저장
        if questions:
            self._save_questions(questions, subject, original_filename)
        
        return questions
    
    def _save_questions(self, questions: List[Dict[str, Any]], subject: str, original_filename: str = None):
        """추출된 문제를 txt와 json으로 저장"""
        try:
            # 파일명 생성
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_filename = original_filename.replace('.pdf', '') if original_filename else f"{subject}_{timestamp}"
            
            # JSON 파일 저장
            json_data = {
                "subject": subject,
                "source_file": original_filename,
                "extraction_date": datetime.now().isoformat(),
                "total_questions": len(questions),
                "questions": questions
            }
            
            json_file = self.questions_dir / f"{base_filename}_questions.json"
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, ensure_ascii=False, indent=2)
            
            # TXT 파일 저장
            txt_file = self.questions_dir / f"{base_filename}_questions.txt"
            with open(txt_file, 'w', encoding='utf-8') as f:
                f.write(f"# {subject} 기출문제\n")
                f.write(f"# 출처: {original_filename}\n")
                f.write(f"# 추출일: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"# 총 문제 수: {len(questions)}개\n\n")
                
                for i, question in enumerate(questions, 1):
                    f.write(f"=== 문제 {question['number']} ===\n")
                    f.write(f"{question['text']}\n\n")
            
            logger.info(f"✅ 문제 저장 완료:")
            logger.info(f"   📄 JSON: {json_file}")
            logger.info(f"   📄 TXT: {txt_file}")
            
        except Exception as e:
            logger.error(f"❌ 문제 저장 중 오류: {e}")
    
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
                chunk["pdf_source"] = Path(pdf_file_path).name
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
        """특정 시험의 추출된 문제 목록 조회"""
        questions = []
        
        try:
            # questions_dir에서 해당 시험의 JSON 파일들 찾기
            for json_file in self.questions_dir.glob(f"*_questions.json"):
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        
                    # 해당 시험의 문제만 필터링
                    if data.get("subject") == subject:
                        source_file = data.get("source_file", "unknown")
                        # 각 문제에 출처 파일 정보 추가
                        for question in data.get("questions", []):
                            question["source_file"] = source_file
                        questions.extend(data.get("questions", []))
                        
                except Exception as e:
                    logger.warning(f"⚠️ JSON 파일 읽기 오류 ({json_file}): {e}")
                    continue
            
            # 문제 번호 순으로 정렬
            questions.sort(key=lambda x: int(x.get("number", 0)) if x.get("number", "0").isdigit() else 0)
            
            logger.info(f"✅ {subject} 시험의 추출된 문제 {len(questions)}개 로드 완료")
            return questions
            
        except Exception as e:
            logger.error(f"❌ 추출된 문제 조회 중 오류: {e}")
            return []
    
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