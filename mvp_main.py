# Standard library imports
import gradio as gr
import os
import json
import random
import tempfile
import logging
import time
import uuid
import hashlib
import re
import traceback
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

# Third-party imports
from dotenv import load_dotenv
import openai

# Optional imports (handled with try-except)
try:
    from pyngrok import ngrok
    NGROK_AVAILABLE = True
except ImportError:
    NGROK_AVAILABLE = False

# Local module imports
from config import Config
from logger import UserLogger
from prompt import ExamPrompts, ChatPrompts, AnalysisPrompts, PDFProcessingPrompts
from vector_store import vector_store
from pdf_processor import pdf_processor
from review_agent_simple import review_agent

# 로거 설정
logger = logging.getLogger(__name__)

# Azure OpenAI 설정 (config.py import 이후에 설정)
openai.api_key = Config.OPENAI_API_KEY
openai.azure_endpoint = Config.AZURE_ENDPOINT
openai.api_type = Config.OPENAI_API_TYPE
openai.api_version = Config.OPENAI_API_VERSION
DEPLOYMENT_NAME = Config.DEPLOYMENT_NAME

class ExamQuestionGenerator:
    def __init__(self):
        self.conversation_history = []
        self.exam_name = "기출문제 RAG 기반 시험 문제 생성 및 질의 응답 챗봇"
        self.difficulties = ["쉬움", "보통", "어려움"]
        self.question_types = ["객관식", "주관식"]
        self.current_question = None
        self.current_answer = None
        self.current_explanation = None
        self.current_context = None
        self.current_metadata = None  # 검색된 문제의 메타데이터
        self.question_mode = "generate"  # "generate" 또는 "exact"
        self.current_exam_name = None  # 현재 선택된 시험 이름
        
        # 시험 관리 데이터
        self.exams = {}  # {exam_name: {pdfs: [], subjects: []}}
        self.exam_names = []  # 시험 이름 목록
        
        # PDF 중복 체크를 위한 해시 저장소
        self.pdf_hashes = {}  # {exam_name: {filename: hash}}
        
        # 오답노트 데이터
        self.wrong_answers = {}  # {exam_name: {question_hash: {question, answer, explanation, wrong_count, last_wrong_date, metadata}}}
        
        # 문제 중복 방지를 위한 히스토리
        self.question_history = {}  # {exam_name: [question_numbers]}
        self.recent_questions = {}  # {exam_name: [recent_question_numbers]}
        
        # 랜덤 시드 초기화 (매번 다른 시드 사용)
        # 완전한 랜덤 시드 생성
        current_time = int(time.time() * 1000000)  # 마이크로초 단위
        process_id = os.getpid()
        unique_id = int(uuid.uuid4().hex[:8], 16)  # UUID의 일부를 정수로 변환
        random_offset = random.randint(1, 999999)
        
        seed = current_time + process_id + unique_id + random_offset
        random.seed(seed)
        logger.info(f"🎲 [초기화] 완전 랜덤 시드 설정: {seed} (시간: {current_time}, PID: {process_id}, UUID: {unique_id}, 오프셋: {random_offset})")
        
        # 시험 데이터 및 PDF 해시 정보 로드
        self._load_exam_data()
        self._load_pdf_hashes()
        self._load_wrong_answers()
        
    def add_exam(self, exam_name: str) -> tuple[str, gr.Dropdown]:
        """새로운 시험 추가"""
        if not exam_name.strip():
            return "❌ 시험 이름을 입력해주세요.", gr.Dropdown(choices=self.get_exam_list())
        
        if exam_name in self.exams:
            return f"❌ '{exam_name}' 시험이 이미 존재합니다.", gr.Dropdown(choices=self.get_exam_list())
        
        self.exams[exam_name] = {
            "pdfs": [],
            "subjects": [],
            "created_at": datetime.now().isoformat()
        }
        self.exam_names.append(exam_name)
        
        # 시험 데이터 저장
        self._save_exam_data()
        
        logger.info(f"✅ [콘솔 로그] 새 시험 추가: {exam_name}")
        return f"✅ '{exam_name}' 시험이 추가되었습니다.", gr.Dropdown(choices=self.get_exam_list())
    
    def remove_exam(self, exam_name: str) -> tuple[str, gr.Dropdown]:
        """시험 제거 (모든 관련 파일 포함)"""
        if exam_name not in self.exams:
            return f"❌ '{exam_name}' 시험을 찾을 수 없습니다.", gr.Dropdown(choices=self.get_exam_list())
        
        try:
            # 1. extracted_questions 폴더에서 관련 파일들 삭제
            questions_dir = Path("extracted_questions")
            if questions_dir.exists():
                for file_path in questions_dir.glob(f"*{exam_name}*"):
                    try:
                        file_path.unlink()
                        logger.info(f"🗑️ 삭제된 파일: {file_path}")
                    except Exception as e:
                        logger.warning(f"⚠️ 파일 삭제 실패 {file_path}: {e}")
            
            # 2. 벡터 DB에서 해당 시험 데이터 삭제
            try:
                vector_store.delete_exam_data(exam_name)
            except Exception as e:
                logger.warning(f"⚠️ 벡터 DB 삭제 실패: {e}")
            
            # 3. 메모리 데이터 삭제
            del self.exams[exam_name]
            self.exam_names.remove(exam_name)
            
            # 4. PDF 해시도 제거
            if exam_name in self.pdf_hashes:
                del self.pdf_hashes[exam_name]
            
            # 5. 오답노트도 제거
            if exam_name in self.wrong_answers:
                del self.wrong_answers[exam_name]
            
            # 6. 문제 히스토리도 제거
            if exam_name in self.recent_questions:
                del self.recent_questions[exam_name]
            
            # 7. 모든 데이터 파일 저장 (업데이트된 상태로)
            self._save_exam_data()
            self._save_pdf_hashes()
            self._save_wrong_answers()
            
            logger.info(f"✅ [콘솔 로그] 시험 완전 제거: {exam_name}")
            return f"✅ '{exam_name}' 시험이 완전히 제거되었습니다.\n\n🗑️ 삭제된 항목:\n- extracted_questions 폴더의 관련 파일들\n- 벡터 DB 데이터\n- PDF 해시 정보\n- 오답노트 데이터\n- 문제 히스토리", gr.Dropdown(choices=self.get_exam_list())
            
        except Exception as e:
            logger.error(f"❌ 시험 제거 중 오류: {e}")
            return f"❌ 시험 제거 중 오류가 발생했습니다: {e}", gr.Dropdown(choices=self.get_exam_list())
    
    def get_exam_list(self) -> List[str]:
        """시험 목록 반환"""
        return self.exam_names
    
    def get_exam_info(self, exam_name: str) -> Dict[str, Any]:
        """시험 정보 반환"""
        if exam_name not in self.exams:
            return {}
        return self.exams[exam_name]
    
    def get_exam_pdfs(self, exam_name: str) -> List[Dict[str, Any]]:
        """시험에 업로드된 PDF 목록 반환"""
        if exam_name not in self.exams:
            return []
        return self.exams[exam_name].get("pdfs", [])
    
    def format_pdf_list(self, exam_name: str) -> str:
        """시험의 PDF 목록을 포맷된 문자열로 반환"""
        logger.debug(f"🔍 [DEBUG] format_pdf_list 호출 - 시험: {exam_name}")
        logger.debug(f"🔍 [DEBUG] 현재 exams 데이터: {self.exams}")
        
        # 시험이 존재하지 않는 경우 처리
        if not exam_name or exam_name not in self.exams:
            return "❌ 선택된 시험이 존재하지 않습니다. 시험을 다시 선택해주세요."
        
        pdfs = self.get_exam_pdfs(exam_name)
        logger.debug(f"🔍 [DEBUG] 가져온 PDF 목록: {pdfs}")
        
        if not pdfs:
            return "📄 업로드된 PDF가 없습니다."
        
        result = f"📚 {exam_name} - 업로드된 PDF 목록:\n\n"
        for i, pdf in enumerate(pdfs, 1):
            uploaded_date = pdf.get("uploaded_at", "").split("T")[0] if pdf.get("uploaded_at") else "날짜 없음"
            result += f"{i}. 📄 {pdf.get('filename', '알 수 없는 파일')}\n"
            result += f"   📊 청크 수: {pdf.get('chunks_count', 0)}개\n"
            result += f"   📅 업로드: {uploaded_date}\n\n"
        
        logger.debug(f"🔍 [DEBUG] 생성된 결과: {result}")
        return result
    
    def update_exam_list(self):
        """시험 목록 업데이트 (Gradio용)"""
        return gr.Dropdown(choices=self.get_exam_list())
    
    def calculate_pdf_hash(self, pdf_file) -> str:
        """PDF 파일의 해시값 계산"""
        
        try:
            if isinstance(pdf_file, str):
                # 파일 경로인 경우, 파일을 읽어서 해시 계산
                with open(pdf_file, 'rb') as f:
                    content = f.read()
            elif hasattr(pdf_file, 'read'):
                # 파일 객체인 경우
                pdf_file.seek(0)  # 파일 포인터를 처음으로
                content = pdf_file.read()
            else:
                # 바이트 데이터인 경우
                content = pdf_file
            
            # SHA-256 해시 계산
            return hashlib.sha256(content).hexdigest()
        except Exception as e:
            logger.warning(f"⚠️ 해시 계산 중 오류: {e}")
            # 오류 발생 시 파일명 기반 해시 생성
            if isinstance(pdf_file, str):
                return hashlib.sha256(pdf_file.encode('utf-8')).hexdigest()
            else:
                return hashlib.sha256(str(pdf_file).encode('utf-8')).hexdigest()
    
    def is_pdf_duplicate(self, exam_name: str, filename: str, pdf_hash: str) -> bool:
        """PDF 중복 체크"""
        if exam_name not in self.pdf_hashes:
            return False
        
        # 파일명과 해시 모두 체크
        for stored_filename, stored_hash in self.pdf_hashes[exam_name].items():
            if stored_filename == filename or stored_hash == pdf_hash:
                return True
        
        return False
    
    def _save_pdf_hashes(self):
        """PDF 해시 정보 저장"""
        try:
            hash_file = Path("pdf_hashes.json")
            with open(hash_file, 'w', encoding='utf-8') as f:
                json.dump(self.pdf_hashes, f, ensure_ascii=False, indent=2)
            logger.info("✅ PDF 해시 정보 저장 완료")
        except Exception as e:
            logger.error(f"❌ PDF 해시 정보 저장 실패: {e}")
    
    def _load_exam_data(self):
        """시험 데이터 로드"""
        try:
            exam_file = Path("exam_data.json")
            if exam_file.exists():
                with open(exam_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.exams = data.get("exams", {})
                    self.exam_names = data.get("exam_names", [])
                logger.info(f"✅ 시험 데이터 로드 완료: {len(self.exam_names)}개 시험")
            else:
                logger.info("📄 시험 데이터 파일이 없습니다. 새로 생성합니다.")
        except Exception as e:
            logger.error(f"❌ 시험 데이터 로드 실패: {e}")
            self.exams = {}
            self.exam_names = []
    
    def _save_exam_data(self):
        """시험 데이터 저장"""
        try:
            exam_file = Path("exam_data.json")
            data = {
                "exams": self.exams,
                "exam_names": self.exam_names,
                "last_updated": datetime.now().isoformat()
            }
            with open(exam_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info("✅ 시험 데이터 저장 완료")
        except Exception as e:
            logger.error(f"❌ 시험 데이터 저장 실패: {e}")
    
    def _load_pdf_hashes(self):
        """PDF 해시 정보 로드"""
        try:
            hash_file = Path("pdf_hashes.json")
            if hash_file.exists():
                with open(hash_file, 'r', encoding='utf-8') as f:
                    self.pdf_hashes = json.load(f)
                logger.info(f"✅ PDF 해시 정보 로드 완료: {len(self.pdf_hashes)}개 시험")
            else:
                logger.info("📄 PDF 해시 정보 파일이 없습니다. 새로 생성합니다.")
        except Exception as e:
            logger.error(f"❌ PDF 해시 정보 로드 실패: {e}")
            self.pdf_hashes = {}
        
    def _load_wrong_answers(self):
        """오답노트 데이터 로드"""
        try:
            wrong_answers_file = Path("wrong_answers.json")
            if wrong_answers_file.exists():
                with open(wrong_answers_file, 'r', encoding='utf-8') as f:
                    self.wrong_answers = json.load(f)
                print(f"✅ 오답노트 데이터 로드 완료: {len(self.wrong_answers)}개 시험")
            else:
                print("📄 오답노트 데이터 파일이 없습니다. 새로 생성합니다.")
        except Exception as e:
            print(f"❌ 오답노트 데이터 로드 실패: {e}")
            self.wrong_answers = {}
        
    def upload_pdf(self, pdf_file, exam_name: str) -> tuple[str, gr.Dropdown]:
        """PDF 파일 업로드 및 벡터 DB 구축"""
        if pdf_file is None:
            return "❌ PDF 파일을 선택해주세요.", gr.Dropdown(choices=self.get_exam_list())
        
        if not exam_name.strip():
            return "❌ 시험 이름을 입력해주세요.", gr.Dropdown(choices=self.get_exam_list())
        
        # 시험이 없으면 자동 생성
        if exam_name not in self.exams:
            self.exams[exam_name] = {
                "pdfs": [],
                "subjects": [],
                "created_at": datetime.now().isoformat()
            }
            self.exam_names.append(exam_name)
            # 시험 데이터 저장
            self._save_exam_data()
        
        try:
            # PDF 해시 계산
            pdf_hash = self.calculate_pdf_hash(pdf_file)
            
            # Gradio에서 전달된 파일 객체 처리
            filename = "uploaded_file.pdf"  # 기본값
            actual_filename = "uploaded_file.pdf"  # 실제 파일명 (사용자에게 표시용)
            actual_file = pdf_file
            
            # 파일 객체 디버깅
            print(f"🔍 [DEBUG] PDF 파일 객체 타입: {type(pdf_file)}")
            
            try:
                # filepath 타입에서는 파일 경로가 직접 전달됨
                if isinstance(pdf_file, str):
                    # 파일 경로에서 파일명 추출
                    filename = Path(pdf_file).name
                    actual_filename = filename
                    actual_file = pdf_file
                elif isinstance(pdf_file, tuple) and len(pdf_file) >= 2:
                    # 튜플 형태로 전달된 경우
                    filename = str(pdf_file[1])  # 두 번째 요소가 파일명
                    actual_filename = filename  # 실제 파일명 저장
                    actual_file = pdf_file[0]  # 첫 번째 요소가 파일 경로
                else:
                    # 기타 경우, 시험 이름과 타임스탬프로 생성
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"{exam_name}_{timestamp}.pdf"
                    actual_filename = f"{exam_name}_기출문제.pdf"
            except Exception as e:
                logger.warning(f"⚠️ 파일명 추출 중 오류: {e}")
                # 오류 발생 시 시험 이름과 타임스탬프로 생성
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{exam_name}_{timestamp}.pdf"
                actual_filename = f"{exam_name}_기출문제.pdf"
            
            print(f"📄 [PDF 업로드] 시험: {exam_name}, 파일: {filename}")
            
            # 중복 체크
            if self.is_pdf_duplicate(exam_name, filename, pdf_hash):
                return f"⚠️ 중복된 PDF 파일입니다!\n\n📊 기존 정보:\n- 시험: {exam_name}\n- 파일명: {filename}\n- 해시: {pdf_hash[:16]}...\n- 상태: 이미 벡터 DB에 저장됨\n\n✅ 기존 벡터 데이터를 재사용합니다. (처리 시간 단축)", gr.Dropdown(choices=self.get_exam_list())
            
            # 임시 파일로 저장
            temp_path = tempfile.mktemp(suffix='.pdf')
            
            try:
                if isinstance(actual_file, str):
                    # 파일 경로인 경우, 파일을 복사
                    import shutil
                    shutil.copy2(actual_file, temp_path)
                elif hasattr(actual_file, 'read') and callable(getattr(actual_file, 'read', None)):
                    # 파일 객체인 경우
                    if hasattr(actual_file, 'seek'):
                        actual_file.seek(0)  # type: ignore # 파일 포인터를 처음으로
                    with open(temp_path, 'wb') as f:
                        f.write(actual_file.read())  # type: ignore
                elif isinstance(actual_file, (bytes, bytearray)):
                    # 바이트 데이터인 경우
                    with open(temp_path, 'wb') as f:
                        f.write(actual_file)
                else:
                    # 기타 경우, 문자열로 변환하여 처리
                    with open(temp_path, 'wb') as f:
                        f.write(str(actual_file).encode('utf-8'))
            except Exception as e:
                logger.warning(f"⚠️ 파일 처리 중 오류: {e}")
                # 오류 발생 시 원본 파일 객체 사용
                if isinstance(pdf_file, str):
                    import shutil
                    shutil.copy2(pdf_file, temp_path)
                elif hasattr(pdf_file, 'read') and callable(getattr(pdf_file, 'read', None)):
                    if hasattr(pdf_file, 'seek'):
                        pdf_file.seek(0)
                    with open(temp_path, 'wb') as f:
                        f.write(pdf_file.read())
                elif isinstance(pdf_file, (bytes, bytearray)):
                    with open(temp_path, 'wb') as f:
                        f.write(pdf_file)
                else:
                    with open(temp_path, 'wb') as f:
                        f.write(str(pdf_file).encode('utf-8'))
            
            # PDF 처리 (실제 파일명 전달)
            result = pdf_processor.process_pdf(temp_path, exam_name, actual_filename)
            
            # 임시 파일 삭제
            os.unlink(temp_path)
            
            if result["success"]:
                # 시험 정보 업데이트 (실제 파일명 사용)
                pdf_info = {
                    "filename": actual_filename,
                    "chunks_count": result["chunks_count"],
                    "uploaded_at": datetime.now().isoformat()
                }
                self.exams[exam_name]["pdfs"].append(pdf_info)
                
                print(f"✅ [DEBUG] PDF 정보 추가: {pdf_info}")
                print(f"✅ [DEBUG] 현재 {exam_name}의 PDF 개수: {len(self.exams[exam_name]['pdfs'])}")
                
                # PDF 해시 저장 (실제 파일명 사용)
                if exam_name not in self.pdf_hashes:
                    self.pdf_hashes[exam_name] = {}
                self.pdf_hashes[exam_name][actual_filename] = pdf_hash
                
                # 해시 정보 영구 저장
                self._save_pdf_hashes()
                
                # 시험 데이터 저장
                self._save_exam_data()
                print(f"✅ [DEBUG] 시험 데이터 저장 완료")
                
                return f"✅ PDF 업로드 완료!\n\n📊 처리 결과:\n- 시험: {exam_name}\n- 파일명: {actual_filename}\n- 생성된 청크: {result['chunks_count']}개\n- 추출된 문제: {result.get('questions_count', 0)}개\n- 해시: {pdf_hash[:16]}...\n\n📝 추출된 문제는 'extracted_questions' 폴더에 저장되었습니다.\n이제 기출문제 기반 문제 생성이 가능합니다.", gr.Dropdown(choices=self.get_exam_list())
            else:
                return f"❌ PDF 처리 실패: {result['error']}", gr.Dropdown(choices=self.get_exam_list())
                
        except Exception as e:
            error_msg = f"PDF 업로드 중 오류 발생: {e}"
            print(f"❌ {error_msg}")
            return error_msg, gr.Dropdown(choices=self.get_exam_list())
    
    def generate_question(self, exam_name: str, question_mode: str = "generate") -> str:
        """시험 문제 생성"""
        print(f"\n🔍 [콘솔 로그] 문제 생성 요청 - 시험: {exam_name}, 모드: {question_mode}")
        
        if not exam_name:
            return "❌ 시험을 선택해주세요."
        
        # 현재 선택된 시험 이름 저장
        self.current_exam_name = exam_name
        
        if not DEPLOYMENT_NAME:
            error_msg = "Error: DEPLOYMENT_NAME 환경 변수가 설정되지 않았습니다."
            print(f"❌ [콘솔 로그] {error_msg}")
            return error_msg
        
        # 랜덤하게 난이도와 문제 유형 선택 (매번 다른 랜덤화)
        import time
        import os
        import uuid
        
        # 매번 새로운 랜덤 시드 생성 (시간 + 프로세스 ID + UUID + 랜덤 값 + 추가 랜덤)
        current_time = int(time.time() * 1000000)  # 마이크로초 단위
        process_id = os.getpid()
        unique_id = int(uuid.uuid4().hex[:8], 16)  # UUID의 일부를 정수로 변환
        random_offset = random.randint(1, 999999)
        additional_random = random.randint(1000000, 9999999)  # 추가 랜덤 값
        
        seed = current_time + process_id + unique_id + random_offset + additional_random
        random.seed(seed)
        logger.info(f"🎲 [문제 생성] 매번 새로운 랜덤 시드 설정: {seed} (시간: {current_time}, PID: {process_id}, UUID: {unique_id}, 오프셋: {random_offset}, 추가: {additional_random})")
        
        # 난수 생성 테스트
        test_random1 = random.randint(1, 1000)
        test_random2 = random.randint(1, 1000)
        logger.info(f"🎲 [문제 생성] 난수 테스트: {test_random1}, {test_random2}")
        
        difficulty = random.choice(self.difficulties)
        question_type = random.choice(self.question_types)
        
        print(f"📊 [콘솔 로그] 선택된 난이도: {difficulty}, 문제 유형: {question_type}")
        
        # RAG 기반 문제 생성
        if question_mode == "generate":
            # 다양한 검색 쿼리 생성
            search_queries = [
                f"{difficulty} {question_type}",
                f"{question_type} {difficulty}",
                f"{exam_name} {difficulty}",
                f"{exam_name} {question_type}",
                f"기출문제 {difficulty}",
                f"기출문제 {question_type}",
                f"{difficulty} 문제",
                f"{question_type} 문제"
            ]
            
            # 랜덤하게 검색 쿼리 선택 (더 나은 랜덤화)
            random.shuffle(search_queries)
            search_query = search_queries[0]
            print(f"🔍 [콘솔 로그] 검색 쿼리: {search_query}")
            
            # 벡터 DB에서 검색
            similar_questions = vector_store.search_similar_questions(search_query, subject=exam_name, n_results=5)
            
            # 추출된 문제에서도 검색
            extracted_questions = pdf_processor.search_extracted_questions_semantic(search_query, exam_name, n_results=3)
            
            # 결과 합치기
            all_questions = similar_questions + extracted_questions
            # 점수 기준으로 정렬
            all_questions = sorted(all_questions, key=lambda x: x.get('score', 0), reverse=True)[:5]
            
            # 그림 포함 문제 필터링
            filtered_questions = []
            for question in all_questions:
                question_content = question.get("content", "")
                # 그림 관련 키워드 체크
                if any(keyword in question_content for keyword in ["그림", "도표", "차트", "이미지", "사진", "화면", "스크린샷"]):
                    print(f"⚠️ [콘솔 로그] RAG 그림 포함 문제 필터링")
                    continue
                filtered_questions.append(question)
            
            if not filtered_questions:
                print(f"⚠️ [콘솔 로그] RAG 필터링 후 문제가 없어 일반 생성으로 전환")
                prompt = ExamPrompts.get_question_generation_prompt(
                    exam_name, difficulty, question_type, exam_name
                )
                self.current_context = None
                self.current_metadata = None
                print("🔄 [콘솔 로그] 일반 문제 생성 중...")
            
            all_questions = filtered_questions
            
            if all_questions:
                # 컨텍스트 구성
                context = "\n\n".join([q["content"] for q in all_questions])
                self.current_context = context
                # 메타데이터 저장 (출처 정보용)
                self.current_metadata = [q["metadata"] for q in all_questions]
                
                # 컨텍스트 품질 검증
                print("🔍 [콘솔 로그] 컨텍스트 품질 검증 중...")
                validation_result = self.validate_context(context, self.current_metadata)
                
                if validation_result["valid"]:
                    print("✅ [콘솔 로그] 컨텍스트 검증 통과")
                    prompt = ExamPrompts.get_rag_question_generation_prompt(
                        exam_name, difficulty, question_type, context, exam_name, self.current_metadata
                    )
                    print("🔄 [콘솔 로그] RAG 기반 문제 생성 중...")
                else:
                    print(f"⚠️ [콘솔 로그] 컨텍스트 검증 실패: {validation_result['reason']}")
                    # 다른 검색 쿼리로 재시도
                    alternative_queries = [
                        f"{exam_name} 기출문제",
                        f"{difficulty} {question_type} 문제",
                        f"{question_type} 문제",
                        f"{difficulty} 문제"
                    ]
                    
                    alt_validation_success = False
                    for alt_query in alternative_queries:
                        print(f"🔍 [콘솔 로그] 대체 검색 쿼리 시도: {alt_query}")
                        
                        # 벡터 DB에서 검색
                        alt_questions = vector_store.search_similar_questions(alt_query, subject=exam_name, n_results=3)
                        
                        # 추출된 문제에서도 검색
                        alt_extracted = pdf_processor.search_extracted_questions_semantic(alt_query, exam_name, n_results=2)
                        
                        # 결과 합치기
                        alt_all_questions = alt_questions + alt_extracted
                        alt_all_questions = sorted(alt_all_questions, key=lambda x: x.get('score', 0), reverse=True)[:3]
                        
                        if alt_all_questions:
                            alt_context = "\n\n".join([q["content"] for q in alt_all_questions])
                            alt_metadata = [q["metadata"] for q in alt_all_questions]
                            
                            alt_validation = self.validate_context(alt_context, alt_metadata)
                            if alt_validation["valid"]:
                                print("✅ [콘솔 로그] 대체 컨텍스트 검증 통과")
                                self.current_context = alt_context
                                self.current_metadata = alt_metadata
                                prompt = ExamPrompts.get_rag_question_generation_prompt(
                                    exam_name, difficulty, question_type, alt_context, exam_name, alt_metadata
                                )
                                print("🔄 [콘솔 로그] 대체 RAG 기반 문제 생성 중...")
                                alt_validation_success = True
                                break
                    
                    if not alt_validation_success:
                        print("⚠️ [콘솔 로그] 모든 컨텍스트 검증 실패, 일반 생성으로 전환")
                        prompt = ExamPrompts.get_question_generation_prompt(
                            exam_name, difficulty, question_type, exam_name
                        )
                        self.current_context = None
                        self.current_metadata = None
                        print("🔄 [콘솔 로그] 일반 문제 생성 중...")
            else:
                # RAG 결과가 없으면 일반 생성
                prompt = ExamPrompts.get_question_generation_prompt(
                    exam_name, difficulty, question_type, exam_name
                )
                self.current_context = None
                self.current_metadata = None
                print("🔄 [콘솔 로그] 일반 문제 생성 중...")
        
        elif question_mode == "exact":
            # 추출된 기출문제에서 랜덤 선택 (모든 PDF에서 균등하게 선택)
            print(f"🔍 [콘솔 로그] 추출된 기출문제에서 선택 중...")
            
            # 추출된 문제 목록 조회
            extracted_questions = pdf_processor.get_extracted_questions(exam_name)
            
            if not extracted_questions:
                print(f"❌ [콘솔 로그] {exam_name} 시험의 추출된 문제가 없습니다.")
                return "❌ 해당 시험의 추출된 기출문제가 없습니다. PDF를 먼저 업로드해주세요."
            
            # 문제 필터링 (그림 포함 문제 제외) - 완화된 필터링
            filtered_questions = []
            for question in extracted_questions:
                question_text = question.get("text", "")
                # 그림 관련 키워드 체크 (더 정확한 필터링)
                # 단순히 "그림"이라는 단어만 있으면 필터링하지 않고, 실제로 그림이 필요한 문제만 필터링
                if any(keyword in question_text for keyword in ["다음 그림", "위의 그림", "아래 그림", "그림과 같이", "그림에서 보는 바와 같이"]):
                    print(f"⚠️ [콘솔 로그] 그림 포함 문제 필터링: {question.get('number', 'unknown')}번")
                    continue
                filtered_questions.append(question)
            
            if not filtered_questions:
                print(f"❌ [콘솔 로그] 필터링 후 사용 가능한 문제가 없습니다.")
                return "❌ 그림이 포함되지 않은 문제가 없습니다. 다른 PDF를 업로드해주세요."
            
            # 문제가 1개만 있을 때 경고
            if len(filtered_questions) == 1:
                print(f"⚠️ [콘솔 로그] 경고: 사용 가능한 문제가 1개만 있습니다. 항상 같은 문제가 출제될 수 있습니다.")
                print(f"⚠️ [콘솔 로그] 추천: 더 많은 PDF를 업로드하거나 'generate' 모드를 사용하세요.")
            
            print(f"✅ [콘솔 로그] 필터링 완료: {len(extracted_questions)}개 → {len(filtered_questions)}개")
            
            # PDF별로 문제 그룹화
            pdf_questions = {}
            for question in filtered_questions:
                source_file = question.get("source_file", "unknown")
                if source_file not in pdf_questions:
                    pdf_questions[source_file] = []
                pdf_questions[source_file].append(question)
            
            logger.info(f"📊 [문제 생성] PDF별 문제 분포: {[(pdf, len(questions)) for pdf, questions in pdf_questions.items()]}")
            
            # 각 PDF 내에서 문제를 랜덤하게 섞기 (매번 다른 순서)
            for pdf_file, questions in pdf_questions.items():
                # 추가 랜덤화를 위한 시드 재설정
                shuffle_seed = int(time.time() * 1000000) + random.randint(1, 999999)
                random.seed(shuffle_seed)
                random.shuffle(questions)
                logger.info(f"🎲 [문제 생성] {pdf_file} 문제 섞기 완료: {len(questions)}개 (시드: {shuffle_seed})")
            
            # 모든 PDF의 문제를 하나의 리스트로 합치고 완전히 랜덤하게 섞기
            all_questions = []
            for pdf_file, questions in pdf_questions.items():
                all_questions.extend(questions)
            
            # 완전히 랜덤하게 섞기 (매번 다른 순서)
            final_shuffle_seed = int(time.time() * 1000000) + random.randint(1, 999999)
            random.seed(final_shuffle_seed)
            random.shuffle(all_questions)
            logger.info(f"🎲 [문제 생성] 전체 문제 랜덤 섞기 완료: {len(all_questions)}개 문제 (시드: {final_shuffle_seed})")
            
            # 최근에 출제된 문제 목록 가져오기
            recent_questions = self.recent_questions.get(exam_name, [])
            logger.info(f"📝 [문제 생성] 최근 출제된 문제: {recent_questions}")
            
            # 중복되지 않는 문제 선택
            available_questions = []
            for question in all_questions:
                question_number = question["number"]
                source_file = question.get("source_file", "unknown")
                # 문제 번호와 출처 파일을 조합하여 고유 식별자 생성
                unique_id = f"{source_file}_{question_number}"
                if unique_id not in recent_questions:
                    available_questions.append(question)
            
            # 사용 가능한 문제가 있으면 랜덤 선택
            if available_questions:
                logger.info(f"🎲 [문제 생성] 사용 가능한 문제 수: {len(available_questions)}개")
                logger.info(f"🎲 [문제 생성] 사용 가능한 문제 목록: {[q['number'] for q in available_questions]}")
                
                # 랜덤 선택 전 시드 재설정
                choice_seed = int(time.time() * 1000000) + random.randint(1, 999999)
                random.seed(choice_seed)
                logger.info(f"🎲 [문제 생성] 문제 선택용 시드 설정: {choice_seed}")
                
                selected_question = random.choice(available_questions)
                question_number = selected_question["number"]
                source_file = selected_question.get("source_file", "unknown")
                unique_id = f"{source_file}_{question_number}"
                logger.info(f"✅ [문제 생성] 중복되지 않는 문제 선택: {question_number}번 (출처: {source_file})")
            else:
                # 모든 문제가 최근에 출제되었다면 최근 목록 초기화
                logger.info(f"🔄 [문제 생성] 모든 문제가 최근에 출제됨, 최근 목록 초기화")
                self.recent_questions[exam_name] = []
                
                logger.info(f"🎲 [문제 생성] 전체 문제 수: {len(all_questions)}개")
                logger.info(f"🎲 [문제 생성] 전체 문제 목록: {[q['number'] for q in all_questions]}")
                
                # 랜덤 선택 전 시드 재설정
                choice_seed = int(time.time() * 1000000) + random.randint(1, 999999)
                random.seed(choice_seed)
                logger.info(f"🎲 [문제 생성] 초기화 후 문제 선택용 시드 설정: {choice_seed}")
                
                selected_question = random.choice(all_questions)
                question_number = selected_question["number"]
                source_file = selected_question.get("source_file", "unknown")
                unique_id = f"{source_file}_{question_number}"
                logger.info(f"✅ [문제 생성] 초기화 후 문제 선택: {question_number}번 (출처: {source_file})")
            
            question_text = selected_question["text"]
            question_number = selected_question["number"]
            
            # 최근 출제 목록에 추가 (최대 10개 유지)
            if exam_name not in self.recent_questions:
                self.recent_questions[exam_name] = []
            self.recent_questions[exam_name].append(unique_id)
            if len(self.recent_questions[exam_name]) > 10:
                self.recent_questions[exam_name].pop(0)
            
            logger.info(f"✅ [문제 생성] 최종 선택된 문제: {question_number}번 (출처: {source_file}) (최근 출제: {len(self.recent_questions[exam_name])}개)")
            
            # 문제 정보를 컨텍스트로 설정
            self.current_context = question_text
            
            # 메타데이터 구성 (출처 정보용)
            # 선택된 문제의 실제 출처 PDF 파일명 사용
            actual_source_file = selected_question.get("source_file", "unknown")
            logger.info(f"📄 [문제 생성] 선택된 문제의 실제 출처: {actual_source_file}")
            
            # 실제 PDF 파일명이 있으면 그대로 사용, 없으면 전체 PDF 목록에서 찾기
            if actual_source_file != "unknown":
                pdf_source_display = actual_source_file
            else:
                # 전체 PDF 목록에서 찾기
                pdf_filenames = []
                if exam_name in self.exams:
                    pdfs = self.exams[exam_name].get("pdfs", [])
                    if pdfs:
                        pdf_filenames = [pdf.get("filename", "추출된 기출문제") for pdf in pdfs]
                        logger.info(f"📄 [문제 생성] 전체 PDF 목록: {pdf_filenames}")
                    else:
                        pdf_filenames = ["추출된 기출문제"]
                else:
                    pdf_filenames = ["추출된 기출문제"]
                
                # 출처 표시용 PDF 파일명
                if len(pdf_filenames) == 1:
                    pdf_source_display = pdf_filenames[0]
                else:
                    pdf_source_display = f"{len(pdf_filenames)}개 PDF 파일"
            
            metadata = {
                "type": "extracted_question",
                "subject": exam_name,
                "question_number": question_number,
                "pdf_source": pdf_source_display,
                "pdf_sources": [pdf_source_display],  # 실제 출처 PDF 파일명 저장
                "extraction_date": selected_question.get("extraction_date", ""),
                "start_line": selected_question.get("start_line", 0),
                "end_line": selected_question.get("end_line", 0)
            }
            self.current_metadata = [metadata]
            logger.info(f"📄 [문제 생성] 최종 PDF 출처: {pdf_source_display}")
            
            # 기출문제 그대로 출제 프롬프트 사용
            prompt = ExamPrompts.get_exact_question_prompt(question_text, exam_name)
            logger.info("🔄 [문제 생성] 추출된 기출문제 그대로 출제 중...")
        
        try:
            print("🤖 [콘솔 로그] Azure OpenAI API 호출 중...")
            response = openai.chat.completions.create(
                model=str(DEPLOYMENT_NAME),
                messages=[
                    {"role": "system", "content": ExamPrompts.get_system_prompts(exam_name)["question_generator"]},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
            )
            result = response.choices[0].message.content
            if result:
                # 결과를 파싱하여 저장
                self._parse_question_result(result)
                
                #Review Agent로 문제 검토
                print("🔍 [콘솔 로그] 문제 검토 시작...")
                review_result = review_agent.review_question(
                    question=self.current_question or "",
                    answer=self.current_answer or "",
                    explanation=self.current_explanation or "",
                    exam_name=exam_name
                )
                
                # 검토 결과에 따른 처리
                if not review_result.get("is_valid", False) and review_result.get("suggestions"):
                    print("⚠️ [콘솔 로그] 문제 검토에서 개선점 발견, 수정 적용 중...")
                    
                    # 수정 제안 적용
                    corrected_result = review_agent.apply_corrections(
                        question=self.current_question or "",
                        answer=self.current_answer or "",
                        explanation=self.current_explanation or "",
                        suggestions=review_result["suggestions"]
                    )
                    
                    if corrected_result:
                        # 수정된 문제로 업데이트
                        self.current_question = corrected_result.get("question", self.current_question)
                        self.current_answer = corrected_result.get("answer", self.current_answer)
                        self.current_explanation = corrected_result.get("explanation", self.current_explanation)
                        print("✅ [콘솔 로그] 문제 수정 완료")
                    else:
                        print("⚠️ [콘솔 로그] 문제 수정 실패, 원본 문제 사용")
                else:
                    print(f"✅ [콘솔 로그] 문제 검토 통과 (점수: {review_result.get('score', 0)})")
                
                question_only = self._get_question_only(self.current_question or result)
                print("✅ [콘솔 로그] 문제 생성 완료")
                print(f"📝 [콘솔 로그] 최종 문제:\n{question_only}")
                return question_only
            else:
                error_msg = "문제 생성에 실패했습니다."
                print(f"❌ [콘솔 로그] {error_msg}")
                return error_msg
        except Exception as e:
            error_msg = f"문제 생성 중 오류가 발생했습니다: {e}"
            print(f"❌ [콘솔 로그] {error_msg}")
            return error_msg
    
    def _parse_question_result(self, result: str):
        """문제 결과를 파싱하여 저장"""
        self.current_question = result
        
        # 정답과 해설 추출
        lines = result.split('\n')
        answer_section = False
        explanation_section = False
        
        for line in lines:
            if "=== 정답 ===" in line:
                answer_section = True
                explanation_section = False
                continue
            elif "=== 해설 ===" in line:
                answer_section = False
                explanation_section = True
                continue
            elif "===" in line:
                answer_section = False
                explanation_section = False
                continue
            
            if answer_section and line.strip():
                self.current_answer = line.strip()
            elif explanation_section and line.strip():
                if not self.current_explanation:
                    self.current_explanation = line.strip()
                else:
                    self.current_explanation += "\n" + line.strip()
        
        print(f"🔍 [콘솔 로그] 정답 파싱 완료: {self.current_answer}")
    
    def _get_question_only(self, result: str) -> str:
        """문제와 보기만 반환 (출처 정보 포함)"""
        lines = result.split('\n')
        question_lines = []
        include_line = True
        in_problem_info = False
        
        for line in lines:
            if "=== 정답 ===" in line:
                include_line = False
                break
            if "=== 문제 정보 ===" in line:
                in_problem_info = True
                question_lines.append(line)
                continue
            elif "===" in line and in_problem_info:
                in_problem_info = False
                question_lines.append(line)
                continue
            
            if include_line:
                # 불필요한 텍스트 제거
                if any(skip_text in line for skip_text in [
                    "아래와 같이 요청하신 형식에 맞추어 정리해드립니다",
                    "위 컨텍스트를 참고하여",
                    "위 기출문제를",
                    "다음 형식으로 응답해주세요",
                    "출처: 기출문제 기반",
                    "출처: 기출문제"
                ]):
                    continue
                
                question_lines.append(line)
                
                # 문제 정보 섹션에 출처 정보 추가
                if in_problem_info and "출처:" in line:
                    # 기존 출처 정보가 있으면 건너뛰기
                    continue
                elif in_problem_info and "유형:" in line:
                    # 유형 다음에 출처 정보 추가
                    source_info = self._get_source_display_info()
                    if source_info:
                        question_lines.append(f"출처: {source_info}")
        
        # Gradio Markdown에서 줄바꿈이 제대로 표시되도록 처리
        result_text = '\n'.join(question_lines)
        
        # === 문제 === 다음에 줄바꿈 추가
        result_text = result_text.replace("=== 문제 ===", "=== 문제 ===\n")
        result_text = result_text.replace("=== 보기 ===", "\n=== 보기 ===\n")
        
        return result_text
    
    def evaluate_answer(self, user_answer: str) -> str:
        """사용자 답변 평가"""
        print(f"\n💭 [콘솔 로그] 답변 평가 요청 - 사용자 답변: '{user_answer}'")
        
        if not self.current_question:
            error_msg = "먼저 문제를 생성해주세요."
            print(f"❌ [콘솔 로그] {error_msg}")
            return error_msg
        
        if not self.current_answer:
            error_msg = "정답 정보를 찾을 수 없습니다."
            print(f"❌ [콘솔 로그] {error_msg}")
            return error_msg
        
        # RAG 기반 평가
        if self.current_context:
            prompt = ExamPrompts.get_rag_answer_evaluation_prompt(
                self.current_question, user_answer, self.current_context, self.current_metadata or []
            )
        else:
            prompt = ExamPrompts.get_answer_evaluation_prompt(
                self.current_question, user_answer
            )
        
        try:
            print("🤖 [콘솔 로그] Azure OpenAI API 호출 중...")
            response = openai.chat.completions.create(
                model=str(DEPLOYMENT_NAME),
                messages=[
                    {"role": "system", "content": ExamPrompts.get_system_prompts(self.current_exam_name or "정보시스템감리사")["answer_evaluator"]},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
            )
            result = response.choices[0].message.content
            if result:
                # 오답 여부 확인 및 저장
                if self._is_wrong_answer(result) and self.current_exam_name:
                    print("❌ [콘솔 로그] 오답 감지, 오답노트에 저장")
                    self.add_wrong_answer(
                        exam_name=self.current_exam_name,
                        question_content=self.current_question or "",
                        correct_answer=self.current_answer or "",
                        explanation=self.current_explanation or "",
                        metadata=self.current_metadata[0] if self.current_metadata else {}
                    )
                
                print(f"✅ [콘솔 로그] 답변 평가 완료")
                return result
            else:
                error_msg = "답변 평가에 실패했습니다."
                print(f"❌ [콘솔 로그] {error_msg}")
                return error_msg
        except Exception as e:
            error_msg = f"답변 평가 중 오류가 발생했습니다: {e}"
            print(f"❌ [콘솔 로그] {error_msg}")
            return error_msg
    
    def _is_wrong_answer(self, evaluation_result: str) -> bool:
        """평가 결과에서 오답 여부 확인"""
        try:
            lines = evaluation_result.split('\n')
            for line in lines:
                if "정답 여부:" in line:
                    return "틀림" in line or "오답" in line or "부정확" in line
            return False
        except:
            return False
    
    def show_solution(self) -> str:
        """정답 및 해설 표시"""
        print("\n🔍 [콘솔 로그] 정답 및 해설 요청")
        
        if not self.current_question:
            error_msg = "먼저 문제를 생성해주세요."
            print(f"❌ [콘솔 로그] {error_msg}")
            return error_msg
        
        if not self.current_answer or not self.current_explanation:
            error_msg = "정답 및 해설 정보를 찾을 수 없습니다."
            print(f"❌ [콘솔 로그] {error_msg}")
            return error_msg
        
        # 출처 정보 추출 (메타데이터 기반)
        source_info = self._extract_source_info()
        
        # 정답을 볼드 처리
        bold_answer = f"**{self.current_answer}**"
        
        # 해설에서 중요한 부분을 볼드 처리
        explanation_lines = self.current_explanation.split('\n')
        bold_explanation_lines = []
        
        for line in explanation_lines:
            # 정답 번호나 키워드를 볼드 처리
            if any(keyword in line for keyword in ['정답', '답', '①', '②', '③', '④', '1)', '2)', '3)', '4)']):
                # 정답 번호를 볼드 처리
                import re
                line = re.sub(r'([①②③④1-4]\)?)', r'**\1**', line)
                bold_explanation_lines.append(line)
            else:
                bold_explanation_lines.append(line)
        
        bold_explanation = '\n'.join(bold_explanation_lines)
        
        solution = f"""
=== 정답 ===

{bold_answer}

=== 해설 ===

{bold_explanation}

=== 출처 ===

{source_info}
        """
        
        print(f"📖 [콘솔 로그] 정답 및 해설 표시:\n{solution.strip()}")
        return solution.strip()
    
    def _extract_source_info(self) -> str:
        """컨텍스트에서 출처 정보 추출"""
        try:
            # 메타데이터에서 출처 정보 추출 (중복 제거)
            unique_sources = set()
            question_numbers = []
            
            if self.current_metadata:
                for metadata in self.current_metadata:
                    # 추출된 문제인지 확인
                    if metadata.get("type") == "extracted_question":
                        question_number = metadata.get("question_number", "")
                        if question_number:
                            question_numbers.append(question_number)
                        
                        # 추출된 문제도 실제 PDF 파일명 사용
                        if metadata.get("pdf_source"):
                            pdf_filename = metadata.get("pdf_source")
                            unique_sources.add(pdf_filename)
                        else:
                            unique_sources.add("추출된 기출문제")  # 파일명이 없는 경우에만 generic 텍스트 사용
                        continue
                    
                    # PDF 소스 정보 추출 (실제 PDF 파일명 사용)
                    if metadata.get("pdf_source"):
                        # 실제 PDF 파일명 사용
                        pdf_filename = metadata.get("pdf_source")
                        unique_sources.add(pdf_filename)
                    elif metadata.get("pdf_sources"):
                        # 기존 방식 호환성
                        pdf_sources = metadata.get("pdf_sources", [])
                        if len(pdf_sources) == 1:
                            unique_sources.add(pdf_sources[0])
                        else:
                            unique_sources.add(f"{len(pdf_sources)}개 PDF 파일")
            
            # 출처 정보 구성
            if unique_sources:
                if len(unique_sources) == 1:
                    # 단일 출처
                    source = list(unique_sources)[0]
                    if question_numbers:
                        # 문제 번호가 있는 경우
                        if len(question_numbers) == 1:
                            return f"{source}, {question_numbers[0]}번 문제"
                        else:
                            question_str = ", ".join([f"{q}번" for q in question_numbers])
                            return f"{source}, {question_str} 문제"
                    else:
                        return f"{source}"
                else:
                    # 여러 출처
                    sources_list = list(unique_sources)
                    sources_str = ", ".join(sources_list)
                    return f"{sources_str}"
            
            # 컨텍스트에서 문제 번호 추출
            if self.current_context:
                problem_number = self._extract_problem_number_from_context()
                if problem_number:
                    return f"문제 번호: {problem_number}"
            
            # 메타데이터나 컨텍스트가 없는 경우
            if self.current_exam_name:
                return f"{self.current_exam_name} 기출문제"
            
            return "기출문제"
                
        except Exception as e:
            print(f"❌ 출처 정보 추출 중 오류: {e}")
            return "[출처 정보 추출 실패]"
    
    def _extract_problem_number_from_context(self) -> str:
        """컨텍스트에서 문제 번호 추출"""
        if not self.current_context:
            return ""
            
        try:
            lines = self.current_context.split('\n')
            
            # 문제 번호 패턴들 (우선순위 순서)
            patterns = [
                r'^(\d+)\s*번\s*[^\d]',  # 21번 다음에 숫자가 아닌 문자가 오는 경우 (줄 시작)
                r'^문제\s*(\d+)\s*[^\d]',  # 문제 21 다음에 숫자가 아닌 문자가 오는 경우 (줄 시작)
                r'^문항\s*(\d+)\s*[^\d]',  # 문항 21 다음에 숫자가 아닌 문자가 오는 경우 (줄 시작)
                r'^(\d+)\s*번$',  # 21번으로 끝나는 경우 (줄 시작)
                r'^문제\s*(\d+)$',  # 문제 21로 끝나는 경우 (줄 시작)
                r'^문항\s*(\d+)$',  # 문항 21로 끝나는 경우 (줄 시작)
                r'^(\d+)\.\s*[^\d]',  # 21. 다음에 숫자가 아닌 문자가 오는 경우 (줄 시작)
                r'\((\d+)\)',  # (21)
                r'문항\s*(\d+)',  # 문항 21
                r'문제\s*(\d+)',  # 문제 21
                r'(\d+)\s*번',  # 21번
            ]
            
            import re
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                for pattern in patterns:
                    match = re.search(pattern, line)
                    if match:
                        number = match.group(1)
                        # 연도가 아닌 문제 번호인지 확인 (1-100 범위)
                        if number.isdigit() and 1 <= int(number) <= 100:
                            print(f"🔍 [DEBUG] 문제 번호 추출: {number} (패턴: {pattern}, 라인: {line})")
                            return number
            
            return ""
            
        except Exception as e:
            print(f"❌ 문제 번호 추출 중 오류: {e}")
            return ""
    
    def _get_source_display_info(self) -> str:
        """출처 표시 정보 생성"""
        try:
            logger.debug(f"🔍 [DEBUG] _get_source_display_info 호출")
            logger.debug(f"🔍 [DEBUG] current_metadata: {self.current_metadata}")
            logger.debug(f"🔍 [DEBUG] current_exam_name: {self.current_exam_name}")
            logger.debug(f"🔍 [DEBUG] current_context: {self.current_context[:200] if self.current_context else 'None'}...")

            if not self.current_metadata:
                # 메타데이터가 없는 경우
                if self.current_exam_name:
                    problem_number = self._extract_problem_number_from_context()
                    if problem_number:
                        return f"{self.current_exam_name}, {problem_number}번 문제"
                    else:
                        return self.current_exam_name
                return ""

            # 출처별 문제 번호 수집 (중복 제거)
            source_problems = {}  # {source: [problem_numbers]}

            for i, metadata in enumerate(self.current_metadata):
                logger.debug(f"🔍 [DEBUG] 메타데이터 {i}: {metadata}")

                # 추출된 문제인지 확인
                if metadata.get("type") == "extracted_question":
                    question_number = metadata.get("question_number", "")
                    # 실제 PDF 파일명 사용
                    pdf_filename = metadata.get("pdf_source", "")
                    if pdf_filename:
                        source_key = pdf_filename
                    else:
                        source_key = "추출된 기출문제"
                    
                    if question_number:
                        if source_key not in source_problems:
                            source_problems[source_key] = []
                        source_problems[source_key].append(question_number)
                        continue

                # 기존 PDF 소스 처리
                pdf_filename = metadata.get("pdf_source", "")
                logger.debug(f"🔍 [DEBUG] PDF 파일명: {pdf_filename}")

                if pdf_filename:
                    # 파일명 그대로 사용
                    source_key = pdf_filename

                    # 문제 번호 추출 (컨텍스트에서)
                    problem_number = self._extract_problem_number_from_context()
                    if problem_number:
                        if source_key not in source_problems:
                            source_problems[source_key] = []
                        source_problems[source_key].append(problem_number)

            # 출처 정보 조합
            if source_problems:
                source_info_list = []

                for source, problem_numbers in source_problems.items():
                    # 중복 제거 및 정렬
                    unique_problems = sorted(list(set(problem_numbers)))

                    if len(unique_problems) == 1:
                        # 단일 문제 번호
                        source_info_list.append(f"{source}, {unique_problems[0]}번 문제")
                    else:
                        # 여러 문제 번호
                        problem_str = ", ".join([f"{p}번" for p in unique_problems])
                        source_info_list.append(f"{source}, {problem_str} 문제")

                if len(source_info_list) == 1:
                    result = source_info_list[0]
                else:
                    result = "; ".join(source_info_list)

                logger.debug(f"🔍 [DEBUG] 최종 출처 정보: {result}")
                return result

            # 출처 정보가 없는 경우 - 파일명만 반환
            if self.current_metadata and self.current_metadata[0].get("pdf_source"):
                pdf_filename = self.current_metadata[0].get("pdf_source")
                logger.debug(f"🔍 [DEBUG] 파일명만 반환: {pdf_filename}")
                return pdf_filename or ""

            # 메타데이터도 없는 경우
            if self.current_exam_name:
                logger.debug(f"🔍 [DEBUG] 시험명만 반환: {self.current_exam_name}")
                return self.current_exam_name

            return ""
        except Exception as e:
            logger.error(f"❌ 출처 정보 생성 중 오류: {e}")
            return ""
    
    def chat_with_ai(self, message: str, history: List[Dict[str, str]]) -> tuple[List[Dict[str, str]], str]:
        """AI와의 일반적인 대화"""
        print(f"\n💬 [콘솔 로그] AI 챗봇 메시지: '{message}'")
        
        if not DEPLOYMENT_NAME:
            error_msg = "Error: DEPLOYMENT_NAME 환경 변수가 설정되지 않았습니다."
            print(f"❌ [콘솔 로그] {error_msg}")
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": error_msg})
            return history, ""
        
        # RAG 기반 대화 시도
        try:
            # 관련 컨텍스트 검색
            similar_chunks = vector_store.search_similar_questions(message, subject=None, n_results=2)
            context = ""
            if similar_chunks:
                context = "\n\n".join([chunk["content"] for chunk in similar_chunks])
            
            if context:
                prompt = ChatPrompts.get_rag_conversation_prompt(message, context, history)
                print("🔄 [콘솔 로그] RAG 기반 대화 중...")
            else:
                prompt = ChatPrompts.get_conversation_prompt(message, history)
                print("🔄 [콘솔 로그] 일반 대화 중...")
            
            print("🤖 [콘솔 로그] Azure OpenAI API 호출 중...")
            response = openai.chat.completions.create(
                model=str(DEPLOYMENT_NAME),
                messages=[
                    {"role": "system", "content": ExamPrompts.get_system_prompts("정보시스템감리사")["chat_assistant"]},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
            )
            ai_response = response.choices[0].message.content
            if ai_response:
                print(f"🤖 [콘솔 로그] AI 응답: {ai_response}")
                history.append({"role": "user", "content": message})
                history.append({"role": "assistant", "content": ai_response})
            else:
                error_msg = "응답을 생성할 수 없습니다."
                print(f"❌ [콘솔 로그] {error_msg}")
                history.append({"role": "user", "content": message})
                history.append({"role": "assistant", "content": error_msg})
            return history, ""
        except Exception as e:
            error_msg = f"오류가 발생했습니다: {e}"
            print(f"❌ [콘솔 로그] {error_msg}")
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": error_msg})
            return history, ""
    
    def validate_context(self, context: str, metadata: Optional[List] = None) -> dict:
        """컨텍스트 품질 검증"""
        logger.info(f"🔍 [콘솔 로그] 컨텍스트 검증 시작")
        
        if not DEPLOYMENT_NAME:
            return {"valid": False, "reason": "DEPLOYMENT_NAME 환경 변수가 설정되지 않았습니다."}
        
        try:
            prompt = ExamPrompts.get_context_validation_prompt(context, metadata or [])
            
            logger.info("🤖 [콘솔 로그] 컨텍스트 검증 API 호출 중...")
            response = openai.chat.completions.create(
                model=str(DEPLOYMENT_NAME),
                messages=[
                    {"role": "system", "content": "당신은 기출문제 컨텍스트 품질 검증 전문가입니다. 정확하고 객관적인 검증을 해주세요."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
            )
            result = response.choices[0].message.content
            
            if result:
                # 검증 결과 파싱
                validation_result = self._parse_validation_result(result)
                logger.info(f"✅ [콘솔 로그] 컨텍스트 검증 완료: {validation_result}")
                return validation_result
            else:
                return {"valid": False, "reason": "검증 결과를 생성할 수 없습니다."}
                
        except Exception as e:
            error_msg = f"컨텍스트 검증 중 오류 발생: {e}"
            logger.error(f"❌ [콘솔 로그] {error_msg}")
            return {"valid": False, "reason": error_msg}
    
    def _parse_validation_result(self, result: str) -> dict:
        """검증 결과 파싱"""
        try:
            lines = result.split('\n')
            validation_result = {
                "valid": False,
                "problem_number": "",
                "question_type": "",
                "options_count": "",
                "issues": [],
                "suggestions": []
            }
            
            in_validation_section = False
            in_issues_section = False
            in_suggestions_section = False
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                if "=== 검증 결과 ===" in line:
                    in_validation_section = True
                    in_issues_section = False
                    in_suggestions_section = False
                    continue
                elif "=== 문제점 ===" in line:
                    in_validation_section = False
                    in_issues_section = True
                    in_suggestions_section = False
                    continue
                elif "=== 개선 제안 ===" in line:
                    in_validation_section = False
                    in_issues_section = False
                    in_suggestions_section = True
                    continue
                elif "===" in line:
                    in_validation_section = False
                    in_issues_section = False
                    in_suggestions_section = False
                    continue
                
                if in_validation_section:
                    if "적합성:" in line:
                        validation_result["valid"] = "적합" in line
                    elif "문제 번호:" in line:
                        validation_result["problem_number"] = line.split(":", 1)[1].strip()
                    elif "문제 유형:" in line:
                        validation_result["question_type"] = line.split(":", 1)[1].strip()
                    elif "보기 개수:" in line:
                        validation_result["options_count"] = line.split(":", 1)[1].strip()
                elif in_issues_section and line:
                    validation_result["issues"].append(line)
                elif in_suggestions_section and line:
                    validation_result["suggestions"].append(line)
            
            return validation_result
            
        except Exception as e:
            print(f"❌ 검증 결과 파싱 중 오류: {e}")
            return {"valid": False, "reason": f"검증 결과 파싱 실패: {e}"}
    
    def _save_wrong_answers(self):
        """오답노트 데이터 저장"""
        try:
            wrong_answers_file = Path("wrong_answers.json")
            with open(wrong_answers_file, 'w', encoding='utf-8') as f:
                json.dump(self.wrong_answers, f, ensure_ascii=False, indent=2)
            print("✅ 오답노트 데이터 저장 완료")
        except Exception as e:
            print(f"❌ 오답노트 데이터 저장 실패: {e}")
    
    def calculate_question_hash(self, question_content: str) -> str:
        """문제 내용의 해시값 계산"""
        return hashlib.sha256(question_content.encode('utf-8')).hexdigest()
    
    def add_wrong_answer(self, exam_name: str, question_content: str, correct_answer: str, explanation: str, metadata: Optional[Dict[str, Any]] = None):
        """오답 추가"""
        if not exam_name or not question_content:
            return
        
        question_hash = self.calculate_question_hash(question_content)
        
        # 메타데이터에 모든 PDF 파일명 추가
        enhanced_metadata = metadata or {}
        if exam_name in self.exams:
            pdfs = self.exams[exam_name].get("pdfs", [])
            if pdfs:
                # 모든 PDF 파일명 수집
                pdf_filenames = [pdf.get("filename", "추출된 기출문제") for pdf in pdfs]
                
                # 출처 표시용 PDF 파일명 (모든 PDF 포함)
                if len(pdf_filenames) == 1:
                    pdf_source_display = pdf_filenames[0]
                else:
                    pdf_source_display = f"{len(pdf_filenames)}개 PDF 파일"
                
                enhanced_metadata["pdf_source"] = pdf_source_display
                enhanced_metadata["pdf_sources"] = pdf_filenames  # 모든 PDF 파일명 저장
                logger.info(f"📄 [오답 추가] PDF 출처 설정: {pdf_source_display} (총 {len(pdf_filenames)}개)")
        
        if exam_name not in self.wrong_answers:
            self.wrong_answers[exam_name] = {}
        
        if question_hash in self.wrong_answers[exam_name]:
            # 기존 오답인 경우 횟수 증가
            self.wrong_answers[exam_name][question_hash]["wrong_count"] += 1
            self.wrong_answers[exam_name][question_hash]["last_wrong_date"] = datetime.now().isoformat()
        else:
            # 새로운 오답인 경우 추가
            self.wrong_answers[exam_name][question_hash] = {
                "question": question_content,
                "answer": correct_answer,
                "explanation": explanation,
                "wrong_count": 1,
                "last_wrong_date": datetime.now().isoformat(),
                "metadata": enhanced_metadata
            }
        
        # 오답 데이터 저장
        self._save_wrong_answers()
        logger.info(f"✅ 오답 추가 완료: {exam_name}, 해시: {question_hash[:8]}...")
    
    def get_wrong_answers(self, exam_name: str) -> list:
        """시험별 오답 목록 조회"""
        if exam_name not in self.wrong_answers:
            return []
        
        wrong_answers_list = []
        for question_hash, data in self.wrong_answers[exam_name].items():
            wrong_answers_list.append({
                "hash": question_hash,
                "question": data["question"],
                "answer": data["answer"],
                "explanation": data["explanation"],
                "wrong_count": data["wrong_count"],
                "last_wrong_date": data["last_wrong_date"],
                "metadata": data.get("metadata", {})
            })
        
        # 최근 오답 순으로 정렬
        wrong_answers_list.sort(key=lambda x: x["last_wrong_date"], reverse=True)
        return wrong_answers_list
    
    def get_wrong_answer_by_hash(self, exam_name: str, question_hash: str) -> dict:
        """해시로 특정 오답 조회"""
        if exam_name not in self.wrong_answers:
            return {}
        if question_hash not in self.wrong_answers[exam_name]:
            return {}
        return self.wrong_answers[exam_name][question_hash]
    
    def remove_wrong_answer(self, exam_name: str, question_hash: str) -> bool:
        """오답 삭제"""
        if exam_name not in self.wrong_answers:
            return False
        
        if question_hash not in self.wrong_answers[exam_name]:
            return False
        
        del self.wrong_answers[exam_name][question_hash]
        self._save_wrong_answers()
        print(f"✅ 오답 삭제 완료: {exam_name}, 해시: {question_hash[:8]}...")
        return True
    
    def clear_wrong_answers(self, exam_name: str) -> bool:
        """시험별 오답 전체 삭제"""
        if exam_name not in self.wrong_answers:
            return False
        
        del self.wrong_answers[exam_name]
        self._save_wrong_answers()
        print(f"✅ {exam_name} 오답 전체 삭제 완료")
        return True
    
    def clear_all_data(self) -> str:
        """모든 데이터 완전 초기화"""
        try:
            # 1. extracted_questions 폴더 전체 삭제
            questions_dir = Path("extracted_questions")
            if questions_dir.exists():
                for file_path in questions_dir.glob("*"):
                    try:
                        file_path.unlink()
                        logger.info(f"🗑️ 삭제된 파일: {file_path}")
                    except Exception as e:
                        logger.warning(f"⚠️ 파일 삭제 실패 {file_path}: {e}")
            
            # 2. faiss_vector_db 폴더 삭제
            vector_db_dir = Path("faiss_vector_db")
            if vector_db_dir.exists():
                for file_path in vector_db_dir.glob("*"):
                    try:
                        file_path.unlink()
                        logger.info(f"🗑️ 삭제된 벡터 DB 파일: {file_path}")
                    except Exception as e:
                        logger.warning(f"⚠️ 벡터 DB 파일 삭제 실패 {file_path}: {e}")
            
            # 3. 데이터 파일들 삭제
            data_files = ["exam_data.json", "pdf_hashes.json", "wrong_answers.json"]
            for file_name in data_files:
                file_path = Path(file_name)
                if file_path.exists():
                    try:
                        file_path.unlink()
                        logger.info(f"🗑️ 삭제된 데이터 파일: {file_path}")
                    except Exception as e:
                        logger.warning(f"⚠️ 데이터 파일 삭제 실패 {file_path}: {e}")
            
            # 4. 메모리 데이터 초기화
            self.exams = {}
            self.exam_names = []
            self.pdf_hashes = {}
            self.wrong_answers = {}
            self.recent_questions = {}
            
            # 5. 벡터 DB 초기화 (메서드가 없으면 무시)
            try:
                if hasattr(vector_store, 'clear_all_data'):
                    vector_store.clear_all_data()
                else:
                    logger.info("ℹ️ 벡터 DB clear_all_data 메서드가 없어 건너뜁니다.")
            except Exception as e:
                logger.warning(f"⚠️ 벡터 DB 초기화 실패: {e}")
            
            logger.info("✅ 모든 데이터 완전 초기화 완료")
            return "✅ 모든 데이터가 완전히 초기화되었습니다.\n\n🗑️ 삭제된 항목:\n- extracted_questions 폴더 전체\n- faiss_vector_db 폴더 전체\n- exam_data.json\n- pdf_hashes.json\n- wrong_answers.json\n- 메모리 데이터\n- 벡터 DB 데이터"
            
        except Exception as e:
            logger.error(f"❌ 데이터 초기화 중 오류: {e}")
            return f"❌ 데이터 초기화 중 오류가 발생했습니다: {e}"
    
    def _extract_keywords(self, message: str) -> list:
        """메시지에서 핵심 키워드 추출"""
        
        # 불용어 목록
        stop_words = {
            '다음', '중에서', '가장', '적절한', '것은', '것으로', '사용하는', '기법', '알려줘',
            '무엇', '어떤', '어떻게', '왜', '언제', '어디서', '누가', '무엇을', '무엇이',
            '이', '가', '을', '를', '의', '에', '에서', '로', '으로', '와', '과', '도', '만',
            '은', '는', '이', '가', '을', '를', '의', '에', '에서', '로', '으로', '와', '과'
        }
        
        # 특수문자 제거 및 소문자 변환
        clean_message = re.sub(r'[^\w\s]', ' ', message)
        words = clean_message.split()
        
        # 불용어 제거 및 길이 필터링
        keywords = [word for word in words if word not in stop_words and len(word) >= 2]
        
        # 빈도수 기반 정렬 (간단한 구현)
        word_count = {}
        for word in keywords:
            word_count[word] = word_count.get(word, 0) + 1
        
        # 빈도수 기준으로 정렬
        sorted_keywords = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
        
        return [word for word, count in sorted_keywords[:5]]  # 상위 5개 키워드 반환
    
    def _deduplicate_chunks(self, chunks: list) -> list:
        """중복 청크 제거"""
        seen_ids = set()
        unique_chunks = []
        
        for chunk in chunks:
            chunk_id = chunk.get('metadata', {}).get('id', '')
            if chunk_id and chunk_id not in seen_ids:
                seen_ids.add(chunk_id)
                unique_chunks.append(chunk)
            elif not chunk_id:  # ID가 없는 경우 내용 기반으로 중복 체크
                content = chunk.get('content', '')[:100]  # 첫 100자로 비교
                if content not in seen_ids:
                    seen_ids.add(content)
                    unique_chunks.append(chunk)
        
        return unique_chunks
    
    def _create_hybrid_prompt(self, message: str, context: str, history: list, exam_name: str) -> str:
        """하이브리드 답변을 위한 프롬프트 생성"""
        conversation_context = ""
        if history:
            conversation_parts = []
            for h in history[-5:]:
                try:
                    if isinstance(h, dict):
                        if h.get('role') == 'user':
                            conversation_parts.append(f"사용자: {h.get('content', '')}")
                        elif h.get('role') == 'assistant':
                            conversation_parts.append(f"도우미: {h.get('content', '')}")
                    elif isinstance(h, list) and len(h) >= 2:
                        conversation_parts.append(f"사용자: {h[0]}")
                        conversation_parts.append(f"도우미: {h[1]}")
                except Exception as e:
                    print(f"⚠️ 히스토리 처리 중 오류: {e}")
                    continue
            
            conversation_context = "\n".join(conversation_parts)
        
        return f"""
다음 {exam_name} 기출문제 컨텍스트와 당신의 지식을 결합하여 답변해주세요.

**답변 전략**:
1. 기출문제 컨텍스트에서 관련 내용을 우선적으로 찾아 답변의 근거로 사용하세요.
2. 기출문제에 관련 내용이 부족한 경우, 당신의 전문 지식을 보완적으로 사용하세요.
3. 기출문제 내용을 정확히 인용하고, 추가 설명은 명확히 구분해주세요.
4. 답변은 구체적이고 자세하게 작성해주세요.

=== 기출문제 컨텍스트 ===
{context}

=== 대화 기록 ===
{conversation_context}

사용자: {message}
도우미: 기출문제와 전문 지식을 바탕으로 답변드리겠습니다.

"""

# 인스턴스 생성
generator = ExamQuestionGenerator()

def create_gradio_interface():
    """Gradio 인터페이스 생성"""
    
    # 동기화 함수 정의 (맨 위에 위치)
    def update_exam_list():
        """시험 목록 업데이트"""
        return gr.Dropdown(choices=generator.get_exam_list())
    
    def update_selected_exam():
        """문제 풀이 탭의 시험 선택 업데이트"""
        return gr.Dropdown(choices=generator.get_exam_list(), value=None)
    
    with gr.Blocks(title="기출문제 RAG 기반 시험 문제 생성 및 질의 응답 챗봇") as demo:
        gr.Markdown("# 🎯 기출문제 RAG 기반 시험 문제 생성 및 질의 응답 챗봇")
        gr.Markdown("### Azure OpenAI와 RAG를 활용한 맞춤형 학습 시스템")
        
        with gr.Tabs():
            # 탭 1: 시험 관리
            with gr.TabItem("📚 시험 관리"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 📋 시험 목록")
                        exam_list = gr.Dropdown(
                            choices=generator.get_exam_list(),
                            label="등록된 시험",
                            interactive=True
                        )
                        
                        with gr.Row():
                            add_exam_btn = gr.Button("시험 추가", variant="primary", size="sm")
                            remove_exam_btn = gr.Button("시험 제거", variant="stop", size="sm")
                        
                        with gr.Row():
                            clear_all_btn = gr.Button("모든 데이터 초기화", variant="stop", size="sm")
                        
                        exam_action_output = gr.Textbox(
                            label="작업 결과",
                            lines=3,
                            interactive=False
                        )
                        
                        gr.Markdown("### 📄 PDF 목록")
                        pdf_list_btn = gr.Button("PDF 목록 보기", variant="secondary", size="sm")
                        pdf_list_output = gr.Textbox(
                            label="업로드된 PDF 목록",
                            lines=8,
                            interactive=False
                        )
                    
                    with gr.Column(scale=2):
                        gr.Markdown("### 📄 PDF 업로드")
                        gr.Markdown("기출문제 PDF를 업로드하면 벡터 DB에 저장되어 RAG 기반 문제 생성이 가능합니다.")
                        
                        exam_name_input = gr.Textbox(
                            label="시험 이름",
                            placeholder="예: 정보시스템감리사, 공무원시험, 토익 등",
                            lines=1
                        )
                        
                        pdf_upload = gr.File(
                            label="기출문제 PDF 업로드",
                            file_types=[".pdf"],
                            type="filepath",
                            file_count="single"
                        )
                        upload_btn = gr.Button("PDF 업로드", variant="primary")
                        
                        upload_output = gr.Textbox(
                            label="업로드 결과",
                            lines=8,
                            interactive=False
                        )
            
            # 탭 2: 문제 생성 및 답변
            with gr.TabItem("📝 문제 풀이"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 🎯 시험 정보")
                        gr.Markdown("**난이도**: 랜덤 선택 (쉬움/보통/어려움)")
                        gr.Markdown("**문제 유형**: 랜덤 선택 (객관식/주관식)")
                        
                        selected_exam = gr.Dropdown(
                            choices=generator.get_exam_list(),
                            label="시험 선택",
                            interactive=True
                        )
                        
                        question_mode = gr.Radio(
                            choices=[
                                ("기출문제 기반 새 문제 생성", "generate"),
                                ("기출문제 그대로 출제", "exact")
                            ],
                            label="문제 생성 모드",
                            value="generate"
                        )
                        
                        generate_btn = gr.Button("문제 생성", variant="primary")
                        
                        # 문제 초기화 버튼 추가
                        reset_btn = gr.Button("문제 초기화", variant="secondary")
                    
                    with gr.Column():
                        gr.Markdown("### 📝 문제")
                        question_output = gr.Markdown(
                            value="문제를 생성해주세요."
                        )
                
                gr.Markdown("---")
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### ✏️ 답변 입력")
                        user_answer_input = gr.Textbox(
                            label="답변 입력",
                            placeholder="답변을 입력하세요...",
                            lines=2
                        )
                        evaluate_btn = gr.Button("답변 확인", variant="secondary")
                    
                    with gr.Column():
                        gr.Markdown("### 📊 평가 결과")
                        evaluation_output = gr.Markdown(
                            value="답변을 확인해주세요."
                        )
                

                
                # 문제 초기화 함수
                def reset_question_history(exam_name):
                    """문제 히스토리 초기화"""
                    if exam_name and exam_name in generator.recent_questions:
                        generator.recent_questions[exam_name] = []
                        print(f"🔄 [콘솔 로그] {exam_name} 시험의 문제 히스토리 초기화 완료")
                        return f"✅ {exam_name} 시험의 문제 히스토리가 초기화되었습니다."
                    return "❌ 시험을 선택해주세요."
                
                # 로딩 상태 표시 함수
                def show_loading_message():
                    """로딩 메시지 표시"""
                    return "🔄 처리 중입니다. 잠시만 기다려주세요..."
                
                def clear_loading_message():
                    """로딩 메시지 제거"""
                    return ""
                
                # 이벤트 연결
                generate_btn.click(
                    fn=show_loading_message,  # 로딩 메시지 표시
                    inputs=[],
                    outputs=question_output
                ).then(
                    fn=generator.generate_question,  # 실제 문제 생성
                    inputs=[selected_exam, question_mode],
                    outputs=question_output
                ).then(
                    fn=lambda: ("", "답변을 확인해주세요."),  # 답변 입력, 결과 초기화
                    inputs=[],
                    outputs=[user_answer_input, evaluation_output]
                )
                
                reset_btn.click(
                    fn=reset_question_history,
                    inputs=[selected_exam],
                    outputs=[question_output]
                )
                
                evaluate_btn.click(
                    fn=show_loading_message,  # 로딩 메시지 표시
                    inputs=[],
                    outputs=evaluation_output
                ).then(
                    fn=generator.evaluate_answer,  # 실제 답변 평가
                    inputs=[user_answer_input],
                    outputs=evaluation_output
                )
            
            # 탭 3: AI 챗봇
            with gr.TabItem("💬 AI 챗봇"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 🎯 현재 선택된 시험")
                        chat_exam_display = gr.Textbox(
                            label="현재 시험",
                            value="시험을 선택해주세요",
                            interactive=False
                        )
                        
                        chat_exam_select = gr.Dropdown(
                            choices=generator.get_exam_list(),
                            label="시험 선택",
                            interactive=True
                        )
                
                chatbot = gr.Chatbot(
                    label="학습 도우미와 대화하기",
                    height=400,
                    type="messages"  # OpenAI 스타일 메시지 형식 사용
                )
                msg = gr.Textbox(
                    label="메시지",
                    placeholder="시험 관련 질문이나 도움이 필요한 내용을 입력하세요...",
                    lines=2
                )
                clear_btn = gr.Button("대화 초기화")
                
                def update_chat_exam_display(exam_name):
                    """채팅 시험 표시 업데이트"""
                    if exam_name:
                        return f"📚 {exam_name}"
                    else:
                        return "시험을 선택해주세요"
                
                def respond(message, history, exam_name):
                    # AI 챗봇 설정 가져오기
                    ai_config = Config.get_ai_chatbot_config()
                    debug_logs = ai_config["debug_logs"]
                    
                    if debug_logs:
                        logger.info(f"💬 [AI 챗봇] 메시지: '{message}', 시험: {exam_name}")
                        logger.info(f"📊 [AI 챗봇] 현재 히스토리 길이: {len(history)}")
                    
                    if not DEPLOYMENT_NAME:
                        error_msg = "Error: DEPLOYMENT_NAME 환경 변수가 설정되지 않았습니다."
                        logger.error(f"❌ [AI 챗봇] {error_msg}")
                        history.append({"role": "user", "content": message})
                        history.append({"role": "assistant", "content": error_msg})
                        return history, ""
                    
                    if not exam_name:
                        error_msg = "❌ 시험을 먼저 선택해주세요. 기출문제 검색을 위해 시험이 필요합니다."
                        logger.error(f"❌ [AI 챗봇] 시험이 선택되지 않음")
                        history.append({"role": "user", "content": message})
                        history.append({"role": "assistant", "content": error_msg})
                        return history, ""
                    
                    # RAG 기반 검색 및 답변
                    try:
                        if debug_logs:
                            logger.info(f"🔍 [AI 챗봇] '{exam_name}' 시험에서 관련 기출문제 검색 중...")
                            logger.info(f"🔍 [AI 챗봇] 검색 쿼리 최적화 시작...")
                        
                        # 원본 질문에서 핵심 키워드 추출
                        keywords = generator._extract_keywords(message)
                        if debug_logs:
                            logger.info(f"🔍 [AI 챗봇] 추출된 키워드: {keywords}")
                        
                        # 다양한 검색 쿼리 생성
                        search_queries = [message] + keywords[:3]  # 원본 질문 + 상위 3개 키워드
                        if debug_logs:
                            logger.info(f"🔍 [AI 챗봇] 검색 쿼리 목록: {search_queries}")
                        
                        # 각 쿼리로 검색하여 최고 점수 결과 수집
                        all_chunks = []
                        all_extracted_questions = []
                        
                        for query in search_queries:
                            if debug_logs:
                                logger.info(f"🔍 [AI 챗봇] 쿼리 '{query}'로 검색 중...")
                            
                            # 벡터 DB에서 검색
                            chunks = vector_store.search_similar_questions(query, subject=exam_name, n_results=5)
                            all_chunks.extend(chunks)
                            
                            # 추출된 문제에서 semantic 검색
                            extracted_results = pdf_processor.search_extracted_questions_semantic(query, exam_name, n_results=3)
                            all_extracted_questions.extend(extracted_results)
                        
                        # 중복 제거 및 점수 기준 정렬
                        unique_chunks = generator._deduplicate_chunks(all_chunks)
                        similar_chunks = sorted(unique_chunks, key=lambda x: x.get('score', 0), reverse=True)[:ai_config["top_k"]]
                        
                        # 추출된 문제도 함께 사용
                        unique_extracted = generator._deduplicate_chunks(all_extracted_questions)
                        top_extracted = sorted(unique_extracted, key=lambda x: x.get('score', 0), reverse=True)[:5]
                        
                        # 벡터 DB 결과와 추출된 문제 결과 합치기
                        combined_context = ""
                        
                        if similar_chunks:
                            if debug_logs:
                                logger.info(f"📄 [AI 챗봇] 벡터 DB 검색 결과: {len(similar_chunks)}개")
                            combined_context += "=== 벡터 DB 검색 결과 ===\n"
                            for i, chunk in enumerate(similar_chunks, 1):
                                combined_context += f"{i}. {chunk.get('content', '')}\n\n"
                        
                        if top_extracted:
                            if debug_logs:
                                logger.info(f"📄 [AI 챗봇] 추출된 문제 검색 결과: {len(top_extracted)}개")
                            combined_context += "=== 추출된 기출문제 ===\n"
                            for i, chunk in enumerate(top_extracted, 1):
                                combined_context += f"{i}. {chunk.get('content', '')}\n\n"
                        
                        if not combined_context:
                            if debug_logs:
                                logger.info(f"❌ [AI 챗봇] 검색 결과 없음")
                            combined_context = "관련 기출문제를 찾을 수 없습니다."
                        
                        if debug_logs:
                            logger.info(f"📝 [AI 챗봇] 최종 컨텍스트 길이: {len(combined_context)}자")
                        
                        # AI 챗봇 응답 생성
                        if debug_logs:
                            logger.info(f"🤖 [AI 챗봇] Azure OpenAI API 호출 시작...")
                        
                        # 하이브리드 프롬프트 생성
                        hybrid_prompt = generator._create_hybrid_prompt(message, combined_context, history, exam_name)
                        
                        if debug_logs:
                            logger.info(f"📝 [AI 챗봇] 하이브리드 프롬프트 생성 완료")
                        
                        # Azure OpenAI API 호출
                        response = openai.chat.completions.create(
                            model=str(DEPLOYMENT_NAME),
                            messages=[
                                {"role": "system", "content": hybrid_prompt},
                                {"role": "user", "content": message}
                            ],
                            temperature=ai_config["temperature"],
                            max_tokens=ai_config["max_tokens"],
                            top_p=ai_config["top_p"],
                            frequency_penalty=ai_config["frequency_penalty"],
                            presence_penalty=ai_config["presence_penalty"]
                        )
                        
                        if debug_logs:
                            logger.info(f"✅ [AI 챗봇] Azure OpenAI API 응답 수신")
                        
                        if response and response.choices:
                            content = response.choices[0].message.content
                            assistant_message = content.strip() if content else "❌ 답변을 생성할 수 없습니다."
                            
                            if debug_logs:
                                logger.info(f"💬 [AI 챗봇] 생성된 답변: {assistant_message[:100]}...")
                            
                            # 히스토리에 메시지 추가
                            history.append({"role": "user", "content": message})
                            history.append({"role": "assistant", "content": assistant_message})
                            
                            if debug_logs:
                                logger.info(f"📝 [AI 챗봇] 히스토리에 메시지 추가 완료")
                        else:
                            error_msg = "❌ 답변을 생성할 수 없습니다."
                            logger.error(f"❌ [AI 챗봇] 답변 생성 실패")
                            history.append({"role": "user", "content": message})
                            history.append({"role": "assistant", "content": error_msg})
                        
                        if debug_logs:
                            logger.info(f"✅ [AI 챗봇] 함수 종료 - 최종 히스토리 길이: {len(history)}")
                        return history, ""
                        
                    except Exception as e:
                        error_msg = f"❌ 오류가 발생했습니다: {e}"
                        logger.error(f"❌ [AI 챗봇] 오류: {e}")
                        logger.error(f"❌ [AI 챗봇] 오류 타입: {type(e)}")
                        logger.error(f"❌ [AI 챗봇] 오류 상세: {traceback.format_exc()}")
                        history.append({"role": "user", "content": message})
                        history.append({"role": "assistant", "content": error_msg})
                        return history, ""
                
                # 이벤트 연결
                chat_exam_select.change(
                    fn=update_chat_exam_display,
                    inputs=[chat_exam_select],
                    outputs=[chat_exam_display]
                )
                
                msg.submit(respond, [msg, chatbot, chat_exam_select], [chatbot, msg])
                clear_btn.click(lambda: [], None, chatbot, queue=False)
            
            # 탭 4: 오답노트
            with gr.TabItem("📝 오답노트"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 📚 시험 선택")
                        wrong_answer_exam = gr.Dropdown(
                            choices=generator.get_exam_list(),
                            label="시험 선택",
                            interactive=True
                        )
                        load_wrong_btn = gr.Button("오답 시험 재도전 하기", variant="primary", size="sm")
                        clear_wrong_btn = gr.Button("오답 전체 삭제", variant="stop", size="sm")
                        gr.Markdown("### 📊 오답 통계")
                        wrong_stats = gr.Textbox(
                            label="오답 통계",
                            lines=4,
                            interactive=False
                        )
                    with gr.Column(scale=2):
                        gr.Markdown("### 📝 오답 재도전 (Sequential Retry)")
                        wrong_state = gr.State({"list": [], "idx": 0})
                        wrong_question_output = gr.Markdown(value="시험을 선택하면 오답 문제가 순서대로 표시됩니다.")
                        wrong_progress = gr.Markdown(value="")
                gr.Markdown("---")
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### ✏️ 답변 입력")
                        wrong_answer_input = gr.Textbox(
                            label="답변 입력",
                            placeholder="답변을 입력하세요...",
                            lines=2
                        )
                        wrong_evaluate_btn = gr.Button("정답 확인", variant="secondary")
                        remember_btn = gr.Button("기억했어요", variant="primary")
                    with gr.Column():
                        gr.Markdown("### 📊 평가 결과")
                        wrong_evaluation_output = gr.Markdown(value="답변을 확인해주세요.")
                # 정답 및 해설 UI 제거됨

                # Sequential retry logic
                def load_wrong_sequential(exam_name):
                    logger.info(f"🔄 [오답노트] 시험 선택됨: {exam_name}")
                    wrongs = generator.get_wrong_answers(exam_name)
                    logger.info(f"📝 [오답노트] 오답 개수: {len(wrongs)}개")
                    
                    if not exam_name:
                        logger.info("❌ [오답노트] 시험명이 없음")
                        return {"list": [], "idx": 0}, "시험을 선택해주세요.", "", "", ""
                    if not wrongs:
                        logger.info("❌ [오답노트] 오답이 없음")
                        return {"list": [], "idx": 0}, "오답이 없습니다!", "", "", ""
                    
                    first = wrongs[0]
                    generator.current_question = first["question"]
                    generator.current_answer = first["answer"]
                    generator.current_explanation = first["explanation"]
                    generator.current_exam_name = exam_name
                    generator.current_metadata = [first["metadata"]] if first.get("metadata") else None
                    generator.current_context = first["question"]
                    
                    # 문제 텍스트에서 정답 부분 제거
                    qtext = first["question"]
                    answer_start = qtext.find("=== 정답 ===")
                    if answer_start != -1:
                        qtext = qtext[:answer_start].strip()
                    qtext = qtext.replace("=== 문제 ===", "=== 문제 ===\n").replace("=== 보기 ===", "\n=== 보기 ===\n")
                    
                    # 출처 정보 추가
                    source_info = ""
                    if first.get("metadata", {}).get("pdf_source"):
                        pdf_source = first["metadata"]["pdf_source"]
                        source_info = f"\n\n📄 **출처**: {pdf_source}"
                        qtext += source_info
                    
                    progress = f"**{1}/{len(wrongs)}**"
                    total_wrong_count = sum(w["wrong_count"] for w in wrongs)
                    stats = f"📝 총 오답 문제: {len(wrongs)}개\n🔄 총 오답 횟수: {total_wrong_count}회"
                    logger.info(f"✅ [오답노트] 첫 번째 오답 문제 로드 완료")
                    return {"list": wrongs, "idx": 0}, qtext, progress, "", stats

                def show_current_wrong(state):
                    wrongs = state["list"]
                    idx = state["idx"]
                    if not wrongs:
                        return "오답이 없습니다!", "", ""
                    cur = wrongs[idx]
                    
                    # 문제 텍스트에서 정답 부분 제거
                    qtext = cur["question"]
                    answer_start = qtext.find("=== 정답 ===")
                    if answer_start != -1:
                        qtext = qtext[:answer_start].strip()
                    qtext = qtext.replace("=== 문제 ===", "=== 문제 ===\n").replace("=== 보기 ===", "\n=== 보기 ===\n")
                    
                    # 출처 정보 추가
                    if cur.get("metadata", {}).get("pdf_source"):
                        pdf_source = cur["metadata"]["pdf_source"]
                        source_info = f"\n\n📄 **출처**: {pdf_source}"
                        qtext += source_info
                    
                    progress = f"**{idx+1}/{len(wrongs)}**"
                    return qtext, progress, ""

                def eval_wrong_answer(user_answer, state, exam_name):
                    wrongs = state["list"]
                    idx = state["idx"]
                    if not wrongs:
                        return state, "오답이 없습니다!", "", "", ""
                    cur = wrongs[idx]
                    generator.current_question = cur["question"]
                    generator.current_answer = cur["answer"]
                    generator.current_explanation = cur["explanation"]
                    generator.current_exam_name = exam_name
                    generator.current_metadata = [cur["metadata"]] if cur.get("metadata") else None
                    generator.current_context = cur["question"]
                    result = generator.evaluate_answer(user_answer)
                    is_wrong = generator._is_wrong_answer(result)
                    if not is_wrong:
                        # Remove from wrong list
                        qhash = cur["hash"]
                        generator.remove_wrong_answer(exam_name, qhash)
                        new_wrongs = generator.get_wrong_answers(exam_name)
                        if not new_wrongs:
                            return {"list": [], "idx": 0}, "정답입니다! 모든 오답을 해결했습니다.", "", "", ""
                        # Stay at same idx (next problem slides in)
                        new_idx = min(idx, len(new_wrongs)-1)
                        next_cur = new_wrongs[new_idx]
                        qtext = next_cur["question"]
                        answer_start = qtext.find("=== 정답 ===")
                        if answer_start != -1:
                            qtext = qtext[:answer_start].strip()
                        qtext = qtext.replace("=== 문제 ===", "=== 문제 ===\n").replace("=== 보기 ===", "\n=== 보기 ===\n")
                        
                        # 출처 정보 추가
                        if next_cur.get("metadata", {}).get("pdf_source"):
                            pdf_source = next_cur["metadata"]["pdf_source"]
                            source_info = f"\n\n📄 **출처**: {pdf_source}"
                            qtext += source_info
                        
                        progress = f"**{new_idx+1}/{len(new_wrongs)}**"
                        return {"list": new_wrongs, "idx": new_idx}, qtext, progress, result, ""
                    else:
                        # Stay on same problem
                        return state, "틀렸습니다. 다시 시도해보세요!", f"**{idx+1}/{len(wrongs)}**", result, ""

                def remember_wrong(state, exam_name):
                    wrongs = state["list"]
                    idx = state["idx"]
                    if not wrongs:
                        return {"list": [], "idx": 0}, "오답이 없습니다!", "", ""
                    cur = wrongs[idx]
                    qhash = cur["hash"]
                    generator.remove_wrong_answer(exam_name, qhash)
                    new_wrongs = generator.get_wrong_answers(exam_name)
                    if not new_wrongs:
                        return {"list": [], "idx": 0}, "모든 오답을 기억했습니다!", "", ""
                    new_idx = min(idx, len(new_wrongs)-1)
                    next_cur = new_wrongs[new_idx]
                    qtext = next_cur["question"]
                    answer_start = qtext.find("=== 정답 ===")
                    if answer_start != -1:
                        qtext = qtext[:answer_start].strip()
                    qtext = qtext.replace("=== 문제 ===", "=== 문제 ===\n").replace("=== 보기 ===", "\n=== 보기 ===\n")
                    
                    # 출처 정보 추가
                    if next_cur.get("metadata", {}).get("pdf_source"):
                        pdf_source = next_cur["metadata"]["pdf_source"]
                        source_info = f"\n\n📄 **출처**: {pdf_source}"
                        qtext += source_info
                    
                    progress = f"**{new_idx+1}/{len(new_wrongs)}**"
                    return {"list": new_wrongs, "idx": new_idx}, qtext, progress, ""

                def show_wrong_solution(state, exam_name):
                    wrongs = state["list"]
                    idx = state["idx"]
                    if not wrongs:
                        return "오답이 없습니다!"
                    cur = wrongs[idx]
                    generator.current_question = cur["question"]
                    generator.current_answer = cur["answer"]
                    generator.current_explanation = cur["explanation"]
                    generator.current_exam_name = exam_name
                    generator.current_metadata = [cur["metadata"]] if cur.get("metadata") else None
                    generator.current_context = cur["question"]
                    return generator.show_solution()

                def clear_all_wrong_answers_seq(exam_name):
                    generator.clear_wrong_answers(exam_name)
                    return {"list": [], "idx": 0}, "모든 오답이 삭제되었습니다.", "", "", ""

                # 이벤트 연결
                wrong_answer_exam.change(
                    fn=load_wrong_sequential,
                    inputs=[wrong_answer_exam],
                    outputs=[wrong_state, wrong_question_output, wrong_progress, wrong_evaluation_output, wrong_stats]
                )
                load_wrong_btn.click(
                    fn=load_wrong_sequential,
                    inputs=[wrong_answer_exam],
                    outputs=[wrong_state, wrong_question_output, wrong_progress, wrong_evaluation_output, wrong_stats]
                )
                wrong_evaluate_btn.click(
                    fn=show_loading_message,  # 로딩 메시지 표시
                    inputs=[],
                    outputs=wrong_evaluation_output
                ).then(
                    fn=eval_wrong_answer,  # 실제 답변 평가
                    inputs=[wrong_answer_input, wrong_state, wrong_answer_exam],
                    outputs=[wrong_state, wrong_question_output, wrong_progress, wrong_evaluation_output]
                ).then(
                    fn=lambda: "",  # 답변 입력 초기화
                    inputs=[],
                    outputs=[wrong_answer_input]
                )
                remember_btn.click(
                    fn=remember_wrong,
                    inputs=[wrong_state, wrong_answer_exam],
                    outputs=[wrong_state, wrong_question_output, wrong_progress, wrong_evaluation_output]
                ).then(
                    fn=lambda: "",  # 답변 입력 초기화
                    inputs=[],
                    outputs=[wrong_answer_input]
                )
                clear_wrong_btn.click(
                    fn=clear_all_wrong_answers_seq,
                    inputs=[wrong_answer_exam],
                    outputs=[wrong_state, wrong_question_output, wrong_progress, wrong_evaluation_output, wrong_stats]
                )
        
        # 시험 관리 탭의 이벤트 연결 (모든 컴포넌트 정의 이후)
        add_exam_btn.click(
            fn=generator.add_exam,
            inputs=[exam_name_input],
            outputs=[exam_action_output, exam_list]
        ).then(
            fn=update_selected_exam,
            inputs=[],
            outputs=selected_exam
        ).then(
            fn=lambda: gr.Dropdown(choices=generator.get_exam_list(), value=None),
            inputs=[],
            outputs=wrong_answer_exam
        ).then(
            fn=lambda: gr.Dropdown(choices=generator.get_exam_list(), value=None),
            inputs=[],
            outputs=chat_exam_select
        )
        
        remove_exam_btn.click(
            fn=generator.remove_exam,
            inputs=[exam_list],
            outputs=[exam_action_output, exam_list]
        ).then(
            fn=update_selected_exam,
            inputs=[],
            outputs=selected_exam
        ).then(
            fn=lambda: gr.Dropdown(choices=generator.get_exam_list(), value=None),
            inputs=[],
            outputs=wrong_answer_exam
        ).then(
            fn=lambda: gr.Dropdown(choices=generator.get_exam_list(), value=None),
            inputs=[],
            outputs=chat_exam_select
        )
        
        clear_all_btn.click(
            fn=generator.clear_all_data,
            inputs=[],
            outputs=[exam_action_output]
        ).then(
            fn=lambda: gr.Dropdown(choices=[]),
            inputs=[],
            outputs=[exam_list]
        ).then(
            fn=lambda: gr.Dropdown(choices=[], value=None),
            inputs=[],
            outputs=selected_exam
        ).then(
            fn=lambda: gr.Dropdown(choices=[], value=None),
            inputs=[],
            outputs=wrong_answer_exam
        ).then(
            fn=lambda: gr.Dropdown(choices=[], value=None),
            inputs=[],
            outputs=chat_exam_select
        )
        
        upload_btn.click(
            fn=generator.upload_pdf,
            inputs=[pdf_upload, exam_name_input],
            outputs=[upload_output, exam_list]
        ).then(
            fn=update_selected_exam,
            inputs=[],
            outputs=selected_exam
        ).then(
            fn=lambda: gr.Dropdown(choices=generator.get_exam_list(), value=None),
            inputs=[],
            outputs=wrong_answer_exam
        ).then(
            fn=lambda: gr.Dropdown(choices=generator.get_exam_list(), value=None),
            inputs=[],
            outputs=chat_exam_select
        )
        
        # 시험 선택 시 PDF 목록 자동 업데이트
        exam_list.change(
            fn=generator.format_pdf_list,
            inputs=[exam_list],
            outputs=[pdf_list_output]
        )
        
        pdf_list_btn.click(
            fn=generator.format_pdf_list,
            inputs=[exam_list],
            outputs=[pdf_list_output]
        )
        
        # 하단 정보
        gr.Markdown("---")
        # gr.Markdown("### 🔧 기술 스택")
        # gr.Markdown("- **Azure OpenAI**: GPT-4 기반 자연어 처리")
        # gr.Markdown("- **Gradio**: 사용자 인터페이스")
        # gr.Markdown("- **Docling**: PDF 텍스트 추출")
        # gr.Markdown("- **FAISS**: 벡터 데이터베이스")
        # gr.Markdown("- **RAG**: 기출문제 검색 및 질의 응답")
        
    return demo

if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(level=logging.INFO)
    
    # 서버 설정 가져오기
    server_config = Config.get_server_config()
    port = server_config["port"]
    use_ngrok = server_config["use_ngrok"]
    
    logger.info("🎯 [콘솔 로그] 정보시스템감리사 문제 생성 챗봇 시작")
    logger.info("🌐 [콘솔 로그] Gradio 웹 인터페이스 실행 중...")
    logger.info(f"🔧 [설정] 포트: {port}, ngrok 사용: {use_ngrok}")
    
    # ngrok 설정
    ngrok_url = None
    
    if use_ngrok:
        if not NGROK_AVAILABLE:
            logger.warning("⚠️ [ngrok] pyngrok이 설치되지 않았습니다. pip install pyngrok")
            logger.info("🔄 Gradio share 모드로 대체 실행...")
            use_ngrok = False
        else:
            try:
                logger.info("🔗 [ngrok] 터널 생성 중...")
                ngrok_url = ngrok.connect(port)
                logger.info(f"🌐 [ngrok] 외부 접속 URL: {ngrok_url}")
                logger.info("=" * 60)
                logger.info(f"✅ 외부에서 이 URL로 접속하세요: {ngrok_url}")
                logger.info("=" * 60)
            except Exception as e:
                logger.warning(f"⚠️ [ngrok] 연결 실패: {e}")
                logger.info("🔄 Gradio share 모드로 대체 실행...")
                use_ngrok = False
    
    # Gradio 인터페이스 실행
    demo = create_gradio_interface()
    
    try:
        demo.launch(
            server_name="0.0.0.0",
            server_port=port,
            share=not use_ngrok,  # ngrok 사용시 share=False, 실패시 share=True
            show_error=True,
            debug=False
        )
    except KeyboardInterrupt:
        logger.info("🛑 [종료] 사용자에 의해 종료되었습니다.")
        if use_ngrok and ngrok_url:
            try:
                ngrok.disconnect(ngrok_url)
                logger.info("🔗 [ngrok] 터널 연결 해제 완료")
            except:
                pass
    except Exception as e:
        logger.error(f"❌ [오류] 실행 중 오류 발생: {e}")
        if use_ngrok and ngrok_url:
            try:
                ngrok.disconnect(ngrok_url)
                logger.info("🔗 [ngrok] 터널 연결 해제 완료")
            except:
                pass