# 기출문제 RAG 기반 시험 문제 생성 및 질의 응답 챗봇 - 설치 가이드

## 1. 프로젝트 개요

Azure OpenAI와 RAG(Retrieval-Augmented Generation)를 활용한 맞춤형 학습 시스템입니다. 기출문제 PDF를 업로드하면 벡터 데이터베이스에 저장하고, 이를 기반으로 새로운 문제를 생성하거나 기출문제를 그대로 출제할 수 있습니다.

### 주요 기능
- 📄 PDF 업로드 및 벡터 DB 구축
- 🎯 기출문제 기반 새 문제 생성
- 📝 기출문제 그대로 출제
- ✏️ 답변 평가 및 피드백
- 💬 AI 챗봇 (RAG 기반 질의응답)
- 📚 오답노트 관리
- 🔍 출처 정보 제공

## 2. 환경 설정

### 2.1 Python 환경
- **Python 3.11 이상** 필요 (권장: Python 3.12)
- 가상환경 사용 권장

```bash
# 가상환경 생성
python -m venv .venv

# 가상환경 활성화 (Windows)
.venv\Scripts\activate

# 가상환경 활성화 (Linux/Mac)
source .venv/bin/activate
```

### 2.2 필요한 라이브러리 설치

```bash
# requirements.txt 사용 (권장)
pip install -r requirements.txt

# 또는 개별 설치
pip install gradio==4.44.0 python-dotenv==1.0.0 openai==1.12.0
pip install docling<2.37.0 PyPDF2==3.0.1 pdfplumber>=0.10.0
pip install faiss-cpu==1.7.4 sentence-transformers>=2.2.0 numpy>=1.24.0
pip install torch>=2.0.0 transformers>=4.30.0 colorama>=0.4.6
```

## 3. Azure OpenAI 설정

### 3.1 Azure OpenAI 서비스 생성
1. Azure Portal에서 OpenAI 서비스 생성
2. 모델 배포 (GPT-4 또는 GPT-4o 권장)
3. API 키 및 엔드포인트 확인

### 3.2 환경 변수 설정
프로젝트 루트에 `.env` 파일을 생성하고 다음 내용을 추가:

```env
OPENAI_API_KEY=your_azure_openai_api_key
AZURE_ENDPOINT=https://your-resource.openai.azure.com/
OPENAI_API_TYPE=azure
OPENAI_API_VERSION=2024-12-01-preview
DEPLOYMENT_NAME=your_deployment_name
```

## 4. 실행 방법

### 4.1 기본 실행
```bash
python mvp_main.py
```

### 4.2 웹 인터페이스 접속
- 브라우저에서 `http://localhost:7860` 접속
- Gradio 인터페이스 확인

## 5. 사용 방법

### 5.1 시험 관리
1. **📚 시험 관리** 탭 선택
2. **시험 추가**: 새로운 시험 이름 입력 후 "시험 추가" 버튼 클릭
3. **PDF 업로드**: 
   - 시험 이름 입력
   - 기출문제 PDF 파일 선택
   - "PDF 업로드" 버튼 클릭
4. **PDF 목록 확인**: "PDF 목록 보기" 버튼으로 업로드된 파일 확인

### 5.2 문제 풀이
1. **📝 문제 풀이** 탭 선택
2. **시험 선택**: 드롭다운에서 원하는 시험 선택
3. **문제 생성 모드 선택**:
   - **기출문제 기반 새 문제 생성**: 기출문제를 참고한 새로운 문제 생성
   - **기출문제 그대로 출제**: 기출문제를 그대로 출제
4. **문제 생성**: "문제 생성" 버튼 클릭
5. **답변 입력**: 생성된 문제에 답변 입력
6. **답변 확인**: "답변 확인" 버튼으로 정답 여부 확인
7. **정답 및 해설**: 정답과 상세한 해설 확인

### 5.3 AI 챗봇
1. **💬 AI 챗봇** 탭 선택
2. **시험 선택**: 질문할 시험 선택
3. **질문 입력**: 시험 관련 질문 입력
4. **RAG 기반 답변**: 기출문제를 참고한 정확한 답변 확인

### 5.4 오답노트
1. **📝 오답노트** 탭 선택
2. **시험 선택**: 오답을 확인할 시험 선택
3. **오답 재도전**: 틀린 문제들을 순서대로 다시 풀기
4. **기억했어요**: 문제를 기억했다고 표시하여 오답 목록에서 제거

## 6. 파일 구조

```
mvp/
├── mvp_main.py              # 메인 애플리케이션 (Gradio 인터페이스)
├── config.py                # 환경 변수 및 설정 관리
├── logger.py                # 로깅 시스템
├── prompt.py                # 프롬프트 정의 (Pythonic Prompting)
├── vector_store.py          # FAISS 벡터 스토어
├── pdf_processor.py         # PDF 처리 모듈 (Docling 활용)
├── review_agent_simple.py   # 문제 검토 에이전트
├── agents/                  # 에이전트 모듈
│   ├── base_agent.py        # 기본 에이전트 클래스
│   └── information_validation_agent.py  # 정보 검증 에이전트
├── requirements.txt         # 필요한 라이브러리
├── INSTALL.md              # 설치 가이드
├── README.md               # 프로젝트 설명
├── .env                    # 환경 변수 (사용자 생성)
├── faiss_vector_db/        # 벡터 데이터베이스 (자동 생성)
├── extracted_questions/    # 추출된 문제 저장소 (자동 생성)
├── logs/                   # 로그 파일 (자동 생성)
├── exam_data.json          # 시험 데이터 (자동 생성)
├── pdf_hashes.json         # PDF 해시 정보 (자동 생성)
└── wrong_answers.json      # 오답노트 데이터 (자동 생성)
```

## 7. 기술 스택

### 7.1 핵심 기술
- **Azure OpenAI**: GPT-4 기반 자연어 처리
- **Gradio**: 사용자 인터페이스
- **FAISS**: 벡터 데이터베이스
- **Sentence Transformers**: 텍스트 임베딩
- **Docling**: PDF 텍스트 추출
- **RAG**: 기출문제 검색 및 질의 응답

### 7.2 GPU 지원
- **Sentence Transformers**: GPU 가속 지원
- **Docling**: GPU 가속 지원 (OCR)
- **FAISS**: CPU 사용 (Python 3.12 호환성)

## 8. 문제 해결

### 8.1 라이브러리 설치 오류
```bash
# FAISS 설치 오류 시
pip install faiss-cpu --no-cache-dir

# sentence-transformers 설치 오류 시
pip install sentence-transformers --no-cache-dir

# Docling 설치 오류 시
pip install docling<2.37.0 --no-cache-dir
```

### 8.2 메모리 부족 오류
- PDF 파일 크기 줄이기
- 청크 크기 조정 (pdf_processor.py에서 chunk_size 수정)

### 8.3 Azure OpenAI 연결 오류
- 환경 변수 확인
- API 키 및 엔드포인트 정확성 확인
- 네트워크 연결 상태 확인

### 8.4 GPU 사용 관련
- CUDA 설치 확인
- PyTorch GPU 버전 설치 확인
- GPU 메모리 부족 시 CPU 모드로 자동 전환

## 9. 성능 최적화

### 9.1 벡터 검색 성능
- FAISS 인덱스 최적화
- 임베딩 모델 선택 (현재: paraphrase-multilingual-MiniLM-L12-v2)

### 9.2 PDF 처리 성능
- GPU 사용으로 OCR 속도 향상
- 청크 크기 최적화

### 9.3 메모리 사용량 최적화
- 대용량 PDF 처리 시 청크 단위 처리
- 불필요한 데이터 정리

## 10. 확장 가능성

### 10.1 새로운 기능 추가
- 사용자 관리 시스템
- 학습 진도 추적
- 맞춤형 학습 경로 제안
- 다중 언어 지원

### 10.2 다른 문서 형식 지원
- Word 문서 (.docx)
- 텍스트 파일 (.txt)
- 이미지 기반 문서 (OCR)

### 10.3 클라우드 배포
- Azure Container Apps
- Azure Functions
- Azure Kubernetes Service 