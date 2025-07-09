# 기출문제 RAG 기반 시험 문제 생성 및 질의 응답 챗봇 - 설치 가이드

## 1. 프로젝트 개요

Azure OpenAI와 RAG(Retrieval-Augmented Generation)를 활용한 맞춤형 학습 시스템입니다. 기출문제 PDF를 업로드하면 벡터 데이터베이스에 저장하고, 이를 기반으로 새로운 문제를 생성하거나 기출문제를 그대로 출제할 수 있습니다.

### 주요 기능
- 📄 **다중 시험 관리**: 여러 시험을 동시에 관리하고 PDF 업로드
- 🎯 **문제 생성 시스템**: 기출문제 기반 새 문제 생성 또는 기출문제 그대로 출제
- ✏️ **답변 평가**: RAG 기반 답변 평가 및 상세한 피드백 제공
- 💬 **AI 챗봇**: 시험별 RAG 기반 질의응답 시스템
- 📝 **오답노트**: Sequential Retry 방식의 체계적인 오답 관리
- 🔍 **출처 추적**: 모든 답변에 대한 기출문제 출처 정보 제공
- 🤖 **에이전트 검증**: Review Agent를 통한 문제 품질 자동 검증

## 2. 환경 설정

### 2.1 Python 환경
- **Python 3.11 이상** 필요 (권장: Python 3.11)
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
pip install gradio==4.44.0 python-dotenv==1.0.0 openai==1.51.2
pip install docling<2.37.0 PyPDF2==3.0.1 pdfplumber>=0.10.0
pip install faiss-cpu==1.7.4 sentence-transformers>=2.2.0 numpy>=1.24.0
pip install torch>=2.0.0 transformers>=4.30.0 colorama>=0.4.6
pip install pyngrok>=7.0.0  # ngrok 터널링 (선택사항)
pip install pandas>=1.5.0 pillow>=9.5.0 tqdm>=4.65.0  # 추가 유틸리티
```

## 3. Azure OpenAI 설정

### 3.1 Azure OpenAI 서비스 생성
1. Azure Portal에서 OpenAI 서비스 생성
2. 모델 배포 (GPT-4 또는 GPT-4o 권장)
3. API 키 및 엔드포인트 확인

### 3.2 환경 변수 설정
프로젝트 루트에 `.env` 파일을 생성하고 다음 내용을 추가:

```bash
# env.sample 파일을 .env로 복사하고 실제 값으로 수정
cp env.sample .env
```

또는 직접 `.env` 파일을 생성하고 다음 내용을 추가:

```env
OPENAI_API_KEY=your_azure_openai_api_key
AZURE_ENDPOINT=https://your-resource.openai.azure.com/
OPENAI_API_TYPE=azure
OPENAI_API_VERSION=2024-12-01-preview
DEPLOYMENT_NAME=your_deployment_name
```

## 4. 로컬 실행 방법

### 4.1 기본 실행
```bash
python mvp_main.py
```

### 4.2 웹 인터페이스 접속
- 브라우저에서 `http://localhost:7860` 접속
- Gradio 인터페이스 확인

## 5. 외부 접속 설정 (ngrok)

### 5.1 ngrok 설치
```bash
# Python 패키지로 설치 (권장)
pip install pyngrok

# 또는 실행파일 다운로드
# https://ngrok.com/download 에서 Windows용 다운로드
# 압축 해제 후 PATH에 추가
```

### 5.2 ngrok 계정 생성 및 설정
```bash
# 1. ngrok 계정 생성 (무료)
# https://dashboard.ngrok.com/signup 접속하여 계정 생성

# 2. authtoken 확인
# Dashboard에서 authtoken 복사

# 3. authtoken 설정
ngrok authtoken YOUR_AUTH_TOKEN_HERE
```

### 5.3 사용 방법

#### 방법 1: .env 파일 설정 (권장)
```bash
# .env 파일에서 설정
# USE_NGROK=true   # ngrok 사용 (기본값)
# USE_NGROK=false  # Gradio.live 사용
# PORT=7860        # 포트 설정 (기본값)

# 기본 실행
python mvp_main.py
```

#### 방법 2: 환경 변수로 임시 설정
```bash
# ngrok 비활성화하고 Gradio.live 사용
set USE_NGROK=false  # Windows
export USE_NGROK=false  # Linux/Mac
python mvp_main.py

# 다른 포트 사용
set PORT=8080  # Windows
export PORT=8080  # Linux/Mac
python mvp_main.py
```

#### 방법 3: 별도 터미널에서 ngrok 실행
```bash
# .env 파일에서 USE_NGROK=false 설정 후
# 터미널 1: 앱 실행 (ngrok 없이)
python mvp_main.py

# 터미널 2: ngrok 터널 생성
ngrok http 7860
```

### 5.4 ngrok 특징
- **안정성**: Gradio.live보다 안정적인 터널링
- **속도**: 빠른 응답 속도
- **무료 제한**: 월 40시간 (무료 계정)
- **URL**: 매번 랜덤 URL 생성 (유료: 고정 도메인 가능)
- **보안**: HTTPS 자동 지원

## 6. 사용 방법

### 6.1 시험 관리
1. **📚 시험 관리** 탭 선택
2. **시험 추가**: 새로운 시험 이름 입력 후 "시험 추가" 버튼 클릭
3. **PDF 업로드**: 
   - 시험 이름 입력
   - 기출문제 PDF 파일 선택
   - "PDF 업로드" 버튼 클릭
4. **PDF 목록 확인**: "PDF 목록 보기" 버튼으로 업로드된 파일 확인
5. **시험 제거**: 시험 선택 후 "시험 제거" 버튼으로 완전 삭제

### 6.2 문제 풀이
1. **📝 문제 풀이** 탭 선택
2. **시험 선택**: 드롭다운에서 원하는 시험 선택
3. **문제 생성 모드 선택**:
   - **기출문제 기반 새 문제 생성**: RAG 기반으로 새로운 문제 생성
   - **기출문제 그대로 출제**: 추출된 기출문제를 그대로 출제
4. **문제 생성**: "문제 생성" 버튼 클릭
5. **답변 입력**: 생성된 문제에 답변 입력
6. **답변 확인**: "답변 확인" 버튼으로 정답 여부 확인
7. **문제 초기화**: "문제 초기화" 버튼으로 문제 히스토리 초기화

### 6.3 AI 챗봇
1. **💬 AI 챗봇** 탭 선택
2. **시험 선택**: 질문할 시험 선택
3. **질문 입력**: 시험 관련 질문 입력
4. **하이브리드 답변**: 기출문제 검색 + LLM 지식 결합 답변 확인
5. **대화 초기화**: "대화 초기화" 버튼으로 대화 기록 삭제

### 6.4 오답노트
1. **📝 오답노트** 탭 선택
2. **시험 선택**: 오답을 확인할 시험 선택
3. **오답 재도전**: "오답 시험 재도전 하기" 버튼으로 Sequential Retry 시작
4. **정답 확인**: 답변 입력 후 "정답 확인" 버튼으로 평가
5. **기억했어요**: "기억했어요" 버튼으로 오답 목록에서 제거
6. **오답 전체 삭제**: "오답 전체 삭제" 버튼으로 모든 오답 삭제

## 7. 파일 구조

```
mvp/
├── mvp_main.py              # 메인 애플리케이션 (Gradio 인터페이스)
├── config.py                # 환경 변수 및 설정 관리
├── logger.py                # 로깅 시스템
├── prompt.py                # 프롬프트 정의 (Pythonic Prompting)
├── vector_store.py          # FAISS 벡터 스토어
├── pdf_processor.py         # PDF 처리 모듈 (Docling 활용)
├── review_agent_simple.py   # 문제 검토 에이전트
├── simple_txt_parser.py     # 텍스트 파싱 유틸리티
├── agents/                  # 에이전트 모듈
│   ├── __init__.py
│   ├── base_agent.py        # 기본 에이전트 클래스
│   └── information_validation_agent.py  # 정보 검증 에이전트
├── requirements.txt         # 필요한 라이브러리
├── startup.sh              # Azure App Service 시작 스크립트
├── web.config              # Azure 웹 설정 파일
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

## 8. 기술 스택

### 8.1 핵심 기술
- **Azure OpenAI**: GPT-4 기반 자연어 처리
- **Gradio**: 사용자 인터페이스 (포트 7860/8000)
- **FAISS**: 벡터 데이터베이스 (CPU 버전)
- **Sentence Transformers**: 텍스트 임베딩 (paraphrase-multilingual-MiniLM-L12-v2)
- **Docling**: PDF 텍스트 추출 및 문제 파싱
- **RAG**: 기출문제 검색 및 질의 응답

### 8.2 에이전트 시스템
- **Review Agent**: 생성된 문제의 품질 자동 검증
- **Information Validation Agent**: 컨텍스트 품질 검증
- **Base Agent**: 에이전트 기본 클래스

### 8.3 GPU 지원
- **Sentence Transformers**: GPU 가속 지원
- **Docling**: GPU 가속 지원 (OCR)
- **FAISS**: CPU 사용 (Python 3.11 호환성)
- **자동 Fallback**: GPU 사용 불가 시 CPU로 자동 전환

## 9. 문제 해결

### 9.1 라이브러리 설치 오류
```bash
# FAISS 설치 오류 시
pip install faiss-cpu --no-cache-dir

# sentence-transformers 설치 오류 시
pip install sentence-transformers --no-cache-dir

# Docling 설치 오류 시
pip install docling<2.37.0 --no-cache-dir
```

### 9.2 메모리 부족 오류
- PDF 파일 크기 줄이기
- 청크 크기 조정 (pdf_processor.py에서 chunk_size 수정)
- 그림 포함 문제 필터링 활성화

### 9.3 Azure OpenAI 연결 오류
- 환경 변수 확인 (.env 파일 또는 Azure App Settings)
- API 키 및 엔드포인트 정확성 확인
- 네트워크 연결 상태 확인
- DEPLOYMENT_NAME 정확성 확인

### 9.4 ngrok 연결 오류
- authtoken 설정 확인
- 방화벽 및 네트워크 설정 확인
- 무료 계정 제한 확인 (월 40시간)
- 대체 옵션: Gradio.live 사용 (USE_NGROK=false)

### 9.5 PDF 처리 오류
- PDF 파일 형식 및 크기 확인
- 그림 포함 문제 필터링 설정
- 문제 번호 패턴 확인 (1-999 범위)

## 10. 성능 최적화

### 10.1 벡터 검색 성능
- FAISS 인덱스 최적화
- 임베딩 모델 최적화 (GPU 가속)
- 검색 결과 중복 제거

### 10.2 PDF 처리 성능
- GPU 사용으로 OCR 속도 향상
- 청크 크기 최적화 (기본값 조정)
- 병렬 처리 활용

### 10.3 메모리 사용량 최적화
- 대용량 PDF 처리 시 청크 단위 처리
- 불필요한 데이터 자동 정리
- 메모리 사용량 모니터링

### 10.4 외부 접속 최적화
- ngrok 유료 계정으로 업그레이드 (고정 도메인, 무제한 시간)
- 방화벽 설정 최적화
- 네트워크 대역폭 고려

## 11. 확장 가능성

### 11.1 새로운 기능 추가
- 사용자 관리 시스템
- 학습 진도 추적
- 맞춤형 학습 경로 제안
- 다중 언어 지원

### 11.2 다른 문서 형식 지원
- Word 문서 (.docx)
- 텍스트 파일 (.txt)
- 이미지 기반 문서 (OCR)

### 11.3 배포 옵션 확장
- Docker 컨테이너화
- 클라우드 플랫폼 (Heroku, Railway, Render)
- VPS 서버 배포
- ngrok 대안 (frp, localtunnel 등)

### 11.4 AI 모델 확장
- 다른 LLM 모델 지원 (Claude, Gemini 등)
- 오픈소스 모델 통합 (Llama, Qwen 등)
- 모델 앙상블 기법 적용

## 12. 보안 고려사항

### 12.1 API 키 관리
- 환경 변수를 통한 민감 정보 보호 (.env 파일)
- 정기적인 API 키 교체
- 외부 시크릿 관리 도구 사용 권장

### 12.2 파일 업로드 보안
- PDF 파일 형식 검증
- 파일 크기 제한
- 악성 파일 스캔

### 12.3 데이터 보호
- 업로드된 PDF 로컬 저장 (암호화 권장)
- 개인정보 포함 문서 주의
- 데이터 백업 및 복원 계획 수립 