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

## 5. Azure App Service 배포

### 5.1 Azure CLI 설치 및 로그인
```bash
# Azure CLI 설치 (Windows)
# https://docs.microsoft.com/en-us/cli/azure/install-azure-cli-windows

# Azure 로그인
az login

# 리소스 그룹 확인
az group list --output table
```

### 5.2 App Service Plan 생성
```bash
# App Service Plan 생성 (Linux B1 SKU)
az appservice plan create \
  --name user05-mvp-plan \
  --resource-group user05-RG \
  --sku B1 \
  --is-linux \
  --location eastus2
```

### 5.3 Web App 생성
```bash
# Web App 생성 (Python 3.11 런타임)
az webapp create \
  --resource-group user05-RG \
  --plan user05-mvp-plan \
  --name user05-mvp-gradio-app \
  --runtime "PYTHON|3.11"
```

### 5.4 배포 설정
```bash
# 빌드 설정 활성화
az webapp config appsettings set \
  --resource-group user05-RG \
  --name user05-mvp-gradio-app \
  --settings SCM_DO_BUILD_DURING_DEPLOYMENT=true

# Git 배포 설정
az webapp deployment source config-local-git \
  --resource-group user05-RG \
  --name user05-mvp-gradio-app

# 배포 자격 증명 설정
az webapp deployment user set \
  --user-name user05-deploy \
  --password Deploy123!
```

### 5.5 Git 원격 저장소 추가 및 배포
```bash
# Git 원격 저장소 추가
git remote add azure https://user05-deploy@user05-mvp-gradio-app.scm.azurewebsites.net/user05-mvp-gradio-app.git

# 변경사항 커밋
git add .
git commit -m "Azure App Service deployment configuration"

# Azure에 배포
git push azure main:master
```

### 5.6 App Service 설정
```bash
# Python 런타임 설정
az webapp config set \
  --resource-group user05-RG \
  --name user05-mvp-gradio-app \
  --linux-fx-version "PYTHON|3.11"

# Startup 명령 설정
az webapp config set \
  --resource-group user05-RG \
  --name user05-mvp-gradio-app \
  --startup-file "python mvp_main.py"

# 포트 설정
az webapp config appsettings set \
  --resource-group user05-RG \
  --name user05-mvp-gradio-app \
  --settings PORT=8000

# 배포 브랜치 설정
az webapp config appsettings set \
  --resource-group user05-RG \
  --name user05-mvp-gradio-app \
  --settings DEPLOYMENT_BRANCH=main
```

### 5.7 로그 설정 및 모니터링
```bash
# 로그 설정 활성화
az webapp log config \
  --resource-group user05-RG \
  --name user05-mvp-gradio-app \
  --application-logging filesystem \
  --level verbose

# 앱 재시작
az webapp restart \
  --resource-group user05-RG \
  --name user05-mvp-gradio-app

# 상태 확인
az webapp show \
  --resource-group user05-RG \
  --name user05-mvp-gradio-app \
  --query "state"

# 실시간 로그 확인
az webapp log tail \
  --resource-group user05-RG \
  --name user05-mvp-gradio-app
```

### 5.8 환경 변수 설정 (Azure)
```bash
# Azure OpenAI 환경 변수 설정
az webapp config appsettings set \
  --resource-group user05-RG \
  --name user05-mvp-gradio-app \
  --settings \
    OPENAI_API_KEY="your_azure_openai_api_key" \
    AZURE_ENDPOINT="https://your-resource.openai.azure.com/" \
    OPENAI_API_TYPE="azure" \
    OPENAI_API_VERSION="2024-12-01-preview" \
    DEPLOYMENT_NAME="your_deployment_name"
```

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

### 9.4 Azure App Service 배포 오류
- Python 런타임 버전 확인 (3.11)
- PORT 환경 변수 설정 (8000)
- startup.sh 파일 권한 확인
- 로그를 통한 오류 진단

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

### 10.4 Azure App Service 최적화
- App Service Plan 스케일링 (B1 → S1/P1V2)
- Always On 설정 활성화
- 로그 레벨 최적화

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

### 11.3 클라우드 배포 확장
- Azure Container Apps
- Azure Functions
- Azure Kubernetes Service
- 다중 리전 배포

### 11.4 AI 모델 확장
- 다른 LLM 모델 지원 (Claude, Gemini 등)
- 오픈소스 모델 통합 (Llama, Qwen 등)
- 모델 앙상블 기법 적용

## 12. 보안 고려사항

### 12.1 API 키 관리
- Azure Key Vault 사용 권장
- 환경 변수를 통한 민감 정보 보호
- 정기적인 API 키 교체

### 12.2 파일 업로드 보안
- PDF 파일 형식 검증
- 파일 크기 제한
- 악성 파일 스캔

### 12.3 데이터 보호
- 업로드된 PDF 암호화 저장
- 사용자 데이터 익명화
- GDPR 준수 고려 