# 🎯 기출문제 RAG 기반 시험 문제 생성 및 질의 응답 챗봇

## 📋 프로젝트 개요

Azure OpenAI와 RAG를 활용한 **맞춤형 학습 시스템**. 기출문제 PDF를 업로드하면 벡터 데이터베이스에 저장하고, 이를 기반으로 새로운 문제를 생성하거나 기출문제를 그대로 출제할 수 있습니다.

### 🎯 핵심 가치
- **개인화 학습**: 기출문제 기반 맞춤형 문제 생성
- **자동화된 피드백**: 즉시 답변 평가 및 상세한 해설 제공
- **출처 추적**: 모든 답변에 대한 기출문제 출처 정보 제공
- **학습 효율성**: 오답노트를 통한 체계적인 학습 관리
- **에이전트 검증**: AI 에이전트 기반 문제 품질 자동 검증

---

## 🚀 주요 기능

### 📚 다중 시험 관리
- **시험 추가/제거**: 여러 시험을 동시에 관리
- **PDF 업로드**: 기출문제 PDF 자동 처리 및 벡터화
- **중복 방지**: PDF 해시 기반 중복 업로드 방지
- **데이터 영속성**: 시험 정보 및 PDF 메타데이터 자동 저장
- **완전 삭제**: 시험 제거 시 모든 관련 데이터 완전 삭제

### 🎯 문제 생성 시스템
#### 1. 기출문제 기반 새 문제 생성
- **RAG 기반 생성**: 기출문제를 참고한 새로운 문제 생성
- **난이도 자동 조절**: 쉬움/보통/어려움 랜덤 출제
- **문제 유형 다양화**: 객관식/주관식 랜덤 출제
- **컨텍스트 검증**: 생성된 문제의 품질 자동 검증 (검토 Agent)
- **하이브리드 검색**: 벡터 DB + 추출된 문제 결합 검색

#### 2. 기출문제 그대로 출제
- **문제 추출**: PDF에서 자동으로 문제 번호 및 내용 추출 (1-999 범위)
- **균등 분배**: 모든 PDF에서 균등하게 문제 선택
- **중복 방지**: 최근 출제된 문제 자동 제외 (히스토리 관리)
- **출처 추적**: 문제별 정확한 PDF 출처 정보 제공
- **그림 필터링**: 그림 포함 문제 자동 필터링

### ✏️ 답변 평가 시스템
- **RAG 기반 평가**: 기출문제 컨텍스트를 활용한 정확한 평가
- **상세한 피드백**: 정답 여부 및 개선점 제시
- **오답 자동 저장**: 틀린 문제를 오답노트에 자동 저장
- **해설 제공**: 정답과 상세한 해설 표시
- **출처 정보**: 평가 근거가 되는 기출문제 출처 제공

### 💬 AI 챗봇 (하이브리드 RAG + LLM)
- **시험별 질의응답**: 선택한 시험의 기출문제 기반 답변
- **하이브리드 답변**: 기출문제 검색 + LLM 지식 결합
- **다중 검색**: 키워드 추출 및 다양한 검색 쿼리 활용
- **정보 검증**: 에이전트 기반 답변 품질 검증
- **출처 정보**: 답변의 근거가 되는 기출문제 출처 제공

### 📝 오답노트 관리 (Sequential Retry)
- **Sequential Retry**: 틀린 문제를 순서대로 재도전
- **오답 통계**: 시험별 오답 횟수 및 통계 정보
- **정답 관리**: 오답 문제 정답 이후 자동 삭제
- **기억 표시**: "기억했어요" 기능으로 수동 삭제 가능
- **전체 삭제**: 시험별 오답 전체 삭제 기능

---

## 🏗️ 시스템 아키텍처

```
[사용자 (Gradio UI)]
        ↓
[다중 시험 관리] → [PDF 업로드] → [Docling PDF 처리] → [문제 추출 (1-999)] → [FAISS 벡터 DB]
        ↓
[문제 생성 모드 선택]
        ↓
[RAG 기반 검색] → [컨텍스트 검증] → [Review Agent] → [Azure OpenAI API]
        ↓
[문제 생성 / 답변 평가]
        ↓
[오답노트 관리] → [Sequential Retry]
        ↓
[Gradio UI 표시]
```

### 🔧 기술 스택

#### 핵심 기술
- **Azure OpenAI**: GPT-4 기반 자연어 처리
- **Gradio**: 직관적인 웹 인터페이스 (포트 7860/8000)
- **FAISS**: 고성능 벡터 데이터베이스 (CPU 버전)
- **Sentence Transformers**: 다국어 텍스트 임베딩 (paraphrase-multilingual-MiniLM-L12-v2)
- **Docling**: 고품질 PDF 텍스트 추출 및 문제 파싱
- **RAG**: 검색 기반 생성 모델

#### 에이전트 시스템
- **Review Agent**: 생성된 문제의 품질 자동 검증
- **Information Validation Agent**: 컨텍스트 품질 검증
- **Base Agent**: 에이전트 기본 클래스 및 확장 가능한 구조

#### GPU 가속 지원
- **Sentence Transformers**: GPU 가속으로 임베딩 속도 향상
- **Docling**: GPU 기반 OCR로 PDF 처리 속도 향상
- **자동 Fallback**: GPU 사용 불가 시 CPU로 자동 전환
- **메모리 최적화**: 대용량 PDF 처리 시 청크 단위 처리

---

## 🎨 사용자 인터페이스

### 📱 Gradio 기반 웹 인터페이스
- **탭 기반 구조**: 직관적인 기능별 탭 구성
  - 📚 **시험 관리**: 시험 추가/제거, PDF 업로드, 목록 관리
  - 📝 **문제 풀이**: 문제 생성, 답변 평가, 문제 초기화
  - 💬 **AI 챗봇**: 시험별 질의응답, 하이브리드 답변
  - 📝 **오답노트**: Sequential Retry, 오답 통계, 전체 삭제
- **반응형 디자인**: 다양한 화면 크기에 최적화
- **실시간 업데이트**: 드롭다운 메뉴 자동 동기화

### 🔄 사용자 플로우
1. **시험 설정** → PDF 업로드 → 벡터 DB 구축 → 문제 추출
2. **문제 생성** → 답변 입력 → 평가 및 피드백 → 오답 저장
3. **AI 챗봇** → 질문 입력 → 하이브리드 답변 → 출처 확인
4. **오답 관리** → Sequential Retry → 정답 확인 → 학습 완료

---

## 🔬 기술적 특징

### 🧠 고급 프롬프팅 (Pythonic Prompting)
- **모듈화된 프롬프트**: 기능별 프롬프트 분리 및 재사용
- **동적 프롬프트 생성**: 컨텍스트에 따른 프롬프트 최적화
- **품질 관리**: 프롬프트 기반 결과 검증
- **하이브리드 프롬프트**: 검색 결과 + LLM 지식 결합

### 🤖 에이전트 기반 시스템
- **정보 검증 에이전트**: 답변 품질 자동 검증
- **문제 검토 에이전트**: 생성된 문제의 적절성 검토
- **확장 가능한 구조**: 새로운 에이전트 쉽게 추가 가능
- **품질 보증**: 다단계 검증을 통한 높은 품질 보장

### 📊 데이터 관리
- **벡터 데이터베이스**: FAISS 기반 고성능 검색
- **메타데이터 관리**: 문제별 상세 정보 저장
- **자동 백업**: 중요 데이터 자동 저장 및 복원
- **해시 기반 중복 방지**: PDF 해시를 통한 중복 업로드 방지

### 🔍 검색 최적화
- **다중 검색 전략**: 키워드 + 의미론적 검색 결합
- **관련성 점수**: 검색 결과의 관련성 자동 평가
- **하이브리드 답변**: 검색 결과 + LLM 지식 결합
- **중복 제거**: 검색 결과 중복 제거 및 최적화

---

## 📈 성능 최적화

### 💾 메모리 효율성
- **청크 기반 처리**: 대용량 PDF 청크 단위 처리
- **자동 정리**: 불필요한 데이터 자동 제거
- **메모리 모니터링**: 실시간 메모리 사용량 추적
- **그림 필터링**: 그림 포함 문제 자동 필터링으로 성능 향상

### 🎯 정확도 향상
- **컨텍스트 검증**: 생성된 컨텍스트 품질 자동 검증
- **다중 검색**: 다양한 검색 쿼리로 정확도 향상
- **에이전트 검증**: AI 에이전트 기반 결과 검증
- **문제 번호 패턴**: 1-999 범위의 정확한 문제 번호 추출

### ⚡ 속도 최적화
- **GPU 가속**: 임베딩 및 OCR 처리 속도 향상
- **병렬 처리**: 다중 검색 쿼리 병렬 실행
- **캐싱**: 검색 결과 캐싱으로 응답 속도 향상
- **최적화된 인덱스**: FAISS 인덱스 최적화

---

## 🚀 빠른 시작

### 📋 요구사항
- Python 3.11 이상
- Azure OpenAI 서비스
- GPU 지원 (선택사항, 성능 향상)

### 🔧 로컬 설치
```bash
# 1. 저장소 클론
git clone https://github.com/ktds-jinseop-sim/ktds_ms_ai.git
cd mvp

# 2. 가상환경 생성 및 활성화
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# 3. 의존성 설치
pip install -r requirements.txt

# 4. 환경 변수 설정 (.env 파일 생성)
# env.sample 파일을 .env로 복사하고 실제 값으로 수정
cp env.sample .env
# 또는 직접 생성
OPENAI_API_KEY=your_api_key
AZURE_ENDPOINT=your_endpoint
DEPLOYMENT_NAME=your_deployment

# 5. 실행
python mvp_main.py
```

### 🌐 웹 접속
- 브라우저에서 `http://localhost:7860` 접속
- Gradio 인터페이스 확인

---

## ☁️ Azure App Service 배포

### 🏗️ 배포 아키텍처
```
[GitHub Repository] → [Azure App Service] → [Gradio Web App]
        ↓
[Azure OpenAI Service] ← [Python 3.11 Runtime] → [FAISS Vector DB]
        ↓
[사용자 접속: https://user05-mvp-gradio-app.azurewebsites.net]
```

### 🔧 Azure CLI 배포 가이드

#### 1. 사전 준비
```bash
# Azure CLI 설치 및 로그인
az login
az group list --output table
```

#### 2. App Service 리소스 생성
```bash
# App Service Plan 생성
az appservice plan create \
  --name user05-mvp-plan \
  --resource-group user05-RG \
  --sku B1 \
  --is-linux \
  --location eastus2

# Web App 생성
az webapp create \
  --resource-group user05-RG \
  --plan user05-mvp-plan \
  --name user05-mvp-gradio-app \
  --runtime "PYTHON|3.11"
```

#### 3. 배포 설정
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

#### 4. Git 배포
```bash
# Git 원격 저장소 추가
git remote add azure https://user05-deploy@user05-mvp-gradio-app.scm.azurewebsites.net/user05-mvp-gradio-app.git

# 배포
git add .
git commit -m "Azure App Service deployment configuration"
git push azure main:master
```

#### 5. 런타임 및 환경 설정
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

# 포트 및 환경 변수 설정
az webapp config appsettings set \
  --resource-group user05-RG \
  --name user05-mvp-gradio-app \
  --settings \
    PORT=8000 \
    DEPLOYMENT_BRANCH=main \
    OPENAI_API_KEY="your_azure_openai_api_key" \
    AZURE_ENDPOINT="https://your-resource.openai.azure.com/" \
    DEPLOYMENT_NAME="your_deployment_name"
```

#### 6. 모니터링 및 로그
```bash
# 로그 설정
az webapp log config \
  --resource-group user05-RG \
  --name user05-mvp-gradio-app \
  --application-logging filesystem \
  --level verbose

# 앱 재시작 및 상태 확인
az webapp restart --resource-group user05-RG --name user05-mvp-gradio-app
az webapp show --resource-group user05-RG --name user05-mvp-gradio-app --query "state"

# 실시간 로그 확인
az webapp log tail --resource-group user05-RG --name user05-mvp-gradio-app
```

### 🌍 배포 결과
- **앱 이름**: user05-mvp-gradio-app
- **URL**: https://user05-mvp-gradio-app.azurewebsites.net
- **상태**: Running
- **위치**: East US 2
- **런타임**: Python 3.11
- **포트**: 8000

---

## 📁 프로젝트 구조

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

---

## 🎯 핵심 특징

### 🚀 기술적 우수성
1. **Prompting**: 모듈화된 프롬프트로 재사용성 극대화
2. **에이전트 기반 검증**: AI 에이전트를 통한 자동 품질 관리
3. **하이브리드 RAG**: 검색 + 생성 모델의 최적 조합
4. **GPU 가속**: 전체 파이프라인 GPU 최적화
5. **다중 시험 관리**: 여러 시험 동시 관리 및 데이터 격리

### 🔮 확장 가능성
1. **다중 언어 지원**: 다양한 언어의 기출문제 처리
2. **사용자 관리**: 개인별 학습 진도 추적
3. **맞춤형 학습**: AI 기반 개인화 학습 경로 제안
4. **클라우드 배포**: Azure, AWS, GCP 등 다양한 클라우드 지원
5. **언어 모델 확장**: Open Source 모델 지원으로 비용 절감

### 🎓 교육적 가치
1. **학습 효율성**: 체계적인 오답 관리로 학습 효과 극대화
2. **자기주도학습**: 개인별 맞춤형 문제 제공
3. **신뢰성**: 기출문제 기반으로 높은 신뢰도
4. **접근성**: 책이 없는 시험도 PDF만으로 준비 가능
5. **출처 추적**: 모든 답변에 대한 명확한 출처 정보

### 💡 혁신적 기능
1. **Sequential Retry**: 오답 문제 순차적 재도전 시스템
2. **그림 필터링**: 그림 포함 문제 자동 필터링
3. **문제 번호 패턴**: 1-999 범위의 정확한 문제 번호 추출
4. **중복 방지**: 해시 기반 PDF 중복 업로드 방지
5. **완전 삭제**: 시험 제거 시 모든 관련 데이터 완전 삭제

---

## 🛠️ 개발 및 운영

### 🔧 개발 환경
- **Python**: 3.11 이상
- **IDE**: VS Code, PyCharm 등
- **Git**: 버전 관리 및 Azure 배포
- **Azure CLI**: 클라우드 배포 및 관리

### 📊 모니터링
- **Azure Application Insights**: 성능 모니터링
- **로그 분석**: 실시간 로그 확인
- **사용자 분석**: 사용 패턴 분석
- **오류 추적**: 자동 오류 감지 및 알림