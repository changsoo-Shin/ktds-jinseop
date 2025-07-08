"""
기본 에이전트 클래스
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from pydantic import BaseModel
import logging
import openai
from config import Config

logger = logging.getLogger(__name__)

class AgentState(BaseModel):
    """에이전트 간 공유 상태"""
    exam_name: Optional[str] = None
    pdf_file: Optional[str] = None
    chunks: Optional[list] = None
    question: Optional[str] = None
    answer: Optional[str] = None
    explanation: Optional[str] = None
    user_answer: Optional[str] = None
    user_query: Optional[str] = None  # 사용자 질문
    context: Optional[str] = None  # RAG 컨텍스트
    evaluation_result: Optional[str] = None
    wrong_answers: Optional[list] = None
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    success: bool = True

class BaseAgent(ABC):
    """모든 에이전트의 기본 클래스"""
    
    def __init__(self, name: str):
        self.name = name
        self.llm = self._setup_llm()
        logger.info(f"✅ {self.name} 에이전트 초기화 완료")
    
    def _setup_llm(self):
        """LLM 설정"""
        try:
            # Azure OpenAI 설정은 이미 mvp_main.py에서 설정됨
            logger.debug(f"🔧 [DEBUG] Azure OpenAI 설정 확인")
            logger.debug(f"🔧 [DEBUG] Endpoint: {Config.AZURE_ENDPOINT}")
            logger.debug(f"🔧 [DEBUG] Deployment: {Config.DEPLOYMENT_NAME}")
            
            return openai
        except Exception as e:
            logger.error(f"❌ LLM 설정 실패: {e}")
            raise
    
    @abstractmethod
    def process(self, state: AgentState) -> AgentState:
        """에이전트 처리 로직 (하위 클래스에서 구현)"""
        pass
    
    def log_activity(self, activity: str, details: Optional[Dict[str, Any]] = None):
        """에이전트 활동 로깅"""
        logger.info(f"🤖 [{self.name}] {activity}")
        if details:
            logger.debug(f"📝 [{self.name}] 상세: {details}")
    
    def handle_error(self, state: AgentState, error: str):
        """에러 처리"""
        state.error = error
        state.success = False
        logger.error(f"❌ [{self.name}] 오류: {error}")
        return state 