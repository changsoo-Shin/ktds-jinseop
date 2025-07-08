"""
정보 검증 및 필터링 에이전트
RAG 검색 결과에서 틀린 답이나 부정확한 정보를 필터링하고 올바른 정보만 제공
"""

import logging
import re
from typing import Dict, Any, List, Tuple
from .base_agent import BaseAgent, AgentState
from prompt import ExamPrompts
import openai
from config import Config

logger = logging.getLogger(__name__)

class InformationValidationAgent(BaseAgent):
    """정보 검증 및 필터링 에이전트"""
    
    def __init__(self):
        super().__init__("Information Validation Agent")
        self.validation_cache = {}  # 검증 결과 캐시
    
    def process(self, state: AgentState) -> AgentState:
        """정보 검증 및 필터링 처리"""
        try:
            self.log_activity("정보 검증 시작", {"query": state.user_query})
            
            if not state.user_query:
                return self.handle_error(state, "사용자 질문이 없습니다.")
            
            # RAG 검색 결과 검증
            metadata_list = state.metadata if isinstance(state.metadata, list) else []
            context_str = state.context or ""
            validated_context = self._validate_rag_results(
                state.user_query, 
                context_str, 
                metadata_list
            )
            
            # 검증된 컨텍스트로 상태 업데이트
            state.context = validated_context["filtered_context"]
            state.metadata = validated_context.get("filtered_metadata", state.metadata)
            
            # 검증 메타데이터 저장
            state.metadata = state.metadata or {}
            state.metadata["validation"] = {
                "original_chunks": validated_context["original_chunks"],
                "filtered_chunks": validated_context["filtered_chunks"],
                "removed_chunks": validated_context["removed_chunks"],
                "validation_reasons": validated_context["validation_reasons"]
            }
            
            self.log_activity("정보 검증 완료", {
                "original_chunks": validated_context["original_chunks"],
                "filtered_chunks": validated_context["filtered_chunks"],
                "removed_chunks": validated_context["removed_chunks"]
            })
            
            return state
            
        except Exception as e:
            return self.handle_error(state, f"정보 검증 중 오류: {e}")
    
    def _validate_rag_results(self, query: str, context: str, metadata: List[Dict] = []) -> Dict[str, Any]:
        """RAG 검색 결과 검증 및 필터링"""
        try:
            if not context:
                return {
                    "filtered_context": "",
                    "filtered_metadata": metadata,
                    "original_chunks": 0,
                    "filtered_chunks": 0,
                    "removed_chunks": 0,
                    "validation_reasons": []
                }
            
            # 컨텍스트를 청크로 분할
            chunks = self._split_context_into_chunks(context)
            original_chunks = len(chunks)
            
            # 각 청크 검증
            validated_chunks = []
            removed_chunks = []
            validation_reasons = []
            
            for i, chunk in enumerate(chunks):
                validation_result = self._validate_chunk(query, chunk, i, metadata)
                
                if validation_result["is_valid"]:
                    validated_chunks.append(chunk)
                    if validation_result["reason"]:
                        validation_reasons.append(f"청크 {i+1}: {validation_result['reason']}")
                else:
                    removed_chunks.append(chunk)
                    validation_reasons.append(f"청크 {i+1} 제거: {validation_result['reason']}")
            
            # 필터링된 컨텍스트 재구성
            filtered_context = "\n\n".join(validated_chunks)
            
            # 메타데이터 필터링 (제거된 청크에 해당하는 메타데이터도 제거)
            metadata_list = metadata or []
            filtered_metadata = self._filter_metadata(metadata_list, len(validated_chunks), original_chunks)
            
            return {
                "filtered_context": filtered_context,
                "filtered_metadata": filtered_metadata,
                "original_chunks": original_chunks,
                "filtered_chunks": len(validated_chunks),
                "removed_chunks": len(removed_chunks),
                "validation_reasons": validation_reasons
            }
            
        except Exception as e:
            logger.error(f"❌ RAG 결과 검증 실패: {e}")
            metadata_list = metadata or []
            return {
                "filtered_context": context,  # 검증 실패시 원본 반환
                "filtered_metadata": metadata_list,
                "original_chunks": 1,
                "filtered_chunks": 1,
                "removed_chunks": 0,
                "validation_reasons": [f"검증 실패: {e}"]
            }
    
    def _split_context_into_chunks(self, context: str) -> List[str]:
        """컨텍스트를 의미있는 청크로 분할"""
        # 기출문제 패턴으로 분할
        patterns = [
            r'\d+\.\s*[^\n]+(?:\n(?!\d+\.)[^\n]*)*',  # 번호가 있는 문제
            r'[①②③④⑤⑥⑦⑧⑨⑩]\s*[^\n]+(?:\n(?![①②③④⑤⑥⑦⑧⑨⑩])[^\n]*)*',  # 한자 번호가 있는 문제
            r'[가-힣]\s*[^\n]+(?:\n(?![가-힣]\s)[^\n]*)*',  # 한글 번호가 있는 문제
        ]
        
        chunks = []
        remaining_text = context
        
        for pattern in patterns:
            matches = re.finditer(pattern, remaining_text, re.MULTILINE)
            for match in matches:
                chunk = match.group().strip()
                if len(chunk) > 10:  # 최소 길이 체크
                    chunks.append(chunk)
        
        # 패턴으로 분할되지 않은 경우 문단 단위로 분할
        if not chunks:
            paragraphs = [p.strip() for p in context.split('\n\n') if p.strip()]
            chunks = [p for p in paragraphs if len(p) > 20]
        
        return chunks
    
    def _validate_chunk(self, query: str, chunk: str, chunk_index: int, metadata: List[Dict] = []) -> Dict[str, Any]:
        """개별 청크 검증"""
        try:
            # 캐시 확인
            cache_key = f"{query}_{hash(chunk)}"
            if cache_key in self.validation_cache:
                return self.validation_cache[cache_key]
            
            # 검증 프롬프트 생성
            prompt = self._create_validation_prompt(query, chunk, chunk_index, metadata)
            
            # 직접 OpenAI API 호출
            response = self.llm.chat.completions.create(
                model=Config.DEPLOYMENT_NAME,
                messages=[
                    {"role": "system", "content": ExamPrompts.get_system_prompts("정보시스템감리사")["validation_assistant"]},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )
            
            response_content = response.choices[0].message.content if response.choices else ""
            if response_content:
                validation_result = self._parse_validation_result(response_content)
            else:
                validation_result = {
                    "is_valid": True,
                    "confidence": 0.5,
                    "reason": "응답이 비어있음"
                }
            
            # 캐시에 저장
            self.validation_cache[cache_key] = validation_result
            
            return validation_result
            
        except Exception as e:
            logger.error(f"❌ 청크 검증 실패: {e}")
            return {
                "is_valid": True,  # 검증 실패시 기본적으로 유효하다고 가정
                "reason": f"검증 실패: {e}",
                "confidence": 0.5
            }
    
    def _create_validation_prompt(self, query: str, chunk: str, chunk_index: int, metadata: List[Dict] = []) -> str:
        """검증 프롬프트 생성"""
        metadata_info = ""
        metadata_list = metadata or []
        if metadata_list and chunk_index < len(metadata_list):
            meta = metadata_list[chunk_index]
            metadata_info = f"\n=== 메타데이터 ===\n"
            if meta.get("pdf_source"):
                metadata_info += f"PDF 파일: {meta.get('pdf_source')}\n"
            if meta.get("subject"):
                metadata_info += f"과목: {meta.get('subject')}\n"
            if meta.get("question_number"):
                metadata_info += f"문제 번호: {meta.get('question_number')}\n"
        
        return f"""
다음 기출문제 청크가 사용자 질문에 대해 올바른 정보를 제공하는지 검증해주세요.

=== 사용자 질문 ===
{query}

=== 검증할 기출문제 청크 ===
{chunk}
{metadata_info}

다음 기준으로 검증해주세요:

1. **정보 정확성**: 청크의 내용이 사실적으로 정확한가?
2. **관련성**: 질문과 관련된 정보를 포함하고 있는가?
3. **완전성**: 질문에 답하기에 충분한 정보를 제공하는가?
4. **오답 포함 여부**: 틀린 답이나 오해를 줄 수 있는 정보가 포함되어 있는가?
5. **모호성**: 모호하거나 불명확한 정보가 있는가?

다음 형식으로 응답해주세요:

=== 검증 결과 ===
유효성: [유효/무효]
신뢰도: [0.0-1.0]
이유: [구체적인 검증 이유]

=== 문제점 분석 ===
[발견된 문제점들]

=== 개선 제안 ===
[개선 방안]
"""
    
    def _parse_validation_result(self, response: str) -> Dict[str, Any]:
        """검증 결과 파싱"""
        try:
            # 유효성 추출
            is_valid = "유효" in response.split("유효성:")[1].split("\n")[0] if "유효성:" in response else True
            
            # 신뢰도 추출
            confidence_match = re.search(r'신뢰도:\s*([0-9.]+)', response)
            confidence = float(confidence_match.group(1)) if confidence_match else 0.5
            
            # 이유 추출
            reason_match = re.search(r'이유:\s*(.+?)(?:\n|$)', response)
            reason = reason_match.group(1).strip() if reason_match else "검증 완료"
            
            return {
                "is_valid": is_valid,
                "confidence": confidence,
                "reason": reason
            }
            
        except Exception as e:
            logger.error(f"❌ 검증 결과 파싱 실패: {e}")
            return {
                "is_valid": True,
                "confidence": 0.5,
                "reason": f"파싱 실패: {e}"
            }
    
    def _filter_metadata(self, metadata: List[Dict], filtered_count: int, original_count: int) -> List[Dict]:
        """메타데이터 필터링"""
        if not metadata:
            return []
        
        # 필터링된 청크 수에 맞춰 메타데이터 조정
        if filtered_count < original_count:
            return metadata[:filtered_count]
        
        return metadata
    
    def validate_single_chunk(self, query: str, chunk: str) -> Dict[str, Any]:
        """단일 청크 검증 (외부에서 호출 가능)"""
        return self._validate_chunk(query, chunk, 0, [])
    
    def get_validation_statistics(self) -> Dict[str, Any]:
        """검증 통계 반환"""
        return {
            "cache_size": len(self.validation_cache),
            "agent_name": self.name
        } 