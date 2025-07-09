"""
간단한 문제 검토 에이전트 (LangChain 없이)
"""

import logging
from typing import Dict, Any, List
import openai
from config import Config

logger = logging.getLogger(__name__)

class SimpleReviewAgent:
    """간단한 문제 검토 에이전트"""
    
    def __init__(self):
        self.name = "Simple Review Agent"
        logger.info(f"✅ {self.name} 초기화 완료")
    
    def review_question(self, question: str, answer: str, explanation: str, exam_name: str = "정보시스템감리사") -> Dict[str, Any]:
        """문제 검토"""
        try:
            logger.info("🔍 문제 검토 시작")
            
            # 검토 프롬프트 생성
            review_prompt = self._create_review_prompt(question, answer, explanation)
            
            # Azure OpenAI 설정은 이미 mvp_main.py에서 설정됨
            
            # Azure OpenAI API 호출
            response = openai.chat.completions.create(
                model=Config.DEPLOYMENT_NAME,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": review_prompt}
                ],
                temperature=0.3,
            )
            
            if response.choices[0].message.content:
                # 검토 결과 파싱
                review_result = self._parse_review_result(response.choices[0].message.content)
                
                logger.info(f"✅ 문제 검토 완료 - 점수: {review_result.get('score', 0)}")
                return review_result
            else:
                logger.error("❌ 문제 검토 실패")
                return {"is_valid": False, "score": 0, "issues": ["검토 실패"], "suggestions": []}
                
        except Exception as e:
            logger.error(f"❌ 문제 검토 중 오류: {e}")
            return {"is_valid": False, "score": 0, "issues": [f"오류: {e}"], "suggestions": []}
    
    def _get_system_prompt(self) -> str:
        """시스템 프롬프트"""
        return """당신은 시험 문제 검토 전문가입니다. 다음 기준으로 문제를 검토해주세요:

1. **문제 명확성**: 문제가 명확하고 모호하지 않은가?
2. **정답 일치성**: 정답이 보기 중에 정확히 존재하는가?
3. **보기 적절성**: 모든 보기가 문제와 관련이 있고 적절한가?
4. **난이도 적절성**: 문제의 난이도가 적절한가?
5. **해설 정확성**: 해설이 정답을 정확히 설명하는가?

검토 결과를 다음 형식으로 응답해주세요:

=== 검토 결과 ===
유효성: [적합/부적합]
점수: [1-10]
문제점: [발견된 문제점들]
개선 제안: [구체적인 수정 제안들]
"""
    
    def _create_review_prompt(self, question: str, answer: str, explanation: str) -> str:
        """검토 프롬프트 생성"""
        return f"""
다음 시험 문제를 검토해주세요:

=== 문제 ===
{question}

=== 정답 ===
{answer}

=== 해설 ===
{explanation}

위 문제를 위의 기준에 따라 철저히 검토하고, 문제가 있다면 구체적인 수정 제안을 해주세요.
"""
    
    def _parse_review_result(self, result: str) -> Dict[str, Any]:
        """검토 결과 파싱"""
        lines = result.split('\n')
        review_result = {
            "is_valid": False,
            "score": 0,
            "issues": [],
            "suggestions": []
        }
        
        in_review_section = False
        in_suggestions_section = False
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            if "=== 검토 결과 ===" in line:
                in_review_section = True
                in_suggestions_section = False
                continue
            elif "=== 개선 제안 ===" in line:
                in_review_section = False
                in_suggestions_section = True
                continue
            elif "===" in line:
                in_review_section = False
                in_suggestions_section = False
                continue
            
            if in_review_section:
                if "유효성:" in line:
                    review_result["is_valid"] = "적합" in line
                elif "점수:" in line:
                    try:
                        score_text = line.split(":")[1].strip()
                        review_result["score"] = int(score_text)
                    except:
                        review_result["score"] = 5
                elif "문제점:" in line:
                    continue
                elif line.startswith("-") or line.startswith("*"):
                    review_result["issues"].append(line[1:].strip())
            elif in_suggestions_section:
                if line.startswith("-") or line.startswith("*"):
                    review_result["suggestions"].append(line[1:].strip())
        
        return review_result
    
    def apply_corrections(self, question: str, answer: str, explanation: str, suggestions: List[str]) -> Dict[str, str]:
        """수정 제안 적용"""
        try:
            correction_prompt = f"""
다음 문제에 대한 수정 제안을 적용해주세요:

=== 원본 문제 ===
{question}

=== 정답 ===
{answer}

=== 해설 ===
{explanation}

=== 수정 제안 ===
{chr(10).join(suggestions)}

위 수정 제안을 반영하여 문제를 개선해주세요. 다음 형식으로 응답해주세요:

=== 수정된 문제 ===
[수정된 문제 내용]

=== 수정된 정답 ===
[수정된 정답]

=== 수정된 해설 ===
[수정된 해설]
"""
            
            # Azure OpenAI 설정은 이미 mvp_main.py에서 설정됨
            
            response = openai.chat.completions.create(
                model=Config.DEPLOYMENT_NAME,
                messages=[
                    {"role": "system", "content": "당신은 시험 문제 수정 전문가입니다. 제안된 수정사항을 반영하여 문제를 개선해주세요."},
                    {"role": "user", "content": correction_prompt}
                ],
                temperature=0.3,
            )
            
            if response.choices[0].message.content:
                return self._parse_corrected_result(response.choices[0].message.content)
            
            return {}
            
        except Exception as e:
            logger.error(f"❌ 수정 적용 실패: {e}")
            return {}
    
    def _parse_corrected_result(self, result: str) -> Dict[str, str]:
        """수정된 결과 파싱"""
        lines = result.split('\n')
        corrected = {
            "question": "",
            "answer": "",
            "explanation": ""
        }
        
        current_section = ""
        
        for line in lines:
            if "=== 수정된 문제 ===" in line:
                current_section = "question"
                continue
            elif "=== 수정된 정답 ===" in line:
                current_section = "answer"
                continue
            elif "=== 수정된 해설 ===" in line:
                current_section = "explanation"
                continue
            elif "===" in line:
                current_section = ""
                continue
            
            if current_section and line.strip():
                if current_section == "question":
                    corrected["question"] += line + "\n"
                elif current_section == "answer":
                    corrected["answer"] += line + "\n"
                elif current_section == "explanation":
                    corrected["explanation"] += line + "\n"
        
        # 마지막 줄바꿈 제거
        for key in corrected:
            corrected[key] = corrected[key].strip()
        
        return corrected

# 전역 인스턴스
review_agent = SimpleReviewAgent() 