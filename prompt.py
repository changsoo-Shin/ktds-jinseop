"""
정보시스템감리사 시험 문제 생성 및 평가를 위한 프롬프트 정의
"""

import logging

# 로거 설정
logger = logging.getLogger(__name__)

class ExamPrompts:
    """시험 관련 프롬프트 클래스"""
    
    # 시스템 프롬프트들 (동적 시험명 지원)
    @staticmethod
    def get_system_prompts(exam_name: str = "정보시스템감리사"):
        """시스템 프롬프트 반환 (시험명 동적 적용)"""
        return {
            "question_generator": f"당신은 {exam_name} 시험 전문가입니다. 실제 기출문제와 유사한 수준의 문제를 생성해주세요.",
            "answer_evaluator": f"당신은 {exam_name} 시험 채점 전문가입니다. 공정하고 정확한 평가를 해주세요.",
            "chat_assistant": f"당신은 {exam_name} 시험 준비를 도와주는 친근한 학습 도우미입니다.",
            "rag_assistant": f"""당신은 {exam_name} 기출문제 데이터베이스 전문가입니다.

**중요한 규칙**:
1. 반드시 제공된 기출문제 컨텍스트만을 기반으로 답변하세요.
2. 기출문제에 없는 내용은 절대 답변하지 마세요.
3. 일반적인 지식이나 다른 정보를 추가하지 마세요.
4. 기출문제 내용을 정확히 인용하고 해석하여 답변하세요.
5. 기출문제에 관련 내용이 없으면 "기출문제에서 해당 내용을 찾을 수 없습니다"라고 답변하세요.

당신의 역할은 기출문제를 정확히 해석하고 관련 내용을 찾아 답변하는 것입니다.""",
            "hybrid_assistant": f"""당신은 {exam_name} 전문가이자 기출문제 분석가입니다.

**중요한 규칙**:
1. 기출문제 컨텍스트를 우선적으로 참고하여 답변하세요.
2. 기출문제에 관련 내용이 있으면 정확히 인용하고 해석하세요.
3. 기출문제 내용이 부족한 경우, 당신의 전문 지식을 보완적으로 사용하세요.
4. 기출문제 내용과 추가 설명을 명확히 구분해주세요.
5. 답변은 구체적이고 자세하게 작성하세요.
6. 기출문제 출처를 명시해주세요.

당신의 역할은 기출문제와 전문 지식을 결합하여 정확하고 유용한 답변을 제공하는 것입니다.""",
            "validation_assistant": f"""당신은 {exam_name} 기출문제 검증 전문가입니다.

**중요한 규칙**:
1. 기출문제의 정확성과 신뢰성을 철저히 검증하세요.
2. 틀린 답이나 오해를 줄 수 있는 정보를 식별하세요.
3. 객관적이고 사실적인 기준으로 검증하세요.
4. 검증 결과에 대한 명확한 근거를 제시하세요.
5. 검증 과정에서 발견된 문제점을 구체적으로 설명하세요.

당신의 역할은 기출문제의 품질을 보장하고 사용자에게 정확한 정보를 제공하는 것입니다."""
        }
    
    @staticmethod
    def get_question_generation_prompt(subject: str, difficulty: str, question_type: str, exam_name: str = "정보시스템감리사") -> str:
        """문제 생성 프롬프트"""
        return f"""
        {exam_name} 시험에 맞는 문제를 생성해주세요.
        
        시험: {exam_name}
        난이도: {difficulty}
        문제 유형: {question_type}
        
        **중요 사항**:
        1. 객관식 문제인 경우 반드시 4개의 보기를 제공해주세요.
        2. 보기가 없는 경우 "보기 없음"이라고 표시하지 말고, 적절한 보기를 생성해주세요.
        3. "아래와 같이 요청하신 형식에 맞추어 정리해드립니다" 같은 불필요한 텍스트는 포함하지 마세요.
        
        다음 형식으로 응답해주세요:
        === 문제 ===
        [문제 내용]
        
        === 보기 ===
        1) [보기1]
        2) [보기2]
        3) [보기3]
        4) [보기4]
        
        === 정답 ===

        [정답 번호]
        
        === 해설 ===

        [문제 해설 및 관련 개념 설명]
        
        === 문제 정보 ===
        난이도: {difficulty}
        유형: {question_type}
        """
    
    @staticmethod
    def get_rag_question_generation_prompt(subject: str, difficulty: str, question_type: str, 
                                          context: str, exam_name: str = "정보시스템감리사", metadata: list | None = None) -> str:
        """RAG 기반 문제 생성 프롬프트"""
        # 메타데이터 정보 구성
        metadata_info = ""
        if metadata:
            metadata_info = "\n=== 메타데이터 정보 ===\n"
            for i, meta in enumerate(metadata, 1):
                metadata_info += f"참고 자료 {i}:\n"
                if meta.get("pdf_source"):
                    metadata_info += f"- PDF 파일: {meta.get('pdf_source')}\n"
                if meta.get("subject"):
                    metadata_info += f"- 과목: {meta.get('subject')}\n"
                if meta.get("created_at"):
                    created_date = meta.get("created_at")[:10]  # YYYY-MM-DD만 추출
                    metadata_info += f"- 생성일: {created_date}\n"
                if meta.get("id"):
                    metadata_info += f"- 문서 ID: {meta.get('id')[:8]}...\n"
                metadata_info += "\n"
        
        return f"""
        다음 기출문제 컨텍스트를 바탕으로 {exam_name} 시험 문제를 생성해주세요.
        
        시험: {exam_name}
        난이도: {difficulty}
        문제 유형: {question_type}
        
        === 기출문제 컨텍스트 ===
        {context}
        {metadata_info}
        
        **중요 사항**:
        1. 컨텍스트에 표가 포함되어 있다면, 표의 내용을 문제에 반드시 포함시켜주세요.
        2. 표는 Markdown 형식(| | |)으로 되어 있으며, 이를 그대로 문제에 포함해야 합니다.
        3. 객관식 문제인 경우 반드시 4개의 보기를 제공해주세요.
        4. 보기가 없는 경우 "보기 없음"이라고 표시하지 말고, 적절한 보기를 생성해주세요.
        5. "아래와 같이 요청하신 형식에 맞추어 정리해드립니다" 같은 불필요한 텍스트는 포함하지 마세요.
        6. 메타데이터 정보를 참고하여 정확한 출처 정보를 문제에 포함시켜주세요.
        
        다음 형식으로 응답해주세요:
        === 문제 ===
        [문제 내용]
        
        === 보기 ===
        1) [보기1]
        2) [보기2]
        3) [보기3]
        4) [보기4]
        
        === 정답 ===

        [정답 번호]
        
        === 해설 ===

        [문제 해설 및 관련 개념 설명]
        
        === 문제 정보 ===
        난이도: {difficulty}
        유형: {question_type}
        """
    
    @staticmethod
    def get_exact_question_prompt(context: str, exam_name: str = "정보시스템감리사") -> str:
        """기출문제 그대로 출제 프롬프트"""
        return f"""
        다음 {exam_name}의 기출문제를 그대로 출제해주세요.
        
        === 기출문제 컨텍스트 ===
        {context}
        
        **중요 사항**:
        1. 객관식 문제인 경우 반드시 4개의 보기를 제공해주세요.
        2. 보기가 없는 경우 "보기 없음"이라고 표시하지 말고, 적절한 보기를 생성해주세요.
        3. "아래와 같이 요청하신 형식에 맞추어 정리해드립니다" 같은 불필요한 텍스트는 포함하지 마세요.
        4. **안전한 내용만 생성**: 폭력, 위험, 해로운 내용은 포함하지 말고 교육적이고 건설적인 내용만 다루세요.
        5. **학습 목적**: 이는 교육 및 학습 목적으로만 사용되는 시험 문제입니다.
        
        다음 형식으로 응답해주세요:
        === 문제 ===
        [기출문제 내용 - 표가 있다면 표도 포함]
        
        === 보기 ===
        1) [보기1]
        2) [보기2]
        3) [보기3]
        4) [보기4]
        
        === 정답 ===

        [정답 번호]
        
        === 해설 ===

        [기출문제 해설]
        
        === 문제 정보 ===
        난이도: [난이도]
        유형: [문제 유형]
        """
    
    @staticmethod
    def get_answer_evaluation_prompt(question: str, user_answer: str) -> str:
        """답변 평가 프롬프트"""
        return f"""
        정보시스템감리사 시험 문제에 대한 답변을 평가해주세요:
        
        문제:
        {question}
        
        사용자 답변: {user_answer}
        
        다음 형식으로 응답해주세요:
        === 평가 결과 ===\n
        정답 여부: [맞음/틀림]
        점수: [점수/만점]
        
        === 피드백 ===\n
        [구체적인 피드백 및 개선점]
        
        === 관련 개념 ===\n
        [문제와 관련된 핵심 개념 설명]
        
        === 출처 ===\n
        [관련 기출문제 출처 - 메타데이터 기반]
        """
    
    @staticmethod
    def get_rag_answer_evaluation_prompt(question: str, user_answer: str, context: str, metadata: list = []) -> str:
        """RAG 기반 답변 평가 프롬프트"""
        # 출처 정보 구성
        source_info = ""
        if metadata:
            sources = []
            for meta in metadata:
                if meta.get("pdf_source"):
                    sources.append(meta.get("pdf_source"))
                elif meta.get("type") == "extracted_question":
                    sources.append("추출된 기출문제")
            
            if sources:
                unique_sources = list(set(sources))
                if len(unique_sources) == 1:
                    source_info = f"{unique_sources[0]}"
                else:
                    source_info = f"{', '.join(unique_sources)}"
        
        return f"""
        다음 기출문제 컨텍스트를 바탕으로 답변을 평가해주세요:
        
        === 기출문제 컨텍스트 ===
        {context}
        
        문제:
        {question}
        
        사용자 답변: {user_answer}
        
        다음 형식으로 응답해주세요:
        === 평가 결과 ===
        정답 여부: [맞음/틀림]
        점수: [점수/만점]
        
        === 피드백 ===
        [구체적인 피드백 및 개선점]
        
        === 관련 개념 ===
        [문제와 관련된 핵심 개념 설명 - 컨텍스트 내용 포함]
        
        === 출처 ===
        {source_info}
        """
    
    @staticmethod
    def get_rag_question_prompt(question: str, context: str) -> str:
        """RAG 기반 질문 프롬프트"""
        return f"""
        다음 컨텍스트를 바탕으로 질문에 답변해주세요:
        
        컨텍스트:
        {context}
        
        질문: {question}
        
        답변:
        """
    
    @staticmethod
    def get_question_improvement_prompt(original_question: str, feedback: str) -> str:
        """문제 개선 프롬프트"""
        return f"""
        다음 문제를 개선해주세요:
        
        원본 문제:
        {original_question}
        
        개선 요청사항:
        {feedback}
        
        개선된 문제:
        """

    @staticmethod
    def get_context_validation_prompt(context: str, metadata: list = []) -> str:
        """컨텍스트 품질 검증 프롬프트"""
        # 메타데이터 정보 구성
        metadata_info = ""
        if metadata:
            metadata_info = "\n=== 메타데이터 정보 ===\n"
            for i, meta in enumerate(metadata, 1):
                metadata_info += f"참고 자료 {i}:\n"
                if meta.get("pdf_source"):
                    metadata_info += f"- PDF 파일: {meta.get('pdf_source')}\n"
                if meta.get("subject"):
                    metadata_info += f"- 과목: {meta.get('subject')}\n"
                metadata_info += "\n"
        
        return f"""
        다음 컨텍스트가 문제 출제에 적합한지 검증해주세요.
        
        === 컨텍스트 ===
        {context}
        {metadata_info}
        
        **검증 기준**:
        1. 컨텍스트가 완전한 문제를 포함하고 있는가?
        2. 문제와 보기가 명확히 구분되어 있는가?
        3. 문제 번호가 명확히 표시되어 있는가?
        4. 객관식 문제의 경우 보기가 4개 모두 있는가?
        5. 정답 정보가 포함되어 있는가?
        6. 컨텍스트가 너무 길거나 짧지 않은가?
        
        다음 형식으로 응답해주세요:
        === 검증 결과 ===
        적합성: [적합/부적합]
        문제 번호: [추출된 문제 번호 또는 "없음"]
        문제 유형: [객관식/주관식/기타]
        보기 개수: [객관식인 경우 보기 개수, 기타는 "해당없음"]
        
        === 문제점 ===
        [부적합한 경우 구체적인 문제점 설명]
        
        === 개선 제안 ===
        [부적합한 경우 개선 방안 제시]
        """

class ChatPrompts:
    """챗봇 관련 프롬프트 클래스"""
    
    @staticmethod
    def get_conversation_prompt(message: str, history: list = None) -> str:
        """대화 프롬프트"""
        context = ""
        if history:
            # 딕셔너리 형태의 히스토리 처리 (안전한 방식)
            conversation_parts = []
            for h in history[-5:]:  # 최근 5개만
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
            
            context = "\n".join(conversation_parts)
        
        return f"""
        {context}
        
        사용자: {message}
        도우미: """
    
    @staticmethod
    def get_rag_conversation_prompt(message: str, context: str, history: list = None) -> str:
        """RAG 기반 대화 프롬프트"""
        conversation_context = ""
        if history:
            # 딕셔너리 형태의 히스토리 처리 (안전한 방식)
            conversation_parts = []
            for h in history[-5:]:  # 최근 5개만
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
        다음 기출문제 컨텍스트를 바탕으로 질문에 답변해주세요.

**중요한 규칙**:
1. 반드시 제공된 기출문제 컨텍스트만을 기반으로 답변하세요.
2. 기출문제에 없는 내용은 절대 답변하지 마세요.
3. 일반적인 지식이나 다른 정보를 추가하지 마세요.
4. 기출문제 내용을 정확히 인용하고 해석하여 답변하세요.
5. 기출문제에 관련 내용이 없으면 "기출문제에서 해당 내용을 찾을 수 없습니다"라고 답변하세요.
6. 답변은 구체적이고 자세하게 작성해주세요.
7. 기출문제의 내용을 그대로 인용하여 답변의 근거를 제시해주세요.

=== 기출문제 컨텍스트 ===
{context}

=== 대화 기록 ===
{conversation_context}

사용자: {message}
도우미: 기출문제를 바탕으로 답변드리겠습니다.

"""

class AnalysisPrompts:
    """분석 관련 프롬프트 클래스"""
    
    @staticmethod
    def get_performance_analysis_prompt(user_answers: list) -> str:
        """성과 분석 프롬프트"""
        return f"""
        다음 사용자의 답변 기록을 분석하여 학습 성과를 평가해주세요:
        
        답변 기록:
        {user_answers}
        
        분석 결과:
        1. 전체 정답률: [%]
        2. 강점 과목: [과목명]
        3. 약점 과목: [과목명]
        4. 개선 권장사항: [구체적인 학습 방향]
        """
    
    @staticmethod
    def get_recommendation_prompt(user_profile: dict) -> str:
        """개인화 추천 프롬프트"""
        return f"""
        다음 사용자 프로필을 바탕으로 맞춤형 학습 추천을 해주세요:
        
        사용자 프로필:
        {user_profile}
        
        추천 사항:
        1. 추천 과목: [과목명]
        2. 추천 난이도: [난이도]
        3. 학습 전략: [구체적인 학습 방법]
        4. 예상 소요 시간: [시간]
        """
    
    @staticmethod
    def get_rag_recommendation_prompt(user_profile: dict, context: str) -> str:
        """RAG 기반 개인화 추천 프롬프트"""
        return f"""
        다음 기출문제 컨텍스트와 사용자 프로필을 바탕으로 맞춤형 학습 추천을 해주세요:
        
        === 기출문제 컨텍스트 ===
        {context}
        
        사용자 프로필:
        {user_profile}
        
        추천 사항:
        1. 추천 과목: [과목명]
        2. 추천 난이도: [난이도]
        3. 학습 전략: [구체적인 학습 방법 - 컨텍스트 내용 포함]
        4. 예상 소요 시간: [시간]
        5. 관련 기출문제: [컨텍스트에서 추천할 문제]
        """

class PDFProcessingPrompts:
    """PDF 처리 관련 프롬프트 클래스"""
    
    @staticmethod
    def get_pdf_summary_prompt(content: str) -> str:
        """PDF 내용 요약 프롬프트"""
        return f"""
        다음 PDF 내용을 요약해주세요:
        
        내용:
        {content}
        
        요약:
        1. 주요 주제: [주제]
        2. 핵심 개념: [개념들]
        3. 중요 내용: [중요한 내용들]
        4. 시험 관련성: [시험과의 연관성]
        """
    
    @staticmethod
    def get_pdf_question_extraction_prompt(content: str) -> str:
        """PDF에서 문제 추출 프롬프트"""
        return f"""
        다음 PDF 내용에서 시험 문제를 추출해주세요:
        
        내용:
        {content}
        
        추출된 문제들:
        [문제 형식으로 정리]
        """ 