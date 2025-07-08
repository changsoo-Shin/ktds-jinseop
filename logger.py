"""
사용자별 로그 관리 시스템
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging
from pathlib import Path
from config import Config

class UserLogger:
    """사용자별 로그 관리 클래스"""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.setup_logging()
    
    def setup_logging(self):
        """로깅 설정"""
        # 설정에서 로깅 설정 가져오기
        log_config = Config.get_logging_config()
        
        # 로그 레벨 설정
        log_level = getattr(logging, log_config["level"].upper(), logging.INFO)
        
        # 핸들러 설정
        handlers = []
        
        # 콘솔 핸들러
        if log_config["console"]:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(log_level)
            # Windows 환경에서 유니코드 이모지 오류 방지
            try:
                import sys
                if sys.platform == "win32":
                    # Windows에서는 이모지가 포함된 메시지를 안전하게 처리
                    class SafeStreamHandler(logging.StreamHandler):
                        def emit(self, record):
                            try:
                                # 이모지를 ASCII 문자로 대체
                                msg = record.getMessage()
                                safe_msg = msg.replace('✅', '[OK]').replace('❌', '[ERROR]').replace('⚠️', '[WARN]').replace('🔍', '[DEBUG]').replace('📄', '[PDF]').replace('📝', '[INFO]').replace('🔄', '[PROCESS]').replace('🤖', '[AI]').replace('📊', '[STATS]').replace('📚', '[EXAM]').replace('💬', '[CHAT]').replace('📖', '[CONTEXT]').replace('📅', '[DATE]')
                                
                                # 새로운 레코드 생성 (원본 레코드 수정 방지)
                                new_record = logging.LogRecord(
                                    name=record.name,
                                    level=record.levelno,
                                    pathname=record.pathname,
                                    lineno=record.lineno,
                                    msg=safe_msg,
                                    args=(),
                                    exc_info=record.exc_info,
                                    func=record.funcName
                                )
                                super().emit(new_record)
                            except UnicodeEncodeError:
                                # 유니코드 오류 시 ASCII로 변환
                                try:
                                    safe_msg = record.getMessage().encode('ascii', 'ignore').decode('ascii')
                                    new_record = logging.LogRecord(
                                        name=record.name,
                                        level=record.levelno,
                                        pathname=record.pathname,
                                        lineno=record.lineno,
                                        msg=safe_msg,
                                        args=(),
                                        exc_info=record.exc_info,
                                        func=record.funcName
                                    )
                                    super().emit(new_record)
                                except:
                                    pass
                    console_handler = SafeStreamHandler()
                    console_handler.setLevel(log_level)
            except:
                pass
            console_handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
            handlers.append(console_handler)
        
        # 파일 핸들러
        if log_config["file"]:
            file_handler = logging.FileHandler(log_config["file_path"], encoding='utf-8')
            file_handler.setLevel(log_level)
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
            handlers.append(file_handler)
        
        # 기본 핸들러 (시스템 로그)
        system_handler = logging.FileHandler(self.log_dir / 'system.log', encoding='utf-8')
        system_handler.setLevel(log_level)
        system_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        handlers.append(system_handler)
        
        # 로깅 설정
        logging.basicConfig(
            level=log_level,
            handlers=handlers
        )
        self.logger = logging.getLogger(__name__)
    
    def get_user_log_file(self, user_id: str) -> Path:
        """사용자별 로그 파일 경로 반환"""
        return self.log_dir / f"user_{user_id}.json"
    
    def log_user_activity(self, user_id: str, activity_type: str, data: Dict[str, Any]):
        """사용자 활동 로그 기록"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id,
            "activity_type": activity_type,
            "data": data
        }
        
        log_file = self.get_user_log_file(user_id)
        
        # 기존 로그 읽기
        logs = []
        if log_file.exists():
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    logs = json.load(f)
            except json.JSONDecodeError:
                logs = []
        
        # 새 로그 추가
        logs.append(log_entry)
        
        # 로그 저장
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(logs, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"User {user_id} activity logged: {activity_type}")
    
    def log_question_generation(self, user_id: str, subject: str, difficulty: str, 
                              question_type: str, generated_question: str):
        """문제 생성 로그"""
        data = {
            "subject": subject,
            "difficulty": difficulty,
            "question_type": question_type,
            "generated_question": generated_question
        }
        self.log_user_activity(user_id, "question_generation", data)
    
    def log_answer_evaluation(self, user_id: str, question: str, user_answer: str, 
                            evaluation_result: str, is_correct: bool, score: float):
        """답변 평가 로그"""
        data = {
            "question": question,
            "user_answer": user_answer,
            "evaluation_result": evaluation_result,
            "is_correct": is_correct,
            "score": score
        }
        self.log_user_activity(user_id, "answer_evaluation", data)
    
    def log_chat_interaction(self, user_id: str, message: str, ai_response: str):
        """챗봇 대화 로그"""
        data = {
            "message": message,
            "ai_response": ai_response
        }
        self.log_user_activity(user_id, "chat_interaction", data)
    
    def get_user_logs(self, user_id: str, activity_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """사용자 로그 조회"""
        log_file = self.get_user_log_file(user_id)
        
        if not log_file.exists():
            return []
        
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                logs = json.load(f)
            
            if activity_type:
                logs = [log for log in logs if log.get("activity_type") == activity_type]
            
            return logs
        except json.JSONDecodeError:
            return []
    
    def get_user_statistics(self, user_id: str) -> Dict[str, Any]:
        """사용자 통계 정보"""
        logs = self.get_user_logs(user_id)
        
        if not logs:
            return {
                "total_activities": 0,
                "questions_generated": 0,
                "answers_evaluated": 0,
                "chat_interactions": 0,
                "correct_answers": 0,
                "total_answers": 0,
                "average_score": 0.0
            }
        
        stats = {
            "total_activities": len(logs),
            "questions_generated": len([log for log in logs if log.get("activity_type") == "question_generation"]),
            "answers_evaluated": len([log for log in logs if log.get("activity_type") == "answer_evaluation"]),
            "chat_interactions": len([log for log in logs if log.get("activity_type") == "chat_interaction"]),
            "correct_answers": 0,
            "total_answers": 0,
            "total_score": 0.0
        }
        
        # 답변 평가 통계 계산
        evaluation_logs = [log for log in logs if log.get("activity_type") == "answer_evaluation"]
        for log in evaluation_logs:
            data = log.get("data", {})
            if data.get("is_correct"):
                stats["correct_answers"] += 1
            stats["total_answers"] += 1
            stats["total_score"] += data.get("score", 0.0)
        
        if stats["total_answers"] > 0:
            stats["average_score"] = stats["total_score"] / stats["total_answers"]
        else:
            stats["average_score"] = 0.0
        
        return stats
    
    def get_user_performance_by_subject(self, user_id: str) -> Dict[str, Dict[str, Any]]:
        """과목별 성과 분석"""
        evaluation_logs = self.get_user_logs(user_id, "answer_evaluation")
        
        subject_stats = {}
        
        for log in evaluation_logs:
            data = log.get("data", {})
            question = data.get("question", "")
            
            # 문제에서 과목 추출 (간단한 방법)
            subject = "기타"  # 기본값
            if "정보시스템 일반" in question:
                subject = "정보시스템 일반"
            elif "정보시스템 감리" in question:
                subject = "정보시스템 감리"
            elif "정보시스템 보안" in question:
                subject = "정보시스템 보안"
            elif "정보시스템 운영" in question:
                subject = "정보시스템 운영"
            elif "정보시스템 개발" in question:
                subject = "정보시스템 개발"
            elif "정보시스템 구축" in question:
                subject = "정보시스템 구축"
            elif "정보시스템 관리" in question:
                subject = "정보시스템 관리"
            elif "정보시스템 설계" in question:
                subject = "정보시스템 설계"
            
            if subject not in subject_stats:
                subject_stats[subject] = {
                    "total_questions": 0,
                    "correct_answers": 0,
                    "total_score": 0.0
                }
            
            subject_stats[subject]["total_questions"] += 1
            if data.get("is_correct"):
                subject_stats[subject]["correct_answers"] += 1
            subject_stats[subject]["total_score"] += data.get("score", 0.0)
        
        # 평균 점수 계산
        for subject in subject_stats:
            total = subject_stats[subject]["total_questions"]
            if total > 0:
                subject_stats[subject]["average_score"] = subject_stats[subject]["total_score"] / total
                subject_stats[subject]["accuracy_rate"] = subject_stats[subject]["correct_answers"] / total
            else:
                subject_stats[subject]["average_score"] = 0.0
                subject_stats[subject]["accuracy_rate"] = 0.0
        
        return subject_stats
    
    def export_user_data(self, user_id: str, export_path: str):
        """사용자 데이터 내보내기"""
        logs = self.get_user_logs(user_id)
        stats = self.get_user_statistics(user_id)
        performance = self.get_user_performance_by_subject(user_id)
        
        export_data = {
            "user_id": user_id,
            "export_timestamp": datetime.now().isoformat(),
            "statistics": stats,
            "performance_by_subject": performance,
            "detailed_logs": logs
        }
        
        with open(export_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"User {user_id} data exported to {export_path}")

# 전역 로거 인스턴스
user_logger = UserLogger() 