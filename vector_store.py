"""
벡터 스토어 관리 시스템
FAISS를 사용한 문서 저장 및 검색
"""

import os
from typing import List, Dict, Any, Optional
import json
from pathlib import Path
import hashlib
from datetime import datetime
import numpy as np
import logging

# 로거 설정
logger = logging.getLogger(__name__)

# FAISS 관련 import
try:
    import faiss
    from sentence_transformers import SentenceTransformer
except ImportError:
    logger.error("FAISS 또는 sentence-transformers가 설치되지 않았습니다.")
    logger.error("pip install faiss-cpu sentence-transformers를 실행해주세요.")
    faiss = None
    SentenceTransformer = None

class VectorStore:
    """FAISS 기반 벡터 스토어 관리 클래스"""
    
    def __init__(self, persist_directory: str = "faiss_vector_db"):
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(exist_ok=True)
        
        # 벡터 모델 및 인덱스 초기화
        self.embedding_model = None
        self.index = None
        self.documents = []
        self.metadata = []
        
        self._initialize_models()
        self._load_existing_data()
    
    def _initialize_models(self):
        """벡터 모델 및 FAISS 인덱스 초기화"""
        if SentenceTransformer is None or faiss is None:
            logger.error("필요한 라이브러리가 설치되지 않았습니다.")
            return
        
        try:
            # GPU 사용 가능 여부 확인 및 설정
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            logger.info(f"🔧 [INFO] 사용할 디바이스: {device}")
            if device == 'cuda':
                logger.info(f"🔧 [INFO] GPU 모델: {torch.cuda.get_device_name(0)}")
            
            # 한국어에 특화된 임베딩 모델 사용
            self.embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', device=device)
            
            # FAISS 인덱스 초기화 (L2 거리 기반)
            dimension = self.embedding_model.get_sentence_embedding_dimension()
            
            # FAISS는 CPU 사용 (Python 3.12에서 GPU FAISS 미지원)
            self.index = faiss.IndexFlatL2(dimension)
            logger.info(f"🔧 [INFO] FAISS CPU 인덱스 사용 (Python 3.12)")
            
            logger.info(f"✅ FAISS 벡터 모델 초기화 완료 (차원: {dimension})")
        except Exception as e:
            logger.error(f"❌ 벡터 모델 초기화 실패: {e}")
    
    def _load_existing_data(self):
        """기존 데이터 로드"""
        try:
            metadata_file = self.persist_directory / "metadata.json"
            index_file = self.persist_directory / "faiss_index.bin"
            
            if metadata_file.exists() and index_file.exists():
                # 메타데이터 로드
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.metadata = data.get("metadata", [])
                    self.documents = [meta.get("text", "") for meta in self.metadata]
                
                # FAISS 인덱스 로드
                self.index = faiss.read_index(str(index_file))
                
                logger.info(f"✅ 기존 데이터 로드 완료 - {len(self.documents)}개 문서")
                return True
            
        except Exception as e:
            logger.error(f"기존 데이터 로드 중 오류: {e}")
        
        return False
    
    def add_exam_question(self, question_data: Dict[str, Any]):
        """시험 문제 추가"""
        if self.embedding_model is None or self.index is None:
            logger.error("벡터 모델이 초기화되지 않았습니다.")
            return None
        
        # 문서 ID 생성
        doc_id = hashlib.md5(
            f"{question_data.get('subject', '')}{question_data.get('question', '')}".encode()
        ).hexdigest()
        
        # 메타데이터 준비
        metadata = {
            "id": doc_id,
            "type": "exam_question",
            "subject": question_data.get("subject", ""),
            "difficulty": question_data.get("difficulty", ""),
            "question_type": question_data.get("question_type", ""),
            "correct_answer": question_data.get("correct_answer", ""),
            "explanation": question_data.get("explanation", ""),
            "source": question_data.get("source", ""),
            "created_at": datetime.now().isoformat()
        }
        
        # 문서 내용 (검색용)
        document_content = f"""
        과목: {question_data.get('subject', '')}
        문제: {question_data.get('question', '')}
        보기: {question_data.get('options', '')}
        정답: {question_data.get('correct_answer', '')}
        해설: {question_data.get('explanation', '')}
        """
        
        try:
            # 벡터화
            embedding = self.embedding_model.encode([document_content])
            
            # FAISS 인덱스에 추가
            self.index.add(embedding.astype('float32'))
            
            # 메타데이터 저장
            metadata["embedding_id"] = len(self.documents)
            self.documents.append(document_content)
            self.metadata.append(metadata)
            
            # 저장
            self._save_data()
            
            logger.info(f"✅ 시험 문제 추가 완료: {doc_id}")
            return doc_id
            
        except Exception as e:
            logger.error(f"문제 추가 중 오류: {e}")
            return None
    
    def add_study_material(self, material_data: Dict[str, Any]):
        """학습 자료 추가"""
        if self.embedding_model is None or self.index is None:
            logger.error("벡터 모델이 초기화되지 않았습니다.")
            return None
        
        doc_id = hashlib.md5(
            f"{material_data.get('title', '')}{material_data.get('content', '')}".encode()
        ).hexdigest()
        
        metadata = {
            "id": doc_id,
            "type": "study_material",
            "title": material_data.get("title", ""),
            "category": material_data.get("category", ""),
            "subject": material_data.get("subject", ""),
            "difficulty": material_data.get("difficulty", ""),
            "source": material_data.get("source", ""),
            "created_at": datetime.now().isoformat()
        }
        
        try:
            # 벡터화
            embedding = self.embedding_model.encode([material_data.get("content", "")])
            
            # FAISS 인덱스에 추가
            self.index.add(embedding.astype('float32'))
            
            # 메타데이터 저장
            metadata["embedding_id"] = len(self.documents)
            self.documents.append(material_data.get("content", ""))
            self.metadata.append(metadata)
            
            # 저장
            self._save_data()
            
            logger.info(f"✅ 학습 자료 추가 완료: {doc_id}")
            return doc_id
            
        except Exception as e:
            logger.error(f"학습 자료 추가 중 오류: {e}")
            return None
    
    def add_user_question(self, user_id: str, question_data: Dict[str, Any]):
        """사용자 질문 추가"""
        if self.embedding_model is None or self.index is None:
            logger.error("벡터 모델이 초기화되지 않았습니다.")
            return None
        
        doc_id = hashlib.md5(
            f"{user_id}{question_data.get('question', '')}{datetime.now().isoformat()}".encode()
        ).hexdigest()
        
        metadata = {
            "id": doc_id,
            "type": "user_question",
            "user_id": user_id,
            "subject": question_data.get("subject", ""),
            "difficulty": question_data.get("difficulty", ""),
            "created_at": datetime.now().isoformat()
        }
        
        try:
            # 벡터화
            embedding = self.embedding_model.encode([question_data.get("question", "")])
            
            # FAISS 인덱스에 추가
            self.index.add(embedding.astype('float32'))
            
            # 메타데이터 저장
            metadata["embedding_id"] = len(self.documents)
            self.documents.append(question_data.get("question", ""))
            self.metadata.append(metadata)
            
            # 저장
            self._save_data()
            
            logger.info(f"✅ 사용자 질문 추가 완료: {doc_id}")
            return doc_id
            
        except Exception as e:
            logger.error(f"사용자 질문 추가 중 오류: {e}")
            return None
    
    def search_similar_questions(self, query: str, subject: Optional[str] = None, 
                               n_results: int = 5) -> List[Dict[str, Any]]:
        """유사한 문제 검색"""
        if self.embedding_model is None or self.index is None:
            logger.error("벡터 모델이 초기화되지 않았습니다.")
            return []
        
        if len(self.documents) == 0:
            logger.error("검색할 문서가 없습니다.")
            return []
        
        try:
            # 쿼리 벡터화
            query_embedding = self.embedding_model.encode([query])
            
            # 검색할 문서 수 제한 (인덱스 크기보다 크면 안됨)
            search_k = min(n_results * 2, len(self.documents))
            
            # FAISS 검색
            distances, indices = self.index.search(query_embedding.astype('float32'), search_k)
            
            # 결과가 비어있는지 확인
            if len(distances) == 0 or len(indices) == 0:
                logger.error("검색 결과가 없습니다.")
                return []
            
            # 결과 필터링 및 포맷팅
            results = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                # 유효한 인덱스인지 확인
                if idx < 0 or idx >= len(self.metadata):
                    continue
                    
                metadata = self.metadata[idx]
                
                # 과목 필터링
                if subject and metadata.get("subject") != subject:
                    continue
                
                # 타입 필터링 (type이 없으면 모든 문서 허용)
                if metadata.get("type") and metadata.get("type") != "exam_question":
                    continue
                
                result = {
                    "id": metadata.get("id"),
                    "content": self.documents[idx],
                    "metadata": metadata,
                    "distance": float(distance),
                    "rank": len(results) + 1
                }
                results.append(result)
                
                if len(results) >= n_results:
                    break
            
            return results
            
        except Exception as e:
            logger.error(f"문제 검색 중 오류: {e}")
            return []
    
    def search_study_materials(self, query: str, subject: Optional[str] = None,
                             n_results: int = 5) -> List[Dict[str, Any]]:
        """학습 자료 검색"""
        if self.embedding_model is None or self.index is None:
            logger.error("벡터 모델이 초기화되지 않았습니다.")
            return []
        
        if len(self.documents) == 0:
            logger.error("검색할 문서가 없습니다.")
            return []
        
        try:
            # 쿼리 벡터화
            query_embedding = self.embedding_model.encode([query])
            
            # 검색할 문서 수 제한
            search_k = min(n_results * 2, len(self.documents))
            
            # FAISS 검색
            distances, indices = self.index.search(query_embedding.astype('float32'), search_k)
            
            # 결과가 비어있는지 확인
            if len(distances) == 0 or len(indices) == 0:
                logger.error("검색 결과가 없습니다.")
                return []
            
            # 결과 필터링 및 포맷팅
            results = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                # 유효한 인덱스인지 확인
                if idx < 0 or idx >= len(self.metadata):
                    continue
                    
                metadata = self.metadata[idx]
                
                # 과목 필터링
                if subject and metadata.get("subject") != subject:
                    continue
                
                # 학습 자료 타입만 필터링
                if metadata.get("type") != "study_material":
                    continue
                
                result = {
                    "id": metadata.get("id"),
                    "content": self.documents[idx],
                    "metadata": metadata,
                    "distance": float(distance),
                    "rank": len(results) + 1
                }
                results.append(result)
                
                if len(results) >= n_results:
                    break
            
            return results
            
        except Exception as e:
            logger.error(f"학습 자료 검색 중 오류: {e}")
            return []
    
    def get_questions_by_subject(self, subject: str, n_results: int = 10) -> List[Dict[str, Any]]:
        """과목별 문제 조회"""
        results = []
        
        for metadata in self.metadata:
            if (metadata.get("subject") == subject and 
                metadata.get("type") == "exam_question"):
                results.append({
                    "id": metadata.get("id"),
                    "content": self.documents[metadata.get("embedding_id", 0)],
                    "metadata": metadata
                })
                if len(results) >= n_results:
                    break
        
        return results
    
    def get_questions_by_difficulty(self, difficulty: str, n_results: int = 10) -> List[Dict[str, Any]]:
        """난이도별 문제 조회"""
        results = []
        
        for metadata in self.metadata:
            if (metadata.get("difficulty") == difficulty and 
                metadata.get("type") == "exam_question"):
                results.append({
                    "id": metadata.get("id"),
                    "content": self.documents[metadata.get("embedding_id", 0)],
                    "metadata": metadata
                })
                if len(results) >= n_results:
                    break
        
        return results
    
    def get_user_questions(self, user_id: str, n_results: int = 10) -> List[Dict[str, Any]]:
        """사용자별 질문 조회"""
        results = []
        
        for metadata in self.metadata:
            if (metadata.get("user_id") == user_id and 
                metadata.get("type") == "user_question"):
                results.append({
                    "id": metadata.get("id"),
                    "content": self.documents[metadata.get("embedding_id", 0)],
                    "metadata": metadata
                })
                if len(results) >= n_results:
                    break
        
        return results
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """컬렉션 통계 정보"""
        stats = {
            "total_documents": len(self.documents),
            "total_metadata": len(self.metadata),
            "index_size": self.index.ntotal if self.index else 0,
            "exam_questions": len([m for m in self.metadata if m.get("type") == "exam_question"]),
            "study_materials": len([m for m in self.metadata if m.get("type") == "study_material"]),
            "user_questions": len([m for m in self.metadata if m.get("type") == "user_question"]),
            "subjects": list(set([m.get("subject", "") for m in self.metadata if m.get("subject")]))
        }
        return stats
    
    def delete_document(self, doc_id: str) -> bool:
        """문서 삭제 (FAISS에서는 복잡하므로 전체 재구성)"""
        try:
            # 해당 문서 찾기
            target_idx = None
            for i, metadata in enumerate(self.metadata):
                if metadata.get("id") == doc_id:
                    target_idx = i
                    break
            
            if target_idx is not None:
                # 문서와 메타데이터 제거
                del self.documents[target_idx]
                del self.metadata[target_idx]
                
                # FAISS 인덱스 재구성
                self._rebuild_index()
                
                logger.info(f"✅ 문서 삭제 완료: {doc_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"문서 삭제 중 오류: {e}")
            return False
    
    def delete_exam_data(self, exam_name: str) -> bool:
        """특정 시험의 모든 데이터 삭제"""
        try:
            # 해당 시험의 문서 ID들 찾기
            doc_ids_to_delete = []
            for meta in self.metadata:
                if meta.get("subject") == exam_name:
                    doc_ids_to_delete.append(meta.get("id"))
            
            if not doc_ids_to_delete:
                logger.info(f"시험 '{exam_name}'의 데이터를 찾을 수 없습니다.")
                return True  # 삭제할 데이터가 없어도 성공으로 처리
            
            # 각 문서 삭제
            for doc_id in doc_ids_to_delete:
                self.delete_document(doc_id)
            
            logger.info(f"✅ 시험 '{exam_name}' 데이터 삭제 완료: {len(doc_ids_to_delete)}개 문서")
            return True
            
        except Exception as e:
            logger.error(f"시험 데이터 삭제 중 오류: {e}")
            return False
    
    def _rebuild_index(self):
        """FAISS 인덱스 재구성"""
        if self.embedding_model is None or not self.documents:
            return
        
        try:
            # 새로운 인덱스 생성
            dimension = self.embedding_model.get_sentence_embedding_dimension()
            self.index = faiss.IndexFlatL2(dimension)
            
            # 모든 문서 벡터화
            embeddings = self.embedding_model.encode(self.documents)
            self.index.add(embeddings.astype('float32'))
            
            # 메타데이터 업데이트
            for i, metadata in enumerate(self.metadata):
                metadata["embedding_id"] = i
            
            # 저장
            self._save_data()
            
            logger.info("✅ FAISS 인덱스 재구성 완료")
            
        except Exception as e:
            logger.error(f"인덱스 재구성 중 오류: {e}")
    
    def _save_data(self):
        """데이터 저장"""
        try:
            metadata_file = self.persist_directory / "metadata.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "total_documents": len(self.documents),
                    "metadata": self.metadata,
                    "last_updated": datetime.now().isoformat()
                }, f, ensure_ascii=False, indent=2)
            
            # FAISS 인덱스 저장
            if self.index:
                index_file = self.persist_directory / "faiss_index.bin"
                faiss.write_index(self.index, str(index_file))
            
            logger.info("✅ 데이터 저장 완료")
            
        except Exception as e:
            logger.error(f"데이터 저장 중 오류: {e}")
    
    def backup_collection(self, backup_path: str):
        """컬렉션 백업"""
        try:
            backup_data = {
                "total_documents": len(self.documents),
                "documents": self.documents,
                "metadata": self.metadata,
                "backup_timestamp": datetime.now().isoformat()
            }
            
            with open(backup_path, 'w', encoding='utf-8') as f:
                json.dump(backup_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"✅ 백업 완료: {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"백업 중 오류: {e}")
            return False
    
    def restore_collection(self, backup_path: str):
        """컬렉션 복원"""
        try:
            with open(backup_path, 'r', encoding='utf-8') as f:
                backup_data = json.load(f)
            
            self.documents = backup_data.get("documents", [])
            self.metadata = backup_data.get("metadata", [])
            
            # FAISS 인덱스 재구성
            self._rebuild_index()
            
            logger.info(f"✅ 복원 완료: {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"복원 중 오류: {e}")
            return False

# 전역 벡터 스토어 인스턴스
vector_store = VectorStore() 