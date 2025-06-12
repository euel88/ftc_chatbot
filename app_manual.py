# 파일 이름: app_manual.py (공정거래위원회 AI 법률 보조원)

# ===== 필수 import 문들을 맨 위로 이동 =====
import streamlit as st
import numpy as np  # SimpleVectorSearch 클래스보다 먼저 import 필요
from typing import List, Dict, Tuple, Optional, Set, TypedDict, Protocol, Iterator, Generator, OrderedDict, Any, Union
import json
import openai
import re
from collections import defaultdict, Counter, OrderedDict
import time
from dataclasses import dataclass, field
import os
import hashlib
from enum import Enum
import asyncio
import logging
from contextlib import contextmanager
import traceback
import gc
import concurrent.futures
from functools import lru_cache, wraps
import pickle
from pathlib import Path

# 선택적 import - 없어도 앱이 실행되도록 처리
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logging.warning("FAISS module not available - using alternative search method")

try:
    from cryptography.fernet import Fernet
    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    CRYPTOGRAPHY_AVAILABLE = False
    logging.warning("cryptography module not available - encryption features disabled")

try:
    import ijson
    IJSON_AVAILABLE = True
except ImportError:
    IJSON_AVAILABLE = False
    logging.warning("ijson module not available - using standard JSON loading")

try:
    import nest_asyncio
    nest_asyncio.apply()
    NEST_ASYNCIO_AVAILABLE = True
except ImportError:
    NEST_ASYNCIO_AVAILABLE = False
    logging.warning("nest_asyncio not available - async features may be limited")

try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("sentence-transformers not available - will use OpenAI embeddings")

# ===== 간단한 벡터 검색 시스템 (FAISS 대체) =====
class SimpleVectorSearch:
    """FAISS를 사용할 수 없을 때를 위한 간단한 벡터 검색
    
    이 클래스는 NumPy만을 사용하여 코사인 유사도 기반의
    벡터 검색을 구현합니다. FAISS보다는 느리지만,
    어떤 환경에서도 작동합니다.
    """
    
    def __init__(self, embeddings: np.ndarray):
        """
        Args:
            embeddings: 문서 임베딩 배열 (n_docs, embedding_dim)
        """
        self.embeddings = embeddings
        # 정규화된 임베딩 저장 (코사인 유사도 계산 최적화)
        self.normalized_embeddings = self._normalize_vectors(embeddings)
        logging.info(f"SimpleVectorSearch initialized with {len(embeddings)} documents")
    
    def _normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """벡터 정규화 (L2 norm = 1)"""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        # 0으로 나누기 방지
        norms = np.where(norms == 0, 1, norms)
        return vectors / norms
    
    def search(self, query_vector: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        코사인 유사도 기반 검색
        
        Args:
            query_vector: 쿼리 벡터 (1, embedding_dim)
            k: 반환할 상위 결과 수
        
        Returns:
            scores: 유사도 점수 배열
            indices: 문서 인덱스 배열
        """
        # 쿼리 벡터 정규화
        query_norm = self._normalize_vectors(query_vector.reshape(1, -1))
        
        # 코사인 유사도 계산 (정규화된 벡터의 내적)
        similarities = np.dot(self.normalized_embeddings, query_norm.T).squeeze()
        
        # 상위 k개 선택
        top_k_indices = np.argpartition(similarities, -k)[-k:]
        top_k_indices = top_k_indices[np.argsort(similarities[top_k_indices])[::-1]]
        
        # FAISS와 동일한 형식으로 반환
        scores = similarities[top_k_indices].reshape(1, -1)
        indices = top_k_indices.reshape(1, -1)
        
        return scores, indices

# ===== 로깅 설정 =====
def setup_logging():
    """구조화된 로깅 시스템 설정"""
    logger = logging.getLogger('ftc_chatbot')
    logger.setLevel(logging.DEBUG)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    )
    
    # 파일 핸들러
    file_handler = logging.FileHandler('ftc_chatbot_debug.log')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    
    # 콘솔 핸들러
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# 전역 로거 설정
logger = setup_logging()

# ===== 사용자 정의 예외 클래스 =====
class RAGPipelineError(Exception):
    """RAG 파이프라인의 기본 예외 클래스"""
    pass

class APIKeyError(RAGPipelineError):
    """API 키 관련 오류"""
    pass

# ===== 타입 정의 =====
class ModelSelection(Enum):
    """사용 가능한 모델들"""
    GPT4O = "gpt-4o"
    GPT4O_MINI = "gpt-4o-mini"
    O4_MINI = "o4-mini"  # 추론 특화 모델

class ProcessStep(Enum):
    """처리 단계"""
    INTENT_ANALYSIS = "intent_analysis"
    DOCUMENT_SEARCH = "document_search"
    ANSWER_GENERATION = "answer_generation"

# ===== 데이터 구조 정의 =====
@dataclass
class SearchResult:
    """검색 결과를 담는 데이터 클래스"""
    chunk_id: str
    content: str
    score: float
    source: str
    page: int
    chunk_type: str
    metadata: Dict
    
    @property
    def document_date(self) -> Optional[str]:
        """문서의 작성/개정 날짜 반환"""
        return self.metadata.get('document_date') or self.metadata.get('revision_date')

@dataclass
class IntentAnalysis:
    """사용자 질문 의도 분석 결과"""
    core_intent: str  # 핵심 의도
    query_type: str  # simple_lookup, complex_analysis, procedural
    target_documents: List[str]  # 검색해야 할 문서들
    key_entities: List[str]  # 핵심 개체들 (회사, 거래 유형 등)
    search_keywords: List[str]  # 검색 키워드
    requires_timeline: bool  # 시간 순서가 중요한지
    requires_calculation: bool  # 계산이 필요한지
    confidence: float  # 분석 신뢰도

# ===== API 관리자 (간소화) =====
class APIManager:
    """OpenAI API 키 관리"""
    
    def __init__(self):
        self._api_key = None
        
    def load_api_key(self) -> str:
        """API 키를 안전하게 로드"""
        # Streamlit secrets 확인 (최우선)
        try:
            if 'OPENAI_API_KEY' in st.secrets:
                self._api_key = st.secrets["OPENAI_API_KEY"]
                logger.info("API key loaded from Streamlit secrets")
                return self._api_key
        except Exception as e:
            logger.debug(f"Streamlit secrets not available: {e}")
        
        # 환경 변수 확인
        if os.environ.get('OPENAI_API_KEY'):
            self._api_key = os.environ.get('OPENAI_API_KEY')
            logger.info("API key loaded from environment variable")
            return self._api_key
        
        raise APIKeyError("API 키를 찾을 수 없습니다. Streamlit secrets 또는 환경 변수를 확인하세요.")
    
    def get_embedding(self, text: str) -> np.ndarray:
        """OpenAI embeddings API를 사용하여 텍스트 임베딩 생성"""
        try:
            response = openai.embeddings.create(
                input=text,
                model="text-embedding-ada-002"
            )
            return np.array(response.data[0].embedding)
        except Exception as e:
            logger.error(f"Failed to get OpenAI embedding: {e}")
            # 실패 시 랜덤 벡터 반환 (임시)
            return np.random.randn(1536)  # OpenAI 임베딩 차원

# ===== 메모리 효율적인 청크 로더 =====
class ChunkLoader:
    """메모리 효율적인 청크 로딩 시스템"""
    
    def __init__(self, filepath: str):
        self.filepath = filepath
        self._chunks = None
        self._chunk_cache = OrderedDict()
        self._cache_size = 1000
        self._load_all_chunks()
    
    def _load_all_chunks(self):
        """전체 청크를 메모리에 로드"""
        logger.info(f"Loading chunks from {self.filepath}")
        try:
            with open(self.filepath, 'r', encoding='utf-8') as f:
                self._chunks = json.load(f)
            logger.info(f"Loaded {len(self._chunks)} chunks into memory")
        except Exception as e:
            logger.error(f"Failed to load chunks: {e}")
            self._chunks = []
    
    def get_chunk(self, idx: int) -> Dict:
        """특정 인덱스의 청크를 가져옴"""
        if self._chunks and 0 <= idx < len(self._chunks):
            return self._chunks[idx]
        else:
            raise IndexError(f"Chunk index {idx} out of range")
    
    def get_total_chunks(self) -> int:
        """전체 청크 수 반환"""
        return len(self._chunks) if self._chunks else 0

# ===== 3단계 프로세스 구현 =====

class Step1_IntentAnalyzer:
    """Step 1: 사용자 질문 의도 파악
    
    이 단계에서는 사용자의 질문을 분석하여:
    - 질문의 핵심 의도를 파악
    - 검색해야 할 문서를 결정
    - 검색 키워드를 추출
    """
    
    def __init__(self, api_manager: APIManager):
        self.api_manager = api_manager
        
    async def analyze_intent(self, query: str) -> IntentAnalysis:
        """사용자 질문의 의도를 분석"""
        
        prompt = f"""
당신은 공정거래위원회 법률 전문가입니다. 
사용자의 질문을 분석하여 정확한 의도를 파악하고, 어떤 매뉴얼을 검색해야 하는지 결정해야 합니다.

사용 가능한 매뉴얼:
1. 대규모내부거래 매뉴얼: 계열사 간 자금거래, 자산거래, 상품용역거래 관련 규정
2. 현황공시 매뉴얼: 기업집단 현황공시, 공시 의무사항
3. 비상장사 중요사항 매뉴얼: 비상장회사의 주식 양도, 합병, 분할 등

사용자 질문: {query}

다음 형식으로 분석 결과를 제공하세요:

{{
    "core_intent": "한 문장으로 요약한 질문의 핵심 의도",
    "query_type": "simple_lookup/complex_analysis/procedural 중 선택",
    "target_documents": ["검색할 매뉴얼 이름들"],
    "key_entities": ["질문에 포함된 핵심 개체들 (예: 계열사, 자금, 이사회 등)"],
    "search_keywords": ["매뉴얼 검색에 사용할 핵심 키워드 5-10개"],
    "requires_timeline": true/false,
    "requires_calculation": true/false,
    "confidence": 0.0-1.0
}}

분석 시 고려사항:
- query_type 판단 기준:
  - simple_lookup: 단순 사실 확인 (기한, 금액 등)
  - complex_analysis: 여러 조건이 결합된 복잡한 상황
  - procedural: 절차나 프로세스에 대한 질문
- 질문에 여러 매뉴얼이 관련될 수 있으니 신중히 판단
- search_keywords는 실제 문서에서 찾을 수 있는 구체적인 용어로 선정
"""
        
        try:
            response = await openai.chat.completions.create(
                model="gpt-4o-mini",  # 의도 분석은 빠른 모델 사용
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            analysis_data = json.loads(response.choices[0].message.content)
            
            return IntentAnalysis(
                core_intent=analysis_data.get("core_intent", ""),
                query_type=analysis_data.get("query_type", "simple_lookup"),
                target_documents=analysis_data.get("target_documents", []),
                key_entities=analysis_data.get("key_entities", []),
                search_keywords=analysis_data.get("search_keywords", []),
                requires_timeline=analysis_data.get("requires_timeline", False),
                requires_calculation=analysis_data.get("requires_calculation", False),
                confidence=analysis_data.get("confidence", 0.8)
            )
            
        except Exception as e:
            logger.error(f"Intent analysis failed: {e}")
            # 기본값 반환
            return IntentAnalysis(
                core_intent=query,
                query_type="simple_lookup",
                target_documents=["대규모내부거래 매뉴얼", "현황공시 매뉴얼", "비상장사 중요사항 매뉴얼"],
                key_entities=[],
                search_keywords=query.split()[:5],
                requires_timeline=False,
                requires_calculation=False,
                confidence=0.5
            )

class Step2_DocumentSearcher:
    """Step 2: 의도에 맞는 매뉴얼 검색
    
    이 단계에서는 Step 1의 분석 결과를 바탕으로:
    - 관련 문서를 벡터 검색
    - 검색 결과를 재정렬 및 필터링
    - 가장 관련성 높은 청크 선별
    """
    
    def __init__(self, index, chunk_loader: ChunkLoader, embedding_model, api_manager: APIManager):
        self.index = index
        self.chunk_loader = chunk_loader
        self.embedding_model = embedding_model
        self.api_manager = api_manager
        self.use_faiss = index is not None
        
        # 매뉴얼별 인덱스 구축
        self.manual_indices = self._build_manual_indices()
        
    def _build_manual_indices(self) -> Dict[str, List[int]]:
        """각 매뉴얼별로 청크 인덱스를 구축"""
        indices = defaultdict(list)
        
        for idx in range(self.chunk_loader.get_total_chunks()):
            try:
                chunk = self.chunk_loader.get_chunk(idx)
                source = chunk.get('source', '').lower()
                
                if '대규모내부거래' in source:
                    indices['대규모내부거래 매뉴얼'].append(idx)
                elif '현황공시' in source or '기업집단' in source:
                    indices['현황공시 매뉴얼'].append(idx)
                elif '비상장' in source:
                    indices['비상장사 중요사항 매뉴얼'].append(idx)
                    
            except Exception as e:
                logger.warning(f"Failed to process chunk {idx}: {e}")
                continue
        
        return dict(indices)
    
    async def search_documents(self, intent: IntentAnalysis, top_k: int = 10) -> List[SearchResult]:
        """의도 분석 결과를 바탕으로 문서 검색"""
        
        # 검색할 인덱스 결정
        search_indices = []
        for doc_name in intent.target_documents:
            if doc_name in self.manual_indices:
                search_indices.extend(self.manual_indices[doc_name])
        
        if not search_indices:
            # 모든 문서에서 검색
            search_indices = list(range(self.chunk_loader.get_total_chunks()))
        
        # 검색 쿼리 생성 (키워드 결합)
        search_query = f"{intent.core_intent} {' '.join(intent.search_keywords)}"
        
        # 벡터 검색 수행
        if self.use_faiss:
            results = await self._vector_search(search_query, search_indices, top_k * 2)
        else:
            results = self._keyword_search(search_query, search_indices, top_k * 2)
        
        # 의도에 맞게 결과 재정렬
        results = self._rerank_by_intent(results, intent)
        
        return results[:top_k]
    
    async def _vector_search(self, query: str, indices: List[int], k: int) -> List[SearchResult]:
        """벡터 기반 검색"""
        # 쿼리 임베딩 생성
        if self.embedding_model is not None:
            query_vector = self.embedding_model.encode([query])[0]
        else:
            query_vector = self.api_manager.get_embedding(query)
        
        query_vector = query_vector.reshape(1, -1).astype(np.float32)
        
        # FAISS 검색
        scores, search_indices = self.index.search(query_vector, k)
        
        # 결과 변환
        results = []
        for idx, score in zip(search_indices[0], scores[0]):
            if idx in indices and 0 <= idx < self.chunk_loader.get_total_chunks():
                chunk = self.chunk_loader.get_chunk(idx)
                result = SearchResult(
                    chunk_id=str(idx),
                    content=chunk['content'],
                    score=float(score),
                    source=chunk.get('source', 'Unknown'),
                    page=chunk.get('page', 0),
                    chunk_type=chunk.get('chunk_type', 'text'),
                    metadata=json.loads(chunk.get('metadata', '{}'))
                )
                results.append(result)
        
        return results
    
    def _keyword_search(self, query: str, indices: List[int], k: int) -> List[SearchResult]:
        """키워드 기반 검색 (폴백)"""
        query_words = set(query.lower().split())
        scored_chunks = []
        
        for idx in indices[:1000]:  # 성능을 위해 제한
            try:
                chunk = self.chunk_loader.get_chunk(idx)
                content_lower = chunk['content'].lower()
                
                # 키워드 매칭 점수
                score = sum(1 for word in query_words if word in content_lower)
                
                if score > 0:
                    scored_chunks.append((idx, score, chunk))
            except Exception as e:
                continue
        
        # 점수순 정렬
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        
        # SearchResult 생성
        results = []
        for idx, score, chunk in scored_chunks[:k]:
            result = SearchResult(
                chunk_id=str(idx),
                content=chunk['content'],
                score=float(score),
                source=chunk.get('source', 'Unknown'),
                page=chunk.get('page', 0),
                chunk_type=chunk.get('chunk_type', 'text'),
                metadata=json.loads(chunk.get('metadata', '{}'))
            )
            results.append(result)
        
        return results
    
    def _rerank_by_intent(self, results: List[SearchResult], intent: IntentAnalysis) -> List[SearchResult]:
        """의도에 맞게 검색 결과 재정렬"""
        
        for result in results:
            # 키워드 매칭 보너스
            keyword_matches = sum(1 for kw in intent.search_keywords 
                                if kw.lower() in result.content.lower())
            result.score += keyword_matches * 0.1
            
            # 개체 매칭 보너스
            entity_matches = sum(1 for entity in intent.key_entities 
                               if entity.lower() in result.content.lower())
            result.score += entity_matches * 0.15
            
            # 특정 조건에 따른 가중치
            if intent.requires_calculation and re.search(r'\d+억|\d+%', result.content):
                result.score *= 1.2
            
            if intent.requires_timeline and re.search(r'\d+일|기한|날짜', result.content):
                result.score *= 1.2
        
        # 재정렬
        results.sort(key=lambda x: x.score, reverse=True)
        return results

class Step3_AnswerGenerator:
    """Step 3: 검색 결과를 바탕으로 답변 생성
    
    이 단계에서는:
    - 검색된 문서와 질문 의도를 결합
    - 적절한 모델 선택 (o4-mini, gpt-4o 등)
    - 체계적이고 정확한 답변 생성
    """
    
    def __init__(self, api_manager: APIManager):
        self.api_manager = api_manager
        
    async def generate_answer(self, 
                            query: str, 
                            intent: IntentAnalysis,
                            search_results: List[SearchResult]) -> str:
        """검색 결과를 바탕으로 답변 생성"""
        
        # 모델 선택
        model = self._select_model(intent)
        
        # 컨텍스트 구성
        context = self._build_context(search_results)
        
        # 프롬프트 구성
        if model == "o4-mini":
            prompt = self._build_reasoning_prompt(query, intent, context)
        else:
            prompt = self._build_standard_prompt(query, intent, context)
        
        # 답변 생성
        try:
            response = await openai.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=2000
            )
            
            answer = response.choices[0].message.content
            
            # 답변 후처리
            answer = self._postprocess_answer(answer, intent)
            
            return answer
            
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            return "죄송합니다. 답변 생성 중 오류가 발생했습니다. 질문을 다시 시도해주세요."
    
    def _select_model(self, intent: IntentAnalysis) -> str:
        """질문 유형에 따라 최적의 모델 선택"""
        
        if intent.query_type == "complex_analysis":
            # 복잡한 분석이 필요한 경우 추론 모델 사용
            return "o4-mini"
        elif intent.query_type == "simple_lookup":
            # 단순 조회는 빠른 모델 사용
            return "gpt-4o-mini"
        else:
            # 절차적 질문은 범용 모델 사용
            return "gpt-4o"
    
    def _build_context(self, search_results: List[SearchResult]) -> str:
        """검색 결과로부터 컨텍스트 구성"""
        context_parts = []
        
        for i, result in enumerate(search_results[:5]):  # 상위 5개만 사용
            context_parts.append(f"""
[참고자료 {i+1}]
출처: {result.source} (페이지 {result.page})
내용: {result.content}
""")
        
        return "\n---\n".join(context_parts)
    
    def _build_reasoning_prompt(self, query: str, intent: IntentAnalysis, context: str) -> str:
        """o4-mini용 추론 중심 프롬프트"""
        return f"""
당신은 한국 공정거래위원회의 법률 전문가입니다.
다음 참고자료를 바탕으로 사용자의 질문에 대해 단계별로 추론하여 답변하세요.

[사용자 질문]
{query}

[질문 의도 분석]
- 핵심 의도: {intent.core_intent}
- 관련 개체: {', '.join(intent.key_entities)}
- 시간순서 중요: {'예' if intent.requires_timeline else '아니오'}
- 계산 필요: {'예' if intent.requires_calculation else '아니오'}

[참고자료]
{context}

[답변 작성 지침]
1. 먼저 질문의 각 요소를 개별적으로 분석하세요
2. 관련 규정과 조건을 단계별로 확인하세요
3. 필요한 경우 계산 과정을 명시하세요
4. 최종 결론을 명확히 제시하세요

답변 형식:
## 핵심 답변
(1-2문장으로 직접적인 답변)

## 상세 분석
### 1. 적용 규정
- 관련 조항과 기준

### 2. 구체적 검토
- 단계별 분석 내용

### 3. 주의사항
- 예외사항이나 특별히 유의할 점

## 결론
(최종 정리 및 실무 지침)
"""
    
    def _build_standard_prompt(self, query: str, intent: IntentAnalysis, context: str) -> str:
        """표준 모델용 프롬프트"""
        return f"""
당신은 한국 공정거래위원회의 법률 전문가입니다.
다음 참고자료를 바탕으로 사용자의 질문에 정확하고 실무적인 답변을 제공하세요.

[사용자 질문]
{query}

[참고자료]
{context}

[답변 지침]
- 핵심 내용을 먼저 제시하고 상세 설명을 추가하세요
- 근거 조항을 명확히 인용하세요
- 실무에 바로 적용할 수 있는 구체적인 답변을 제공하세요
- 관련 주의사항이 있다면 반드시 언급하세요

답변:
"""
    
    def _postprocess_answer(self, answer: str, intent: IntentAnalysis) -> str:
        """답변 후처리 - 최신 정보 확인 등"""
        
        # 최신 규정 정보 추가
        if "대규모내부거래" in ' '.join(intent.target_documents):
            if "50억" in answer or "30억" in answer:
                answer += "\n\n⚠️ **중요**: 대규모내부거래 기준 금액은 2023년 1월 1일부터 100억원으로 변경되었습니다."
        
        if "공시" in answer and "7일" in answer:
            if "영업일" not in answer:
                answer += "\n\n📌 **참고**: 공시 기한은 영업일 기준 7일입니다."
        
        return answer

# ===== 통합 RAG 파이프라인 =====
class IntegratedRAGPipeline:
    """3단계 프로세스를 통합한 RAG 파이프라인"""
    
    def __init__(self, embedding_model, index, chunk_loader: ChunkLoader, api_manager: APIManager):
        self.step1_analyzer = Step1_IntentAnalyzer(api_manager)
        self.step2_searcher = Step2_DocumentSearcher(index, chunk_loader, embedding_model, api_manager)
        self.step3_generator = Step3_AnswerGenerator(api_manager)
        
    async def process_query(self, query: str) -> Tuple[str, Dict]:
        """전체 프로세스 실행"""
        stats = {
            "process_times": {},
            "intent_analysis": None,
            "search_results_count": 0
        }
        
        try:
            # Step 1: 의도 분석
            start_time = time.time()
            intent = await self.step1_analyzer.analyze_intent(query)
            stats["process_times"]["intent_analysis"] = time.time() - start_time
            stats["intent_analysis"] = intent
            
            # Step 2: 문서 검색
            start_time = time.time()
            search_results = await self.step2_searcher.search_documents(intent)
            stats["process_times"]["document_search"] = time.time() - start_time
            stats["search_results_count"] = len(search_results)
            
            # Step 3: 답변 생성
            start_time = time.time()
            answer = await self.step3_generator.generate_answer(query, intent, search_results)
            stats["process_times"]["answer_generation"] = time.time() - start_time
            
            stats["total_time"] = sum(stats["process_times"].values())
            
            return answer, stats
            
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            return "처리 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.", stats

# ===== 비동기 실행 헬퍼 =====
def run_async_in_streamlit(coro):
    """Streamlit 환경에서 비동기 함수 실행"""
    try:
        # 이벤트 루프 확인 및 실행
        try:
            loop = asyncio.get_running_loop()
            if NEST_ASYNCIO_AVAILABLE:
                import nest_asyncio
                nest_asyncio.apply()
            return asyncio.run(coro)
        except RuntimeError:
            return asyncio.run(coro)
    except Exception as e:
        logger.error(f"Async execution failed: {str(e)}")
        raise

# ===== Streamlit UI =====
st.set_page_config(
    page_title="공정거래위원회 AI 법률 보조원", 
    page_icon="⚖️", 
    layout="wide"
)

# CSS 스타일
st.markdown("""
<style>
    /* 메인 헤더 */
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #1f4788 0%, #2a5298 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    
    .main-header p {
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
        font-size: 1.1rem;
    }
    
    /* 프로세스 단계 표시 */
    .process-step {
        display: inline-block;
        padding: 0.5rem 1rem;
        margin: 0.25rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 500;
    }
    
    .step-active {
        background-color: #2a5298;
        color: white;
    }
    
    .step-completed {
        background-color: #28a745;
        color: white;
    }
    
    .step-pending {
        background-color: #e0e0e0;
        color: #666;
    }
    
    /* 의도 분석 결과 */
    .intent-box {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    /* 숨기기 */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
</style>
""", unsafe_allow_html=True)

# 모델 및 데이터 로딩
@st.cache_resource(show_spinner=False)
def load_models_and_data():
    """필요한 모델과 데이터 로드"""
    try:
        # API 관리자 초기화
        api_manager = APIManager()
        api_manager.load_api_key()
        openai.api_key = api_manager._api_key
        
        # 필수 파일 확인
        required_files = ["all_manual_chunks.json"]
        missing_files = [f for f in required_files if not os.path.exists(f)]
        
        if missing_files:
            st.error(f"❌ 필수 파일이 없습니다: {', '.join(missing_files)}")
            return None
        
        with st.spinner("🤖 AI 시스템을 준비하는 중..."):
            # FAISS 인덱스 로드 시도
            index = None
            if os.path.exists("manuals_vector_db.index") and FAISS_AVAILABLE:
                try:
                    import faiss
                    index = faiss.read_index("manuals_vector_db.index")
                    logger.info("FAISS index loaded successfully")
                except Exception as e:
                    logger.warning(f"Failed to load FAISS index: {e}")
            
            # 청크 로더 초기화
            chunk_loader = ChunkLoader("all_manual_chunks.json")
            
            # 임베딩 모델 로드 시도
            embedding_model = None
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                try:
                    embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
                    logger.info("Sentence transformer model loaded")
                except Exception as e:
                    logger.warning(f"Failed to load sentence transformer: {e}")
            
            return embedding_model, index, chunk_loader, api_manager
        
    except Exception as e:
        st.error(f"시스템 초기화 실패: {str(e)}")
        return None

def main():
    """메인 애플리케이션"""
    
    # 헤더
    st.markdown("""
    <div class="main-header">
        <h1>⚖️ 공정거래위원회 AI 법률 보조원</h1>
        <p>대규모내부거래, 현황공시, 비상장사 중요사항 관련 전문 상담</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 모델 로드
    models = load_models_and_data()
    if not models or len(models) != 4:
        st.stop()
    
    embedding_model, index, chunk_loader, api_manager = models
    
    # RAG 파이프라인 초기화
    pipeline = IntegratedRAGPipeline(embedding_model, index, chunk_loader, api_manager)
    
    # 세션 상태 초기화
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # 채팅 인터페이스
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.write(message["content"])
            else:
                st.write(message["answer"])
                
                # 프로세스 정보 표시
                if "stats" in message:
                    stats = message["stats"]
                    if stats.get("intent_analysis"):
                        with st.expander("🔍 질문 분석 결과"):
                            intent = stats["intent_analysis"]
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write(f"**핵심 의도**: {intent.core_intent}")
                                st.write(f"**질문 유형**: {intent.query_type}")
                                st.write(f"**검색 대상**: {', '.join(intent.target_documents)}")
                            with col2:
                                st.write(f"**핵심 키워드**: {', '.join(intent.search_keywords[:5])}")
                                st.write(f"**관련 개체**: {', '.join(intent.key_entities)}")
                                st.write(f"**신뢰도**: {intent.confidence:.1%}")
                    
                    # 처리 시간 표시
                    if "process_times" in stats:
                        times = stats["process_times"]
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("의도 분석", f"{times.get('intent_analysis', 0):.1f}초")
                        with col2:
                            st.metric("문서 검색", f"{times.get('document_search', 0):.1f}초")
                        with col3:
                            st.metric("답변 생성", f"{times.get('answer_generation', 0):.1f}초")
    
    # 새 질문 입력
    if prompt := st.chat_input("공정거래 관련 질문을 입력하세요"):
        # 사용자 메시지 추가
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.write(prompt)
        
        # AI 응답 생성
        with st.chat_message("assistant"):
            # 프로세스 진행 상황 표시
            progress_container = st.container()
            with progress_container:
                col1, col2, col3 = st.columns(3)
                with col1:
                    step1_placeholder = st.empty()
                    step1_placeholder.markdown('<span class="process-step step-active">1️⃣ 질문 분석 중...</span>', unsafe_allow_html=True)
                with col2:
                    step2_placeholder = st.empty()
                    step2_placeholder.markdown('<span class="process-step step-pending">2️⃣ 문서 검색</span>', unsafe_allow_html=True)
                with col3:
                    step3_placeholder = st.empty()
                    step3_placeholder.markdown('<span class="process-step step-pending">3️⃣ 답변 생성</span>', unsafe_allow_html=True)
            
            # 파이프라인 실행
            answer, stats = run_async_in_streamlit(pipeline.process_query(prompt))
            
            # 프로세스 완료 표시
            step1_placeholder.markdown('<span class="process-step step-completed">1️⃣ 질문 분석 ✓</span>', unsafe_allow_html=True)
            step2_placeholder.markdown('<span class="process-step step-completed">2️⃣ 문서 검색 ✓</span>', unsafe_allow_html=True)
            step3_placeholder.markdown('<span class="process-step step-completed">3️⃣ 답변 생성 ✓</span>', unsafe_allow_html=True)
            
            # 답변 표시
            st.write(answer)
            
            # 메시지 저장
            st.session_state.messages.append({
                "role": "assistant",
                "answer": answer,
                "stats": stats
            })
    
    # 사이드바
    with st.sidebar:
        st.header("📚 사용 가이드")
        
        st.subheader("질문 예시")
        
        example_questions = [
            "대규모내부거래 이사회 의결 후 공시 기한은?",
            "계열사 간 자금 대여 시 이사회 의결이 필요한 금액은?",
            "비상장회사가 주식을 양도할 때 필요한 절차는?",
            "기업집단 현황공시는 언제 해야 하나요?",
            "여러 계열사와 동시에 거래할 때 주의사항은?"
        ]
        
        for q in example_questions:
            if st.button(q, key=q):
                st.session_state.new_question = q
                st.rerun()
        
        st.divider()
        
        st.subheader("🔧 시스템 정보")
        st.caption("이 시스템은 3단계 프로세스로 작동합니다:")
        st.caption("1️⃣ **질문 의도 분석**: 사용자의 질문을 이해하고 검색 전략 수립")
        st.caption("2️⃣ **문서 검색**: 관련 매뉴얼에서 필요한 정보 검색")
        st.caption("3️⃣ **답변 생성**: 검색 결과를 바탕으로 정확한 답변 생성")
        
        st.divider()
        
        st.caption("💡 각 단계에서 최적화된 AI 모델이 사용됩니다:")
        st.caption("• **의도 분석**: GPT-4o-mini (빠른 분석)")
        st.caption("• **답변 생성**: o4-mini, GPT-4o (질문 복잡도에 따라)")
    
    # 새 질문 처리
    if "new_question" in st.session_state:
        st.session_state.messages.append({"role": "user", "content": st.session_state.new_question})
        del st.session_state.new_question
        st.rerun()

if __name__ == "__main__":
    main()
