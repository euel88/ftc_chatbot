# 파일 이름: app_manual.py (공정거래위원회 AI 법률 보조원 - 완전한 버전)

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

class IndexError(RAGPipelineError):
    """인덱스 관련 오류"""
    pass

class EmbeddingError(RAGPipelineError):
    """임베딩 생성 관련 오류"""
    pass

class GPTAnalysisError(RAGPipelineError):
    """GPT 분석 실패 관련 오류"""
    pass

class APIKeyError(RAGPipelineError):
    """API 키 관련 오류"""
    pass

# ===== 에러 컨텍스트 매니저 =====
@contextmanager
def error_context(operation_name: str, fallback_value=None):
    """에러 처리를 위한 컨텍스트 매니저
    
    이 컨텍스트 매니저는 작업 중 발생하는 에러를 우아하게 처리하고,
    사용자에게 친숙한 메시지를 보여주면서도 개발자를 위한 상세 정보를 로깅합니다.
    """
    try:
        logger.debug(f"Starting operation: {operation_name}")
        yield
    except Exception as e:
        logger.error(f"Error in {operation_name}: {str(e)}")
        logger.debug(f"Full traceback:\n{traceback.format_exc()}")
        
        if 'st' in globals():
            st.error(f"⚠️ 작업 중 오류 발생: {operation_name}")
            with st.expander("🔍 상세 오류 정보"):
                st.code(traceback.format_exc())
        
        if fallback_value is not None:
            return fallback_value
        raise

# ===== API 관리자 =====
class APIManager:
    """OpenAI API 키와 호출을 관리하는 클래스"""
    
    def __init__(self):
        self._api_key = None
        self._last_call_time = {}
        self._embedding_cache = {}
        
        # 각 모델의 속도 제한 설정
        self._rate_limits = {
            'gpt-4o': {'calls_per_minute': 60, 'tokens_per_minute': 150000},
            'gpt-4o-mini': {'calls_per_minute': 500, 'tokens_per_minute': 200000},
            'o4-mini': {'calls_per_minute': 30, 'tokens_per_minute': 100000}
        }
    
    def load_api_key(self) -> str:
        """API 키를 안전하게 로드
        
        우선순위:
        1. Streamlit secrets (개발/프로덕션 환경)
        2. 환경 변수 (로컬 테스트)
        3. 암호화된 파일 (고급 보안 - 선택적)
        """
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
        
        # 암호화된 파일 확인 (선택적 - cryptography 모듈이 있을 때만)
        if CRYPTOGRAPHY_AVAILABLE:
            try:
                from cryptography.fernet import Fernet
                encrypted_key_path = Path('.api_key.enc')
                if encrypted_key_path.exists():
                    cipher_key = os.environ.get('API_CIPHER_KEY')
                    if cipher_key:
                        cipher = Fernet(cipher_key.encode())
                        with open(encrypted_key_path, 'rb') as f:
                            encrypted_key = f.read()
                        self._api_key = cipher.decrypt(encrypted_key).decode()
                        logger.info("API key loaded from encrypted file")
                        return self._api_key
            except Exception as e:
                logger.error(f"Failed to decrypt API key: {e}")
        
        raise APIKeyError("API 키를 찾을 수 없습니다. Streamlit secrets 또는 환경 변수를 확인하세요.")
    
    def rate_limit(self, model: str = 'gpt-4o'):
        """API 호출 속도 제한 데코레이터"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                current_time = time.time()
                model_key = f"{model}_last_call"
                
                # 마지막 호출 시간 확인
                if model_key in self._last_call_time:
                    elapsed = current_time - self._last_call_time[model_key]
                    min_interval = 60.0 / self._rate_limits[model]['calls_per_minute']
                    
                    if elapsed < min_interval:
                        sleep_time = min_interval - elapsed
                        logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f}s")
                        time.sleep(sleep_time)
                
                # 호출 실행
                result = func(*args, **kwargs)
                
                # 마지막 호출 시간 업데이트
                self._last_call_time[model_key] = time.time()
                
                return result
            return wrapper
        return decorator
    
    def get_embedding(self, text: str) -> np.ndarray:
        """OpenAI embeddings API를 사용하여 텍스트 임베딩 생성"""
        # 캐시 확인
        cache_key = hashlib.md5(text.encode()).hexdigest()
        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]
        
        try:
            response = openai.embeddings.create(
                input=text,
                model="text-embedding-ada-002"
            )
            embedding = np.array(response.data[0].embedding)
            
            # 캐시에 저장
            self._embedding_cache[cache_key] = embedding
            
            # 캐시 크기 제한
            if len(self._embedding_cache) > 1000:
                # 가장 오래된 항목 제거
                oldest_key = next(iter(self._embedding_cache))
                del self._embedding_cache[oldest_key]
            
            return embedding
        except Exception as e:
            logger.error(f"Failed to get OpenAI embedding: {e}")
            # 실패 시 랜덤 벡터 반환 (임시)
            return np.random.randn(1536)  # OpenAI 임베딩 차원

# ===== 메모리 효율적인 청크 로더 =====
class ChunkLoader:
    """메모리 효율적인 청크 로딩 시스템
    
    이 클래스는 대용량 JSON 파일을 효율적으로 처리합니다.
    ijson이 있으면 스트리밍 방식을, 없으면 일반 방식을 사용합니다.
    """
    
    def __init__(self, filepath: str):
        self.filepath = filepath
        self._chunks = None
        self._chunk_cache = OrderedDict()
        self._cache_size = 1000
        self._use_streaming = False
        
        # ijson 사용 가능 여부 확인
        if IJSON_AVAILABLE:
            self._use_streaming = True
            logger.info("Using streaming JSON parser (ijson)")
        else:
            self._use_streaming = False
            logger.info("ijson not available, using standard JSON loading")
        
        self._initialize()
    
    def _initialize(self):
        """초기화 - 스트리밍 또는 일반 방식 선택"""
        if self._use_streaming:
            try:
                self._build_streaming_index()
            except Exception as e:
                logger.warning(f"Streaming index failed: {e}, falling back to standard loading")
                self._use_streaming = False
                self._load_all_chunks()
        else:
            self._load_all_chunks()
    
    def _load_all_chunks(self):
        """전체 청크를 메모리에 로드 (폴백 방식)"""
        logger.info(f"Loading all chunks from {self.filepath}")
        try:
            with open(self.filepath, 'r', encoding='utf-8') as f:
                self._chunks = json.load(f)
            logger.info(f"Loaded {len(self._chunks)} chunks into memory")
        except Exception as e:
            logger.error(f"Failed to load chunks: {e}")
            self._chunks = []
    
    def _build_streaming_index(self):
        """스트리밍 방식으로 인덱스 구축 (ijson 필요)"""
        import ijson
        logger.info(f"Building streaming index for {self.filepath}")
        self._index = {}
        
        with open(self.filepath, 'rb') as f:
            parser = ijson.items(f, 'item')
            for idx, item in enumerate(parser):
                # 인덱스만 저장하고 실제 데이터는 나중에 로드
                self._index[idx] = idx
        
        logger.info(f"Streaming index built: {len(self._index)} chunks found")
    
    def get_chunk(self, idx: int) -> Dict:
        """특정 인덱스의 청크를 가져옴"""
        # 캐시 확인
        if idx in self._chunk_cache:
            self._chunk_cache.move_to_end(idx)
            return self._chunk_cache[idx]
        
        # 스트리밍 방식이 아니면 메모리에서 직접 반환
        if not self._use_streaming:
            if self._chunks and 0 <= idx < len(self._chunks):
                chunk = self._chunks[idx]
                self._add_to_cache(idx, chunk)
                return chunk
            else:
                raise IndexError(f"Chunk index {idx} out of range")
        
        # 스트리밍 방식으로 특정 청크 로드
        try:
            import ijson
            with open(self.filepath, 'rb') as f:
                parser = ijson.items(f, 'item')
                for i, item in enumerate(parser):
                    if i == idx:
                        self._add_to_cache(idx, item)
                        return item
            raise IndexError(f"Chunk index {idx} not found")
        except Exception as e:
            logger.error(f"Failed to load chunk {idx}: {e}")
            # 폴백: 전체 로드 시도
            if not self._chunks:
                self._load_all_chunks()
            if self._chunks and 0 <= idx < len(self._chunks):
                return self._chunks[idx]
            raise
    
    def _add_to_cache(self, idx: int, chunk: Dict):
        """캐시에 청크 추가 (LRU 정책)"""
        if len(self._chunk_cache) >= self._cache_size:
            self._chunk_cache.popitem(last=False)
        self._chunk_cache[idx] = chunk
    
    def iter_chunks(self, indices: List[int] = None) -> Iterator[Dict]:
        """필요한 청크들을 순차적으로 반환"""
        if not self._use_streaming and self._chunks:
            # 메모리에 로드된 경우
            if indices is None:
                indices = range(len(self._chunks))
            for idx in indices:
                if 0 <= idx < len(self._chunks):
                    yield self._chunks[idx]
        else:
            # 스트리밍 방식
            if indices is None:
                indices = range(self.get_total_chunks())
            for idx in indices:
                yield self.get_chunk(idx)
    
    def get_total_chunks(self) -> int:
        """전체 청크 수 반환"""
        if not self._use_streaming and self._chunks:
            return len(self._chunks)
        elif hasattr(self, '_index'):
            return len(self._index)
        else:
            # 카운트를 위해 한 번 스캔
            if not self._chunks:
                self._load_all_chunks()
            return len(self._chunks) if self._chunks else 0

# ===== 타입 정의 =====
class ModelSelection(Enum):
    """사용 가능한 모델들"""
    GPT4O = "gpt-4o"
    GPT4O_MINI = "gpt-4o-mini"
    O4_MINI = "o4-mini"  # 추론 특화 모델

class ChunkDict(TypedDict):
    """청크 데이터의 타입 정의"""
    chunk_id: str
    content: str
    source: str
    page: int
    chunk_type: str
    metadata: str

class AnalysisResult(TypedDict):
    """GPT 분석 결과의 타입 정의"""
    query_analysis: dict
    legal_concepts: list
    search_strategy: dict
    answer_requirements: dict
    model_used: str

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
    
    @property
    def is_latest(self) -> bool:
        """최신 자료 여부 확인"""
        return self.metadata.get('is_latest', False)

@dataclass
class SearchError:
    """검색 중 발생한 에러 정보"""
    error_type: str
    message: str
    timestamp: float
    context: Dict[str, Any]
    severity: str  # 'critical', 'warning', 'info'

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
    complexity_reason: str  # 복잡도 판단 이유
    confidence: float  # 분석 신뢰도

class QueryComplexity(Enum):
    """질문 복잡도 레벨"""
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"

# ===== 복잡도 평가기 =====
class ComplexityAssessor:
    """질문의 복잡도를 평가하여 처리 방식을 결정"""
    
    def __init__(self):
        self.simple_indicators = [
            r'언제', r'며칠', r'기한', r'날짜', r'금액', r'%', r'얼마',
            r'정의[가는]?', r'무엇', r'뜻[이은]?', r'의미[가는]?'
        ]
        
        self.complex_indicators = [
            r'동시에', r'여러', r'복합', r'연관', r'영향',
            r'만[약일].*경우', r'[AB].*동시.*[CD]', r'거래.*여러',
            r'전체적', r'종합적', r'분석', r'검토', r'평가',
            r'리스크', r'위험', r'대응', r'전략'
        ]
        
        self.medium_indicators = [
            r'어떻게', r'방법', r'절차', r'과정',
            r'주의', r'예외', r'특별', r'고려'
        ]
    
    def assess(self, query: str) -> Tuple[QueryComplexity, float, Dict]:
        """질문의 복잡도를 평가하고 관련 정보 반환"""
        query_lower = query.lower()
        
        simple_score = sum(1 for pattern in self.simple_indicators 
                         if re.search(pattern, query_lower))
        complex_score = sum(2 for pattern in self.complex_indicators 
                          if re.search(pattern, query_lower))
        medium_score = sum(1.5 for pattern in self.medium_indicators 
                         if re.search(pattern, query_lower))
        
        # 길이에 따른 가중치
        if len(query) > 150:
            complex_score += 2
        elif len(query) > 100:
            complex_score += 1
        elif len(query) < 30:
            simple_score += 0.5
            
        # 복수 질문 여부
        if query.count('?') > 1 or re.search(r'그리고.*[?]', query_lower):
            complex_score += 1.5
            
        total_score = simple_score + medium_score + complex_score
        
        # 복잡도 결정
        if total_score == 0:
            complexity = QueryComplexity.MEDIUM
            confidence = 0.5
        elif complex_score > simple_score * 2:
            complexity = QueryComplexity.COMPLEX
            confidence = min(complex_score / (total_score + 1), 0.9)
        elif simple_score > complex_score * 2:
            complexity = QueryComplexity.SIMPLE
            confidence = min(simple_score / (total_score + 1), 0.9)
        else:
            complexity = QueryComplexity.MEDIUM
            confidence = 0.6
            
        analysis = {
            'simple_score': simple_score,
            'medium_score': medium_score,
            'complex_score': complex_score,
            'query_length': len(query),
            'confidence': confidence
        }
        
        return complexity, confidence, analysis

# ===== LRU 캐시 구현 =====
class LRUCache:
    """시간 기반 만료를 지원하는 LRU 캐시 구현"""
    def __init__(self, max_size: int = 100, ttl: int = 3600):
        self.cache = OrderedDict()
        self.max_size = max_size
        self.ttl = ttl
        
    def get(self, key: str):
        """캐시에서 값을 가져오고, 만료된 항목은 제거"""
        if key not in self.cache:
            return None
            
        value, timestamp = self.cache[key]
        if time.time() - timestamp > self.ttl:
            del self.cache[key]
            return None
            
        # LRU: 최근 사용 항목을 끝으로 이동
        self.cache.move_to_end(key)
        return value
        
    def put(self, key: str, value):
        """캐시에 값 저장"""
        if key in self.cache:
            del self.cache[key]
            
        if len(self.cache) >= self.max_size:
            # 가장 오래된 항목 제거
            self.cache.popitem(last=False)
            
        self.cache[key] = (value, time.time())
        
    def clear_expired(self):
        """만료된 모든 항목 제거"""
        current_time = time.time()
        expired_keys = [
            key for key, (_, timestamp) in self.cache.items()
            if current_time - timestamp > self.ttl
        ]
        for key in expired_keys:
            del self.cache[key]

# ===== 문서 버전 관리 =====
class DocumentVersionManager:
    """문서의 버전과 최신성을 관리하는 시스템"""
    
    def __init__(self):
        self.regulation_changes = {
            '대규모내부거래_금액기준': [
                {'date': '2023-01-01', 'old_value': '50억원', 'new_value': '100억원',
                 'description': '자본금 및 자본총계 중 큰 금액의 5% 이상 또는 100억원 이상'},
                {'date': '2020-01-01', 'old_value': '30억원', 'new_value': '50억원',
                 'description': '자본금 및 자본총계 중 큰 금액의 5% 이상 또는 50억원 이상'}
            ],
            '공시_기한': [
                # 현행 규정: 영업일 7일
                {'date': '2019-01-01', 'old_value': '7일', 'new_value': '영업일 7일',
                 'description': '이사회 의결 후 공시 기한 (영업일 기준으로 명확화)'}
            ]
        }
        
        self.critical_patterns = {
            '금액': r'(\d+)억\s*원',
            '비율': r'(\d+(?:\.\d+)?)\s*%',
            '기한': r'(\d+)\s*일',
            '날짜': r'(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일'
        }
    
    def extract_document_date(self, chunk: Dict) -> Optional[str]:
        """문서에서 작성/개정 날짜 추출"""
        content = chunk.get('content', '')
        metadata = json.loads(chunk.get('metadata', '{}'))
        
        # 메타데이터에 날짜가 있으면 우선 사용
        if 'document_date' in metadata:
            return metadata['document_date']
        
        # 내용에서 날짜 패턴 추출
        date_patterns = [
            r'(\d{4})년\s*(\d{1,2})월\s*개정',
            r'시행일\s*:\s*(\d{4})년\s*(\d{1,2})월',
            r'(\d{4})\.\s*(\d{1,2})\.\s*(\d{1,2})',
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, content)
            if match:
                return self._normalize_date(match.group(0))
        
        return None
    
    def _normalize_date(self, date_str: str) -> str:
        """날짜 문자열을 표준 형식으로 변환"""
        # 숫자만 추출
        numbers = re.findall(r'\d+', date_str)
        if len(numbers) >= 2:
            year = numbers[0] if len(numbers[0]) == 4 else '20' + numbers[0]
            month = numbers[1].zfill(2)
            day = numbers[2].zfill(2) if len(numbers) > 2 else '01'
            return f"{year}-{month}-{day}"
        return None
    
    def check_for_outdated_info(self, content: str, document_date: str = None) -> List[Dict]:
        """구버전 정보가 포함되어 있는지 확인"""
        warnings = []
        
        # 대규모내부거래 금액 기준 확인
        amount_match = re.search(r'(\d+)억\s*원.*대규모내부거래', content)
        if amount_match:
            amount = int(amount_match.group(1))
            if amount == 50:
                warnings.append({
                    'type': 'outdated_amount',
                    'found': '50억원',
                    'current': '100억원',
                    'regulation': '대규모내부거래 금액 기준',
                    'changed_date': '2023-01-01',
                    'severity': 'critical'
                })
            elif amount == 30:
                warnings.append({
                    'type': 'outdated_amount',
                    'found': '30억원',
                    'current': '100억원',
                    'regulation': '대규모내부거래 금액 기준',
                    'changed_date': '2023-01-01',
                    'severity': 'critical'
                })
        
        # 공시 기한 확인 - 영업일 명시 여부 체크
        deadline_match = re.search(r'의결.*?(\d+)\s*일.*?공시', content)
        if deadline_match:
            # "영업일" 명시 여부 확인
            context = content[max(0, deadline_match.start()-20):deadline_match.end()+20]
            if '영업일' not in context and '영업일' not in content[max(0, deadline_match.start()-50):deadline_match.end()+50]:
                # 영업일이 명시되지 않은 경우 경고
                warnings.append({
                    'type': 'unclear_deadline',
                    'found': f'{deadline_match.group(1)}일',
                    'current': '영업일 7일',
                    'regulation': '공시 기한',
                    'changed_date': '2019-01-01',
                    'severity': 'warning',
                    'note': '영업일 기준임을 명확히 해야 함'
                })
        
        return warnings

# ===== 충돌 해결 시스템 =====
class ConflictResolver:
    """상충하는 정보를 해결하는 시스템"""
    
    def __init__(self, version_manager: DocumentVersionManager):
        self.version_manager = version_manager
    
    def resolve_conflicts(self, results: List[SearchResult], query: str) -> List[SearchResult]:
        """검색 결과 중 상충하는 정보를 해결하고 최신 정보를 우선시"""
        
        # 각 결과의 날짜와 구버전 정보 확인
        has_outdated = False
        for result in results:
            doc_date = result.document_date
            warnings = self.version_manager.check_for_outdated_info(result.content, doc_date)
            
            if warnings:
                result.metadata['warnings'] = warnings
                result.metadata['has_outdated_info'] = True
                has_outdated = True
            else:
                result.metadata['has_outdated_info'] = False
        
        # 중요 정보 추출 및 충돌 확인
        critical_info = self._extract_critical_info(results, query)
        if critical_info:
            conflicts = self._find_conflicts(critical_info)
            if conflicts:
                results = self._prioritize_latest_info(results, conflicts)
        
        # 최신 정보를 우선으로 정렬
        results.sort(key=lambda r: (
            not r.metadata.get('has_outdated_info', False),  # 최신 정보 우선
            r.document_date or '1900-01-01',  # 최신 날짜 우선
            r.score  # 관련도 점수
        ), reverse=True)
        
        return results, has_outdated
    
    def _extract_critical_info(self, results: List[SearchResult], query: str) -> Dict:
        """결과에서 중요 정보 추출"""
        critical_info = defaultdict(list)
        
        for i, result in enumerate(results):
            # 금액 정보
            amounts = re.findall(r'(\d+)억\s*원', result.content)
            for amount in amounts:
                critical_info['amounts'].append({
                    'value': amount + '억원',
                    'result_index': i,
                    'context': result.content[:100]
                })
            
            # 비율 정보
            percentages = re.findall(r'(\d+(?:\.\d+)?)\s*%', result.content)
            for pct in percentages:
                critical_info['percentages'].append({
                    'value': pct + '%',
                    'result_index': i,
                    'context': result.content[:100]
                })
            
            # 기한 정보
            deadlines = re.findall(r'(\d+)\s*일', result.content)
            for deadline in deadlines:
                critical_info['deadlines'].append({
                    'value': deadline + '일',
                    'result_index': i,
                    'context': result.content[:100]
                })
        
        return dict(critical_info)
    
    def _find_conflicts(self, critical_info: Dict) -> List[Dict]:
        """중요 정보 간 충돌 찾기"""
        conflicts = []
        
        # 대규모내부거래 금액 충돌 확인
        if 'amounts' in critical_info:
            amount_values = set()
            for item in critical_info['amounts']:
                if '대규모내부거래' in item['context']:
                    amount_values.add(item['value'])
            
            if len(amount_values) > 1 and ('50억원' in amount_values or '30억원' in amount_values):
                conflicts.append({
                    'type': 'amount_conflict',
                    'values': list(amount_values),
                    'correct_value': '100억원'
                })
        
        # 공시 기한 명확성 확인 (충돌이 아닌 명확성 체크로 변경)
        if 'deadlines' in critical_info:
            unclear_deadlines = []
            for item in critical_info['deadlines']:
                if '공시' in item['context'] and '의결' in item['context']:
                    # 영업일 명시 여부 확인
                    if '영업일' not in item['context']:
                        unclear_deadlines.append(item['value'])
            
            if unclear_deadlines:
                conflicts.append({
                    'type': 'deadline_clarity',
                    'values': list(unclear_deadlines),
                    'correct_value': '영업일 7일',
                    'issue': '영업일 기준임을 명시해야 함'
                })
        
        return conflicts
    
    def _prioritize_latest_info(self, results: List[SearchResult], conflicts: List[Dict]) -> List[SearchResult]:
        """충돌이 있을 때 최신 정보를 우선시"""
        for conflict in conflicts:
            if conflict['type'] == 'amount_conflict':
                # 구버전 금액이 포함된 결과의 점수 감소
                for i, result in enumerate(results):
                    if any(old_val in result.content for old_val in ['50억원', '30억원']):
                        results[i].score *= 0.5  # 점수를 절반으로 감소
                        results[i].metadata['score_reduced'] = True
                        results[i].metadata['reduction_reason'] = 'outdated_amount'
            
            elif conflict['type'] == 'deadline_clarity':
                # 영업일이 명시되지 않은 결과의 점수 약간 감소
                for i, result in enumerate(results):
                    deadline_match = re.search(r'의결.*?(\d+)\s*일.*?공시', result.content)
                    if deadline_match:
                        context = result.content[max(0, deadline_match.start()-50):deadline_match.end()+50]
                        if '영업일' not in context:
                            results[i].score *= 0.85  # 점수를 15% 감소
                            results[i].metadata['score_reduced'] = True
                            results[i].metadata['reduction_reason'] = 'unclear_deadline_specification'
                            results[i].metadata['clarification_needed'] = '영업일 기준임을 명시 필요'
        
        return results

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
        self.complexity_assessor = ComplexityAssessor()
        self.analysis_cache = LRUCache(max_size=100, ttl=3600)
        
    async def analyze_intent(self, query: str) -> IntentAnalysis:
        """사용자 질문의 의도를 분석"""
        
        # 캐시 확인
        cache_key = hashlib.md5(query.encode()).hexdigest()
        cached = self.analysis_cache.get(cache_key)
        if cached:
            logger.info("Using cached intent analysis")
            return cached
        
        # 복잡도 평가
        complexity, confidence, complexity_analysis = self.complexity_assessor.assess(query)
        
        # 모델 선택 (의도 분석은 빠른 모델 사용)
        model = "gpt-4o-mini"
        
        prompt = f"""
당신은 공정거래위원회 법률 전문가입니다. 
사용자의 질문을 분석하여 정확한 의도를 파악하고, 어떤 매뉴얼을 검색해야 하는지 결정해야 합니다.

사용 가능한 매뉴얼:
1. 대규모내부거래 매뉴얼: 계열사 간 자금거래, 자산거래, 상품용역거래 관련 규정
2. 현황공시 매뉴얼: 기업집단 현황공시, 공시 의무사항
3. 비상장사 중요사항 매뉴얼: 비상장회사의 주식 양도, 합병, 분할 등

사용자 질문: {query}
질문 복잡도: {complexity.value} (신뢰도: {confidence:.1%})

다음 형식으로 분석 결과를 제공하세요:

{{
    "core_intent": "한 문장으로 요약한 질문의 핵심 의도",
    "query_type": "simple_lookup/complex_analysis/procedural 중 선택",
    "target_documents": ["검색할 매뉴얼 이름들"],
    "key_entities": ["질문에 포함된 핵심 개체들 (예: 계열사, 자금, 이사회 등)"],
    "search_keywords": ["매뉴얼 검색에 사용할 핵심 키워드 5-10개"],
    "requires_timeline": true/false,
    "requires_calculation": true/false,
    "complexity_reason": "복잡도 판단의 구체적 이유",
    "confidence": 0.0-1.0
}}

분석 시 고려사항:
- query_type 판단 기준:
  - simple_lookup: 단순 사실 확인 (기한, 금액 등)
  - complex_analysis: 여러 조건이 결합된 복잡한 상황
  - procedural: 절차나 프로세스에 대한 질문
- 질문에 여러 매뉴얼이 관련될 수 있으니 신중히 판단
- search_keywords는 실제 문서에서 찾을 수 있는 구체적인 용어로 선정
- 법률 용어와 일상 용어를 모두 포함 (예: '대여'와 '자금거래' 모두 포함)
"""
        
        try:
            response = await openai.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            analysis_data = json.loads(response.choices[0].message.content)
            
            intent = IntentAnalysis(
                core_intent=analysis_data.get("core_intent", ""),
                query_type=analysis_data.get("query_type", "simple_lookup"),
                target_documents=analysis_data.get("target_documents", []),
                key_entities=analysis_data.get("key_entities", []),
                search_keywords=analysis_data.get("search_keywords", []),
                requires_timeline=analysis_data.get("requires_timeline", False),
                requires_calculation=analysis_data.get("requires_calculation", False),
                complexity_reason=analysis_data.get("complexity_reason", ""),
                confidence=analysis_data.get("confidence", 0.8)
            )
            
            # 캐시 저장
            self.analysis_cache.put(cache_key, intent)
            
            return intent
            
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
                complexity_reason="자동 분석 실패",
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
        self._embedding_cache = {}
        
        # SimpleVectorSearch 초기화 (FAISS가 없을 때)
        if not self.use_faiss:
            self._initialize_simple_search()
        
        # 매뉴얼별 인덱스 구축
        self.manual_indices = self._build_manual_indices()
        
        # 문서 버전 관리 및 충돌 해결
        self.version_manager = DocumentVersionManager()
        self.conflict_resolver = ConflictResolver(self.version_manager)
        
    def _initialize_simple_search(self):
        """SimpleVectorSearch 초기화"""
        embeddings_file = "chunk_embeddings.npy"
        if os.path.exists(embeddings_file):
            try:
                embeddings = np.load(embeddings_file)
                self.simple_search = SimpleVectorSearch(embeddings)
                logger.info(f"Loaded {len(embeddings)} pre-computed embeddings")
            except Exception as e:
                logger.warning(f"Failed to load embeddings: {e}")
                self._create_simple_search()
        else:
            self._create_simple_search()
    
    def _create_simple_search(self):
        """간단한 검색 시스템 생성"""
        logger.info("Creating embeddings for simple search...")
        
        total_chunks = self.chunk_loader.get_total_chunks()
        max_chunks = min(total_chunks, 1000)  # 데모를 위해 제한
        
        embeddings = []
        
        if self.embedding_model is not None:
            # sentence-transformers 사용 가능
            for i in range(max_chunks):
                try:
                    chunk = self.chunk_loader.get_chunk(i)
                    text = chunk['content'][:500]
                    embedding = self.embedding_model.encode([text])
                    embeddings.append(embedding[0])
                except Exception as e:
                    logger.warning(f"Failed to create embedding for chunk {i}: {e}")
                    embeddings.append(np.random.randn(384))
        else:
            # OpenAI embeddings 사용
            logger.info("Using OpenAI embeddings...")
            for i in range(max_chunks):
                try:
                    chunk = self.chunk_loader.get_chunk(i)
                    text = chunk['content'][:500]
                    embedding = self.api_manager.get_embedding(text)
                    embeddings.append(embedding)
                except Exception as e:
                    logger.warning(f"Failed to create OpenAI embedding for chunk {i}: {e}")
                    embeddings.append(np.random.randn(1536))
        
        embeddings_array = np.array(embeddings, dtype=np.float32)
        
        # 임베딩 저장
        try:
            np.save("chunk_embeddings.npy", embeddings_array)
            logger.info("Saved embeddings for future use")
        except Exception as e:
            logger.warning(f"Failed to save embeddings: {e}")
        
        self.simple_search = SimpleVectorSearch(embeddings_array)
    
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
                else:
                    indices['기타'].append(idx)
                    
                # 문서 날짜 추출 및 저장
                doc_date = self.version_manager.extract_document_date(chunk)
                if doc_date:
                    metadata = json.loads(chunk.get('metadata', '{}'))
                    metadata['document_date'] = doc_date
                    chunk['metadata'] = json.dumps(metadata)
                    
            except Exception as e:
                logger.warning(f"Failed to process chunk {idx}: {e}")
                continue
        
        return dict(indices)
    
    async def search_documents(self, intent: IntentAnalysis, top_k: int = 10) -> Tuple[List[SearchResult], bool]:
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
        
        # 충돌 해결 및 최신 정보 우선시
        results, has_outdated = self.conflict_resolver.resolve_conflicts(results, search_query)
        
        return results[:top_k], has_outdated
    
    async def _vector_search(self, query: str, indices: List[int], k: int) -> List[SearchResult]:
        """벡터 기반 검색"""
        # 쿼리 임베딩 생성
        query_vector = self._get_query_embedding(query)
        query_vector = query_vector.reshape(1, -1).astype(np.float32)
        
        # 검색 수행
        if self.use_faiss:
            scores, search_indices = self.index.search(query_vector, k)
        else:
            scores, search_indices = self.simple_search.search(query_vector, k)
        
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
    
    def _get_query_embedding(self, query: str) -> np.ndarray:
        """쿼리 임베딩 생성 (캐시 활용)"""
        cache_key = hashlib.md5(query.encode()).hexdigest()
        
        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]
        
        if self.embedding_model is not None:
            embedding = self.embedding_model.encode([query])[0]
        else:
            embedding = self.api_manager.get_embedding(query)
        
        self._embedding_cache[cache_key] = embedding
        
        # 캐시 크기 제한
        if len(self._embedding_cache) > 1000:
            oldest_key = next(iter(self._embedding_cache))
            del self._embedding_cache[oldest_key]
        
        return embedding
    
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
            
            # 질문 유형에 따른 가중치
            if intent.query_type == "procedural" and re.search(r'절차|단계|과정', result.content):
                result.score *= 1.15
        
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
                            search_results: List[SearchResult],
                            has_outdated_info: bool = False) -> str:
        """검색 결과를 바탕으로 답변 생성"""
        
        # 모델 선택
        model = self._select_model(intent)
        
        # 컨텍스트 구성
        context = self._build_context(search_results)
        
        # 프롬프트 구성
        if model == "o4-mini":
            prompt = self._build_reasoning_prompt(query, intent, context, has_outdated_info)
        else:
            prompt = self._build_standard_prompt(query, intent, context, has_outdated_info)
        
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
            answer = self._postprocess_answer(answer, intent, has_outdated_info)
            
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
            # 구버전 정보 경고 포함
            warnings = result.metadata.get('warnings', [])
            warning_text = ""
            if warnings:
                warning_text = "\n⚠️ 주의: 이 문서에는 개정 전 정보가 포함되어 있을 수 있습니다."
            
            context_parts.append(f"""
[참고자료 {i+1}]
출처: {result.source} (페이지 {result.page}){warning_text}
내용: {result.content}
""")
        
        return "\n---\n".join(context_parts)
    
    def _build_reasoning_prompt(self, query: str, intent: IntentAnalysis, context: str, has_outdated_info: bool) -> str:
        """o4-mini용 추론 중심 프롬프트"""
        
        outdated_warning = ""
        if has_outdated_info:
            outdated_warning = """
[중요 사항]
일부 참고자료에 개정 전 정보가 포함되어 있을 수 있습니다.
- 대규모내부거래 금액 기준: 현재 100억원 (2023년 1월 1일부터)
- 공시 기한: 영업일 기준 7일
상충하는 정보가 있다면 최신 정보를 우선시하세요.
"""
        
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
{outdated_warning}

[답변 작성 지침]
1. 먼저 질문의 각 요소를 개별적으로 분석하세요
2. 관련 규정과 조건을 단계별로 확인하세요
3. 필요한 경우 계산 과정을 명시하세요
4. 최신 규정을 기준으로 정확한 정보를 제공하세요
5. 실무에 바로 적용할 수 있는 구체적인 지침을 포함하세요

답변 형식:
## 핵심 답변
(1-2문장으로 직접적인 답변)

## 상세 분석
### 1. 적용 규정
- 관련 조항과 기준

### 2. 구체적 검토
- 단계별 분석 내용
- 필요시 계산 과정

### 3. 주의사항
- 예외사항이나 특별히 유의할 점

## 결론
(최종 정리 및 실무 지침)
"""
    
    def _build_standard_prompt(self, query: str, intent: IntentAnalysis, context: str, has_outdated_info: bool) -> str:
        """표준 모델용 프롬프트"""
        
        outdated_warning = ""
        if has_outdated_info:
            outdated_warning = """
⚠️ 참고: 일부 자료에 개정 전 정보가 있을 수 있습니다. 
최신 규정 - 대규모내부거래: 100억원, 공시기한: 영업일 7일
"""
        
        return f"""
당신은 한국 공정거래위원회의 법률 전문가입니다.
다음 참고자료를 바탕으로 사용자의 질문에 정확하고 실무적인 답변을 제공하세요.

[사용자 질문]
{query}

[참고자료]
{context}
{outdated_warning}

[답변 지침]
- 핵심 내용을 먼저 제시하고 상세 설명을 추가하세요
- 근거 조항을 명확히 인용하세요
- 실무에 바로 적용할 수 있는 구체적인 답변을 제공하세요
- 관련 주의사항이 있다면 반드시 언급하세요
- 최신 규정을 기준으로 답변하세요

답변:
"""
    
    def _postprocess_answer(self, answer: str, intent: IntentAnalysis, has_outdated_info: bool) -> str:
        """답변 후처리 - 최신 정보 확인 등"""
        
        # 최신 규정 정보 추가 (필요한 경우만)
        if has_outdated_info and "대규모내부거래" in ' '.join(intent.target_documents):
            if "50억" in answer or "30억" in answer:
                answer += "\n\n⚠️ **중요**: 대규모내부거래 기준 금액은 2023년 1월 1일부터 100억원으로 변경되었습니다."
        
        if has_outdated_info and "공시" in answer and "7일" in answer:
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
            "search_results_count": 0,
            "has_outdated_info": False
        }
        
        try:
            # Step 1: 의도 분석
            start_time = time.time()
            intent = await self.step1_analyzer.analyze_intent(query)
            stats["process_times"]["intent_analysis"] = time.time() - start_time
            stats["intent_analysis"] = intent
            
            # Step 2: 문서 검색
            start_time = time.time()
            search_results, has_outdated = await self.step2_searcher.search_documents(intent)
            stats["process_times"]["document_search"] = time.time() - start_time
            stats["search_results_count"] = len(search_results)
            stats["has_outdated_info"] = has_outdated
            stats["search_results"] = search_results  # 상세 정보 표시를 위해 저장
            
            # Step 3: 답변 생성
            start_time = time.time()
            answer = await self.step3_generator.generate_answer(query, intent, search_results, has_outdated)
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
    
    /* 경고 메시지 */
    .outdated-warning {
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
        border-radius: 4px;
        padding: 0.75rem 1.25rem;
        margin: 0.5rem 0;
        color: #856404;
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
        with st.expander("🔍 상세 오류 정보"):
            st.code(traceback.format_exc())
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
                
                # 구버전 정보 경고
                if message.get("stats", {}).get("has_outdated_info"):
                    st.markdown("""
                    <div class="outdated-warning">
                    ⚠️ 일부 참고 자료에 개정 전 정보가 포함되어 있을 수 있습니다.
                    </div>
                    """, unsafe_allow_html=True)
                
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
                    
                    # 검색된 문서 정보
                    if "search_results" in stats and stats["search_results"]:
                        with st.expander("📚 참고한 문서"):
                            for i, result in enumerate(stats["search_results"][:3]):
                                st.caption(f"**{i+1}. {result.source}** (페이지 {result.page}, 관련도: {result.score:.2f})")
                                if result.metadata.get('has_outdated_info'):
                                    st.warning("⚠️ 이 문서에는 개정 전 정보가 포함되어 있을 수 있습니다.")
    
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
            
            # 구버전 정보 경고
            if stats.get("has_outdated_info"):
                st.markdown("""
                <div class="outdated-warning">
                ⚠️ 일부 참고 자료에 개정 전 정보가 포함되어 있을 수 있습니다.
                </div>
                """, unsafe_allow_html=True)
            
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
        
        st.divider()
        
        st.caption("⚠️ **최신 규정 안내**")
        st.caption("• 대규모내부거래 기준: 100억원")
        st.caption("• 공시 기한: 영업일 7일")
    
    # 새 질문 처리
    if "new_question" in st.session_state:
        st.session_state.messages.append({"role": "user", "content": st.session_state.new_question})
        del st.session_state.new_question
        st.rerun()

if __name__ == "__main__":
    main()
