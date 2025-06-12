# 파일 이름: app_improved.py (공정거래위원회 AI 법률 보조원 - 하이브리드 개선 버전)

import streamlit as st
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
import json
import openai
import numpy as np
from typing import List, Dict, Tuple, Optional, Set, TypedDict, Protocol, Iterator, Generator, OrderedDict, Any, Union
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
from cryptography.fernet import Fernet
import ijson  # 스트리밍 JSON 파서를 위해 필요

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

# ===== 보안 강화된 API 관리자 =====
class SecureAPIManager:
    """API 키와 호출을 안전하게 관리하는 클래스
    
    이 클래스는 API 키의 안전한 저장과 로드, 그리고 API 호출의 
    속도 제한 및 비용 추적을 담당합니다. 프로덕션 환경에서의 
    보안과 비용 관리를 위해 필수적입니다.
    """
    
    def __init__(self):
        self._api_key = None
        self._last_call_time = {}
        self._call_counts = {}
        self._cost_tracker = defaultdict(float)
        
        # 각 모델의 속도 제한 설정
        self._rate_limits = {
            'gpt-4o': {'calls_per_minute': 60, 'tokens_per_minute': 150000},
            'gpt-4o-mini': {'calls_per_minute': 500, 'tokens_per_minute': 200000},
            'o4-mini': {'calls_per_minute': 30, 'tokens_per_minute': 100000}  # 추론 모델은 더 보수적으로
        }
        
        # 모델별 비용 (1K 토큰당)
        self._model_costs = {
            'gpt-4o': {'input': 0.0025, 'output': 0.01},
            'gpt-4o-mini': {'input': 0.00015, 'output': 0.0006},
            'o4-mini': {'input': 0.001, 'output': 0.004}  # 추론 모델의 예상 비용
        }
    
    def load_api_key(self) -> str:
        """API 키를 안전하게 로드
        
        우선순위:
        1. Streamlit secrets (개발 환경)
        2. 환경 변수 (프로덕션 환경)
        3. 암호화된 파일 (고급 보안 환경)
        """
        # Streamlit secrets 확인
        try:
            if 'st' in globals() and hasattr(st, 'secrets') and 'OPENAI_API_KEY' in st.secrets:
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
        
        # 암호화된 파일 확인 (프로덕션용)
        encrypted_key_path = Path('.api_key.enc')
        if encrypted_key_path.exists():
            try:
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
        
        raise APIKeyError("API 키를 찾을 수 없습니다. Streamlit secrets, 환경 변수, 또는 암호화된 파일을 확인하세요.")
    
    def rate_limit(self, model: str = 'gpt-4o'):
        """API 호출 속도 제한 데코레이터
        
        이 데코레이터는 OpenAI API의 속도 제한을 준수하도록 보장합니다.
        너무 빠른 연속 호출을 방지하여 API 오류를 예방합니다.
        """
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
                
                # 호출 횟수 추적
                self._track_usage(model, args, kwargs, result)
                
                return result
            return wrapper
        return decorator
    
    def _track_usage(self, model: str, args, kwargs, result):
        """API 사용량 및 비용 추적"""
        try:
            # 토큰 수 추정 (실제로는 tiktoken 라이브러리 사용 권장)
            if hasattr(result, 'usage'):
                input_tokens = result.usage.prompt_tokens
                output_tokens = result.usage.completion_tokens
                
                # 비용 계산
                input_cost = (input_tokens / 1000) * self._model_costs[model]['input']
                output_cost = (output_tokens / 1000) * self._model_costs[model]['output']
                total_cost = input_cost + output_cost
                
                # 비용 추적
                self._cost_tracker[model] += total_cost
                self._cost_tracker['total'] += total_cost
                
                logger.debug(f"API call to {model}: {input_tokens} input + {output_tokens} output tokens = ${total_cost:.4f}")
        except Exception as e:
            logger.warning(f"Failed to track usage: {e}")
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """현재 사용 통계 반환"""
        return {
            'costs': dict(self._cost_tracker),
            'last_calls': dict(self._last_call_time),
            'estimated_monthly_cost': self._cost_tracker['total'] * 30  # 대략적인 월 비용 추정
        }

# ===== 메모리 효율적인 청크 로더 =====
class ChunkLoader:
    """메모리 효율적인 청크 로딩 시스템
    
    이 클래스는 대용량 JSON 파일을 전체 메모리에 로드하지 않고
    필요한 부분만 스트리밍 방식으로 읽을 수 있게 합니다.
    수 GB의 문서도 효율적으로 처리할 수 있습니다.
    """
    
    def __init__(self, filepath: str):
        self.filepath = filepath
        self._index = None
        self._chunk_cache = OrderedDict()  # LRU 캐시
        self._cache_size = 1000  # 캐시할 청크 수
        self._build_index()
    
    def _build_index(self):
        """청크의 파일 내 위치를 인덱싱
        
        이 메서드는 파일을 한 번 스캔하여 각 청크의 시작 위치를 기록합니다.
        이후 특정 청크를 읽을 때 해당 위치로 직접 이동할 수 있습니다.
        """
        logger.info(f"Building index for {self.filepath}")
        self._index = {}
        
        with open(self.filepath, 'rb') as f:
            # 배열의 시작 '[' 찾기
            while True:
                char = f.read(1)
                if char == b'[':
                    break
            
            # 각 청크의 위치 기록
            idx = 0
            bracket_count = 0
            in_string = False
            escape_next = False
            chunk_start = f.tell()
            
            while True:
                char = f.read(1)
                if not char:
                    break
                
                if escape_next:
                    escape_next = False
                    continue
                
                if char == b'\\':
                    escape_next = True
                    continue
                
                if char == b'"' and not escape_next:
                    in_string = not in_string
                
                if not in_string:
                    if char == b'{':
                        if bracket_count == 0:
                            chunk_start = f.tell() - 1
                        bracket_count += 1
                    elif char == b'}':
                        bracket_count -= 1
                        if bracket_count == 0:
                            self._index[idx] = chunk_start
                            idx += 1
        
        logger.info(f"Index built: {len(self._index)} chunks found")
    
    def get_chunk(self, idx: int) -> Dict:
        """특정 인덱스의 청크를 가져옴
        
        캐시를 먼저 확인하고, 없으면 파일에서 읽습니다.
        LRU 캐시를 사용하여 자주 사용되는 청크는 메모리에 유지합니다.
        """
        # 캐시 확인
        if idx in self._chunk_cache:
            # LRU: 최근 사용 항목을 끝으로 이동
            self._chunk_cache.move_to_end(idx)
            return self._chunk_cache[idx]
        
        # 인덱스 확인
        if idx not in self._index:
            raise IndexError(f"Chunk index {idx} not found")
        
        # 파일에서 청크 읽기
        with open(self.filepath, 'rb') as f:
            f.seek(self._index[idx])
            
            # 청크 파싱
            chunk_str = ''
            bracket_count = 0
            in_string = False
            escape_next = False
            
            while True:
                char = f.read(1).decode('utf-8', errors='ignore')
                if not char:
                    break
                
                chunk_str += char
                
                if escape_next:
                    escape_next = False
                    continue
                
                if char == '\\':
                    escape_next = True
                    continue
                
                if char == '"' and not escape_next:
                    in_string = not in_string
                
                if not in_string:
                    if char == '{':
                        bracket_count += 1
                    elif char == '}':
                        bracket_count -= 1
                        if bracket_count == 0:
                            break
            
            chunk = json.loads(chunk_str)
            
            # 캐시에 저장
            self._add_to_cache(idx, chunk)
            
            return chunk
    
    def _add_to_cache(self, idx: int, chunk: Dict):
        """캐시에 청크 추가 (LRU 정책)"""
        if len(self._chunk_cache) >= self._cache_size:
            # 가장 오래된 항목 제거
            self._chunk_cache.popitem(last=False)
        
        self._chunk_cache[idx] = chunk
    
    def iter_chunks(self, indices: List[int] = None) -> Iterator[Dict]:
        """필요한 청크들을 순차적으로 반환
        
        메모리 효율성을 위해 제너레이터를 사용합니다.
        """
        if indices is None:
            indices = range(len(self._index))
        
        for idx in indices:
            yield self.get_chunk(idx)
    
    def get_total_chunks(self) -> int:
        """전체 청크 수 반환"""
        return len(self._index)

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
    model_used: str  # 어떤 모델을 사용했는지 기록

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

class QueryComplexity(Enum):
    """질문 복잡도 레벨"""
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"

# ===== 하이브리드 GPT 전략 =====
class HybridGPTStrategy:
    """작업 복잡도에 따라 적절한 GPT 모델을 선택하는 전략
    
    이 클래스는 각 작업의 특성을 분석하여 최적의 모델을 선택합니다.
    비용과 성능의 균형을 맞추는 것이 목표입니다.
    """
    
    def __init__(self, api_manager: SecureAPIManager):
        self.api_manager = api_manager
        
        # 모델별 특성 정의
        self.model_characteristics = {
            ModelSelection.GPT4O: {
                "strength": "범용성, 긴 컨텍스트, 창의성",
                "weakness": "높은 비용, 상대적으로 느림",
                "best_for": ["긴 문서 요약", "복잡한 설명", "창의적 답변"]
            },
            ModelSelection.GPT4O_MINI: {
                "strength": "빠른 속도, 낮은 비용, 단순 작업에 충분",
                "weakness": "제한된 컨텍스트, 복잡한 추론 약함",
                "best_for": ["단순 사실 확인", "짧은 답변", "분류 작업"]
            },
            ModelSelection.O4_MINI: {
                "strength": "뛰어난 추론 능력, 논리적 분석",
                "weakness": "창의성 부족, 속도 보통",
                "best_for": ["복잡한 논리 분석", "다단계 추론", "조건부 판단"]
            }
        }
    
    def select_model_for_analysis(self, query: str, complexity: QueryComplexity) -> ModelSelection:
        """질문 분석을 위한 모델 선택
        
        질문의 복잡도와 특성을 고려하여 최적의 모델을 선택합니다.
        """
        query_lower = query.lower()
        
        # 추론이 많이 필요한 패턴 감지
        reasoning_patterns = [
            r'만[약일].*경우',
            r'동시에.*그리고',
            r'[AB].*면서.*[CD]',
            r'각각.*어떻게',
            r'종합적으로',
            r'여러.*고려'
        ]
        
        needs_reasoning = any(re.search(pattern, query_lower) for pattern in reasoning_patterns)
        
        # 단순 정보 요청 패턴
        simple_patterns = [
            r'^.*기한[은이]',
            r'^.*금액[은이]',
            r'^.*날짜[는가]',
            r'^언제'
        ]
        
        is_simple = any(re.search(pattern, query_lower) for pattern in simple_patterns)
        
        # 모델 선택 로직
        if needs_reasoning and complexity in [QueryComplexity.MEDIUM, QueryComplexity.COMPLEX]:
            logger.info(f"Selected O4_MINI for reasoning-intensive analysis")
            return ModelSelection.O4_MINI
        elif is_simple or complexity == QueryComplexity.SIMPLE:
            logger.info(f"Selected GPT4O_MINI for simple analysis")
            return ModelSelection.GPT4O_MINI
        else:
            logger.info(f"Selected GPT4O for general analysis")
            return ModelSelection.GPT4O
    
    def select_model_for_answer(self, 
                               query: str, 
                               search_results: List[SearchResult],
                               analysis: Dict) -> ModelSelection:
        """답변 생성을 위한 모델 선택
        
        질문의 특성과 검색 결과의 복잡도를 고려하여 선택합니다.
        """
        # 분석 결과에서 필요한 정보 추출
        requirements = analysis.get('answer_requirements', {})
        complexity = analysis.get('query_analysis', {}).get('actual_complexity', 'medium')
        
        # 간단한 사실 확인인 경우
        if (requirements.get('needs_specific_numbers') and 
            not requirements.get('needs_multiple_perspectives') and
            complexity == 'simple'):
            return ModelSelection.GPT4O_MINI
        
        # 복잡한 추론이 필요한 경우
        if (requirements.get('needs_multiple_perspectives') or
            requirements.get('needs_exceptions') or
            '종합' in query or '리스크' in query):
            return ModelSelection.O4_MINI
        
        # 일반적인 경우
        return ModelSelection.GPT4O
    
    def estimate_cost(self, model: ModelSelection, estimated_tokens: int) -> float:
        """예상 비용 계산"""
        costs = self.api_manager._model_costs[model.value]
        # 입력과 출력을 대략 7:3으로 가정
        input_cost = (estimated_tokens * 0.7 / 1000) * costs['input']
        output_cost = (estimated_tokens * 0.3 / 1000) * costs['output']
        return input_cost + output_cost

# ===== 향상된 질문 분석기 =====
class EnhancedQueryAnalyzer:
    """하이브리드 모델 전략을 사용하는 질문 분석기
    
    이 클래스는 o4-mini의 추론 능력과 GPT-4o의 범용성을 
    적절히 활용하여 질문을 분석합니다.
    """
    
    def __init__(self, api_manager: SecureAPIManager, hybrid_strategy: HybridGPTStrategy):
        self.api_manager = api_manager
        self.hybrid_strategy = hybrid_strategy
        self.analysis_cache = LRUCache(max_size=100, ttl=3600)
    
    @api_manager.rate_limit('gpt-4o')  # 기본값, 실제로는 동적으로 변경됨
    async def analyze_query(self, query: str, available_chunks_info: Dict) -> AnalysisResult:
        """질문을 분석하고 최적의 검색 전략 수립
        
        먼저 질문의 복잡도를 평가한 후, 적절한 모델을 선택하여
        상세한 분석을 수행합니다.
        """
        # 캐시 확인
        cache_key = hashlib.md5(f"{query}_{json.dumps(available_chunks_info)}".encode()).hexdigest()
        cached = self.analysis_cache.get(cache_key)
        if cached:
            logger.info("Using cached analysis")
            return cached
        
        # 1단계: 복잡도 평가
        complexity_assessor = ComplexityAssessor()
        complexity, confidence, _ = complexity_assessor.assess(query)
        
        # 2단계: 모델 선택
        selected_model = self.hybrid_strategy.select_model_for_analysis(query, complexity)
        
        # 3단계: 선택된 모델로 분석 수행
        if selected_model == ModelSelection.O4_MINI:
            analysis = await self._analyze_with_reasoning_model(query, available_chunks_info, complexity)
        else:
            analysis = await self._analyze_with_standard_model(query, available_chunks_info, complexity, selected_model)
        
        # 4단계: 사용된 모델 기록
        analysis['model_used'] = selected_model.value
        analysis['model_selection_reason'] = f"Complexity: {complexity.value}, Confidence: {confidence:.2f}"
        
        # 캐시 저장
        self.analysis_cache.put(cache_key, analysis)
        
        return analysis
    
    async def _analyze_with_reasoning_model(self, query: str, chunks_info: Dict, complexity: QueryComplexity) -> AnalysisResult:
        """o4-mini를 사용한 추론 중심 분석
        
        추론 모델의 강점을 활용하여 복잡한 질문을 체계적으로 분해합니다.
        """
        prompt = f"""
        법률 전문가로서 다음 질문을 체계적으로 분석해주세요.
        
        질문: {query}
        
        사용 가능한 문서:
        - 대규모내부거래 매뉴얼: {chunks_info.get('대규모내부거래', 0)}개 섹션
        - 현황공시 매뉴얼: {chunks_info.get('현황공시', 0)}개 섹션
        - 비상장사 중요사항 매뉴얼: {chunks_info.get('비상장사 중요사항', 0)}개 섹션
        
        분석 과정:
        
        1. 질문 구조 분해
           - 핵심 주체들 식별 (회사, 계열사 등)
           - 거래 유형 파악
           - 시간적 요소 확인
        
        2. 법적 쟁점 도출
           - 각 거래/행위별 적용 법규
           - 필수 확인 사항
           - 잠재적 리스크
        
        3. 검색 전략 수립
           - 우선 검색할 매뉴얼과 이유
           - 핵심 키워드 도출 과정
           - 필요한 정보의 깊이
        
        각 단계에서 "왜" 그런 판단을 했는지 명확히 설명하고,
        다음 JSON 형식으로 응답하세요:
        
        {{
            "query_analysis": {{
                "core_intent": "한 문장으로 요약한 질문의 핵심",
                "actual_complexity": "{complexity.value}",
                "complexity_reason": "복잡도 판단의 구체적 근거",
                "reasoning_chain": ["추론 단계 1", "추론 단계 2", ...]
            }},
            "legal_concepts": [
                {{
                    "concept": "관련 법률 개념",
                    "relevance": "primary/secondary",
                    "specific_aspects": ["구체적 검토 사항들"],
                    "why_relevant": "이 개념이 중요한 이유"
                }}
            ],
            "search_strategy": {{
                "approach": "direct_lookup/focused_search/comprehensive_analysis",
                "primary_manual": "주 검색 대상 매뉴얼",
                "search_keywords": ["도출된 키워드들"],
                "keyword_derivation": "키워드 도출 과정 설명",
                "expected_chunks_needed": 숫자,
                "rationale": "이 전략의 논리적 근거"
            }},
            "answer_requirements": {{
                "needs_specific_numbers": true/false,
                "needs_process_steps": true/false,
                "needs_timeline": true/false,
                "needs_exceptions": true/false,
                "needs_multiple_perspectives": true/false,
                "critical_points": ["답변에 꼭 포함되어야 할 요소들"]
            }}
        }}
        """
        
        try:
            response = await openai.chat.completions.create(
                model="o4-mini",  # 추론 특화 모델 사용
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,  # 추론 모델은 낮은 temperature가 효과적
                max_tokens=2000,
                response_format={"type": "json_object"}
            )
            
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            logger.error(f"o4-mini analysis failed: {e}")
            # 폴백으로 GPT-4o 사용
            return await self._analyze_with_standard_model(query, chunks_info, complexity, ModelSelection.GPT4O)
    
    async def _analyze_with_standard_model(self, query: str, chunks_info: Dict, 
                                         complexity: QueryComplexity, model: ModelSelection) -> AnalysisResult:
        """표준 GPT 모델을 사용한 분석"""
        prompt = f"""
        공정거래법 전문가로서 다음 질문을 분석하고 검색 전략을 수립하세요.
        
        질문: {query}
        복잡도: {complexity.value}
        
        사용 가능한 문서:
        {json.dumps(chunks_info, ensure_ascii=False)}
        
        다음 JSON 형식으로 간결하게 응답하세요:
        {{
            "query_analysis": {{
                "core_intent": "질문의 핵심 의도",
                "actual_complexity": "{complexity.value}",
                "complexity_reason": "복잡도 판단 이유"
            }},
            "legal_concepts": [
                {{
                    "concept": "대규모내부거래/현황공시/비상장사 중요사항",
                    "relevance": "primary/secondary",
                    "specific_aspects": ["관련 측면들"]
                }}
            ],
            "search_strategy": {{
                "approach": "direct_lookup/focused_search/comprehensive_analysis",
                "primary_manual": "주 검색 매뉴얼",
                "search_keywords": ["키워드1", "키워드2"],
                "expected_chunks_needed": 10,
                "rationale": "전략 선택 이유"
            }},
            "answer_requirements": {{
                "needs_specific_numbers": true/false,
                "needs_process_steps": true/false,
                "needs_timeline": true/false,
                "needs_exceptions": true/false,
                "needs_multiple_perspectives": true/false
            }}
        }}
        """
        
        response = await openai.chat.completions.create(
            model=model.value,
            messages=[{"role": "user", "content": prompt}],
            temperature=0 if model == ModelSelection.GPT4O_MINI else 0.3,
            max_tokens=1000,
            response_format={"type": "json_object"}
        )
        
        return json.loads(response.choices[0].message.content)

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

# ===== 검색 전략 인터페이스 =====
class SearchStrategy(Protocol):
    """검색 전략의 인터페이스"""
    def prepare_indices(self, manual: str, limit: int) -> List[int]:
        """검색할 인덱스 준비"""
        ...
    
    def enhance_query(self, query: str, keywords: List[str]) -> str:
        """쿼리 향상"""
        ...
    
    def filter_results(self, results: List[SearchResult], requirements: Dict) -> List[SearchResult]:
        """결과 필터링"""
        ...

# ===== 기본 검색 실행기 =====
class BaseSearchExecutor:
    """공통 검색 로직을 담당하는 기본 클래스
    
    이 클래스는 모든 검색 전략에서 공통으로 사용되는 로직을 구현합니다.
    코드 중복을 줄이고 일관성을 보장합니다.
    """
    
    def __init__(self, index: faiss.Index, chunk_loader: ChunkLoader, embedding_model):
        self.index = index
        self.chunk_loader = chunk_loader
        self.embedding_model = embedding_model
        self._embedding_cache = {}
    
    async def execute_search(self,
                           query: str,
                           indices: List[int],
                           top_k: int,
                           strategy: SearchStrategy) -> Tuple[List[SearchResult], Dict]:
        """공통 검색 실행 로직"""
        start_time = time.time()
        stats = {
            'search_method': strategy.__class__.__name__,
            'errors': []
        }
        
        try:
            # 1. 인덱스 검증
            if not indices:
                raise ValueError("No indices provided for search")
            
            # 2. 쿼리 임베딩 생성 (캐시 활용)
            query_vector = self._get_cached_embedding(query)
            
            # 3. FAISS 검색 실행
            k_search = min(len(indices), max(1, top_k * 3))
            scores, search_indices = self._safe_faiss_search(query_vector, k_search)
            
            # 4. 결과 변환
            results = await self._convert_to_search_results(
                scores[0], search_indices[0], set(indices)
            )
            
            # 5. 전략별 후처리
            if hasattr(strategy, 'filter_results'):
                results = strategy.filter_results(results, {})
            
            # 6. 정렬 및 제한
            results.sort(key=lambda x: x.score, reverse=True)
            results = results[:top_k]
            
            stats.update({
                'search_time': time.time() - start_time,
                'searched_chunks': len(indices),
                'results_count': len(results),
                'cache_hit': query in self._embedding_cache
            })
            
            return results, stats
            
        except Exception as e:
            logger.error(f"Search execution error: {str(e)}")
            stats['errors'].append({
                'type': type(e).__name__,
                'message': str(e)
            })
            stats['search_time'] = time.time() - start_time
            return [], stats
    
    def _get_cached_embedding(self, query: str) -> np.ndarray:
        """캐시된 임베딩 또는 새로 생성"""
        if query not in self._embedding_cache:
            embedding = self.embedding_model.encode([query])
            self._embedding_cache[query] = np.array(embedding, dtype=np.float32)
            
            # 캐시 크기 제한
            if len(self._embedding_cache) > 1000:
                # 가장 오래된 항목 제거
                oldest_key = next(iter(self._embedding_cache))
                del self._embedding_cache[oldest_key]
        
        return self._embedding_cache[query]
    
    def _safe_faiss_search(self, query_vector: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """안전한 FAISS 검색"""
        try:
            return self.index.search(query_vector, k)
        except Exception as e:
            logger.error(f"FAISS search failed: {e}")
            # 빈 결과 반환
            return np.array([[0.0]]), np.array([[-1]])
    
    async def _convert_to_search_results(self, 
                                       scores: np.ndarray, 
                                       indices: np.ndarray,
                                       valid_indices: set) -> List[SearchResult]:
        """인덱스를 SearchResult 객체로 변환
        
        ChunkLoader를 사용하여 필요한 청크만 메모리에 로드합니다.
        """
        results = []
        
        for idx, score in zip(indices, scores):
            if idx < 0 or idx >= self.chunk_loader.get_total_chunks():
                continue
            
            if idx not in valid_indices:
                continue
            
            try:
                # ChunkLoader를 통해 청크 가져오기
                chunk = self.chunk_loader.get_chunk(idx)
                result = SearchResult(
                    chunk_id=chunk.get('chunk_id', str(idx)),
                    content=chunk.get('content', ''),
                    score=float(score),
                    source=chunk.get('source', 'Unknown'),
                    page=chunk.get('page', 0),
                    chunk_type=chunk.get('chunk_type', 'unknown'),
                    metadata=json.loads(chunk.get('metadata', '{}'))
                )
                results.append(result)
            except Exception as e:
                logger.warning(f"Failed to create SearchResult for index {idx}: {e}")
                continue
        
        return results

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
        for result in results:
            doc_date = result.document_date
            warnings = self.version_manager.check_for_outdated_info(result.content, doc_date)
            
            if warnings:
                result.metadata['warnings'] = warnings
                result.metadata['has_outdated_info'] = True
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
        
        return results
    
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
        
        # 공시 기한 충돌 확인
        if 'deadlines' in critical_info:
            deadline_values = set()
            for item in critical_info['deadlines']:
                if '공시' in item['context'] and '의결' in item['context']:
                    deadline_values.add(item['value'])
            
            if len(deadline_values) > 1 and '7일' in deadline_values:
                conflicts.append({
                    'type': 'deadline_conflict',
                    'values': list(deadline_values),
                    'correct_value': '5일'
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
            
            elif conflict['type'] == 'deadline_conflict':
                # 구버전 기한이 포함된 결과의 점수 감소
                for i, result in enumerate(results):
                    if '7일' in result.content and '공시' in result.content:
                        results[i].score *= 0.7
                        results[i].metadata['score_reduced'] = True
                        results[i].metadata['reduction_reason'] = 'outdated_deadline'
        
        return results

# ===== 검색 전략 구현 =====
class DirectSearchStrategy(SearchStrategy):
    """직접 검색 전략 - 단순한 질문에 적합"""
    def __init__(self, manual_indices: Dict[str, List[int]]):
        self.manual_indices = manual_indices
    
    def prepare_indices(self, manual: str, limit: int = 100) -> List[int]:
        return self.manual_indices.get(manual, [])[:limit]
    
    def enhance_query(self, query: str, keywords: List[str]) -> str:
        return f"{query} {' '.join(keywords)}"

class FocusedSearchStrategy(SearchStrategy):
    """집중 검색 전략 - 특정 주제에 대한 깊이 있는 검색"""
    def __init__(self, manual_indices: Dict[str, List[int]], chunk_loader: ChunkLoader):
        self.manual_indices = manual_indices
        self.chunk_loader = chunk_loader
    
    def prepare_indices(self, manual: str, limit: int = 200) -> List[int]:
        return self.manual_indices.get(manual, [])[:limit]
    
    def enhance_query(self, query: str, keywords: List[str]) -> str:
        return f"{query} {' '.join(keywords)}"
    
    def filter_results(self, results: List[SearchResult], requirements: Dict) -> List[SearchResult]:
        """요구사항에 따른 결과 필터링"""
        if requirements.get('needs_specific_numbers'):
            # 숫자가 포함된 결과 우선
            number_results = []
            other_results = []
            
            for result in results:
                if re.search(r'\d+억|\d+%', result.content):
                    result.score *= 1.2  # 가중치 부여
                    number_results.append(result)
                else:
                    other_results.append(result)
            
            return number_results + other_results
        
        return results

# ===== 견고한 검색 파이프라인 =====
class RobustSearchPipeline:
    """에러 처리가 강화된 검색 파이프라인
    
    이 클래스는 검색 과정에서 발생할 수 있는 다양한 에러를
    우아하게 처리하고, 사용자에게 안정적인 서비스를 제공합니다.
    """
    
    def __init__(self, base_executor: BaseSearchExecutor):
        self.base_executor = base_executor
        self.error_history = []
        self.max_retries = 3
        self.timeout_seconds = 30
    
    async def search_with_retry(self, 
                               query: str,
                               indices: List[int],
                               top_k: int,
                               strategy: SearchStrategy) -> Tuple[List[SearchResult], Dict]:
        """재시도 로직을 포함한 안전한 검색"""
        for attempt in range(self.max_retries):
            try:
                # 타임아웃 설정
                results, stats = await asyncio.wait_for(
                    self.base_executor.execute_search(query, indices, top_k, strategy),
                    timeout=self.timeout_seconds
                )
                
                # 결과 검증
                if self._validate_results(results, stats):
                    return results, stats
                else:
                    raise ValueError("Invalid search results")
                    
            except asyncio.TimeoutError:
                error = SearchError(
                    error_type="timeout",
                    message=f"Search timed out after {self.timeout_seconds}s",
                    timestamp=time.time(),
                    context={"query": query, "attempt": attempt + 1},
                    severity="warning"
                )
                self._log_error(error)
                
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # 지수 백오프
                    continue
                    
            except Exception as e:
                error = SearchError(
                    error_type=type(e).__name__,
                    message=str(e),
                    timestamp=time.time(),
                    context={
                        "query": query, 
                        "attempt": attempt + 1,
                        "traceback": traceback.format_exc()
                    },
                    severity="critical" if attempt == self.max_retries - 1 else "warning"
                )
                self._log_error(error)
                
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(1)
                    continue
        
        # 모든 시도 실패 시 안전한 기본값 반환
        return self._get_fallback_results(query), self._get_error_stats()
    
    def _validate_results(self, results: List[SearchResult], stats: Dict) -> bool:
        """검색 결과의 유효성 검증"""
        if not isinstance(results, list):
            return False
        
        if not stats or not isinstance(stats, dict):
            return False
        
        # 결과가 너무 적거나 많은 경우
        if len(results) == 0 and stats.get('searched_chunks', 0) > 0:
            logger.warning("No results despite searching chunks")
            return False
        
        # 각 결과의 유효성 검증
        for result in results:
            if not isinstance(result, SearchResult):
                return False
            if not result.content or len(result.content) < 10:
                return False
            if result.score < 0 or result.score > 1000:
                return False
        
        return True
    
    def _log_error(self, error: SearchError):
        """에러 로깅 및 기록"""
        self.error_history.append(error)
        
        # 최근 100개 에러만 유지
        if len(self.error_history) > 100:
            self.error_history = self.error_history[-100:]
        
        # 로깅
        if error.severity == "critical":
            logger.error(f"{error.error_type}: {error.message}")
        else:
            logger.warning(f"{error.error_type}: {error.message}")
    
    def _get_fallback_results(self, query: str) -> List[SearchResult]:
        """에러 시 반환할 기본 결과"""
        return [
            SearchResult(
                chunk_id="error_fallback",
                content="죄송합니다. 검색 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.",
                score=0.0,
                source="System",
                page=0,
                chunk_type="error",
                metadata={"error": True, "query": query}
            )
        ]
    
    def _get_error_stats(self) -> Dict:
        """에러 발생 시 통계 정보"""
        recent_errors = [e for e in self.error_history[-10:]]
        return {
            "error": True,
            "error_count": len(self.error_history),
            "recent_errors": [
                {"type": e.error_type, "time": e.timestamp} 
                for e in recent_errors
            ],
            "search_time": 0,
            "searched_chunks": 0
        }

# ===== 개선된 하이브리드 RAG 파이프라인 =====
class ImprovedHybridRAGPipeline:
    """모든 개선사항이 반영된 하이브리드 RAG 파이프라인
    
    이 클래스는 다음의 개선사항을 포함합니다:
    1. 하이브리드 모델 전략 (GPT-4o, GPT-4o-mini, o4-mini)
    2. 메모리 효율적인 청크 로딩
    3. 견고한 에러 처리
    4. 코드 중복 제거
    5. 문서 버전 관리 및 충돌 해결
    """
    
    def __init__(self, embedding_model, reranker_model, index, chunk_loader: ChunkLoader,
                 api_manager: SecureAPIManager):
        self.embedding_model = embedding_model
        self.reranker_model = reranker_model
        self.index = index
        self.chunk_loader = chunk_loader
        self.api_manager = api_manager
        
        # 하이브리드 전략 초기화
        self.hybrid_strategy = HybridGPTStrategy(api_manager)
        self.query_analyzer = EnhancedQueryAnalyzer(api_manager, self.hybrid_strategy)
        
        # 문서 버전 관리
        self.version_manager = DocumentVersionManager()
        self.conflict_resolver = ConflictResolver(self.version_manager)
        
        # 매뉴얼 인덱스 구축
        self.manual_indices = self._build_manual_indices()
        self.chunks_info = {
            category: len(indices) 
            for category, indices in self.manual_indices.items()
        }
        
        # 검색 실행기 및 전략
        self.search_executor = BaseSearchExecutor(index, chunk_loader, embedding_model)
        self.robust_search = RobustSearchPipeline(self.search_executor)
        
        self.search_strategies = {
            'direct': DirectSearchStrategy(self.manual_indices),
            'focused': FocusedSearchStrategy(self.manual_indices, chunk_loader)
        }
        
        # 캐시
        self.search_cache = LRUCache(max_size=50, ttl=1800)
        
        logger.info(f"ImprovedHybridRAGPipeline initialized with {chunk_loader.get_total_chunks()} chunks")
    
    def _build_manual_indices(self) -> Dict[str, List[int]]:
        """각 매뉴얼별로 청크 인덱스를 구축
        
        메모리 효율성을 위해 청크 내용을 모두 로드하지 않고
        필요한 최소한의 정보만으로 인덱스를 구축합니다.
        """
        indices = defaultdict(list)
        
        # 청크 메타데이터만 빠르게 스캔
        for idx in range(self.chunk_loader.get_total_chunks()):
            try:
                chunk = self.chunk_loader.get_chunk(idx)
                source = chunk.get('source', '').lower()
                
                if '대규모내부거래' in source:
                    indices['대규모내부거래'].append(idx)
                elif '현황공시' in source or '기업집단' in source:
                    indices['현황공시'].append(idx)
                elif '비상장' in source:
                    indices['비상장사 중요사항'].append(idx)
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
    
    async def process_query(self, query: str, top_k: int = 5) -> Tuple[List[SearchResult], Dict]:
        """개선된 쿼리 처리 프로세스
        
        하이브리드 모델 전략을 사용하여 질문을 분석하고
        최적의 검색 방법을 선택합니다.
        """
        start_time = time.time()
        
        # 캐시 확인
        cache_key = hashlib.md5(f"{query}_{top_k}".encode()).hexdigest()
        cached = self.search_cache.get(cache_key)
        if cached:
            logger.info(f"Cache hit for query: {query[:50]}...")
            return cached['results'], cached['stats']
        
        # 1단계: 하이브리드 모델을 사용한 질문 분석
        try:
            gpt_analysis = await self.query_analyzer.analyze_query(query, self.chunks_info)
            analysis_time = time.time() - start_time
        except Exception as e:
            logger.error(f"Query analysis failed: {str(e)}")
            return self._fallback_search(query, top_k)
        
        # 2단계: 분석 결과에 따른 검색 전략 선택
        search_approach = gpt_analysis['search_strategy']['approach']
        
        # 3단계: 선택된 전략으로 검색 실행
        if search_approach == 'direct_lookup':
            results, search_stats = await self._execute_direct_search(query, gpt_analysis, top_k)
        elif search_approach == 'focused_search':
            results, search_stats = await self._execute_focused_search(query, gpt_analysis, top_k)
        else:  # comprehensive_analysis
            results, search_stats = await self._execute_comprehensive_search(query, gpt_analysis, top_k)
        
        # 4단계: 충돌 해결 및 최신 정보 우선시
        results = self.conflict_resolver.resolve_conflicts(results, query)
        
        # 5단계: 통계 정보 구성
        stats = {
            'gpt_analysis': gpt_analysis,
            'analysis_time': analysis_time,
            'model_used_for_analysis': gpt_analysis.get('model_used', 'unknown'),
            'search_approach': search_approach,
            'total_time': time.time() - start_time,
            'has_outdated_warnings': any(r.metadata.get('has_outdated_info', False) for r in results),
            **search_stats
        }
        
        # 캐시 저장 (에러가 없고 빠른 응답인 경우만)
        if not stats.get('error') and stats['total_time'] < 5:
            self.search_cache.put(cache_key, {
                'results': results,
                'stats': stats
            })
        
        return results, stats
    
    async def _execute_direct_search(self, query: str, analysis: Dict, top_k: int) -> Tuple[List[SearchResult], Dict]:
        """직접 검색 실행"""
        strategy = self.search_strategies['direct']
        primary_manual = analysis['search_strategy']['primary_manual']
        
        indices = strategy.prepare_indices(primary_manual)
        enhanced_query = strategy.enhance_query(query, analysis['search_strategy']['search_keywords'])
        
        return await self.robust_search.search_with_retry(enhanced_query, indices, top_k, strategy)
    
    async def _execute_focused_search(self, query: str, analysis: Dict, top_k: int) -> Tuple[List[SearchResult], Dict]:
        """집중 검색 실행"""
        strategy = self.search_strategies['focused']
        primary_manual = analysis['search_strategy']['primary_manual']
        expected_chunks = analysis['search_strategy'].get('expected_chunks_needed', 10)
        
        indices = strategy.prepare_indices(primary_manual, expected_chunks * 2)
        enhanced_query = strategy.enhance_query(query, analysis['search_strategy']['search_keywords'])
        
        results, stats = await self.robust_search.search_with_retry(
            enhanced_query, indices, top_k * 2, strategy
        )
        
        # 요구사항에 따른 추가 필터링
        requirements = analysis.get('answer_requirements', {})
        filtered_results = strategy.filter_results(results, requirements)
        
        return filtered_results[:top_k], stats
    
    async def _execute_comprehensive_search(self, query: str, analysis: Dict, top_k: int) -> Tuple[List[SearchResult], Dict]:
        """종합 검색 실행 - 여러 매뉴얼에 걸친 검색"""
        all_results = []
        total_searched = 0
        
        for concept in analysis['legal_concepts']:
            if concept['relevance'] in ['primary', 'secondary']:
                manual = concept['concept']
                if manual in self.manual_indices:
                    strategy = self.search_strategies['focused']
                    indices = strategy.prepare_indices(manual, 50)
                    enhanced_query = f"{query} {' '.join(concept['specific_aspects'])}"
                    
                    partial_results, _ = await self.robust_search.search_with_retry(
                        enhanced_query, indices, top_k // 2, strategy
                    )
                    all_results.extend(partial_results)
                    total_searched += len(indices)
        
        # 중복 제거 및 정렬
        seen_chunks = set()
        unique_results = []
        for result in sorted(all_results, key=lambda x: x.score, reverse=True):
            if result.chunk_id not in seen_chunks:
                seen_chunks.add(result.chunk_id)
                unique_results.append(result)
                if len(unique_results) >= top_k:
                    break
        
        stats = {
            'searched_chunks': total_searched,
            'search_method': 'comprehensive_multi_manual'
        }
        
        return unique_results, stats
    
    def _fallback_search(self, query: str, top_k: int) -> Tuple[List[SearchResult], Dict]:
        """분석 실패 시 기본 검색"""
        logger.warning("Falling back to basic search")
        
        # 간단한 키워드 기반 검색
        all_indices = list(range(min(100, self.chunk_loader.get_total_chunks())))
        
        try:
            query_vector = self.embedding_model.encode([query])
            query_vector = np.array(query_vector, dtype=np.float32)
            
            scores, indices = self.index.search(query_vector, min(len(all_indices), top_k * 3))
            
            results = []
            for idx, score in zip(indices[0], scores[0]):
                if 0 <= idx < self.chunk_loader.get_total_chunks():
                    chunk = self.chunk_loader.get_chunk(idx)
                    results.append(SearchResult(
                        chunk_id=chunk.get('chunk_id', str(idx)),
                        content=chunk['content'],
                        score=float(score),
                        source=chunk['source'],
                        page=chunk['page'],
                        chunk_type=chunk.get('chunk_type', 'unknown'),
                        metadata=json.loads(chunk.get('metadata', '{}'))
                    ))
                    if len(results) >= top_k:
                        break
            
            return results, {
                'search_method': 'fallback',
                'error': 'analysis_failed',
                'searched_chunks': len(all_indices)
            }
            
        except Exception as e:
            logger.error(f"Fallback search also failed: {e}")
            return [], {'error': 'complete_failure'}

# ===== 개선된 답변 생성 함수 =====
async def generate_answer_with_hybrid_model(query: str, 
                                          results: List[SearchResult], 
                                          stats: Dict,
                                          api_manager: SecureAPIManager,
                                          hybrid_strategy: HybridGPTStrategy) -> str:
    """하이브리드 모델 전략을 사용한 답변 생성
    
    질문의 특성과 분석 결과를 바탕으로 최적의 모델을 선택하여
    고품질의 답변을 생성합니다.
    """
    # 모델 선택
    gpt_analysis = stats.get('gpt_analysis', {})
    selected_model = hybrid_strategy.select_model_for_answer(query, results, gpt_analysis)
    
    logger.info(f"Selected {selected_model.value} for answer generation")
    
    # 컨텍스트 구성
    context_parts = []
    for i, result in enumerate(results[:5]):
        # 구버전 정보 경고 포함
        warnings = result.metadata.get('warnings', [])
        warning_text = ""
        if warnings:
            warning_text = "\n⚠️ 주의: 이 문서에는 개정 전 정보가 포함되어 있을 수 있습니다."
        
        context_parts.append(f"""
[참고 {i+1}] {result.source} (페이지 {result.page}){warning_text}
{result.content}
""")
    
    context = "\n---\n".join(context_parts)
    
    # 모델별 프롬프트 최적화
    if selected_model == ModelSelection.O4_MINI:
        # 추론 모델용 상세 프롬프트
        system_prompt = """당신은 논리적 추론에 뛰어난 한국 공정거래위원회 법률 전문가입니다.
제공된 자료를 바탕으로 단계별 추론을 통해 정확한 답변을 제공하세요.

답변 구조:
1. 핵심 답변 (직접적이고 명확하게)
2. 법적 근거와 추론 과정
   - "첫째, ..." (각 논점을 단계별로)
   - "둘째, ..."
   - "따라서, ..."
3. 실무 적용 지침
4. 주의사항 및 예외

특히 복잡한 상황에서는 각 요소를 개별적으로 분석한 후 종합하세요."""
        
        user_prompt = f"""다음 자료를 바탕으로 질문에 대해 논리적으로 추론하여 답변하세요.

[참고 자료]
{context}

[질문]
{query}

[추가 고려사항]
- 최신 개정사항 반영 (대규모내부거래 기준: 100억원, 공시기한: 5일)
- 상충하는 정보가 있다면 최신 정보를 우선시
- 불확실한 부분은 명시적으로 표현"""
        
    elif selected_model == ModelSelection.GPT4O_MINI:
        # 간단한 모델용 간결한 프롬프트
        system_prompt = """당신은 한국 공정거래위원회 전문가입니다.
간결하고 정확한 답변을 제공하세요."""
        
        user_prompt = f"""[참고 자료]
{context}

[질문]
{query}

핵심만 간단명료하게 답변하세요."""
        
    else:  # GPT-4o
        # 범용 모델용 균형잡힌 프롬프트
        system_prompt = """당신은 한국 공정거래위원회 전문가입니다.
제공된 자료를 근거로 정확하고 실무적인 답변을 제공하세요.

답변은 다음 구조를 따라주세요:
1. 핵심 답변 (1-2문장)
2. 상세 설명 (근거 조항 포함)
3. 주의사항 또는 예외사항 (있는 경우)"""
        
        user_prompt = f"""다음 자료를 바탕으로 질문에 답변해주세요.

[참고 자료]
{context}

[질문]
{query}

정확하고 실무적인 답변을 부탁드립니다."""
    
    # API 호출 (속도 제한 적용)
    @api_manager.rate_limit(selected_model.value)
    async def call_api():
        return await openai.chat.completions.create(
            model=selected_model.value,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1 if selected_model == ModelSelection.O4_MINI else 0.3,
            max_tokens=1500
        )
    
    response = await call_api()
    answer = response.choices[0].message.content
    
    # 구버전 정보 경고 추가
    if stats.get('has_outdated_warnings'):
        answer += "\n\n⚠️ **중요**: 일부 참고 자료에 개정 전 정보가 포함되어 있을 수 있습니다. 최신 규정을 확인하시기 바랍니다."
    
    return answer

# ===== 비동기 실행 헬퍼 =====
def run_async_in_streamlit(coro):
    """Streamlit 환경에서 비동기 함수를 안전하게 실행"""
    try:
        loop = asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, coro)
            return future.result()
    except RuntimeError:
        return asyncio.run(coro)

# ===== 페이지 설정 및 스타일링 =====
st.set_page_config(
    page_title="전략기획부 AI 법률 보조원", 
    page_icon="⚖️", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS 스타일
st.markdown("""
<style>
    /* Streamlit 기본 요소 숨기기 */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* 메인 헤더 스타일 */
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
    
    /* 채팅 컨테이너 스타일 */
    .chat-container {
        max-width: 900px;
        margin: 0 auto;
    }
    
    /* 메트릭 스타일 개선 */
    [data-testid="metric-container"] {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* 복잡도 표시 스타일 */
    .complexity-indicator {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 16px;
        font-size: 0.85rem;
        font-weight: 500;
        margin-left: 8px;
    }
    
    .complexity-simple {
        background-color: #d4edda;
        color: #155724;
    }
    
    .complexity-medium {
        background-color: #fff3cd;
        color: #856404;
    }
    
    .complexity-complex {
        background-color: #f8d7da;
        color: #721c24;
    }
    
    /* 모델 표시 스타일 */
    .model-indicator {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-left: 4px;
    }
    
    .model-gpt4o {
        background-color: #e3f2fd;
        color: #1565c0;
    }
    
    .model-gpt4o-mini {
        background-color: #f3e5f5;
        color: #6a1b9a;
    }
    
    .model-o4-mini {
        background-color: #e8f5e9;
        color: #2e7d32;
    }
    
    /* 경고 메시지 스타일 */
    .outdated-warning {
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
        border-radius: 4px;
        padding: 0.75rem 1.25rem;
        margin: 0.5rem 0;
        color: #856404;
    }
</style>
""", unsafe_allow_html=True)

# ===== 모델 및 데이터 로딩 =====
@st.cache_resource(show_spinner=False)
def load_models_and_data():
    """필요한 모델과 데이터 로드
    
    개선된 버전에서는 ChunkLoader를 사용하여
    메모리 효율적으로 데이터를 관리합니다.
    """
    try:
        # 필수 파일 확인
        required_files = ["manuals_vector_db.index", "all_manual_chunks.json"]
        missing_files = [f for f in required_files if not os.path.exists(f)]
        
        if missing_files:
            st.error(f"❌ 필수 파일이 없습니다: {', '.join(missing_files)}")
            st.info("💡 prepare_pdfs_ftc.py를 먼저 실행하여 데이터를 준비하세요.")
            return None
        
        with st.spinner("🤖 AI 시스템을 준비하는 중... (최초 1회만 수행됩니다)"):
            # API 관리자 초기화
            api_manager = SecureAPIManager()
            api_manager.load_api_key()
            openai.api_key = api_manager._api_key
            
            # FAISS 인덱스 로드
            index = faiss.read_index("manuals_vector_db.index")
            
            # 메모리 효율적인 청크 로더 초기화
            chunk_loader = ChunkLoader("all_manual_chunks.json")
            
            # 임베딩 모델 로드
            try:
                embedding_model = SentenceTransformer('jhgan/ko-sroberta-multitask')
                logger.info("Korean embedding model loaded successfully")
            except Exception as e:
                st.warning("한국어 모델 로드 실패. 대체 모델을 사용합니다.")
                embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
            
            # Reranker 모델 로드 (선택적)
            try:
                reranker_model = CrossEncoder('Dongjin-kr/ko-reranker')
                logger.info("Korean reranker model loaded successfully")
            except:
                logger.warning("Reranker model load failed, proceeding without it")
                reranker_model = None
        
        return embedding_model, reranker_model, index, chunk_loader, api_manager
        
    except Exception as e:
        st.error(f"시스템 초기화 실패: {str(e)}")
        logger.error(f"System initialization failed: {str(e)}")
        return None

# ===== 메인 UI =====
def main():
    """메인 애플리케이션 함수"""
    
    # 헤더 표시
    st.markdown("""
    <div class="main-header">
        <h1>⚖️ 전략기획부 AI 법률 보조원</h1>
        <p>공정거래위원회 규정 및 매뉴얼 기반 하이브리드 AI Q&A 시스템</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 모델 및 데이터 로드
    models = load_models_and_data()
    if not models or len(models) != 5:
        st.stop()
    
    embedding_model, reranker_model, index, chunk_loader, api_manager = models
    
    # 개선된 RAG 파이프라인 초기화
    rag = ImprovedHybridRAGPipeline(
        embedding_model, reranker_model, index, chunk_loader, api_manager
    )
    
    # 세션 상태 초기화
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "api_usage" not in st.session_state:
        st.session_state.api_usage = api_manager.get_usage_stats()
    
    # 채팅 컨테이너
    chat_container = st.container()
    
    with chat_container:
        # 이전 메시지 표시
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                if message["role"] == "user":
                    st.write(message["content"])
                else:
                    if isinstance(message["content"], dict):
                        st.write(message["content"]["answer"])
                        
                        # 복잡도 및 모델 정보 표시
                        complexity = message["content"].get("complexity", "unknown")
                        model_used = message["content"].get("model_used", "unknown")
                        
                        complexity_html = f'<span class="complexity-indicator complexity-{complexity}">{complexity.upper()}</span>'
                        model_html = f'<span class="model-indicator model-{model_used.replace("-", "")}">{model_used}</span>'
                        
                        st.markdown(f"복잡도: {complexity_html} | 사용 모델: {model_html}", unsafe_allow_html=True)
                        
                        # 성능 메트릭
                        if "total_time" in message["content"]:
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("🔍 검색", f"{message['content']['search_time']:.1f}초")
                            with col2:
                                st.metric("✍️ 답변 생성", f"{message['content']['generation_time']:.1f}초")
                            with col3:
                                st.metric("⏱️ 전체", f"{message['content']['total_time']:.1f}초")
                        
                        # 구버전 정보 경고
                        if message["content"].get("has_outdated_warnings"):
                            st.markdown("""
                            <div class="outdated-warning">
                            ⚠️ 일부 참고 자료에 개정 전 정보가 포함되어 있을 수 있습니다.
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.write(message["content"])
        
        # 새 질문 입력
        if prompt := st.chat_input("질문을 입력하세요 (예: 대규모내부거래 공시 기한은?)"):
            # 사용자 메시지 추가
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)
            
            # AI 응답 생성
            with st.chat_message("assistant"):
                total_start_time = time.time()
                
                # 검색 수행
                search_start_time = time.time()
                with st.spinner("🔍 AI가 질문을 분석하고 최적의 검색 전략을 수립하는 중..."):
                    results, stats = run_async_in_streamlit(rag.process_query(prompt, top_k=5))
                search_time = time.time() - search_start_time
                
                # 답변 생성
                generation_start_time = time.time()
                with st.spinner("💭 AI가 답변을 생성하는 중..."):
                    answer = run_async_in_streamlit(
                        generate_answer_with_hybrid_model(
                            prompt, results, stats, api_manager, rag.hybrid_strategy
                        )
                    )
                generation_time = time.time() - generation_start_time
                
                total_time = time.time() - total_start_time
                
                # 답변 표시
                st.write(answer)
                
                # 분석 정보 표시
                gpt_analysis = stats.get('gpt_analysis', {})
                complexity = gpt_analysis.get('query_analysis', {}).get('actual_complexity', 'unknown')
                model_used = gpt_analysis.get('model_used', 'unknown')
                
                complexity_html = f'<span class="complexity-indicator complexity-{complexity}">{complexity.upper()}</span>'
                model_html = f'<span class="model-indicator model-{model_used.replace("-", "")}">{model_used}</span>'
                
                st.markdown(f"질문 복잡도: {complexity_html} | 분석 모델: {model_html}", unsafe_allow_html=True)
                
                # 구버전 정보 경고
                if stats.get('has_outdated_warnings'):
                    st.markdown("""
                    <div class="outdated-warning">
                    ⚠️ 일부 참고 자료에 개정 전 정보가 포함되어 있을 수 있습니다.
                    </div>
                    """, unsafe_allow_html=True)
                
                # 성능 메트릭
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("🔍 검색", f"{search_time:.1f}초")
                with col2:
                    st.metric("✍️ 답변 생성", f"{generation_time:.1f}초")
                with col3:
                    st.metric("⏱️ 전체", f"{total_time:.1f}초")
                
                # 상세 정보
                with st.expander("🔍 상세 정보 보기"):
                    if gpt_analysis:
                        st.subheader("🤖 AI 질문 분석")
                        
                        # 분석 내용 표시
                        col1, col2 = st.columns(2)
                        with col1:
                            st.json({
                                "핵심 의도": gpt_analysis.get('query_analysis', {}).get('core_intent', ''),
                                "실제 복잡도": complexity,
                                "선택 이유": gpt_analysis.get('model_selection_reason', ''),
                                "검색 전략": gpt_analysis.get('search_strategy', {}).get('approach', '')
                            })
                        
                        with col2:
                            st.json({
                                "주요 매뉴얼": gpt_analysis.get('search_strategy', {}).get('primary_manual', ''),
                                "검색 키워드": gpt_analysis.get('search_strategy', {}).get('search_keywords', []),
                                "필요 청크 수": gpt_analysis.get('search_strategy', {}).get('expected_chunks_needed', 0)
                            })
                        
                        # 추론 체인 표시 (o4-mini 사용 시)
                        reasoning_chain = gpt_analysis.get('query_analysis', {}).get('reasoning_chain', [])
                        if reasoning_chain:
                            st.subheader("🧠 추론 과정")
                            for i, step in enumerate(reasoning_chain, 1):
                                st.write(f"{i}. {step}")
                    
                    # 검색된 문서
                    st.subheader("📚 참고 자료")
                    for i, result in enumerate(results[:3]):
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.caption(f"**{result.source}** - 페이지 {result.page}")
                        with col2:
                            st.caption(f"관련도: {result.score:.2f}")
                        
                        # 문서 날짜 및 경고
                        if result.document_date:
                            st.caption(f"📅 문서 날짜: {result.document_date}")
                        
                        if result.metadata.get('has_outdated_info'):
                            warnings = result.metadata.get('warnings', [])
                            for warning in warnings:
                                st.warning(f"⚠️ 구버전 정보: {warning['found']} → 현재: {warning['current']}")
                        
                        # 내용 미리보기
                        with st.container():
                            content = result.content[:300] + "..." if len(result.content) > 300 else result.content
                            st.text(content)
                    
                    # API 사용 통계
                    st.subheader("📊 API 사용 통계")
                    usage_stats = api_manager.get_usage_stats()
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("GPT-4o 비용", f"${usage_stats['costs'].get('gpt-4o', 0):.4f}")
                    with col2:
                        st.metric("GPT-4o-mini 비용", f"${usage_stats['costs'].get('gpt-4o-mini', 0):.4f}")
                    with col3:
                        st.metric("o4-mini 비용", f"${usage_stats['costs'].get('o4-mini', 0):.4f}")
                    
                    st.info(f"💰 예상 월 비용: ${usage_stats.get('estimated_monthly_cost', 0):.2f}")
                
                # 응답 데이터 저장
                response_data = {
                    "answer": answer,
                    "search_time": search_time,
                    "generation_time": generation_time,
                    "total_time": total_time,
                    "complexity": complexity,
                    "model_used": model_used,
                    "has_outdated_warnings": stats.get('has_outdated_warnings', False)
                }
                st.session_state.messages.append({"role": "assistant", "content": response_data})
    
    # 구분선
    st.divider()
    
    # 면책 조항
    st.caption("⚠️ 본 답변은 AI가 생성한 참고자료입니다. 중요한 사항은 반드시 원문을 확인하시기 바랍니다.")
    
    # 사이드바
    with st.sidebar:
        st.header("💡 예시 질문")
        
        # 복잡도별 예시 질문
        st.subheader("🟢 단순 질문")
        if st.button("대규모내부거래 공시 기한은?"):
            st.session_state.new_question = "대규모내부거래 이사회 의결 후 공시 기한은 며칠인가요?"
            st.rerun()
        if st.button("이사회 의결 금액 기준은?"):
            st.session_state.new_question = "대규모내부거래에서 이사회 의결이 필요한 거래 금액은?"
            st.rerun()
            
        st.subheader("🟡 중간 복잡도")
        if st.button("계열사 거래 시 주의사항은?"):
            st.session_state.new_question = "계열사와 자금거래를 할 때 어떤 절차를 거쳐야 하고 주의할 점은 무엇인가요?"
            st.rerun()
        if st.button("비상장사 주식 양도 절차는?"):
            st.session_state.new_question = "비상장회사가 주식을 양도할 때 필요한 절차와 공시 의무는 어떻게 되나요?"
            st.rerun()
            
        st.subheader("🔴 복잡한 질문 (추론 필요)")
        if st.button("복합 거래 분석"):
            st.session_state.new_question = "A회사가 B계열사에 자금을 대여하면서 동시에 C계열사의 주식을 취득하는 경우, 각각 어떤 규제가 적용되고 공시는 어떻게 해야 하나요?"
            st.rerun()
        if st.button("종합적 리스크 검토"):
            st.session_state.new_question = "우리 회사가 여러 계열사와 동시에 거래를 진행할 때 대규모내부거래 규제와 관련하여 종합적으로 검토해야 할 리스크와 대응 전략은?"
            st.rerun()
        
        st.divider()
        
        # 시스템 정보
        st.subheader("🔧 시스템 정보")
        st.caption("이 시스템은 질문의 복잡도에 따라 최적의 AI 모델을 자동으로 선택합니다:")
        st.caption("• **o4-mini**: 복잡한 추론")
        st.caption("• **GPT-4o**: 종합적 분석")
        st.caption("• **GPT-4o-mini**: 단순 조회")
        
        st.divider()
        
        # 성능 통계
        if st.button("📊 성능 통계 보기"):
            with st.container():
                st.subheader("API 사용 현황")
                usage = api_manager.get_usage_stats()
                
                total_cost = sum(usage['costs'].values())
                st.metric("총 비용", f"${total_cost:.4f}")
                
                for model, cost in usage['costs'].items():
                    if model != 'total':
                        st.caption(f"{model}: ${cost:.4f}")
    
    # 새 질문 처리
    if "new_question" in st.session_state:
        st.session_state.messages.append({"role": "user", "content": st.session_state.new_question})
        del st.session_state.new_question
        st.rerun()

if __name__ == "__main__":
    main()
