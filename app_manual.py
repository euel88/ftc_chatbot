# 파일 이름: app_manual.py (공정거래위원회 AI 법률 보조원 - 최적화 버전)

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
from functools import lru_cache
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

# ===== 로깅 설정 =====
# 로깅은 시스템의 작동 상황을 기록하는 일기장과 같습니다.
# 문제가 발생했을 때 원인을 찾거나 성능을 개선하는 데 도움이 됩니다.
def setup_logging():
    """구조화된 로깅 시스템 설정"""
    logger = logging.getLogger('ftc_chatbot')
    logger.setLevel(logging.DEBUG)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    )
    
    # 파일 핸들러 - 로그를 파일에 저장
    file_handler = logging.FileHandler('ftc_chatbot_debug.log')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    
    # 콘솔 핸들러 - 개발 중 확인용
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# 전역 로거 설정
logger = setup_logging()

# ===== 사용자 정의 예외 클래스 =====
# 예외 클래스는 프로그램에서 발생할 수 있는 다양한 오류 상황을 
# 체계적으로 관리하기 위한 도구입니다.
class RAGPipelineError(Exception):
    """RAG 파이프라인의 기본 예외 클래스"""
    pass

class IndexError(RAGPipelineError):
    """인덱스 관련 오류"""
    pass

class EmbeddingError(RAGPipelineError):
    """임베딩 생성 관련 오류"""
    pass

class ModelSelectionError(RAGPipelineError):
    """모델 선택 관련 오류"""
    pass

# ===== 에러 컨텍스트 매니저 =====
@contextmanager
def error_context(operation_name: str, fallback_value=None):
    """
    에러 처리를 위한 컨텍스트 매니저
    
    이는 마치 안전망과 같아서, 작업 중 문제가 발생해도
    프로그램이 완전히 중단되지 않고 적절히 대응할 수 있게 합니다.
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

# ===== 페이지 설정 및 스타일링 =====
st.set_page_config(
    page_title="전략기획부 AI 법률 보조원", 
    page_icon="⚖️", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS 스타일 - UI를 보기 좋게 만드는 디자인 설정
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
    
    /* 비용 효율성 표시 */
    .cost-efficiency {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.8rem;
        margin-left: 4px;
    }
    
    .cost-saved {
        background-color: #d4edda;
        color: #155724;
    }
    
    .cost-normal {
        background-color: #e2e3e5;
        color: #383d41;
    }
    
    .cost-high {
        background-color: #f8d7da;
        color: #721c24;
    }
</style>
""", unsafe_allow_html=True)

# OpenAI API 설정
try:
    openai.api_key = st.secrets["OPENAI_API_KEY"]
except:
    st.error("⚠️ OpenAI API 키를 설정해주세요.")
    st.stop()

# ===== 타입 정의 =====
# 타입 정의는 데이터의 구조를 명확히 하여 코드의 안정성을 높입니다.
class ChunkDict(TypedDict):
    """청크 데이터의 타입 정의"""
    chunk_id: str
    content: str
    source: str
    page: int
    chunk_type: str
    metadata: str

class AnalysisResult(TypedDict):
    """분석 결과의 타입 정의"""
    query_analysis: dict
    complexity_score: float
    recommended_model: str
    search_strategy: dict

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

class QueryComplexity(Enum):
    """질문 복잡도 레벨"""
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"

# ===== LRU 캐시 구현 =====
class LRUCache:
    """
    시간 기반 만료를 지원하는 LRU 캐시 구현
    
    캐시는 자주 사용되는 데이터를 메모리에 보관하여
    반복적인 계산을 피하고 성능을 향상시키는 기술입니다.
    """
    def __init__(self, max_size: int = 100, ttl: int = 3600):
        self.cache = OrderedDict()
        self.max_size = max_size
        self.ttl = ttl  # Time To Live (초 단위)
        
    def get(self, key: str):
        """캐시에서 값을 가져오고, 만료된 항목은 제거"""
        if key not in self.cache:
            return None
            
        value, timestamp = self.cache[key]
        if time.time() - timestamp > self.ttl:
            del self.cache[key]
            return None
            
        # 최근 사용된 항목을 끝으로 이동
        self.cache.move_to_end(key)
        return value
        
    def put(self, key: str, value):
        """캐시에 값 저장"""
        if key in self.cache:
            del self.cache[key]
            
        # 캐시가 가득 차면 가장 오래된 항목 제거
        if len(self.cache) >= self.max_size:
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

# ===== 비동기 실행 헬퍼 =====
def run_async_in_streamlit(coro):
    """
    Streamlit 환경에서 비동기 함수를 안전하게 실행
    
    Streamlit은 기본적으로 동기적으로 작동하므로,
    비동기 함수를 실행하려면 특별한 처리가 필요합니다.
    """
    try:
        loop = asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, coro)
            return future.result()
    except RuntimeError:
        return asyncio.run(coro)

# ===== 비용 관리 시스템 =====
class BudgetManager:
    """
    API 사용 비용을 추적하고 관리하는 시스템
    
    이는 마치 가계부를 작성하는 것과 같아서,
    얼마나 사용했고 얼마나 남았는지를 항상 파악할 수 있게 합니다.
    """
    def __init__(self, daily_budget: float = 50.0):
        self.daily_budget = daily_budget
        self.reset_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        # 모델별 비용 정보 (1M 토큰당 달러)
        self.model_costs = {
            'gpt-4o-mini': {
                'input': 0.15,
                'cached': 0.075,
                'output': 0.60
            },
            'o4-mini': {
                'input': 1.10,
                'cached': 0.275,
                'output': 4.40
            },
            'gpt-4o': {
                'input': 2.50,
                'cached': 1.25,
                'output': 10.00
            }
        }
    
    def calculate_cost(self, model: str, input_tokens: int, 
                      output_tokens: int, cached: bool = False) -> float:
        """토큰 수를 기반으로 비용 계산"""
        costs = self.model_costs[model]
        input_cost = costs['cached' if cached else 'input'] * (input_tokens / 1_000_000)
        output_cost = costs['output'] * (output_tokens / 1_000_000)
        return input_cost + output_cost
    
    def get_current_status(self) -> Dict:
        """현재 예산 상황 반환"""
        # 세션 상태에서 오늘의 사용량 가져오기
        if 'daily_cost' not in st.session_state:
            st.session_state.daily_cost = 0.0
            
        # 날짜가 바뀌었으면 리셋
        current_date = datetime.now().date()
        if 'last_reset_date' not in st.session_state:
            st.session_state.last_reset_date = current_date
        elif st.session_state.last_reset_date != current_date:
            st.session_state.daily_cost = 0.0
            st.session_state.last_reset_date = current_date
            
        used = st.session_state.daily_cost
        remaining = self.daily_budget - used
        
        return {
            'used': used,
            'remaining': remaining,
            'remaining_ratio': remaining / self.daily_budget,
            'is_budget_critical': remaining < self.daily_budget * 0.2
        }
    
    def add_usage(self, cost: float):
        """사용 비용 추가"""
        st.session_state.daily_cost = st.session_state.get('daily_cost', 0.0) + cost

# ===== 모델 선택 시스템 =====
class SimplifiedModelSelector:
    """
    세 가지 모델(gpt-4o-mini, o4-mini, gpt-4o)만을 사용하는 
    단순화되고 효율적인 모델 선택 시스템
    
    이 시스템은 마치 의사가 환자의 증상을 보고 
    적절한 검사를 결정하는 것과 같이 작동합니다.
    """
    
    def __init__(self):
        self.model_profiles = {
            'gpt-4o-mini': {
                'cost_per_1k': 0.00015,
                'strengths': ['speed', 'simple_queries', 'fact_checking'],
                'max_tokens': 2000,
                'decision_threshold': 0.3,
                'performance_score': 0.6
            },
            'o4-mini': {
                'cost_per_1k': 0.0011,
                'strengths': ['reasoning', 'analysis', 'multi_step'],
                'max_tokens': 4000,
                'decision_threshold': 0.85,
                'performance_score': 0.85  # 추론 능력이 뛰어남
            },
            'gpt-4o': {
                'cost_per_1k': 0.0025,
                'strengths': ['long_context', 'creative', 'fallback'],
                'max_tokens': 8000,
                'decision_threshold': 0.95,
                'performance_score': 0.75  # o4-mini보다 비싸지만 추론은 약함
            }
        }
        
        self.budget_manager = BudgetManager()
        
    def select_model(self, query: str, initial_assessment: Dict) -> Tuple[str, Dict]:
        """
        질문에 가장 적합한 모델을 선택합니다.
        
        선택 과정은 다음과 같습니다:
        1. 질문의 복잡도를 평가
        2. 현재 예산 상황 확인
        3. 각 모델의 강점과 비용을 고려하여 최적 선택
        """
        
        # 질문의 특성을 점수화 (0~1)
        complexity_score = self._calculate_complexity_score(query, initial_assessment)
        
        # 현재 예산 상황
        budget_status = self.budget_manager.get_current_status()
        
        # 명확한 규칙 기반 선택
        if complexity_score < self.model_profiles['gpt-4o-mini']['decision_threshold']:
            selected_model = 'gpt-4o-mini'
            reason = "간단한 사실 확인 또는 정의 질문"
            
        elif complexity_score < self.model_profiles['o4-mini']['decision_threshold']:
            selected_model = 'o4-mini'
            reason = "추론과 분석이 필요한 표준 질문"
            
        else:
            # 특별한 경우를 확인
            if self._requires_long_context(query):
                selected_model = 'gpt-4o'
                reason = "긴 문맥 처리가 필요한 특수 케이스"
            elif self._is_creative_task(query):
                selected_model = 'gpt-4o'
                reason = "창의적 해석이 필요한 특수 케이스"
            else:
                # 대부분의 복잡한 질문도 o4-mini가 더 효과적
                selected_model = 'o4-mini'
                reason = "복잡하지만 o4-mini의 추론 능력으로 충분"
        
        # 예산이 부족한 경우 하위 모델로 대체
        if budget_status['is_budget_critical']:
            if selected_model == 'gpt-4o':
                selected_model = 'o4-mini'
                reason += " (예산 제약으로 대체)"
            elif selected_model == 'o4-mini' and budget_status['remaining_ratio'] < 0.1:
                selected_model = 'gpt-4o-mini'
                reason += " (예산 제약으로 대체)"
        
        selection_info = {
            'model': selected_model,
            'reason': reason,
            'complexity_score': complexity_score,
            'budget_remaining': budget_status['remaining_ratio'],
            'estimated_cost': self._estimate_query_cost(selected_model)
        }
        
        return selected_model, selection_info
    
    def _calculate_complexity_score(self, query: str, assessment: Dict) -> float:
        """
        질문의 복잡도를 0에서 1 사이의 점수로 계산합니다.
        
        복잡도 평가 요소:
        - 질문의 길이
        - 특정 키워드의 존재
        - 문장 구조의 복잡성
        - 여러 주체나 조건의 언급
        """
        score = 0.0
        
        # 길이 기반 점수 (0~0.3)
        query_length = len(query)
        if query_length < 50:
            score += 0.1
        elif query_length < 150:
            score += 0.2
        else:
            score += 0.3
            
        # 키워드 기반 점수 (0~0.4)
        complex_keywords = ['만약', '경우', '동시에', '여러', '비교', '분석', '전략', '종합적']
        keyword_count = sum(1 for keyword in complex_keywords if keyword in query)
        score += min(keyword_count * 0.1, 0.4)
        
        # 구조적 복잡도 (0~0.3)
        if '?' in query and query.count('?') > 1:
            score += 0.1
        if any(conj in query for conj in ['그리고', '또한', '하지만', '그러나']):
            score += 0.1
        if re.search(r'[A-Z].*[A-Z]', query):  # 여러 주체가 언급됨
            score += 0.1
            
        return min(score, 1.0)
    
    def _requires_long_context(self, query: str) -> bool:
        """긴 문맥 처리가 필요한지 판단"""
        return len(query) > 1000 or '전체' in query or '모든' in query
    
    def _is_creative_task(self, query: str) -> bool:
        """창의적 작업인지 판단"""
        creative_keywords = ['시나리오', '스토리', '창의', '제안', '아이디어']
        return any(keyword in query for keyword in creative_keywords)
    
    def _estimate_query_cost(self, model: str) -> float:
        """질문 처리 예상 비용"""
        avg_tokens = 3000  # 평균 토큰 수
        return self.model_profiles[model]['cost_per_1k'] * (avg_tokens / 1000) * 2  # 입출력 모두 고려

# ===== 캐싱 전략 시스템 =====
class SmartCacheStrategy:
    """
    모델별 특성을 고려한 지능형 캐싱 전략
    
    캐싱은 마치 자주 찾는 책을 책상 위에 올려놓는 것과 같아서,
    반복적인 질문에 대해 빠르고 저렴하게 답변할 수 있게 합니다.
    """
    
    def __init__(self):
        self.cache_configs = {
            'gpt-4o-mini': {
                'ttl': 7200,  # 2시간 - 저렴하므로 오래 보관
                'similarity_threshold': 0.85,
                'cache_benefit_ratio': 0.5  # 50% 비용 절감
            },
            'o4-mini': {
                'ttl': 3600,  # 1시간
                'similarity_threshold': 0.9,
                'cache_benefit_ratio': 0.75  # 75% 비용 절감
            },
            'gpt-4o': {
                'ttl': 2400,  # 40분
                'similarity_threshold': 0.92,
                'cache_benefit_ratio': 0.5
            }
        }
        
        self.query_cache = LRUCache(max_size=200, ttl=7200)
        self.embedding_cache = LRUCache(max_size=500, ttl=10800)
    
    def should_use_cache(self, query: str, model: str, 
                        similar_cached_query: Optional[Dict]) -> bool:
        """캐시 사용 여부를 결정"""
        if not similar_cached_query:
            return False
        
        config = self.cache_configs[model]
        
        # 유사도가 임계값을 넘는지 확인
        similarity = similar_cached_query.get('similarity', 0)
        if similarity < config['similarity_threshold']:
            return False
        
        # 캐시 나이 확인
        cache_age = time.time() - similar_cached_query.get('timestamp', 0)
        if cache_age > config['ttl']:
            return False
        
        return True
    
    def get_cached_result(self, query: str) -> Optional[Dict]:
        """캐시된 결과 반환"""
        # 정확히 일치하는 질문 먼저 확인
        cache_key = hashlib.md5(query.encode()).hexdigest()
        exact_match = self.query_cache.get(cache_key)
        if exact_match:
            return exact_match
        
        # 유사한 질문 찾기 (실제로는 더 정교한 유사도 계산 필요)
        # 여기서는 간단한 예시만 제공
        return None
    
    def store_result(self, query: str, result: Dict, model: str):
        """결과를 캐시에 저장"""
        cache_key = hashlib.md5(query.encode()).hexdigest()
        result['cached_at'] = time.time()
        result['model'] = model
        self.query_cache.put(cache_key, result)

# ===== 질문 분석기 =====
class EnhancedQueryAnalyzer:
    """
    GPT-4o-mini를 활용한 효율적인 질문 분석기
    
    이 클래스는 질문을 분석하여 어떤 정보가 필요한지,
    어떻게 검색해야 하는지를 파악합니다.
    """
    
    def __init__(self):
        self.analysis_cache = LRUCache(max_size=100, ttl=3600)
        
    async def analyze_query(self, query: str) -> Dict:
        """질문을 분석하고 검색 전략 수립"""
        
        # 캐시 확인
        cache_key = hashlib.md5(query.encode()).hexdigest()
        cached = self.analysis_cache.get(cache_key)
        if cached:
            return cached
        
        prompt = f"""
        당신은 공정거래법 전문가입니다. 다음 질문을 분석해주세요.
        
        질문: {query}
        
        다음 형식의 JSON으로 응답해주세요:
        {{
            "query_type": "simple/standard/complex",
            "main_topic": "대규모내부거래/현황공시/비상장사 중요사항/기타",
            "required_info": ["필요한 정보 1", "필요한 정보 2"],
            "search_keywords": ["검색 키워드 1", "검색 키워드 2"],
            "expected_answer_type": "fact/process/analysis/comparison"
        }}
        """
        
        try:
            response = openai.chat.completions.create(
                model="gpt-4o-mini",  # 분석에는 저렴한 모델 사용
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=500,
                response_format={"type": "json_object"}
            )
            
            analysis = json.loads(response.choices[0].message.content)
            
            # 캐시 저장
            self.analysis_cache.put(cache_key, analysis)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Query analysis error: {str(e)}")
            # 폴백 분석
            return {
                "query_type": "standard",
                "main_topic": "기타",
                "required_info": [],
                "search_keywords": query.split()[:5],
                "expected_answer_type": "fact"
            }

# ===== 하이브리드 RAG 파이프라인 =====
class OptimizedHybridRAGPipeline:
    """
    비용 최적화가 적용된 하이브리드 RAG 파이프라인
    
    이 클래스는 전체 시스템의 핵심으로, 질문을 받아
    적절한 모델을 선택하고, 검색을 수행하며, 답변을 생성합니다.
    """
    
    def __init__(self, embedding_model, reranker_model, index, chunks):
        if not chunks:
            raise ValueError("No chunks provided to pipeline")
        
        if index.ntotal == 0:
            raise ValueError("FAISS index is empty")
        
        self.embedding_model = embedding_model
        self.reranker_model = reranker_model
        self.index = index
        self.chunks = chunks
        
        # 시스템 구성 요소들
        self.model_selector = SimplifiedModelSelector()
        self.query_analyzer = EnhancedQueryAnalyzer()
        self.cache_strategy = SmartCacheStrategy()
        self.budget_manager = BudgetManager()
        
        # 매뉴얼별 인덱스 구축
        self.manual_indices = self._build_manual_indices()
        
        # 성능 추적
        self.performance_history = []
        
        logger.info(f"Pipeline initialized with {len(chunks)} chunks")
    
    def _build_manual_indices(self) -> Dict[str, List[int]]:
        """각 매뉴얼별로 청크 인덱스를 미리 구축"""
        indices = defaultdict(list)
        
        for idx, chunk in enumerate(self.chunks):
            source = chunk.get('source', '').lower()
            
            if '대규모내부거래' in source:
                indices['대규모내부거래'].append(idx)
            elif '현황공시' in source or '기업집단' in source:
                indices['현황공시'].append(idx)
            elif '비상장' in source:
                indices['비상장사 중요사항'].append(idx)
            else:
                indices['기타'].append(idx)
        
        return dict(indices)
    
    async def process_query(self, query: str, top_k: int = 5) -> Tuple[List[SearchResult], Dict]:
        """
        질문을 처리하는 메인 메서드
        
        처리 과정:
        1. 캐시 확인
        2. 질문 분석
        3. 모델 선택
        4. 검색 수행
        5. 답변 생성
        """
        start_time = time.time()
        
        # 1. 캐시 확인
        cached_result = self.cache_strategy.get_cached_result(query)
        if cached_result:
            logger.info(f"Cache hit for query: {query[:50]}...")
            return cached_result['results'], {
                'cache_hit': True,
                'total_time': 0.1,
                'model_used': cached_result['model']
            }
        
        # 2. 질문 분석
        analysis = await self.query_analyzer.analyze_query(query)
        
        # 3. 모델 선택
        selected_model, selection_info = self.model_selector.select_model(query, analysis)
        
        # 4. 검색 수행
        search_results = await self._perform_search(query, analysis, top_k)
        
        # 5. 통계 정보 구성
        stats = {
            'query_analysis': analysis,
            'selected_model': selected_model,
            'selection_info': selection_info,
            'search_time': time.time() - start_time,
            'cache_hit': False,
            'total_results': len(search_results)
        }
        
        # 6. 캐시 저장 (간단한 질문만)
        if analysis['query_type'] == 'simple':
            self.cache_strategy.store_result(query, {
                'results': search_results,
                'stats': stats
            }, selected_model)
        
        return search_results, stats
    
    async def _perform_search(self, query: str, analysis: Dict, top_k: int) -> List[SearchResult]:
        """실제 검색 수행"""
        # 관련 매뉴얼 확인
        main_topic = analysis.get('main_topic', '기타')
        relevant_indices = self.manual_indices.get(main_topic, [])
        
        if not relevant_indices:
            relevant_indices = list(range(min(len(self.chunks), 300)))
        
        # 검색어 확장
        search_keywords = analysis.get('search_keywords', [])
        enhanced_query = f"{query} {' '.join(search_keywords)}"
        
        # 임베딩 생성
        query_embedding = self.embedding_model.encode([enhanced_query])
        query_vector = np.array(query_embedding, dtype=np.float32)
        
        # FAISS 검색
        k_search = min(len(relevant_indices), max(1, top_k * 3))
        scores, indices = self.index.search(query_vector, k_search)
        
        # 결과 구성
        results = []
        seen_chunks = set()
        
        for idx, score in zip(indices[0], scores[0]):
            if idx in relevant_indices and idx not in seen_chunks:
                seen_chunks.add(idx)
                chunk = self.chunks[idx]
                
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
        
        # 리랭킹 (옵션)
        if self.reranker_model and len(results) > 0:
            # 리랭킹 로직 (필요시 구현)
            pass
        
        return results

# ===== 답변 생성 함수 =====
async def generate_answer(query: str, results: List[SearchResult], stats: Dict) -> Tuple[str, Dict]:
    """
    선택된 모델을 사용하여 답변 생성
    
    이 함수는 검색 결과를 바탕으로 사용자 질문에 대한
    완성된 답변을 생성합니다.
    """
    
    # 선택된 모델 확인
    model = stats.get('selected_model', 'gpt-4o-mini')
    
    # 컨텍스트 구성
    context_parts = []
    for i, result in enumerate(results[:5]):  # 상위 5개 결과만 사용
        context_parts.append(f"""
[참고 {i+1}] {result.source} (페이지 {result.page})
{result.content}
""")
    
    context = "\n---\n".join(context_parts)
    
    # 질문 유형에 따른 프롬프트 조정
    query_type = stats.get('query_analysis', {}).get('query_type', 'standard')
    
    if query_type == 'simple':
        instruction = "간결하고 명확하게 답변해주세요."
        max_tokens = 500
    elif query_type == 'complex':
        instruction = "단계별로 상세하게 설명해주세요."
        max_tokens = 1500
    else:
        instruction = "정확하고 실무적인 답변을 제공해주세요."
        max_tokens = 1000
    
    # 시스템 프롬프트
    system_prompt = f"""당신은 한국 공정거래위원회 전문가입니다.
제공된 자료를 근거로 정확한 답변을 제공하세요.

{instruction}

답변 구조:
1. 핵심 답변 (1-2문장)
2. 상세 설명 (필요시)
3. 주의사항 (있는 경우)"""
    
    # 메시지 구성
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"""다음 자료를 바탕으로 질문에 답변해주세요.

[참고 자료]
{context}

[질문]
{query}"""}
    ]
    
    # API 호출 시작 시간
    api_start = time.time()
    
    try:
        # 모델에 따른 temperature 설정
        temperature = 0.1 if query_type == 'simple' else 0.3
        
        response = openai.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        answer = response.choices[0].message.content
        
        # 토큰 사용량 추정 (실제로는 response에서 가져와야 함)
        estimated_tokens = len(context) / 4 + len(answer) / 4
        
        # 비용 계산
        cost = stats['selection_info']['estimated_cost']
        
        # 예산에 추가
        if 'budget_manager' in globals():
            budget_manager = BudgetManager()
            budget_manager.add_usage(cost)
        
        generation_stats = {
            'generation_time': time.time() - api_start,
            'model': model,
            'estimated_tokens': estimated_tokens,
            'cost': cost
        }
        
        return answer, generation_stats
        
    except Exception as e:
        logger.error(f"Answer generation error: {str(e)}")
        return "죄송합니다. 답변 생성 중 오류가 발생했습니다.", {
            'generation_time': time.time() - api_start,
            'error': str(e)
        }

# ===== 성능 시각화 함수들 =====
def create_complexity_gauge(score: float) -> go.Figure:
    """복잡도를 게이지 차트로 표시"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "질문 복잡도"},
        gauge = {
            'axis': {'range': [None, 1]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 0.3], 'color': "lightgreen"},
                {'range': [0.3, 0.7], 'color': "yellow"},
                {'range': [0.7, 1], 'color': "lightcoral"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 0.85
            }
        }
    ))
    
    fig.update_layout(height=200, margin=dict(l=20, r=20, t=40, b=20))
    return fig

def create_budget_pie_chart(used: float, remaining: float) -> go.Figure:
    """예산 사용 현황을 파이 차트로 표시"""
    fig = go.Figure(data=[go.Pie(
        labels=['사용됨', '남음'],
        values=[used, remaining],
        hole=.3,
        marker_colors=['#ff6b6b', '#51cf66']
    )])
    
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
        hoverinfo='label+value+percent'
    )
    
    fig.update_layout(
        title="일일 예산 현황",
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        showlegend=False
    )
    
    return fig

# ===== 모델 및 데이터 로딩 =====
@st.cache_resource(show_spinner=False)
def load_models_and_data():
    """필요한 모델과 데이터 로드"""
    try:
        required_files = ["manuals_vector_db.index", "all_manual_chunks.json"]
        missing_files = [f for f in required_files if not os.path.exists(f)]
        
        if missing_files:
            st.error(f"❌ 필수 파일이 없습니다: {', '.join(missing_files)}")
            st.info("💡 prepare_pdfs_ftc.py를 먼저 실행하여 데이터를 준비하세요.")
            return None, None, None, None
        
        with st.spinner("🤖 AI 시스템을 준비하는 중... (최초 1회만 수행됩니다)"):
            # FAISS 인덱스 로드
            index = faiss.read_index("manuals_vector_db.index")
            
            # 청크 데이터 로드
            with open("all_manual_chunks.json", "r", encoding="utf-8") as f:
                chunks = json.load(f)
            
            # 임베딩 모델 로드
            try:
                embedding_model = SentenceTransformer('jhgan/ko-sroberta-multitask')
            except Exception as e:
                st.warning("한국어 모델 로드 실패. 대체 모델을 사용합니다.")
                embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
            
            # 리랭커 모델 로드 (선택적)
            try:
                reranker_model = CrossEncoder('Dongjin-kr/ko-reranker')
            except:
                reranker_model = None
                logger.warning("Reranker model not loaded")
        
        return embedding_model, reranker_model, index, chunks
        
    except Exception as e:
        st.error(f"시스템 초기화 실패: {str(e)}")
        logger.error(f"Initialization error: {str(e)}")
        return None, None, None, None

# ===== 메인 UI =====
def main():
    """메인 애플리케이션 함수"""
    
    # 헤더 표시
    st.markdown("""
    <div class="main-header">
        <h1>⚖️ 전략기획부 AI 법률 보조원</h1>
        <p>공정거래위원회 규정 및 매뉴얼 기반 통합 AI Q&A 시스템</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 모델 및 데이터 로드
    models = load_models_and_data()
    if not all(models):
        st.stop()
    
    embedding_model, reranker_model, index, chunks = models
    
    # RAG 파이프라인 초기화
    if 'rag_pipeline' not in st.session_state:
        st.session_state.rag_pipeline = OptimizedHybridRAGPipeline(
            embedding_model, reranker_model, index, chunks
        )
    
    rag = st.session_state.rag_pipeline
    
    # 대화 기록 초기화
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # 채팅 컨테이너
    chat_container = st.container()
    
    with chat_container:
        # 이전 대화 표시
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                if message["role"] == "user":
                    st.write(message["content"])
                else:
                    if isinstance(message["content"], dict):
                        st.write(message["content"]["answer"])
                        
                        # 메타 정보 표시
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            model_used = message["content"].get("model_used", "unknown")
                            model_emoji = {
                                'gpt-4o-mini': '🟢',
                                'o4-mini': '🟡',
                                'gpt-4o': '🔵'
                            }.get(model_used, '⚪')
                            st.caption(f"{model_emoji} {model_used}")
                        
                        with col2:
                            cost = message["content"].get("cost", 0)
                            if cost < 0.01:
                                cost_class = "cost-saved"
                            elif cost < 0.05:
                                cost_class = "cost-normal"
                            else:
                                cost_class = "cost-high"
                            st.caption(f'<span class="cost-efficiency {cost_class}">${cost:.4f}</span>', 
                                     unsafe_allow_html=True)
                        
                        with col3:
                            total_time = message["content"].get("total_time", 0)
                            st.caption(f"⏱️ {total_time:.1f}초")
                    else:
                        st.write(message["content"])
        
        # 사용자 입력
        if prompt := st.chat_input("질문을 입력하세요 (예: 대규모내부거래 공시 기한은?)"):
            # 사용자 메시지 추가
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.write(prompt)
            
            # AI 응답
            with st.chat_message("assistant"):
                total_start_time = time.time()
                
                # 검색 수행
                with st.spinner("🔍 최적의 AI 모델을 선택하고 검색을 수행하는 중..."):
                    results, search_stats = run_async_in_streamlit(
                        rag.process_query(prompt, top_k=5)
                    )
                
                # 답변 생성
                with st.spinner("💭 답변을 생성하는 중..."):
                    answer, generation_stats = run_async_in_streamlit(
                        generate_answer(prompt, results, search_stats)
                    )
                
                # 답변 표시
                st.write(answer)
                
                # 통계 정보
                total_time = time.time() - total_start_time
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    model_used = search_stats.get('selected_model', 'unknown')
                    model_emoji = {
                        'gpt-4o-mini': '🟢',
                        'o4-mini': '🟡',
                        'gpt-4o': '🔵'
                    }.get(model_used, '⚪')
                    st.caption(f"{model_emoji} {model_used}")
                
                with col2:
                    cost = generation_stats.get('cost', 0)
                    if cost < 0.01:
                        cost_class = "cost-saved"
                    elif cost < 0.05:
                        cost_class = "cost-normal"
                    else:
                        cost_class = "cost-high"
                    st.caption(f'<span class="cost-efficiency {cost_class}">${cost:.4f}</span>', 
                             unsafe_allow_html=True)
                
                with col3:
                    st.caption(f"⏱️ {total_time:.1f}초")
                
                # 상세 정보 (접을 수 있음)
                with st.expander("🔍 상세 정보 보기"):
                    st.subheader("📊 처리 과정")
                    
                    # 복잡도 게이지
                    complexity_score = search_stats.get('selection_info', {}).get('complexity_score', 0)
                    fig = create_complexity_gauge(complexity_score)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 모델 선택 이유
                    st.info(f"**선택 이유**: {search_stats.get('selection_info', {}).get('reason', 'N/A')}")
                    
                    # 참고 자료
                    st.subheader("📚 참고 자료")
                    for i, result in enumerate(results[:3]):
                        st.caption(f"**{result.source}** - 페이지 {result.page}")
                        with st.container():
                            content_preview = result.content[:300] + "..." if len(result.content) > 300 else result.content
                            st.text(content_preview)
                
                # 응답 데이터 저장
                response_data = {
                    "answer": answer,
                    "model_used": model_used,
                    "cost": generation_stats.get('cost', 0),
                    "total_time": total_time,
                    "complexity_score": complexity_score
                }
                st.session_state.messages.append({"role": "assistant", "content": response_data})
    
    # 사이드바
    with st.sidebar:
        st.header("💰 비용 관리 대시보드")
        
        # 예산 현황
        budget_manager = BudgetManager()
        budget_status = budget_manager.get_current_status()
        
        # 파이 차트
        fig = create_budget_pie_chart(budget_status['used'], budget_status['remaining'])
        st.plotly_chart(fig, use_container_width=True)
        
        # 모델별 사용 통계
        if 'model_usage_stats' not in st.session_state:
            st.session_state.model_usage_stats = {
                'gpt-4o-mini': {'count': 0, 'total_cost': 0},
                'o4-mini': {'count': 0, 'total_cost': 0},
                'gpt-4o': {'count': 0, 'total_cost': 0}
            }
        
        st.subheader("📊 모델별 사용 현황")
        stats_df = pd.DataFrame(st.session_state.model_usage_stats).T
        if not stats_df.empty:
            stats_df.columns = ['사용 횟수', '총 비용($)']
            st.dataframe(stats_df)
        
        st.divider()
        
        # 예시 질문
        st.header("💡 예시 질문")
        
        st.subheader("🟢 간단한 질문")
        example_simple = [
            "대규모내부거래 공시 기한은?",
            "이사회 의결 금액 기준은?",
            "현황공시는 언제 해야 하나요?"
        ]
        for example in example_simple:
            if st.button(example, key=f"simple_{example}"):
                st.session_state.new_question = example
                st.rerun()
        
        st.subheader("🟡 표준 질문")
        example_standard = [
            "계열사와 자금거래 시 절차는?",
            "비상장사 주식 양도 시 필요한 서류는?",
            "대규모내부거래 면제 조건은?"
        ]
        for example in example_standard:
            if st.button(example, key=f"standard_{example}"):
                st.session_state.new_question = example
                st.rerun()
        
        st.subheader("🔵 복잡한 질문")
        example_complex = [
            "A회사가 B계열사에 자금을 대여하면서 동시에 C계열사의 지분을 취득하는 경우 적용되는 규제는?",
            "여러 계열사와 동시에 거래할 때 검토해야 할 사항들을 종합적으로 설명해주세요"
        ]
        for example in example_complex:
            if st.button(example, key=f"complex_{example}"):
                st.session_state.new_question = example
                st.rerun()
        
        st.divider()
        
        # 비용 절감 팁
        st.info("""
        💡 **비용 절감 팁**
        - 간단한 정의는 자동으로 저렴한 모델 사용
        - 유사한 질문은 캐시 활용
        - o4-mini가 대부분의 분석에 최적
        """)
    
    # 페이지 하단
    st.divider()
    st.caption("⚠️ 본 답변은 AI가 생성한 참고자료입니다. 중요한 사항은 반드시 원문을 확인하시기 바랍니다.")
    
    # 새 질문 처리
    if "new_question" in st.session_state:
        # 입력창에 질문 설정하는 방법이 streamlit에서는 직접적으로 불가능하므로
        # 메시지에 추가하고 rerun
        st.session_state.messages.append({"role": "user", "content": st.session_state.new_question})
        del st.session_state.new_question
        st.rerun()

if __name__ == "__main__":
    main()
