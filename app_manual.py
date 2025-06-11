# 파일 이름: app_manual.py (공정거래위원회 AI 법률 보조원 - 완전 수정 버전)

import streamlit as st
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
import json
import openai
import numpy as np
from typing import List, Dict, Tuple, Optional, Set, TypedDict, Protocol, Iterator, Generator, OrderedDict, Any, Union
import re
from collections import defaultdict, Counter
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

# ===== 에러 컨텍스트 매니저 =====
@contextmanager
def error_context(operation_name: str, fallback_value=None):
    """에러 처리를 위한 컨텍스트 매니저"""
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
</style>
""", unsafe_allow_html=True)

# OpenAI API 설정
try:
    openai.api_key = st.secrets["OPENAI_API_KEY"]
except:
    st.error("⚠️ OpenAI API 키를 설정해주세요.")
    st.stop()

# ===== 타입 정의 =====
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

class QueryComplexity(Enum):
    """질문 복잡도 레벨"""
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"

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
            
        self.cache.move_to_end(key)
        return value
        
    def put(self, key: str, value):
        """캐시에 값 저장"""
        if key in self.cache:
            del self.cache[key]
            
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
    """Streamlit 환경에서 비동기 함수를 안전하게 실행"""
    try:
        loop = asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, coro)
            return future.result()
    except RuntimeError:
        return asyncio.run(coro)

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
                {'date': '2022-07-01', 'old_value': '7일', 'new_value': '5일',
                 'description': '이사회 의결 후 공시 기한 단축'}
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
        
        if 'document_date' in metadata:
            return metadata['document_date']
        
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
        date_str = re.sub(r'[^\d]', '-', date_str)
        parts = date_str.split('-')
        if len(parts) >= 2:
            year = parts[0] if len(parts[0]) == 4 else '20' + parts[0]
            month = parts[1].zfill(2)
            day = parts[2].zfill(2) if len(parts) > 2 else '01'
            return f"{year}-{month}-{day}"
        return None
    
    def check_for_outdated_info(self, content: str, document_date: str = None) -> List[Dict]:
        """구버전 정보가 포함되어 있는지 확인"""
        warnings = []
        
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
        
        return warnings

# ===== 충돌 해결 시스템 =====
class ConflictResolver:
    """상충하는 정보를 해결하는 시스템"""
    
    def __init__(self, version_manager: DocumentVersionManager):
        self.version_manager = version_manager
    
    def resolve_conflicts(self, results: List[SearchResult], query: str) -> List[SearchResult]:
        """검색 결과 중 상충하는 정보를 해결하고 최신 정보를 우선시"""
        
        for result in results:
            doc_date = result.document_date
            warnings = self.version_manager.check_for_outdated_info(result.content, doc_date)
            
            if warnings:
                result.metadata['warnings'] = warnings
                result.metadata['has_outdated_info'] = True
            else:
                result.metadata['has_outdated_info'] = False
        
        critical_info = self._extract_critical_info(results, query)
        if critical_info:
            conflicts = self._find_conflicts(critical_info)
            if conflicts:
                results = self._prioritize_latest_info(results, conflicts)
        
        results.sort(key=lambda r: (
            not r.metadata.get('has_outdated_info', False),
            r.document_date or '1900-01-01',
            r.score
        ), reverse=True)
        
        return results
    
    def _extract_critical_info(self, results: List[SearchResult], query: str) -> Dict:
        """결과에서 중요 정보 추출"""
        critical_info = defaultdict(list)
        
        for i, result in enumerate(results):
            amounts = re.findall(r'(\d+)억\s*원', result.content)
            for amount in amounts:
                critical_info['amounts'].append({
                    'value': amount + '억원',
                    'result_index': i,
                    'context': result.content[:100]
                })
            
            percentages = re.findall(r'(\d+(?:\.\d+)?)\s*%', result.content)
            for pct in percentages:
                critical_info['percentages'].append({
                    'value': pct + '%',
                    'result_index': i,
                    'context': result.content[:100]
                })
        
        return dict(critical_info)
    
    def _find_conflicts(self, critical_info: Dict) -> List[Dict]:
        """중요 정보 간 충돌 찾기"""
        conflicts = []
        
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
        
        return conflicts
    
    def _prioritize_latest_info(self, results: List[SearchResult], conflicts: List[Dict]) -> List[SearchResult]:
        """충돌이 있을 때 최신 정보를 우선시"""
        for conflict in conflicts:
            if conflict['type'] == 'amount_conflict':
                for i, result in enumerate(results):
                    if any(old_val in result.content for old_val in ['50억원', '30억원']):
                        results[i].score *= 0.5
                        results[i].metadata['score_reduced'] = True
                        results[i].metadata['reduction_reason'] = 'outdated_amount'
        
        return results

# ===== GPT-4o 질문 분석기 =====
class GPT4oQueryAnalyzer:
    """GPT-4o를 활용한 통합 질문 분석 및 검색 전략 수립"""
    
    def __init__(self):
        self.analysis_cache = LRUCache(max_size=100, ttl=3600)
    
    def analyze_and_strategize(self, query: str, available_chunks_info: Dict) -> Dict:
        """GPT-4o로 질문을 분석하고 최적의 검색 전략 수립"""
        
        cache_data = f"{query}_{json.dumps(available_chunks_info, sort_keys=True)}"
        cache_key = hashlib.md5(cache_data.encode()).hexdigest()
        
        cached_analysis = self.analysis_cache.get(cache_key)
        if cached_analysis:
            logger.info(f"Cache hit for query analysis: {query[:50]}...")
            return cached_analysis
        
        prompt = f"""
        당신은 공정거래법 전문가입니다. 다음 질문을 분석하고 최적의 검색 전략을 수립해주세요.
        
        질문: {query}
        
        사용 가능한 문서 정보:
        - 대규모내부거래 매뉴얼: {available_chunks_info.get('대규모내부거래', 0)}개 청크
        - 현황공시 매뉴얼: {available_chunks_info.get('현황공시', 0)}개 청크
        - 비상장사 중요사항 매뉴얼: {available_chunks_info.get('비상장사 중요사항', 0)}개 청크
        
        다음 형식의 JSON으로 응답해주세요:
        {{
            "query_analysis": {{
                "core_intent": "질문의 핵심 의도 (한 문장)",
                "actual_complexity": "simple/medium/complex",
                "complexity_reason": "복잡도 판단 이유"
            }},
            "legal_concepts": [
                {{
                    "concept": "대규모내부거래/현황공시/비상장사 중요사항",
                    "relevance": "primary/secondary",
                    "specific_aspects": ["금액기준", "절차", "공시의무"]
                }}
            ],
            "search_strategy": {{
                "approach": "direct_lookup/focused_search/comprehensive_analysis",
                "primary_manual": "주로 검색할 매뉴얼",
                "search_keywords": ["핵심 검색어1", "핵심 검색어2"],
                "expected_chunks_needed": 10,
                "rationale": "이 전략을 선택한 이유"
            }},
            "answer_requirements": {{
                "needs_specific_numbers": true,
                "needs_process_steps": false,
                "needs_timeline": false,
                "needs_exceptions": false,
                "needs_multiple_perspectives": false
            }}
        }}
        """
        
        try:
            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=1000,
                response_format={"type": "json_object"}
            )
            
            analysis = json.loads(response.choices[0].message.content)
            
            self.analysis_cache.put(cache_key, analysis)
            
            if len(self.analysis_cache.cache) % 10 == 0:
                self.analysis_cache.clear_expired()
            
            return analysis
            
        except Exception as e:
            logger.error(f"GPT-4o analysis error: {str(e)}")
            return self._get_fallback_strategy(query)
    
    def _get_fallback_strategy(self, query: str) -> Dict:
        """GPT 분석 실패 시 기본 전략"""
        return {
            "query_analysis": {
                "core_intent": "질문 분석 실패 - 기본 검색 수행",
                "actual_complexity": "medium",
                "complexity_reason": "자동 분석 실패로 중간 복잡도 가정"
            },
            "legal_concepts": [],
            "search_strategy": {
                "approach": "focused_search",
                "primary_manual": "대규모내부거래",
                "search_keywords": query.split()[:5],
                "expected_chunks_needed": 30,
                "rationale": "기본 검색 전략"
            },
            "answer_requirements": {
                "needs_specific_numbers": True,
                "needs_process_steps": True,
                "needs_timeline": True,
                "needs_exceptions": False,
                "needs_multiple_perspectives": False
            }
        }

# ===== 질문 분류기 =====
class QuestionClassifier:
    """질문을 분석하여 어떤 매뉴얼을 우선 검색할지 결정"""
    
    def __init__(self):
        self.categories = {
            '대규모내부거래': {
                'keywords': ['대규모내부거래', '내부거래', '이사회 의결', '이사회', '의결', 
                           '계열사', '계열회사', '특수관계인', '자금', '대여', '차입', '보증',
                           '자금거래', '유가증권', '자산거래', '50억', '거래금액'],
                'patterns': [r'이사회.*의결', r'계열.*거래', r'내부.*거래'],
                'manual_pattern': '대규모내부거래.*매뉴얼',
                'priority': 1
            },
            '현황공시': {
                'keywords': ['현황공시', '기업집단', '소속회사', '동일인', '친족', 
                           '지분율', '임원', '순환출자', '상호출자', '지배구조',
                           '계열편입', '계열제외', '주주현황', '임원현황'],
                'patterns': [r'기업집단.*현황', r'소속.*회사', r'지분.*변동'],
                'manual_pattern': '기업집단현황공시.*매뉴얼',
                'priority': 2
            },
            '비상장사 중요사항': {
                'keywords': ['비상장', '중요사항', '주식', '양도', '양수', '합병', 
                           '분할', '영업양도', '임원변경', '증자', '감자',
                           '정관변경', '해산', '청산'],
                'patterns': [r'비상장.*공시', r'주식.*양도', r'중요.*사항'],
                'manual_pattern': '비상장사.*중요사항.*매뉴얼',
                'priority': 3
            }
        }
    
    def classify(self, question: str) -> Tuple[str, float]:
        """질문을 분류하고 신뢰도를 반환"""
        question_lower = question.lower()
        scores = {}
        
        for category, info in self.categories.items():
            score = 0
            matched_keywords = []
            
            for i, keyword in enumerate(info['keywords']):
                if keyword in question_lower:
                    weight = 1.0 if i < 5 else 0.7
                    score += weight
                    matched_keywords.append(keyword)
            
            for pattern in info.get('patterns', []):
                if re.search(pattern, question_lower):
                    score += 1.5
            
            scores[category] = score
        
        if scores:
            best_category = max(scores, key=scores.get)
            max_possible_score = len(self.categories[best_category]['keywords']) + \
                               len(self.categories[best_category].get('patterns', [])) * 1.5
            confidence = min(scores[best_category] / max_possible_score, 1.0)
            
            if confidence < 0.15:
                return None, 0.0
                
            return best_category, confidence
        
        return None, 0.0

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
        
        if len(query) > 150:
            complex_score += 2
        elif len(query) > 100:
            complex_score += 1
        elif len(query) < 30:
            simple_score += 0.5
            
        if re.search(r'[AB]회사.*[CD]회사', query_lower):
            complex_score += 2
        if '?' in query and query.count('?') > 1:
            complex_score += 1
            
        total_score = simple_score + medium_score + complex_score
        
        if total_score == 0:
            complexity = QueryComplexity.MEDIUM
            confidence = 0.5
        elif complex_score > simple_score * 3:
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
            'estimated_cost_multiplier': self._estimate_cost_multiplier(complexity)
        }
        
        return complexity, confidence, analysis
    
    def _estimate_cost_multiplier(self, complexity: QueryComplexity) -> float:
        """복잡도에 따른 예상 비용 배수"""
        multipliers = {
            QueryComplexity.SIMPLE: 1.0,
            QueryComplexity.MEDIUM: 3.0,
            QueryComplexity.COMPLEX: 10.0
        }
        return multipliers[complexity]

# ===== 하이브리드 RAG 파이프라인 =====
class HybridRAGPipeline:
    """GPT-4o 기반 하이브리드 파이프라인"""
    
    def __init__(self, embedding_model, reranker_model, index, chunks):
        if not chunks:
            raise ValueError("No chunks provided to HybridRAGPipeline")
        
        if index.ntotal == 0:
            raise ValueError("FAISS index is empty")
        
        # 임베딩 차원 검증
        test_embedding = embedding_model.encode(["test"])
        if len(test_embedding[0]) != index.d:
            raise ValueError(f"Embedding dimension {len(test_embedding[0])} doesn't match index dimension {index.d}")
        
        self.embedding_model = embedding_model
        self.reranker_model = reranker_model
        self.index = index
        self.chunks = chunks
        
        self.classifier = QuestionClassifier()
        self.complexity_assessor = ComplexityAssessor()
        self.gpt4o_analyzer = GPT4oQueryAnalyzer()
        
        self.version_manager = DocumentVersionManager()
        self.conflict_resolver = ConflictResolver(self.version_manager)
        
        self.manual_indices = self._build_manual_indices()
        
        self.chunks_info = {
            category: len(indices) 
            for category, indices in self.manual_indices.items()
        }
        
        self.search_cache = LRUCache(max_size=50, ttl=1800)
        self.embedding_cache = LRUCache(max_size=100, ttl=3600)
        
        self._extract_chunk_dates()
        
        logger.info(f"HybridRAGPipeline initialized with {len(chunks)} chunks")
    
    def _extract_chunk_dates(self):
        """모든 청크의 날짜 정보를 미리 추출"""
        for chunk in self.chunks:
            doc_date = self.version_manager.extract_document_date(chunk)
            if doc_date:
                metadata = json.loads(chunk.get('metadata', '{}'))
                metadata['document_date'] = doc_date
                chunk['metadata'] = json.dumps(metadata)
    
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
    
    def _get_query_embedding(self, query: str) -> np.ndarray:
        """쿼리 임베딩을 캐시와 함께 가져오기"""
        cached_embedding = self.embedding_cache.get(query)
        if cached_embedding is not None:
            return cached_embedding
            
        embedding = self.embedding_model.encode([query])
        embedding = np.array(embedding, dtype=np.float32)
        
        self.embedding_cache.put(query, embedding)
        
        return embedding
    
    async def process_query(self, query: str, top_k: int = 5) -> Tuple[List[SearchResult], Dict]:
        """GPT-4o가 질문을 분석하여 최적의 처리 방식을 선택"""
        start_time = time.time()
        
        analysis_start = time.time()
        try:
            gpt_analysis = self.gpt4o_analyzer.analyze_and_strategize(
                query, self.chunks_info
            )
            analysis_time = time.time() - analysis_start
        except Exception as e:
            logger.error(f"GPT-4o analysis failed: {str(e)}, falling back to rule-based")
            return self._fallback_process_query(query, top_k)
        
        actual_complexity = gpt_analysis['query_analysis']['actual_complexity']
        search_approach = gpt_analysis['search_strategy']['approach']
        
        stats = {
            'gpt_analysis': gpt_analysis,
            'analysis_time': analysis_time,
            'actual_complexity': actual_complexity,
            'search_approach': search_approach
        }
        
        if search_approach == 'direct_lookup':
            results, search_stats = await self._gpt_guided_direct_search(
                query, gpt_analysis, top_k
            )
            stats['processing_mode'] = 'gpt_guided_direct'
            
        elif search_approach == 'focused_search':
            results, search_stats = await self._gpt_guided_focused_search(
                query, gpt_analysis, top_k
            )
            stats['processing_mode'] = 'gpt_guided_focused'
            
        else:  # comprehensive_analysis
            results, search_stats = await self._gpt_guided_comprehensive_search(
                query, gpt_analysis, top_k
            )
            stats['processing_mode'] = 'gpt_guided_comprehensive'
        
        results = self.conflict_resolver.resolve_conflicts(results, query)
        
        outdated_warnings = []
        for result in results:
            if result.metadata.get('has_outdated_info'):
                outdated_warnings.extend(result.metadata.get('warnings', []))
        
        stats.update(search_stats)
        stats['total_time'] = time.time() - start_time
        stats['outdated_warnings'] = outdated_warnings
        stats['has_version_conflicts'] = len(outdated_warnings) > 0
        
        return results, stats
    
    async def _gpt_guided_direct_search(self, query: str, gpt_analysis: Dict, 
                                       top_k: int) -> Tuple[List[SearchResult], Dict]:
        """GPT 분석을 기반으로 한 직접 검색"""
        start_time = time.time()
        
        primary_manual = gpt_analysis['search_strategy']['primary_manual']
        search_keywords = gpt_analysis['search_strategy']['search_keywords']
        
        target_indices = self.manual_indices.get(primary_manual, [])[:100]
        
        if not target_indices:
            logger.warning(f"No indices for manual '{primary_manual}', using all chunks")
            target_indices = list(range(min(len(self.chunks), 50)))
            if not target_indices:
                return [], {
                    'search_time': time.time() - start_time,
                    'searched_chunks': 0,
                    'search_method': 'direct_vector',
                    'warning': 'No chunks available'
                }
        
        enhanced_query = f"{query} {' '.join(search_keywords)}"
        query_vector = self._get_query_embedding(enhanced_query)
        
        k_search = min(len(target_indices), max(1, top_k * 3))
        
        try:
            scores, indices = self.index.search(query_vector, k_search)
        except Exception as e:
            logger.error(f"FAISS search error in direct search: {str(e)}")
            return [], {
                'search_time': time.time() - start_time,
                'searched_chunks': len(target_indices),
                'search_method': 'direct_vector',
                'error': str(e)
            }
        
        results = []
        target_set = set(target_indices)
        
        for idx, score in zip(indices[0], scores[0]):
            if idx in target_set and idx < len(self.chunks):
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
        
        stats = {
            'search_time': time.time() - start_time,
            'searched_chunks': len(target_indices),
            'search_method': 'direct_vector'
        }
        
        return results, stats
    
    async def _gpt_guided_focused_search(self, query: str, gpt_analysis: Dict, 
                                        top_k: int) -> Tuple[List[SearchResult], Dict]:
        """GPT 분석을 기반으로 한 집중 검색 (완전히 수정된 버전)"""
        start_time = time.time()
        stats = {
            'search_time': 0,
            'searched_chunks': 0,
            'search_method': 'focused_vector',
            'errors': []
        }
        
        try:
            if not query or not isinstance(query, str):
                raise ValueError(f"Invalid query: {query}")
            if not isinstance(top_k, int) or top_k <= 0:
                raise ValueError(f"Invalid top_k value: {top_k}")
                
            if not gpt_analysis or 'search_strategy' not in gpt_analysis:
                logger.warning("Invalid or missing GPT analysis, using defaults")
                gpt_analysis = self._get_default_analysis(query)
                
            primary_manual = gpt_analysis['search_strategy'].get('primary_manual', '')
            search_keywords = gpt_analysis['search_strategy'].get('search_keywords', [])
            expected_chunks = gpt_analysis['search_strategy'].get('expected_chunks_needed', 10)
            
            logger.info(f"Search strategy - Manual: {primary_manual}, Keywords: {search_keywords}")
            
            search_limit = min(expected_chunks * 2, 200)
            target_indices = self.manual_indices.get(primary_manual, [])[:search_limit]
            
            if not target_indices:
                logger.warning(f"No target indices found for manual '{primary_manual}'")
                stats['errors'].append({
                    'type': 'missing_indices',
                    'manual': primary_manual,
                    'action': 'fallback_to_all_chunks'
                })
                
                target_indices = list(range(min(len(self.chunks), 100)))
                if not target_indices:
                    logger.error("No chunks available for search")
                    return [], stats
            
            requirements = gpt_analysis.get('answer_requirements', {})
            if requirements.get('needs_specific_numbers'):
                try:
                    filtered_indices = [
                        idx for idx in target_indices 
                        if idx < len(self.chunks) and
                        re.search(r'\d+억|\d+%', self.chunks[idx].get('content', ''))
                    ]
                    if filtered_indices:
                        target_indices = filtered_indices
                        logger.debug(f"Filtered to {len(filtered_indices)} chunks with numbers")
                except Exception as e:
                    logger.warning(f"Error during number filtering: {str(e)}")
                    stats['errors'].append({
                        'type': 'filter_error',
                        'error': str(e)
                    })
            
            try:
                enhanced_query = f"{query} {' '.join(search_keywords)}"
                
                with error_context("Creating query embedding"):
                    query_vector = self._get_query_embedding(enhanced_query)
                    
                k_search = min(len(target_indices), max(1, top_k * 5))
                logger.debug(f"Performing FAISS search with k={k_search}")
                
                scores, indices = self.index.search(query_vector, k_search)
                
            except Exception as e:
                logger.error(f"FAISS search error: {str(e)}")
                stats['errors'].append({
                    'type': 'faiss_error',
                    'error': str(e),
                    'k_search': k_search,
                    'target_indices_count': len(target_indices)
                })
                return [], stats
            
            results = []
            target_set = set(target_indices)
            invalid_results_count = 0
            
            for idx, score in zip(indices[0], scores[0]):
                try:
                    if idx < 0 or idx >= len(self.chunks):
                        invalid_results_count += 1
                        continue
                        
                    if idx not in target_set:
                        continue
                        
                    chunk = self.chunks[idx]
                    
                    if not chunk or 'content' not in chunk:
                        logger.warning(f"Invalid chunk at index {idx}")
                        invalid_results_count += 1
                        continue
                    
                    relevance_boost = self._calculate_gpt_relevance(
                        chunk['content'], gpt_analysis
                    )
                    
                    result = SearchResult(
                        chunk_id=chunk.get('chunk_id', str(idx)),
                        content=chunk.get('content', ''),
                        score=float(score) * (1 + relevance_boost),
                        source=chunk.get('source', 'Unknown'),
                        page=chunk.get('page', 0),
                        chunk_type=chunk.get('chunk_type', 'unknown'),
                        metadata=json.loads(chunk.get('metadata', '{}'))
                    )
                    
                    results.append(result)
                    
                    if len(results) >= top_k * 2:
                        break
                        
                except Exception as e:
                    logger.error(f"Error processing result at index {idx}: {str(e)}")
                    invalid_results_count += 1
                    continue
            
            if invalid_results_count > 0:
                logger.warning(f"Found {invalid_results_count} invalid results during processing")
                stats['errors'].append({
                    'type': 'invalid_results',
                    'count': invalid_results_count
                })
            
            results.sort(key=lambda x: x.score, reverse=True)
            results = results[:top_k]
            
            stats.update({
                'search_time': time.time() - start_time,
                'searched_chunks': len(target_indices),
                'results_count': len(results),
                'has_errors': len(stats['errors']) > 0
            })
            
            logger.info(f"Search completed in {stats['search_time']:.2f}s, found {len(results)} results")
            
            return results, stats
            
        except Exception as e:
            logger.error(f"Unexpected error in focused search: {str(e)}")
            logger.debug(f"Full traceback:\n{traceback.format_exc()}")
            
            stats['errors'].append({
                'type': 'unexpected_error',
                'error': str(e),
                'traceback': traceback.format_exc()
            })
            stats['search_time'] = time.time() - start_time
            
            return [], stats
    
    async def _gpt_guided_comprehensive_search(self, query: str, gpt_analysis: Dict, 
                                              top_k: int) -> Tuple[List[SearchResult], Dict]:
        """GPT 분석을 기반으로 한 종합 검색"""
        start_time = time.time()
        
        all_results = []
        
        for concept in gpt_analysis['legal_concepts']:
            if concept['relevance'] in ['primary', 'secondary']:
                manual = concept['concept']
                if manual in self.manual_indices:
                    partial_results = await self._search_in_manual(
                        query, manual, concept['specific_aspects'], top_k // 2
                    )
                    all_results.extend(partial_results)
        
        seen_chunks = set()
        unique_results = []
        for result in sorted(all_results, key=lambda x: x.score, reverse=True):
            if result.chunk_id not in seen_chunks:
                seen_chunks.add(result.chunk_id)
                unique_results.append(result)
                if len(unique_results) >= top_k:
                    break
        
        stats = {
            'search_time': time.time() - start_time,
            'searched_chunks': sum(len(self.manual_indices.get(c['concept'], [])) 
                                 for c in gpt_analysis['legal_concepts'] 
                                 if c['relevance'] in ['primary', 'secondary']),
            'search_method': 'comprehensive_multi_manual'
        }
        
        return unique_results, stats
    
    def _calculate_gpt_relevance(self, content: str, gpt_analysis: Dict) -> float:
        """GPT 분석 결과와 청크 내용의 관련성 계산"""
        relevance_boost = 0.0
        content_lower = content.lower()
        
        for keyword in gpt_analysis['search_strategy']['search_keywords']:
            if keyword.lower() in content_lower:
                relevance_boost += 0.1
        
        requirements = gpt_analysis['answer_requirements']
        if requirements.get('needs_specific_numbers') and re.search(r'\d+억|\d+%', content):
            relevance_boost += 0.2
        if requirements.get('needs_timeline') and re.search(r'\d+일|기한', content):
            relevance_boost += 0.2
        if requirements.get('needs_process_steps') and re.search(r'절차|단계|순서', content):
            relevance_boost += 0.15
        
        return min(relevance_boost, 0.5)
    
    async def _search_in_manual(self, query: str, manual: str, aspects: List[str], 
                               limit: int) -> List[SearchResult]:
        """특정 매뉴얼 내에서 검색"""
        indices = self.manual_indices.get(manual, [])[:100]
        
        if not indices:
            logger.warning(f"No indices found for manual '{manual}'")
            return []
        
        enhanced_query = f"{query} {' '.join(aspects)}"
        query_vector = self._get_query_embedding(enhanced_query)
        
        k_search = min(len(indices), max(1, limit * 3))
        
        try:
            scores, search_indices = self.index.search(query_vector, k_search)
        except Exception as e:
            logger.error(f"FAISS search error in manual search: {str(e)}")
            return []
        
        results = []
        indices_set = set(indices)
        
        for idx, score in zip(search_indices[0], scores[0]):
            if idx in indices_set and idx < len(self.chunks):
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
                if len(results) >= limit:
                    break
        
        return results
    
    def _fallback_process_query(self, query: str, top_k: int) -> Tuple[List[SearchResult], Dict]:
        """GPT 분석 실패 시 기존 방식으로 폴백"""
        results, stats = self._fast_traditional_search(query, top_k)
        stats['processing_mode'] = 'fallback_traditional'
        stats['gpt_failure'] = True
        return results, stats
    
    def _fast_traditional_search(self, query: str, top_k: int) -> Tuple[List[SearchResult], Dict]:
        """기존의 빠른 벡터 검색 방식"""
        start_time = time.time()
        
        cache_key = hashlib.md5(f"{query}_{top_k}_traditional".encode()).hexdigest()
        cached = self.search_cache.get(cache_key)
        if cached:
            stats = cached['stats'].copy()
            stats['cache_hit'] = True
            return cached['results'], stats
        
        category, cat_confidence = self.classifier.classify(query)
        
        if category and cat_confidence > 0.3:
            primary_indices = self.manual_indices.get(category, [])
            if len(primary_indices) > 200:
                primary_indices = primary_indices[:200]
            secondary_indices = []
        else:
            primary_indices = list(range(min(len(self.chunks), 300)))
            secondary_indices = []
        
        query_vector = self._get_query_embedding(query)
        
        k_search = min(len(primary_indices), max(1, top_k * 5))
        scores, indices = self.index.search(query_vector, k_search)
        
        results = []
        seen_chunks = set()
        
        if primary_indices:
            primary_set = set(primary_indices)
            for idx, score in zip(indices[0], scores[0]):
                if idx in primary_set and idx not in seen_chunks and idx < len(self.chunks):
                    seen_chunks.add(idx)
                    chunk = self.chunks[idx]
                    result = SearchResult(
                        chunk_id=chunk.get('chunk_id', str(idx)),
                        content=chunk['content'],
                        score=float(score),
                        source=chunk['source'],
                        page=chunk['page'],
                        chunk_type=chunk.get('chunk_type', 'unknown'),
                        metadata=json.loads(chunk.get('metadata', '{}'))
                    )
                    results.append(result)
                    if len(results) >= top_k:
                        break
        
        stats = {
            'search_time': time.time() - start_time,
            'category': category,
            'category_confidence': cat_confidence,
            'cache_hit': False,
            'searched_chunks': len(primary_indices)
        }
        
        if stats['search_time'] < 0.5 and len(self.search_cache.cache) < self.search_cache.max_size:
            self.search_cache.put(cache_key, {
                'results': results,
                'stats': stats,
                'timestamp': time.time()
            })
        
        return results, stats
    
    def _get_default_analysis(self, query: str) -> Dict:
        """GPT 분석 실패 시 사용할 기본 분석 생성"""
        return {
            'query_analysis': {
                'core_intent': query,
                'actual_complexity': 'medium',
                'complexity_reason': 'Default analysis due to GPT failure'
            },
            'search_strategy': {
                'primary_manual': '대규모내부거래',
                'search_keywords': query.split()[:5],
                'expected_chunks_needed': 20,
                'approach': 'focused_search'
            },
            'answer_requirements': {
                'needs_specific_numbers': True,
                'needs_process_steps': True
            }
        }

# ===== 답변 생성 함수 =====
def determine_temperature(query: str, complexity: str) -> float:
    """질문 유형과 복잡도에 따라 최적의 temperature 결정"""
    query_lower = query.lower()
    
    base_temps = {
        'simple': 0.1,
        'medium': 0.3,
        'complex': 0.5
    }
    
    temp = base_temps.get(complexity, 0.3)
    
    if any(keyword in query_lower for keyword in ['언제', '며칠', '기한', '날짜', '금액', '%']):
        temp = min(temp, 0.1)
    elif any(keyword in query_lower for keyword in ['전략', '대응', '리스크', '주의', '권장']):
        temp = max(temp, 0.7)
    
    return temp

def generate_answer(query: str, results: List[SearchResult], stats: Dict) -> str:
    """GPT-4o를 활용한 고품질 답변 생성"""
    
    has_outdated = stats.get('has_version_conflicts', False)
    outdated_warnings = stats.get('outdated_warnings', [])
    
    context_parts = []
    latest_info_parts = []
    outdated_info_parts = []
    
    for i, result in enumerate(results[:5]):
        context_str = f"""
[참고 {i+1}] {result.source} (페이지 {result.page})
{result.content}
"""
        if result.metadata.get('has_outdated_info'):
            outdated_info_parts.append(context_str)
        else:
            latest_info_parts.append(context_str)
    
    context_parts = latest_info_parts + outdated_info_parts
    context = "\n---\n".join(context_parts)
    
    critical_updates = ""
    if has_outdated:
        critical_updates = "\n\n[중요 법규 변경사항]"
        for warning in outdated_warnings:
            if warning['severity'] == 'critical':
                critical_updates += f"\n- {warning['regulation']}: {warning['found']} → {warning['current']} (변경일: {warning['changed_date']})"
    
    gpt_analysis = stats.get('gpt_analysis', {})
    complexity = gpt_analysis.get('query_analysis', {}).get('actual_complexity', 'medium')
    temperature = determine_temperature(query, complexity)
    
    mode_instructions = {
        'gpt_guided_direct': "GPT-4o가 선택한 직접 검색 결과를 바탕으로 간결하고 정확한 답변을 제공하세요.",
        'gpt_guided_focused': "GPT-4o가 분석한 핵심 주제에 대해 상세하고 실무적인 답변을 제공하세요.",
        'gpt_guided_comprehensive': "GPT-4o가 파악한 여러 관련 주제를 종합하여 포괄적인 답변을 제공하세요.",
        'fallback_traditional': "제공된 참고 자료를 바탕으로 간결하고 정확한 답변을 제공하세요."
    }
    
    mode = stats.get('processing_mode', 'fallback_traditional')
    extra_instruction = mode_instructions.get(mode, "")
    
    category = stats.get('category')
    if not category and gpt_analysis:
        primary_manual = gpt_analysis.get('search_strategy', {}).get('primary_manual')
        category = primary_manual
    
    if category:
        category_instructions = {
            '대규모내부거래': "특히 이사회 의결 요건, 공시 기한, 면제 조건을 명확히 설명하세요. 금액 기준은 반드시 최신 기준(100억원 이상 또는 자본금 및 자본총계 중 큰 금액의 5% 이상)을 사용하세요.",
            '현황공시': "공시 주체, 시기, 제출 서류를 구체적으로 안내하세요.",
            '비상장사 중요사항': "공시 대상 거래, 기한, 제출 방법을 상세히 설명하세요."
        }
        extra_instruction += f"\n{category_instructions.get(category, '')}"
    
    system_prompt = f"""당신은 한국 공정거래위원회 전문가입니다.
제공된 자료만을 근거로 정확하고 실무적인 답변을 제공하세요.

질문 복잡도: {complexity}
처리 방식: {mode}

중요: 법규가 변경된 경우 반드시 최신 정보를 기준으로 답변하세요. 
특히 대규모내부거래 금액 기준은 2023년부터 100억원 이상으로 변경되었습니다.

답변은 다음 구조를 따라주세요:
1. 핵심 답변 (1-2문장) - 최신 법규 기준
2. 상세 설명 (근거 조항 포함)
3. 주의사항 또는 예외사항 (있는 경우)
4. 법규 변경사항 (중요한 변경이 있었던 경우)

{extra_instruction}"""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"""다음 자료를 바탕으로 질문에 답변해주세요.
{critical_updates}

[참고 자료]
{context}

[질문]
{query}

{"간결하고 명확하게" if complexity == 'simple' else "상세하고 실무적으로"} 답변해주세요.
구버전 정보와 최신 정보가 상충하는 경우, 반드시 최신 정보를 기준으로 답변하세요."""}
    ]
    
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=temperature,
        max_tokens=1500
    )
    
    return response.choices[0].message.content

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
            index = faiss.read_index("manuals_vector_db.index")
            with open("all_manual_chunks.json", "r", encoding="utf-8") as f:
                chunks = json.load(f)
            
            try:
                embedding_model = SentenceTransformer('jhgan/ko-sroberta-multitask')
            except Exception as e:
                st.warning("한국어 모델 로드 실패. 대체 모델을 사용합니다.")
                embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
            
            try:
                reranker_model = CrossEncoder('Dongjin-kr/ko-reranker')
            except:
                reranker_model = None
        
        return embedding_model, reranker_model, index, chunks
        
    except Exception as e:
        st.error(f"시스템 초기화 실패: {str(e)}")
        return None, None, None, None

# ===== 메인 UI =====
def main():
    st.markdown("""
    <div class="main-header">
        <h1>⚖️ 전략기획부 AI 법률 보조원</h1>
        <p>공정거래위원회 규정 및 매뉴얼 기반 GPT-4o 통합 Q&A 시스템</p>
    </div>
    """, unsafe_allow_html=True)
    
    models = load_models_and_data()
    if not all(models):
        st.stop()
    
    embedding_model, reranker_model, index, chunks = models
    
    rag = HybridRAGPipeline(embedding_model, reranker_model, index, chunks)
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                if message["role"] == "user":
                    st.write(message["content"])
                else:
                    if isinstance(message["content"], dict):
                        st.write(message["content"]["answer"])
                        
                        complexity = message["content"].get("complexity", "unknown")
                        complexity_html = f'<span class="complexity-indicator complexity-{complexity}">{complexity.upper()}</span>'
                        st.markdown(f"처리 복잡도: {complexity_html}", unsafe_allow_html=True)
                        
                        if "total_time" in message["content"]:
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("🔍 검색", f"{message['content']['search_time']:.1f}초")
                            with col2:
                                st.metric("✍️ 답변 생성", f"{message['content']['generation_time']:.1f}초")
                            with col3:
                                st.metric("⏱️ 전체", f"{message['content']['total_time']:.1f}초")
                    else:
                        st.write(message["content"])
        
        if prompt := st.chat_input("질문을 입력하세요 (예: 대규모내부거래 공시 기한은?)"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)
            
            with st.chat_message("assistant"):
                total_start_time = time.time()
                
                search_start_time = time.time()
                with st.spinner("🔍 GPT-4o가 질문을 분석하고 최적의 검색 전략을 수립하는 중..."):
                    results, stats = run_async_in_streamlit(rag.process_query(prompt, top_k=5))
                search_time = time.time() - search_start_time
                
                generation_start_time = time.time()
                with st.spinner("💭 답변을 생성하는 중..."):
                    answer = generate_answer(prompt, results, stats)
                generation_time = time.time() - generation_start_time
                
                total_time = time.time() - total_start_time
                
                st.write(answer)
                
                gpt_analysis = stats.get('gpt_analysis', {})
                complexity = gpt_analysis.get('query_analysis', {}).get('actual_complexity', 'unknown')
                mode = stats.get('processing_mode', 'unknown')
                complexity_html = f'<span class="complexity-indicator complexity-{complexity}">{complexity.upper()}</span>'
                st.markdown(f"질문 복잡도: {complexity_html} | 처리 방식: **{mode}**", unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("🔍 검색", f"{search_time:.1f}초")
                with col2:
                    st.metric("✍️ 답변 생성", f"{generation_time:.1f}초")
                with col3:
                    st.metric("⏱️ 전체", f"{total_time:.1f}초")
                
                with st.expander("🔍 상세 정보 보기"):
                    if stats.get('has_version_conflicts'):
                        st.error("⚠️ **중요: 법규 변경사항 발견**")
                        for warning in stats.get('outdated_warnings', []):
                            if warning['severity'] == 'critical':
                                st.warning(f"""
                                📌 **{warning['regulation']}** 변경
                                - 이전: {warning['found']}
                                - 현재: **{warning['current']}** ✅
                                - 변경일: {warning['changed_date']}
                                """)
                        st.info("💡 본 시스템은 최신 법규를 기준으로 답변을 제공합니다.")
                    
                    if gpt_analysis:
                        st.subheader("🤖 GPT-4o 질문 분석")
                        st.json({
                            "핵심 의도": gpt_analysis.get('query_analysis', {}).get('core_intent', ''),
                            "실제 복잡도": gpt_analysis.get('query_analysis', {}).get('actual_complexity', ''),
                            "검색 전략": gpt_analysis.get('search_strategy', {}).get('approach', ''),
                            "주요 매뉴얼": gpt_analysis.get('search_strategy', {}).get('primary_manual', '')
                        })
                    
                    mode_descriptions = {
                        'gpt_guided_direct': "GPT-4o가 단순한 질문으로 판단하여 직접 검색을 수행했습니다.",
                        'gpt_guided_focused': "GPT-4o가 특정 주제에 대한 집중 검색을 수행했습니다.",
                        'gpt_guided_comprehensive': "GPT-4o가 여러 주제에 걸친 종합 분석을 수행했습니다.",
                        'fallback_traditional': "GPT-4o 분석이 실패하여 기존 방식으로 처리했습니다."
                    }
                    st.info(f"🎯 **처리 방식**: {mode_descriptions.get(mode, '알 수 없음')}")
                    
                    if stats.get('searched_chunks'):
                        st.info(f"🔍 {stats['searched_chunks']}개 문서를 검색했습니다.")
                    
                    st.subheader("📚 참고 자료")
                    for i, result in enumerate(results[:3]):
                        version_indicator = ""
                        if result.metadata.get('has_outdated_info'):
                            version_indicator = " ⚠️ **[구버전 정보 포함]**"
                        
                        st.caption(f"**{result.source}** - 페이지 {result.page} (관련도: {result.score:.2f}){version_indicator}")
                        
                        if result.document_date:
                            st.caption(f"📅 문서 날짜: {result.document_date}")
                        
                        with st.container():
                            content = result.content[:300] + "..." if len(result.content) > 300 else result.content
                            
                            if '50억원' in content or '30억원' in content:
                                content = re.sub(r'(50억원|30억원)', r'~~\1~~ → **100억원**', content)
                            
                            st.text(content)
                    
                    if total_time < 5:
                        st.success("⚡ 매우 빠른 응답 속도!")
                    elif total_time < 10:
                        st.info("✅ 적절한 응답 속도")
                    else:
                        st.warning("⏰ 응답 시간이 다소 길었습니다 (복잡한 질문으로 인한 정상적인 처리)")
                
                response_data = {
                    "answer": answer,
                    "search_time": search_time,
                    "generation_time": generation_time,
                    "total_time": total_time,
                    "complexity": complexity,
                    "processing_mode": mode
                }
                st.session_state.messages.append({"role": "assistant", "content": response_data})
    
    st.divider()
    st.caption("⚠️ 본 답변은 AI가 생성한 참고자료입니다. 중요한 사항은 반드시 원문을 확인하시기 바랍니다.")
    
    with st.sidebar:
        st.header("💡 예시 질문")
        
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
            
        st.subheader("🔴 복잡한 질문")
        if st.button("복합 거래 분석"):
            st.session_state.new_question = "A회사가 B계열사에 자금을 대여하면서 동시에 C계열사의 주식을 취득하는 경우, 각각 어떤 규제가 적용되고 공시는 어떻게 해야 하나요?"
            st.rerun()
        if st.button("종합적 리스크 검토"):
            st.session_state.new_question = "우리 회사가 여러 계열사와 동시에 거래를 진행할 때 대규모내부거래 규제와 관련하여 종합적으로 검토해야 할 리스크와 대응 전략은?"
            st.rerun()
        
        st.divider()
        st.caption("💡 GPT-4o가 모든 질문의 핵심을 파악하여 최적의 답변을 제공합니다.")
    
    if "new_question" in st.session_state:
        st.session_state.messages.append({"role": "user", "content": st.session_state.new_question})
        del st.session_state.new_question
        st.rerun()

if __name__ == "__main__":
    main()
