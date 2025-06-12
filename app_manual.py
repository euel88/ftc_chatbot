# 파일 이름: app_manual.py (공정거래위원회 AI 법률 보조원 - 개선된 버전)

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
from datetime import datetime, timedelta

# ===== 로깅 설정 =====
def setup_logging():
    """구조화된 로깅 시스템 설정
    
    로깅은 프로그램의 작동 상황을 기록하는 일기장과 같습니다.
    문제가 발생했을 때 원인을 찾거나, 성능을 개선하는 데 필수적입니다.
    """
    logger = logging.getLogger('ftc_chatbot')
    logger.setLevel(logging.DEBUG)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    )
    
    # 파일 핸들러 - 영구적인 로그 기록
    file_handler = logging.FileHandler('ftc_chatbot_debug.log')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    
    # 콘솔 핸들러 - 개발 중 실시간 확인
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
    """인덱스 관련 오류 - FAISS 인덱스 문제 시 발생"""
    pass

class EmbeddingError(RAGPipelineError):
    """임베딩 생성 관련 오류 - 벡터 변환 실패 시 발생"""
    pass

class GPTAnalysisError(RAGPipelineError):
    """GPT 분석 실패 관련 오류 - API 호출 문제 시 발생"""
    pass

class ModelSelectionError(RAGPipelineError):
    """모델 선택 관련 오류 - 적절한 모델을 찾을 수 없을 때 발생"""
    pass

# ===== 에러 컨텍스트 매니저 =====
@contextmanager
def error_context(operation_name: str, fallback_value=None):
    """에러 처리를 위한 컨텍스트 매니저
    
    이는 마치 작업장의 안전망과 같아서, 작업 중 문제가 발생해도
    전체 시스템이 멈추지 않고 적절히 대응할 수 있게 합니다.
    """
    try:
        logger.debug(f"Starting operation: {operation_name}")
        yield
    except Exception as e:
        logger.error(f"Error in {operation_name}: {str(e)}")
        logger.debug(f"Full traceback:\n{traceback.format_exc()}")
        
        # Streamlit UI에 에러 표시
        if 'st' in globals():
            st.error(f"⚠️ 작업 중 오류 발생: {operation_name}")
            with st.expander("🔍 상세 오류 정보"):
                st.code(traceback.format_exc())
        
        # 폴백 값이 있으면 반환, 없으면 예외 재발생
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

# CSS 스타일 - UI의 시각적 디자인을 정의
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
    
    /* 경고 메시지 스타일 */
    .version-warning {
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
        color: #856404;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    
    /* 중요도 표시 스타일 */
    .importance-indicator {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.8rem;
        margin-left: 4px;
    }
    
    .importance-critical {
        background-color: #dc3545;
        color: white;
    }
    
    .importance-high {
        background-color: #fd7e14;
        color: white;
    }
    
    .importance-normal {
        background-color: #6c757d;
        color: white;
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
    """청크 데이터의 타입 정의
    
    청크는 긴 문서를 작은 조각으로 나눈 것입니다.
    각 청크는 검색과 참조가 가능한 독립적인 정보 단위입니다.
    """
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
    """검색 결과를 담는 데이터 클래스
    
    이 클래스는 마치 도서관에서 찾은 책의 정보 카드와 같습니다.
    어떤 내용이 어디에 있는지, 얼마나 관련성이 높은지를 담고 있습니다.
    """
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

class QueryImportance(Enum):
    """질문 중요도 레벨 - 법적 리스크와 금액을 고려"""
    CRITICAL = "critical"  # 법적 리스크가 높거나 대규모 금액
    HIGH = "high"         # 중요한 법적 판단이 필요
    NORMAL = "normal"     # 일반적인 질문

# ===== LRU 캐시 구현 =====
class LRUCache:
    """시간 기반 만료를 지원하는 LRU 캐시 구현
    
    LRU(Least Recently Used) 캐시는 가장 오래 사용하지 않은 항목을
    자동으로 제거하는 똑똑한 보관함입니다. 자주 사용하는 정보는
    빠르게 접근할 수 있도록 메모리에 보관합니다.
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
        # 만료 시간 확인
        if time.time() - timestamp > self.ttl:
            del self.cache[key]
            return None
            
        # 최근 사용된 항목을 끝으로 이동 (LRU 구현)
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
        """만료된 모든 항목 제거 - 주기적인 정리를 위해"""
        current_time = time.time()
        expired_keys = [
            key for key, (_, timestamp) in self.cache.items()
            if current_time - timestamp > self.ttl
        ]
        for key in expired_keys:
            del self.cache[key]

# ===== 비동기 실행 헬퍼 =====
def run_async_in_streamlit(coro):
    """Streamlit 환경에서 비동기 함수를 안전하게 실행
    
    Streamlit은 기본적으로 동기적으로 작동하므로,
    비동기 함수를 실행하려면 특별한 처리가 필요합니다.
    이 함수가 그 다리 역할을 합니다.
    """
    try:
        loop = asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, coro)
            return future.result()
    except RuntimeError:
        return asyncio.run(coro)

# ===== 문서 버전 관리 시스템 =====
class DocumentVersionManager:
    """문서의 버전과 최신성을 관리하는 시스템
    
    법률은 시간에 따라 변경되므로, 어떤 정보가 최신인지
    파악하는 것이 매우 중요합니다. 이 클래스는 마치
    도서관의 개정판 관리 시스템과 같은 역할을 합니다.
    """
    
    def __init__(self):
        # 주요 규정 변경 이력
        self.regulation_changes = {
            '대규모내부거래_금액기준': [
                {
                    'date': '2023-01-01', 
                    'old_value': '50억원', 
                    'new_value': '100억원',
                    'description': '자본금 및 자본총계 중 큰 금액의 5% 이상 또는 100억원 이상'
                },
                {
                    'date': '2020-01-01', 
                    'old_value': '30억원', 
                    'new_value': '50억원',
                    'description': '자본금 및 자본총계 중 큰 금액의 5% 이상 또는 50억원 이상'
                }
            ],
            '공시_기한': [
                {
                    'date': '2022-07-01', 
                    'old_value': '7일', 
                    'new_value': '5일',
                    'description': '이사회 의결 후 공시 기한 단축'
                }
            ]
        }
        
        # 중요 정보 패턴 정의
        self.critical_patterns = {
            '금액': r'(\d+)억\s*원',
            '비율': r'(\d+(?:\.\d+)?)\s*%',
            '기한': r'(\d+)\s*일',
            '날짜': r'(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일'
        }
    
    def extract_document_date(self, chunk: Dict) -> Optional[str]:
        """문서에서 작성/개정 날짜 추출
        
        문서의 날짜는 그 문서의 유효성을 판단하는 중요한 기준입니다.
        여러 패턴을 사용해 날짜를 찾아냅니다.
        """
        content = chunk.get('content', '')
        metadata = json.loads(chunk.get('metadata', '{}'))
        
        # 메타데이터에 날짜가 있으면 우선 사용
        if 'document_date' in metadata:
            return metadata['document_date']
        
        # 내용에서 날짜 패턴 찾기
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
        """날짜 문자열을 표준 형식(YYYY-MM-DD)으로 변환"""
        # 숫자가 아닌 문자를 하이픈으로 변경
        date_str = re.sub(r'[^\d]', '-', date_str)
        parts = date_str.split('-')
        if len(parts) >= 2:
            year = parts[0] if len(parts[0]) == 4 else '20' + parts[0]
            month = parts[1].zfill(2)
            day = parts[2].zfill(2) if len(parts) > 2 else '01'
            return f"{year}-{month}-{day}"
        return None
    
    def check_for_outdated_info(self, content: str, document_date: str = None) -> List[Dict]:
        """구버전 정보가 포함되어 있는지 확인
        
        이는 마치 식품의 유통기한을 확인하는 것과 같습니다.
        오래된 정보는 사용자에게 해가 될 수 있으므로
        신중하게 확인하고 경고합니다.
        """
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
        
        # 공시 기한 확인
        deadline_match = re.search(r'이사회.*의결.*(\d+)일', content)
        if deadline_match:
            days = int(deadline_match.group(1))
            if days == 7:
                warnings.append({
                    'type': 'outdated_deadline',
                    'found': '7일',
                    'current': '5일',
                    'regulation': '대규모내부거래 공시 기한',
                    'changed_date': '2022-07-01',
                    'severity': 'high'
                })
        
        return warnings

# ===== 충돌 해결 시스템 =====
class ConflictResolver:
    """상충하는 정보를 해결하는 시스템
    
    여러 문서에서 서로 다른 정보를 제공할 때,
    어떤 것이 맞는지 판단하는 것은 매우 중요합니다.
    이 클래스는 마치 재판관과 같은 역할을 합니다.
    """
    
    def __init__(self, version_manager: DocumentVersionManager):
        self.version_manager = version_manager
    
    def resolve_conflicts(self, results: List[SearchResult], query: str) -> List[SearchResult]:
        """검색 결과 중 상충하는 정보를 해결하고 최신 정보를 우선시
        
        이 과정은 다음과 같이 진행됩니다:
        1. 각 결과의 날짜와 내용을 확인
        2. 구버전 정보가 있는지 검사
        3. 충돌하는 정보를 찾아내기
        4. 최신 정보를 우선순위로 재정렬
        """
        
        # 각 결과에 대해 구버전 정보 확인
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
        
        # 최종 정렬: 최신 정보와 높은 점수를 우선시
        results.sort(key=lambda r: (
            not r.metadata.get('has_outdated_info', False),  # 최신 정보 우선
            r.document_date or '1900-01-01',  # 최신 날짜 우선
            r.score  # 관련성 점수
        ), reverse=True)
        
        return results
    
    def _extract_critical_info(self, results: List[SearchResult], query: str) -> Dict:
        """결과에서 중요 정보 추출
        
        금액, 비율, 기한 등 법적으로 중요한 수치 정보를 추출합니다.
        """
        critical_info = defaultdict(list)
        
        for i, result in enumerate(results):
            # 금액 정보 추출
            amounts = re.findall(r'(\d+)억\s*원', result.content)
            for amount in amounts:
                critical_info['amounts'].append({
                    'value': amount + '억원',
                    'result_index': i,
                    'context': result.content[:100]
                })
            
            # 비율 정보 추출
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
        
        # 대규모내부거래 금액 기준 충돌 확인
        if 'amounts' in critical_info:
            amount_values = set()
            for item in critical_info['amounts']:
                if '대규모내부거래' in item['context']:
                    amount_values.add(item['value'])
            
            # 여러 다른 금액이 언급되고, 그 중 구버전이 있는 경우
            if len(amount_values) > 1 and ('50억원' in amount_values or '30억원' in amount_values):
                conflicts.append({
                    'type': 'amount_conflict',
                    'values': list(amount_values),
                    'correct_value': '100억원'
                })
        
        return conflicts
    
    def _prioritize_latest_info(self, results: List[SearchResult], conflicts: List[Dict]) -> List[SearchResult]:
        """충돌이 있을 때 최신 정보를 우선시
        
        구버전 정보가 포함된 결과의 점수를 낮춰서
        자연스럽게 하위로 밀려나게 합니다.
        """
        for conflict in conflicts:
            if conflict['type'] == 'amount_conflict':
                for i, result in enumerate(results):
                    # 구버전 금액이 포함된 경우 점수 감소
                    if any(old_val in result.content for old_val in ['50억원', '30억원']):
                        results[i].score *= 0.5  # 점수를 절반으로
                        results[i].metadata['score_reduced'] = True
                        results[i].metadata['reduction_reason'] = 'outdated_amount'
        
        return results

# ===== 중요도 평가 시스템 =====
class ImportanceAssessor:
    """질문의 중요도를 평가하는 시스템
    
    법적 리스크와 금액을 고려하여 질문의 중요도를 판단합니다.
    이는 응급실에서 환자의 중증도를 분류하는 것과 같은 원리입니다.
    """
    
    def __init__(self):
        # 법적 리스크 키워드
        self.legal_risk_keywords = {
            'critical': ['공시의무', '이사회의결', '법적책임', '제재', '과태료', '위반', '처벌'],
            'high': ['대규모내부거래', '계열사거래', '특수관계인', '공정거래법', '신고의무'],
            'medium': ['절차', '기한', '보고', '제출', '통지']
        }
        
        # 금액 임계값 (억원)
        self.amount_thresholds = {
            'critical': 100,  # 100억원 이상
            'high': 50,       # 50억원 이상
            'medium': 10      # 10억원 이상
        }
    
    def assess_importance(self, query: str) -> Tuple[QueryImportance, Dict]:
        """질문의 중요도를 평가
        
        금액과 법적 리스크를 종합적으로 고려하여
        질문이 얼마나 중요한지 판단합니다.
        """
        importance_score = 0
        factors = []
        
        query_lower = query.lower()
        
        # 금액 확인
        amount_matches = re.findall(r'(\d+)억\s*원', query)
        max_amount = 0
        for amount_str in amount_matches:
            amount = int(amount_str)
            max_amount = max(max_amount, amount)
            
            if amount >= self.amount_thresholds['critical']:
                importance_score += 10
                factors.append(f"{amount}억원 - 대규모 거래")
            elif amount >= self.amount_thresholds['high']:
                importance_score += 7
                factors.append(f"{amount}억원 - 중규모 거래")
            elif amount >= self.amount_thresholds['medium']:
                importance_score += 4
                factors.append(f"{amount}억원 거래")
        
        # 법적 리스크 키워드 확인
        for level, keywords in self.legal_risk_keywords.items():
            for keyword in keywords:
                if keyword in query_lower:
                    if level == 'critical':
                        importance_score += 8
                        factors.append(f"핵심 법적 개념: {keyword}")
                    elif level == 'high':
                        importance_score += 5
                        factors.append(f"중요 법적 개념: {keyword}")
                    else:
                        importance_score += 2
                        factors.append(f"법적 개념: {keyword}")
        
        # 복합 거래 여부
        if '동시에' in query_lower or ('그리고' in query_lower and '거래' in query_lower):
            importance_score += 3
            factors.append("복합 거래 분석 필요")
        
        # 중요도 레벨 결정
        if importance_score >= 15 or max_amount >= self.amount_thresholds['critical']:
            importance = QueryImportance.CRITICAL
        elif importance_score >= 8 or max_amount >= self.amount_thresholds['high']:
            importance = QueryImportance.HIGH
        else:
            importance = QueryImportance.NORMAL
        
        return importance, {
            'score': importance_score,
            'factors': factors,
            'max_amount': max_amount
        }

# ===== 모델 선택 시스템 =====
class OptimizedModelSelector:
    """질문의 중요도와 복잡도를 고려한 최적화된 모델 선택 시스템
    
    이 시스템은 마치 병원에서 환자의 상태에 따라
    적절한 의료진을 배정하는 것과 같습니다.
    중요한 질문에는 더 정확한 모델을 사용합니다.
    """
    
    def __init__(self):
        self.model_profiles = {
            'gpt-4o-mini': {
                'accuracy_score': 0.65,
                'speed_score': 0.95,
                'best_for': ['simple_facts', 'basic_queries'],
                'description': '간단한 사실 확인에 최적화된 경제적 모델'
            },
            'o4-mini': {
                'accuracy_score': 0.75,
                'speed_score': 0.93,
                'best_for': ['standard_queries', 'basic_analysis'],
                'description': '표준적인 질문에 대한 빠른 처리'
            },
            'o3-mini': {
                'accuracy_score': 0.78,
                'speed_score': 0.90,
                'best_for': ['intermediate_queries', 'moderate_complexity'],
                'description': '중간 복잡도의 분석이 가능한 모델'
            },
            'gpt-4o': {
                'accuracy_score': 0.90,
                'speed_score': 0.80,
                'best_for': ['complex_analysis', 'legal_interpretation', 'critical_decisions'],
                'description': '복잡한 법률 해석과 중요한 판단에 최적화된 모델'
            }
        }
    
    def select_model(self, query: str, complexity: QueryComplexity, 
                    importance: QueryImportance, initial_analysis: Dict) -> Tuple[str, Dict]:
        """질문에 가장 적합한 모델을 선택
        
        중요도가 높은 질문은 복잡도와 관계없이 상위 모델을 사용합니다.
        이는 수술의 난이도와 관계없이 중요한 수술은
        경험 많은 의사가 집도하는 것과 같은 원리입니다.
        """
        
        # 중요도가 CRITICAL인 경우 무조건 최고 모델 사용
        if importance == QueryImportance.CRITICAL:
            selected_model = 'gpt-4o'
            reason = "법적 리스크가 높거나 대규모 금액이 관련된 중요한 질문"
        
        # 중요도가 HIGH인 경우 상위 모델 사용
        elif importance == QueryImportance.HIGH:
            if complexity == QueryComplexity.COMPLEX:
                selected_model = 'gpt-4o'
                reason = "중요하고 복잡한 법률 해석이 필요한 질문"
            else:
                selected_model = 'o3-mini'
                reason = "중요도가 높은 표준적인 질문"
        
        # 일반적인 경우 복잡도에 따라 선택
        else:
            if complexity == QueryComplexity.SIMPLE:
                selected_model = 'gpt-4o-mini'
                reason = "간단한 사실 확인 또는 기본 정보 요청"
            elif complexity == QueryComplexity.MEDIUM:
                selected_model = 'o4-mini'
                reason = "표준적인 절차나 규정에 대한 질문"
            else:
                selected_model = 'o3-mini'
                reason = "복잡하지만 중요도가 낮은 질문"
        
        selection_info = {
            'model': selected_model,
            'reason': reason,
            'complexity': complexity.value,
            'importance': importance.value,
            'accuracy_score': self.model_profiles[selected_model]['accuracy_score']
        }
        
        return selected_model, selection_info

# ===== 캐싱 전략 시스템 =====
class SmartCacheStrategy:
    """모델별 특성을 고려한 지능형 캐싱 전략
    
    캐싱은 자주 찾는 책을 책상 위에 올려놓는 것과 같습니다.
    한 번 찾은 정보를 빠르게 다시 사용할 수 있어
    시간과 비용을 크게 절약할 수 있습니다.
    """
    
    def __init__(self):
        self.cache_configs = {
            'gpt-4o-mini': {
                'ttl': 7200,  # 2시간 - 저렴하므로 오래 보관
                'similarity_threshold': 0.85
            },
            'o4-mini': {
                'ttl': 5400,  # 1.5시간
                'similarity_threshold': 0.88
            },
            'o3-mini': {
                'ttl': 3600,  # 1시간
                'similarity_threshold': 0.90
            },
            'gpt-4o': {
                'ttl': 1800,  # 30분 - 중요한 질문은 자주 업데이트
                'similarity_threshold': 0.95
            }
        }
        
        # 질문-답변 캐시
        self.query_cache = LRUCache(max_size=200, ttl=7200)
        # 임베딩 캐시
        self.embedding_cache = LRUCache(max_size=500, ttl=10800)
        # 분석 결과 캐시
        self.analysis_cache = LRUCache(max_size=100, ttl=3600)
    
    def get_cached_result(self, query: str) -> Optional[Dict]:
        """캐시된 결과 반환"""
        cache_key = self._generate_cache_key(query)
        return self.query_cache.get(cache_key)
    
    def store_result(self, query: str, result: Dict, model: str, importance: QueryImportance):
        """결과를 캐시에 저장
        
        중요한 질문의 결과는 캐싱하지 않거나 짧게 보관합니다.
        이는 중요한 정보일수록 최신성이 더 중요하기 때문입니다.
        """
        # CRITICAL 중요도의 질문은 캐싱하지 않음
        if importance == QueryImportance.CRITICAL:
            return
        
        cache_key = self._generate_cache_key(query)
        result['cached_at'] = time.time()
        result['model'] = model
        self.query_cache.put(cache_key, result)
    
    def _generate_cache_key(self, query: str) -> str:
        """쿼리에 대한 고유한 캐시 키 생성"""
        return hashlib.md5(query.encode()).hexdigest()

# ===== GPT 질문 분석기 =====
class AdaptiveQueryAnalyzer:
    """질문의 중요도에 따라 적응적으로 분석하는 시스템
    
    중요한 질문은 더 정교한 분석을 수행합니다.
    이는 중요한 서류는 더 꼼꼼히 검토하는 것과 같은 원리입니다.
    """
    
    def __init__(self, cache_strategy: SmartCacheStrategy, importance_assessor: ImportanceAssessor):
        self.cache_strategy = cache_strategy
        self.importance_assessor = importance_assessor
    
    def analyze_and_strategize(self, query: str, available_chunks_info: Dict, 
                              importance: QueryImportance) -> Dict:
        """질문을 분석하고 최적의 검색 전략 수립
        
        중요한 질문은 더 정교한 분석을 위해 상위 모델을 사용합니다.
        """
        
        # 캐시 확인
        cache_key = hashlib.md5(f"analysis_{query}".encode()).hexdigest()
        cached = self.cache_strategy.analysis_cache.get(cache_key)
        if cached:
            logger.info(f"Analysis cache hit for query: {query[:50]}...")
            return cached
        
        # 중요도에 따라 분석 모델 선택
        if importance == QueryImportance.CRITICAL:
            analysis_model = "gpt-4o"  # 중요한 질문은 최고 모델로 분석
        else:
            analysis_model = "gpt-4o-mini"  # 일반 질문은 경제적 모델 사용
        
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
                model=analysis_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=1000,
                response_format={"type": "json_object"}
            )
            
            analysis = json.loads(response.choices[0].message.content)
            
            # 캐시 저장 (중요한 질문은 캐시하지 않음)
            if importance != QueryImportance.CRITICAL:
                self.cache_strategy.analysis_cache.put(cache_key, analysis)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Query analysis error: {str(e)}")
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
    """질문을 분석하여 어떤 매뉴얼을 우선 검색할지 결정
    
    이는 도서관에서 어느 섹션으로 가야 할지
    안내하는 것과 같습니다. 질문의 키워드와 패턴을
    분석하여 가장 관련성 높은 매뉴얼을 찾습니다.
    """
    
    def __init__(self):
        self.categories = {
            '대규모내부거래': {
                'keywords': ['대규모내부거래', '내부거래', '이사회 의결', '이사회', '의결', 
                           '계열사', '계열회사', '특수관계인', '자금', '대여', '차입', '보증',
                           '자금거래', '유가증권', '자산거래', '50억', '거래금액', '100억'],
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
        """질문을 분류하고 신뢰도를 반환
        
        각 카테고리별로 점수를 계산하고,
        가장 높은 점수를 받은 카테고리를 선택합니다.
        신뢰도는 얼마나 확실한지를 나타냅니다.
        """
        question_lower = question.lower()
        scores = {}
        
        for category, info in self.categories.items():
            score = 0
            matched_keywords = []
            
            # 키워드 매칭 (중요도에 따라 가중치 부여)
            for i, keyword in enumerate(info['keywords']):
                if keyword in question_lower:
                    weight = 1.0 if i < 5 else 0.7  # 앞쪽 키워드가 더 중요
                    score += weight
                    matched_keywords.append(keyword)
            
            # 패턴 매칭
            for pattern in info.get('patterns', []):
                if re.search(pattern, question_lower):
                    score += 1.5
            
            scores[category] = score
        
        if scores:
            best_category = max(scores, key=scores.get)
            max_possible_score = len(self.categories[best_category]['keywords']) + \
                               len(self.categories[best_category].get('patterns', [])) * 1.5
            confidence = min(scores[best_category] / max_possible_score, 1.0)
            
            # 너무 낮은 신뢰도는 분류하지 않음
            if confidence < 0.15:
                return None, 0.0
                
            return best_category, confidence
        
        return None, 0.0

# ===== 복잡도 평가기 =====
class ComplexityAssessor:
    """질문의 복잡도를 평가하여 처리 방식을 결정
    
    이는 요리의 난이도를 평가하는 것과 같습니다.
    재료의 수, 조리 단계, 필요한 기술 등을
    종합적으로 고려하여 난이도를 결정합니다.
    """
    
    def __init__(self):
        # 복잡도 지표들
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
        
        # 각 지표별 점수 계산
        simple_score = sum(1 for pattern in self.simple_indicators 
                         if re.search(pattern, query_lower))
        complex_score = sum(2 for pattern in self.complex_indicators 
                          if re.search(pattern, query_lower))
        medium_score = sum(1.5 for pattern in self.medium_indicators 
                         if re.search(pattern, query_lower))
        
        # 길이에 따른 추가 점수
        if len(query) > 150:
            complex_score += 2
        elif len(query) > 100:
            complex_score += 1
        elif len(query) < 30:
            simple_score += 0.5
            
        # 특수 패턴 확인
        if re.search(r'[AB]회사.*[CD]회사', query_lower):
            complex_score += 2
        if '?' in query and query.count('?') > 1:
            complex_score += 1
            
        total_score = simple_score + medium_score + complex_score
        
        # 복잡도 레벨 결정
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
            
        # 정규화된 점수 (0-1)
        normalized_score = min(total_score / 10, 1.0)
        
        analysis = {
            'simple_score': simple_score,
            'medium_score': medium_score,
            'complex_score': complex_score,
            'total_score': total_score,
            'normalized_score': normalized_score,
            'query_length': len(query)
        }
        
        return complexity, confidence, analysis

# ===== 하이브리드 RAG 파이프라인 =====
class OptimizedHybridRAGPipeline:
    """중요도 기반 적응적 처리를 수행하는 하이브리드 RAG 파이프라인
    
    이 클래스는 전체 시스템의 핵심입니다.
    질문의 중요도에 따라 다른 수준의 처리를 수행하여
    정확성과 효율성의 균형을 맞춥니다.
    """
    
    def __init__(self, embedding_model, reranker_model, index, chunks):
        # 기본 구성 요소 검증
        if not chunks:
            raise ValueError("No chunks provided to HybridRAGPipeline")
        
        if index.ntotal == 0:
            raise ValueError("FAISS index is empty")
        
        # 임베딩 차원 검증
        test_embedding = embedding_model.encode(["test"])
        if len(test_embedding[0]) != index.d:
            raise ValueError(f"Embedding dimension {len(test_embedding[0])} doesn't match index dimension {index.d}")
        
        # 기본 구성 요소
        self.embedding_model = embedding_model
        self.reranker_model = reranker_model
        self.index = index
        self.chunks = chunks
        
        # 시스템 구성 요소들
        self.cache_strategy = SmartCacheStrategy()
        self.version_manager = DocumentVersionManager()
        self.conflict_resolver = ConflictResolver(self.version_manager)
        self.importance_assessor = ImportanceAssessor()
        
        self.classifier = QuestionClassifier()
        self.complexity_assessor = ComplexityAssessor()
        self.query_analyzer = AdaptiveQueryAnalyzer(self.cache_strategy, self.importance_assessor)
        self.model_selector = OptimizedModelSelector()
        
        # 매뉴얼별 인덱스 구축
        self.manual_indices = self._build_manual_indices()
        
        # 청크 정보
        self.chunks_info = {
            category: len(indices) 
            for category, indices in self.manual_indices.items()
        }
        
        # 모든 청크의 날짜 정보 추출
        self._extract_chunk_dates()
        
        logger.info(f"HybridRAGPipeline initialized with {len(chunks)} chunks")
        logger.info(f"Manual distribution: {self.chunks_info}")
    
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
        # 캐시 확인
        cached_embedding = self.cache_strategy.embedding_cache.get(query)
        if cached_embedding is not None:
            return cached_embedding
            
        # 새로 생성
        embedding = self.embedding_model.encode([query])
        embedding = np.array(embedding, dtype=np.float32)
        
        # 캐시 저장
        self.cache_strategy.embedding_cache.put(query, embedding)
        
        return embedding
    
    async def process_query(self, query: str, top_k: int = 5) -> Tuple[List[SearchResult], Dict]:
        """질문을 처리하는 메인 메서드
        
        중요한 질문은 더 신중하게, 일반적인 질문은 효율적으로 처리합니다.
        """
        start_time = time.time()
        
        # 1. 중요도 평가
        importance, importance_info = self.importance_assessor.assess_importance(query)
        logger.info(f"Query importance: {importance.value} - {importance_info['factors']}")
        
        # 2. 캐시 확인 (중요한 질문은 캐시 사용 안 함)
        if importance != QueryImportance.CRITICAL:
            cached_result = self.cache_strategy.get_cached_result(query)
            if cached_result:
                logger.info(f"Cache hit for query: {query[:50]}...")
                return cached_result['results'], cached_result['stats']
        
        # 3. 질문 분류 및 복잡도 평가
        category, cat_confidence = self.classifier.classify(query)
        complexity, comp_confidence, complexity_analysis = self.complexity_assessor.assess(query)
        
        # 4. 적응적 질문 분석 (중요한 질문은 더 정교하게 분석)
        analysis_start = time.time()
        try:
            gpt_analysis = self.query_analyzer.analyze_and_strategize(
                query, self.chunks_info, importance
            )
            analysis_time = time.time() - analysis_start
        except Exception as e:
            logger.error(f"Query analysis failed: {str(e)}, falling back to rule-based")
            gpt_analysis = self._get_fallback_analysis(query, category)
            analysis_time = time.time() - analysis_start
        
        # 5. 모델 선택 (중요도 우선 고려)
        selected_model, selection_info = self.model_selector.select_model(
            query, complexity, importance, gpt_analysis
        )
        
        # 6. 중요도에 따른 검색 전략 실행
        if importance == QueryImportance.CRITICAL:
            # 중요한 질문은 더 광범위한 검색
            results, search_stats = await self._comprehensive_search_for_critical(
                query, gpt_analysis, top_k
            )
        else:
            # 일반적인 검색 전략
            search_approach = gpt_analysis['search_strategy']['approach']
            
            if search_approach == 'direct_lookup':
                results, search_stats = await self._gpt_guided_direct_search(
                    query, gpt_analysis, top_k
                )
            elif search_approach == 'focused_search':
                results, search_stats = await self._gpt_guided_focused_search(
                    query, gpt_analysis, top_k
                )
            else:  # comprehensive_analysis
                results, search_stats = await self._gpt_guided_comprehensive_search(
                    query, gpt_analysis, top_k
                )
        
        # 7. 충돌 해결 및 최신 정보 우선시
        results = self.conflict_resolver.resolve_conflicts(results, query)
        
        # 8. 구버전 경고 수집
        outdated_warnings = []
        for result in results:
            if result.metadata.get('has_outdated_info'):
                outdated_warnings.extend(result.metadata.get('warnings', []))
        
        # 9. 통계 정보 구성
        stats = {
            'total_time': time.time() - start_time,
            'analysis_time': analysis_time,
            'gpt_analysis': gpt_analysis,
            'category': category,
            'category_confidence': cat_confidence,
            'complexity': complexity.value,
            'complexity_confidence': comp_confidence,
            'complexity_analysis': complexity_analysis,
            'importance': importance.value,
            'importance_info': importance_info,
            'selected_model': selected_model,
            'selection_info': selection_info,
            'search_approach': search_stats.get('search_method', 'unknown'),
            'outdated_warnings': outdated_warnings,
            'has_version_conflicts': len(outdated_warnings) > 0,
            **search_stats
        }
        
        # 10. 캐시 저장 (중요하지 않은 질문만)
        if importance != QueryImportance.CRITICAL and not outdated_warnings:
            self.cache_strategy.store_result(query, {
                'results': results,
                'stats': stats
            }, selected_model, importance)
        
        return results, stats
    
    async def _comprehensive_search_for_critical(self, query: str, gpt_analysis: Dict, 
                                               top_k: int) -> Tuple[List[SearchResult], Dict]:
        """중요한 질문에 대한 포괄적 검색
        
        중요한 질문은 더 많은 문서를 검색하고,
        여러 관점에서 정보를 수집합니다.
        """
        start_time = time.time()
        all_results = []
        
        # 모든 관련 매뉴얼에서 광범위하게 검색
        for manual in self.manual_indices.keys():
            if manual != '기타':
                partial_results = await self._search_in_manual(
                    query, manual, [], top_k * 2  # 더 많은 결과 수집
                )
                all_results.extend(partial_results)
        
        # 중복 제거 및 정렬
        seen_chunks = set()
        unique_results = []
        for result in sorted(all_results, key=lambda x: x.score, reverse=True):
            if result.chunk_id not in seen_chunks:
                seen_chunks.add(result.chunk_id)
                unique_results.append(result)
                if len(unique_results) >= top_k * 3:  # 더 많은 결과 보관
                    break
        
        # 상위 결과만 반환
        final_results = unique_results[:top_k]
        
        stats = {
            'search_time': time.time() - start_time,
            'searched_chunks': sum(len(indices) for indices in self.manual_indices.values()),
            'search_method': 'comprehensive_critical',
            'total_results_found': len(unique_results),
            'results_returned': len(final_results)
        }
        
        return final_results, stats
    
    async def _gpt_guided_direct_search(self, query: str, gpt_analysis: Dict, 
                                       top_k: int) -> Tuple[List[SearchResult], Dict]:
        """GPT 분석을 기반으로 한 직접 검색"""
        start_time = time.time()
        
        primary_manual = gpt_analysis['search_strategy']['primary_manual']
        search_keywords = gpt_analysis['search_strategy']['search_keywords']
        
        # 대상 인덱스 선택
        target_indices = self.manual_indices.get(primary_manual, [])[:100]
        
        if not target_indices:
            logger.warning(f"No indices for manual '{primary_manual}', using all chunks")
            target_indices = list(range(min(len(self.chunks), 50)))
        
        # 검색어 확장
        enhanced_query = f"{query} {' '.join(search_keywords)}"
        query_vector = self._get_query_embedding(enhanced_query)
        
        # FAISS 검색
        k_search = min(len(target_indices), max(1, top_k * 3))
        
        try:
            scores, indices = self.index.search(query_vector, k_search)
        except Exception as e:
            logger.error(f"FAISS search error: {str(e)}")
            return [], {'search_time': time.time() - start_time, 'error': str(e)}
        
        # 결과 구성
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
        """GPT 분석을 기반으로 한 집중 검색"""
        start_time = time.time()
        
        primary_manual = gpt_analysis['search_strategy'].get('primary_manual', '')
        search_keywords = gpt_analysis['search_strategy'].get('search_keywords', [])
        expected_chunks = gpt_analysis['search_strategy'].get('expected_chunks_needed', 10)
        
        # 검색 범위 설정
        search_limit = min(expected_chunks * 2, 200)
        target_indices = self.manual_indices.get(primary_manual, [])[:search_limit]
        
        if not target_indices:
            target_indices = list(range(min(len(self.chunks), 100)))
        
        # 답변 요구사항에 따른 필터링
        requirements = gpt_analysis.get('answer_requirements', {})
        if requirements.get('needs_specific_numbers'):
            filtered_indices = [
                idx for idx in target_indices 
                if idx < len(self.chunks) and
                re.search(r'\d+억|\d+%', self.chunks[idx].get('content', ''))
            ]
            if filtered_indices:
                target_indices = filtered_indices
        
        # 검색 수행
        enhanced_query = f"{query} {' '.join(search_keywords)}"
        query_vector = self._get_query_embedding(enhanced_query)
        
        k_search = min(len(target_indices), max(1, top_k * 5))
        
        try:
            scores, indices = self.index.search(query_vector, k_search)
        except Exception as e:
            logger.error(f"FAISS search error: {str(e)}")
            return [], {'search_time': time.time() - start_time, 'error': str(e)}
        
        # 결과 구성 및 관련성 부스팅
        results = []
        target_set = set(target_indices)
        
        for idx, score in zip(indices[0], scores[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
                
            if idx not in target_set:
                continue
                
            chunk = self.chunks[idx]
            
            # GPT 분석과의 관련성 계산
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
        
        # 점수 기준 정렬 및 상위 선택
        results.sort(key=lambda x: x.score, reverse=True)
        results = results[:top_k]
        
        stats = {
            'search_time': time.time() - start_time,
            'searched_chunks': len(target_indices),
            'search_method': 'focused_vector',
            'results_count': len(results)
        }
        
        return results, stats
    
    async def _gpt_guided_comprehensive_search(self, query: str, gpt_analysis: Dict, 
                                              top_k: int) -> Tuple[List[SearchResult], Dict]:
        """GPT 분석을 기반으로 한 종합 검색"""
        start_time = time.time()
        
        all_results = []
        
        # 모든 관련 법적 개념에 대해 검색
        for concept in gpt_analysis['legal_concepts']:
            if concept['relevance'] in ['primary', 'secondary']:
                manual = concept['concept']
                if manual in self.manual_indices:
                    partial_results = await self._search_in_manual(
                        query, manual, concept['specific_aspects'], top_k // 2
                    )
                    all_results.extend(partial_results)
        
        # 중복 제거 및 상위 결과 선택
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
        
        # 검색 키워드 매칭
        for keyword in gpt_analysis['search_strategy']['search_keywords']:
            if keyword.lower() in content_lower:
                relevance_boost += 0.1
        
        # 답변 요구사항과의 매칭
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
    
    def _get_fallback_analysis(self, query: str, category: str = None) -> Dict:
        """GPT 분석 실패 시 폴백 분석"""
        primary_manual = category or "대규모내부거래"
        
        return {
            'query_analysis': {
                'core_intent': query,
                'actual_complexity': 'medium',
                'complexity_reason': 'GPT 분석 실패로 기본값 사용'
            },
            'legal_concepts': [{
                'concept': primary_manual,
                'relevance': 'primary',
                'specific_aspects': []
            }],
            'search_strategy': {
                'approach': 'focused_search',
                'primary_manual': primary_manual,
                'search_keywords': query.split()[:5],
                'expected_chunks_needed': 20,
                'rationale': '기본 검색 전략'
            },
            'answer_requirements': {
                'needs_specific_numbers': True,
                'needs_process_steps': True,
                'needs_timeline': False,
                'needs_exceptions': False,
                'needs_multiple_perspectives': False
            }
        }

# ===== 답변 품질 검증 시스템 =====
class AnswerQualityValidator:
    """생성된 답변의 품질을 검증하는 시스템
    
    이는 제품 출하 전 품질 검사와 같습니다.
    답변이 정확하고 완전한지 확인합니다.
    """
    
    def validate_answer(self, query: str, answer: str, importance: QueryImportance) -> Dict:
        """답변의 품질을 검증
        
        중요한 질문의 답변은 더 엄격하게 검증합니다.
        """
        issues = []
        quality_score = 1.0
        
        # 불확실한 표현 체크
        uncertain_phrases = ['확실하지 않', '아마도', '추정', '~것 같습니다', '~수도 있습니다']
        for phrase in uncertain_phrases:
            if phrase in answer:
                issues.append(f"불확실한 표현 사용: {phrase}")
                quality_score -= 0.1
        
        # 법적 근거 확인
        legal_references = re.findall(r'(제\d+조|시행령|규정|매뉴얼)', answer)
        if len(legal_references) == 0:
            issues.append("법적 근거 부족")
            quality_score -= 0.2
        
        # 답변 길이 확인
        if len(answer) < 100:
            issues.append("답변이 너무 짧음")
            quality_score -= 0.15
        
        # 중요한 질문의 경우 추가 검증
        if importance == QueryImportance.CRITICAL:
            # 구체적인 수치 확인
            numbers = re.findall(r'\d+', answer)
            if len(numbers) == 0:
                issues.append("구체적인 수치 정보 부족")
                quality_score -= 0.1
            
            # 주의사항 포함 여부
            if not any(word in answer for word in ['주의', '유의', '예외', '단서']):
                issues.append("주의사항이나 예외사항 미포함")
                quality_score -= 0.1
        
        return {
            'quality_score': max(quality_score, 0),
            'issues': issues,
            'passed': quality_score >= 0.7,
            'needs_revision': quality_score < 0.7 and importance == QueryImportance.CRITICAL
        }

# ===== 답변 생성 함수 =====
def determine_temperature(query: str, complexity: str, model: str, importance: str) -> float:
    """질문 유형, 복잡도, 모델, 중요도에 따라 최적의 temperature 결정
    
    Temperature는 AI의 창의성 수준을 조절합니다.
    중요한 질문일수록 낮은 temperature를 사용하여
    일관되고 정확한 답변을 생성합니다.
    """
    query_lower = query.lower()
    
    # 중요도별 기본 temperature
    if importance == 'critical':
        base_temp = 0.1  # 매우 낮게 설정
    elif importance == 'high':
        base_temp = 0.2
    else:
        base_temp = 0.3
    
    # 질문 유형에 따른 미세 조정
    if any(keyword in query_lower for keyword in ['언제', '며칠', '기한', '날짜', '금액', '%']):
        base_temp = min(base_temp, 0.1)  # 사실 확인은 정확성이 중요
    elif any(keyword in query_lower for keyword in ['전략', '대응', '리스크', '주의', '권장']):
        base_temp = min(base_temp + 0.1, 0.5)  # 전략적 조언도 신중하게
    
    return base_temp

async def generate_answer(query: str, results: List[SearchResult], stats: Dict) -> Tuple[str, Dict]:
    """선택된 모델을 사용하여 고품질 답변 생성
    
    중요한 질문은 더 신중하게 답변을 생성하고,
    필요시 품질 검증 후 재생성합니다.
    """
    
    # 선택된 모델 및 중요도 확인
    model = stats.get('selected_model', 'gpt-4o-mini')
    importance = stats.get('importance', 'normal')
    
    # 컨텍스트 구성
    context_parts = []
    
    # 구버전 경고가 있으면 먼저 표시
    if stats.get('has_version_conflicts'):
        context_parts.append("⚠️ 주의: 일부 참고 자료에 구버전 정보가 포함되어 있을 수 있습니다.\n")
    
    # 검색 결과 포함
    for i, result in enumerate(results[:5]):
        # 구버전 정보 표시
        warning_marker = ""
        if result.metadata.get('has_outdated_info'):
            warnings = result.metadata.get('warnings', [])
            if warnings:
                warning_marker = " ⚠️ [구버전 정보 포함]"
        
        context_parts.append(f"""
[참고 {i+1}] {result.source} (페이지 {result.page}){warning_marker}
{result.content}
""")
    
    context = "\n---\n".join(context_parts)
    
    # 복잡도 정보 활용
    complexity = stats.get('complexity', 'medium')
    temperature = determine_temperature(query, complexity, model, importance)
    
    # 중요한 질문에 대한 특별 지시사항
    if importance == 'critical':
        extra_instructions = """
특히 다음 사항들을 반드시 포함하세요:
1. 정확한 법적 근거 (조항 명시)
2. 구체적인 수치와 기한
3. 주의사항 및 예외사항
4. 실무 적용 시 확인해야 할 체크리스트"""
    else:
        extra_instructions = ""
    
    # 시스템 프롬프트
    system_prompt = f"""당신은 한국 공정거래위원회 전문가입니다.
제공된 자료를 근거로 정확하고 실무적인 답변을 제공하세요.

답변은 다음 구조를 따라주세요:
1. 핵심 답변 (1-2문장)
2. 상세 설명 (근거 조항 포함)
3. 주의사항 또는 예외사항 (있는 경우)
4. 실무 적용 시 권장사항 (필요한 경우)

{extra_instructions}

중요: 구버전 정보가 포함된 경우, 반드시 최신 기준을 명시하세요."""
    
    # 메시지 구성
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"""다음 자료를 바탕으로 질문에 답변해주세요.

[참고 자료]
{context}

[질문]
{query}

{"정확하고 신중하게" if importance == 'critical' else "명확하고 실무적으로"} 답변해주세요."""}
    ]
    
    # API 호출 시작
    api_start = time.time()
    
    try:
        response = openai.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=2000 if importance == 'critical' else 1500
        )
        
        answer = response.choices[0].message.content
        
        # 중요한 질문의 경우 품질 검증
        if importance == 'critical':
            validator = AnswerQualityValidator()
            validation_result = validator.validate_answer(query, answer, QueryImportance.CRITICAL)
            
            if validation_result['needs_revision']:
                logger.warning(f"Answer quality issues detected: {validation_result['issues']}")
                # 필요시 재생성 로직 추가 가능
        
        generation_stats = {
            'generation_time': time.time() - api_start,
            'model': model,
            'temperature': temperature
        }
        
        return answer, generation_stats
        
    except Exception as e:
        logger.error(f"Answer generation error: {str(e)}")
        
        # 폴백 답변
        fallback_answer = f"""죄송합니다. 답변 생성 중 오류가 발생했습니다.

오류 내용: {str(e)}

다음 참고 자료를 직접 확인해 주시기 바랍니다:
"""
        for i, result in enumerate(results[:3]):
            fallback_answer += f"\n{i+1}. {result.source} (페이지 {result.page})"
        
        return fallback_answer, {
            'generation_time': time.time() - api_start,
            'error': str(e),
            'model': model
        }

# ===== 모델 및 데이터 로딩 =====
@st.cache_resource(show_spinner=False)
def load_models_and_data():
    """필요한 모델과 데이터 로드"""
    try:
        # 필수 파일 확인
        required_files = ["manuals_vector_db.index", "all_manual_chunks.json"]
        missing_files = [f for f in required_files if not os.path.exists(f)]
        
        if missing_files:
            st.error(f"❌ 필수 파일이 없습니다: {', '.join(missing_files)}")
            st.info("💡 prepare_pdfs_ftc.py를 먼저 실행하여 데이터를 준비하세요.")
            return None, None, None, None
        
        with st.spinner("🤖 AI 시스템을 준비하는 중... (최초 1회만 수행됩니다)"):
            # FAISS 인덱스 로드
            index = faiss.read_index("manuals_vector_db.index")
            logger.info(f"Loaded FAISS index with {index.ntotal} vectors")
            
            # 청크 데이터 로드
            with open("all_manual_chunks.json", "r", encoding="utf-8") as f:
                chunks = json.load(f)
            logger.info(f"Loaded {len(chunks)} chunks")
            
            # 임베딩 모델 로드
            try:
                embedding_model = SentenceTransformer('jhgan/ko-sroberta-multitask')
                logger.info("Loaded Korean embedding model")
            except Exception as e:
                st.warning("한국어 모델 로드 실패. 대체 모델을 사용합니다.")
                embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
                logger.warning(f"Using fallback embedding model: {str(e)}")
            
            # 리랭커 모델 로드 (선택적)
            try:
                reranker_model = CrossEncoder('Dongjin-kr/ko-reranker')
                logger.info("Loaded Korean reranker model")
            except:
                reranker_model = None
                logger.warning("Reranker model not loaded")
        
        return embedding_model, reranker_model, index, chunks
        
    except Exception as e:
        st.error(f"시스템 초기화 실패: {str(e)}")
        logger.error(f"Initialization error: {str(e)}")
        logger.debug(traceback.format_exc())
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
        with st.spinner("시스템을 초기화하는 중..."):
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
                        # 답변 표시
                        st.write(message["content"]["answer"])
                        
                        # 구버전 경고 표시
                        if message["content"].get("has_version_conflicts"):
                            st.markdown("""
                            <div class="version-warning">
                            ⚠️ <strong>주의:</strong> 일부 참고 자료에 구버전 정보가 포함되어 있을 수 있습니다. 
                            최신 규정을 확인해주세요.
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # 메타 정보 표시
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            model_used = message["content"].get("model_used", "unknown")
                            model_emoji = {
                                'gpt-4o-mini': '🟢',
                                'o4-mini': '🟡',
                                'o3-mini': '🟠',
                                'gpt-4o': '🔵'
                            }.get(model_used, '⚪')
                            st.caption(f"{model_emoji} {model_used}")
                        
                        with col2:
                            importance = message["content"].get("importance", "normal")
                            importance_class = f"importance-{importance}"
                            st.markdown(f'<span class="importance-indicator {importance_class}">{importance.upper()}</span>', 
                                      unsafe_allow_html=True)
                        
                        with col3:
                            complexity = message["content"].get("complexity", "unknown")
                            complexity_class = f"complexity-{complexity}"
                            st.markdown(f'<span class="complexity-indicator {complexity_class}">{complexity.upper()}</span>', 
                                      unsafe_allow_html=True)
                        
                        with col4:
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
                search_start_time = time.time()
                with st.spinner("🔍 질문을 분석하고 최적의 검색 전략을 수립하는 중..."):
                    results, search_stats = run_async_in_streamlit(
                        rag.process_query(prompt, top_k=5)
                    )
                search_time = time.time() - search_start_time
                
                # 답변 생성
                generation_start_time = time.time()
                with st.spinner(f"💭 {search_stats.get('selected_model', 'AI')}로 답변을 생성하는 중..."):
                    answer, generation_stats = run_async_in_streamlit(
                        generate_answer(prompt, results, search_stats)
                    )
                generation_time = time.time() - generation_start_time
                
                # 답변 표시
                st.write(answer)
                
                # 구버전 경고 표시
                if search_stats.get('has_version_conflicts'):
                    st.markdown("""
                    <div class="version-warning">
                    ⚠️ <strong>주의:</strong> 일부 참고 자료에 구버전 정보가 포함되어 있을 수 있습니다. 
                    최신 규정을 확인해주세요.
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # 구체적인 경고 내용
                    for warning in search_stats.get('outdated_warnings', []):
                        st.warning(f"구버전: {warning['found']} → 현재: {warning['current']} ({warning['regulation']})")
                
                # 통계 정보
                total_time = time.time() - total_start_time
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    model_used = search_stats.get('selected_model', 'unknown')
                    model_emoji = {
                        'gpt-4o-mini': '🟢',
                        'o4-mini': '🟡',
                        'o3-mini': '🟠',
                        'gpt-4o': '🔵'
                    }.get(model_used, '⚪')
                    st.metric("모델", f"{model_emoji} {model_used}")
                
                with col2:
                    st.metric("검색", f"{search_time:.1f}초")
                
                with col3:
                    st.metric("생성", f"{generation_time:.1f}초")
                
                with col4:
                    st.metric("전체", f"{total_time:.1f}초")
                
                # 중요도와 복잡도 표시
                importance = search_stats.get('importance', 'normal')
                complexity = search_stats.get('complexity', 'unknown')
                
                col1, col2 = st.columns(2)
                with col1:
                    importance_html = f'<span class="importance-indicator importance-{importance}">중요도: {importance.upper()}</span>'
                    st.markdown(importance_html, unsafe_allow_html=True)
                
                with col2:
                    complexity_html = f'<span class="complexity-indicator complexity-{complexity}">복잡도: {complexity.upper()}</span>'
                    st.markdown(complexity_html, unsafe_allow_html=True)
                
                # 상세 정보 (접을 수 있음)
                with st.expander("🔍 상세 정보 보기"):
                    # 탭 구성
                    tab1, tab2, tab3 = st.tabs(["📊 분석 과정", "📚 참고 자료", "🤖 AI 분석"])
                    
                    with tab1:
                        # 중요도 정보
                        importance_info = search_stats.get('importance_info', {})
                        if importance_info:
                            st.subheader("🎯 중요도 평가")
                            st.write(f"**중요도 점수**: {importance_info.get('score', 0)}")
                            st.write("**평가 요인**:")
                            for factor in importance_info.get('factors', []):
                                st.write(f"  • {factor}")
                        
                        # 모델 선택 이유
                        selection_info = search_stats.get('selection_info', {})
                        if selection_info:
                            st.info(f"**모델 선택 이유**: {selection_info.get('reason', 'N/A')}")
                        
                        # 검색 전략
                        st.markdown("**검색 전략**")
                        st.json({
                            "접근 방식": search_stats.get('search_approach', 'N/A'),
                            "검색된 청크 수": search_stats.get('searched_chunks', 0)
                        })
                    
                    with tab2:
                        for i, result in enumerate(results[:5]):
                            with st.container():
                                # 출처 정보
                                st.markdown(f"**[{i+1}] {result.source}** - 페이지 {result.page}")
                                
                                # 문서 날짜 및 경고
                                if result.document_date:
                                    st.caption(f"📅 문서 날짜: {result.document_date}")
                                
                                if result.metadata.get('has_outdated_info'):
                                    st.error("⚠️ 이 문서에는 구버전 정보가 포함되어 있습니다.")
                                
                                # 내용 미리보기
                                content_preview = result.content[:300] + "..." if len(result.content) > 300 else result.content
                                st.text(content_preview)
                                
                                # 관련도 점수
                                st.caption(f"관련도 점수: {result.score:.2f}")
                                st.divider()
                    
                    with tab3:
                        # GPT 분석 결과
                        gpt_analysis = search_stats.get('gpt_analysis', {})
                        
                        st.markdown("**질문 분석**")
                        st.json(gpt_analysis.get('query_analysis', {}))
                        
                        st.markdown("**법적 개념**")
                        st.json(gpt_analysis.get('legal_concepts', []))
                        
                        st.markdown("**답변 요구사항**")
                        requirements = gpt_analysis.get('answer_requirements', {})
                        req_text = []
                        if requirements.get('needs_specific_numbers'):
                            req_text.append("✓ 구체적인 수치 필요")
                        if requirements.get('needs_process_steps'):
                            req_text.append("✓ 단계별 절차 필요")
                        if requirements.get('needs_timeline'):
                            req_text.append("✓ 시간 순서 필요")
                        st.write("\n".join(req_text) if req_text else "특별한 요구사항 없음")
                
                # 응답 데이터 저장
                response_data = {
                    "answer": answer,
                    "model_used": model_used,
                    "total_time": total_time,
                    "complexity": complexity,
                    "importance": importance,
                    "has_version_conflicts": search_stats.get('has_version_conflicts', False),
                    "search_stats": search_stats,
                    "generation_stats": generation_stats
                }
                st.session_state.messages.append({"role": "assistant", "content": response_data})
    
    # 사이드바
    with st.sidebar:
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
        
        st.subheader("🔵 중요한 질문")
        example_important = [
            "계열회사 A가 B에게 채권 500억원 규모를 매각하였습니다. 매각이익은 200억원 입니다. 공시대상은?",
            "A회사가 B계열사에 100억원을 대여하면서 동시에 C계열사의 지분을 취득하는 경우 적용되는 규제는?",
            "여러 계열사와 동시에 거래할 때 검토해야 할 사항들을 종합적으로 설명해주세요"
        ]
        for example in example_important:
            if st.button(example, key=f"important_{example}"):
                st.session_state.new_question = example
                st.rerun()
        
        st.divider()
        
        # 시스템 정보
        with st.expander("ℹ️ 시스템 정보"):
            st.info("""
            **중요도 기반 모델 선택**
            - 🔴 Critical: 대규모 금액, 법적 리스크 → gpt-4o
            - 🟠 High: 중요한 법적 판단 → o3-mini/gpt-4o
            - 🟢 Normal: 일반 질문 → gpt-4o-mini/o4-mini
            
            **질문 분석 프로세스**
            1. 중요도 평가 (금액, 법적 리스크)
            2. 복잡도 분석
            3. 최적 모델 선택
            4. 품질 보증
            """)
        
        # 리셋 버튼
        if st.button("🔄 대화 초기화", type="secondary"):
            st.session_state.messages = []
            st.rerun()
    
    # 페이지 하단
    st.divider()
    st.caption("⚠️ 본 답변은 AI가 생성한 참고자료입니다. 중요한 사항은 반드시 원문을 확인하시기 바랍니다.")
    st.caption("📅 시스템 기준일: 2025년 1월 (최신 규정 반영)")
    
    # 새 질문 처리
    if "new_question" in st.session_state:
        st.session_state.messages.append({"role": "user", "content": st.session_state.new_question})
        del st.session_state.new_question
        st.rerun()

if __name__ == "__main__":
    main()
