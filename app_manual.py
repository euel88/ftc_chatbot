# 파일 이름: app_hybrid_gpt.py (공정거래위원회 AI 법률 보조원 - GPT 하이브리드 통합 버전)

import streamlit as st
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
import json
import openai
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
import re
from collections import defaultdict, Counter
import time
from dataclasses import dataclass
import os
import hashlib
from enum import Enum

# ===== 1. 페이지 설정 및 스타일링 =====
st.set_page_config(
    page_title="전략기획부 AI 법률 보조원", 
    page_icon="⚖️", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 깔끔한 UI를 위한 CSS (기술적 정보 숨김)
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
    
    /* 답변 메시지 스타일 */
    .stChatMessage {
        border-radius: 10px;
        margin-bottom: 1rem;
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

# ===== 2. 데이터 구조 정의 =====
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
    SIMPLE = "simple"      # 단순 사실 확인
    MEDIUM = "medium"      # 중간 복잡도
    COMPLEX = "complex"    # 복잡한 분석 필요

# ===== 3. 문서 버전 관리 및 최신성 검증 시스템 (새로운 기능) =====
class DocumentVersionManager:
    """문서의 버전과 최신성을 관리하는 시스템"""
    
    def __init__(self):
        # 중요 법규 변경사항 데이터베이스
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
        
        # 핵심 수치 패턴 (정규표현식)
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
        
        # 메타데이터에서 날짜 확인
        if 'document_date' in metadata:
            return metadata['document_date']
        
        # 문서 내용에서 날짜 패턴 찾기
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
        # 간단한 정규화 (실제로는 더 정교한 처리 필요)
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
        
        return warnings

class ConflictResolver:
    """상충하는 정보를 해결하는 시스템"""
    
    def __init__(self, version_manager: DocumentVersionManager):
        self.version_manager = version_manager
    
    def resolve_conflicts(self, results: List[SearchResult], query: str) -> List[SearchResult]:
        """검색 결과 중 상충하는 정보를 해결하고 최신 정보를 우선시"""
        
        # 1. 각 결과의 날짜와 구버전 정보 확인
        for result in results:
            doc_date = result.document_date
            warnings = self.version_manager.check_for_outdated_info(result.content, doc_date)
            
            # 메타데이터에 경고 추가
            if warnings:
                result.metadata['warnings'] = warnings
                result.metadata['has_outdated_info'] = True
            else:
                result.metadata['has_outdated_info'] = False
        
        # 2. 중요 수치에 대한 충돌 검사
        critical_info = self._extract_critical_info(results, query)
        if critical_info:
            conflicts = self._find_conflicts(critical_info)
            if conflicts:
                results = self._prioritize_latest_info(results, conflicts)
        
        # 3. 최신 정보를 포함한 결과를 상위로 재정렬
        results.sort(key=lambda r: (
            not r.metadata.get('has_outdated_info', False),  # 구버전 정보가 없는 것 우선
            r.document_date or '1900-01-01',  # 최신 문서 우선
            r.score  # 원래 점수
        ), reverse=True)
        
        return results
    
    def _extract_critical_info(self, results: List[SearchResult], query: str) -> Dict:
        """결과에서 중요 정보 추출"""
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
        
        # 금액 충돌 확인 (예: 50억 vs 100억)
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
        # 구버전 정보를 포함한 결과의 점수를 낮춤
        for conflict in conflicts:
            if conflict['type'] == 'amount_conflict':
                for i, result in enumerate(results):
                    if any(old_val in result.content for old_val in ['50억원', '30억원']):
                        # 구버전 정보를 포함한 결과의 점수를 50% 감소
                        results[i].score *= 0.5
                        results[i].metadata['score_reduced'] = True
                        results[i].metadata['reduction_reason'] = 'outdated_amount'
        
        return results

# ===== 3-1. 질문 복잡도 평가기 (기존 코드) =====
class ComplexityAssessor:
    """질문의 복잡도를 평가하여 처리 방식을 결정"""
    
    def __init__(self):
        # 복잡도 판단 기준
        self.simple_indicators = [
            # 단순 사실 확인
            r'언제', r'며칠', r'기한', r'날짜', r'금액', r'%', r'얼마',
            r'정의[가는]?', r'무엇', r'뜻[이은]?', r'의미[가는]?'
        ]
        
        self.complex_indicators = [
            # 복잡한 분석 필요
            r'동시에', r'여러', r'복합', r'연관', r'영향',
            r'만[약일].*경우', r'[AB].*동시.*[CD]', r'거래.*여러',
            r'전체적', r'종합적', r'분석', r'검토', r'평가',
            r'리스크', r'위험', r'대응', r'전략'
        ]
        
        self.medium_indicators = [
            # 중간 복잡도
            r'어떻게', r'방법', r'절차', r'과정',
            r'주의', r'예외', r'특별', r'고려'
        ]
        
    def assess(self, query: str) -> Tuple[QueryComplexity, float, Dict]:
        """질문의 복잡도를 평가하고 관련 정보 반환"""
        query_lower = query.lower()
        
        # 점수 계산
        simple_score = sum(1 for pattern in self.simple_indicators 
                         if re.search(pattern, query_lower))
        complex_score = sum(2 for pattern in self.complex_indicators 
                          if re.search(pattern, query_lower))
        medium_score = sum(1.5 for pattern in self.medium_indicators 
                         if re.search(pattern, query_lower))
        
        # 추가 복잡도 요인
        # 1. 질문 길이
        if len(query) > 100:
            complex_score += 1
        elif len(query) < 30:
            simple_score += 0.5
            
        # 2. 특수 패턴
        if re.search(r'[AB]회사.*[CD]회사', query_lower):
            complex_score += 2  # 여러 회사 관련
        if '?' in query and query.count('?') > 1:
            complex_score += 1  # 여러 질문
            
        # 최종 복잡도 결정
        total_score = simple_score + medium_score + complex_score
        
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
            
        # 분석 정보
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

# ===== 4. 질문 분류기 (기존 코드 유지) =====
class QuestionClassifier:
    """질문을 분석하여 어떤 매뉴얼을 우선 검색할지 결정"""
    
    def __init__(self):
        # 각 카테고리별 핵심 키워드와 패턴
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
            
            # 키워드 매칭 (가중치 적용)
            for i, keyword in enumerate(info['keywords']):
                if keyword in question_lower:
                    # 앞쪽 키워드일수록 높은 가중치
                    weight = 1.0 if i < 5 else 0.7
                    score += weight
                    matched_keywords.append(keyword)
            
            # 패턴 매칭 (정규표현식)
            for pattern in info.get('patterns', []):
                if re.search(pattern, question_lower):
                    score += 1.5
            
            scores[category] = score
        
        # 가장 높은 점수의 카테고리 선택
        if scores:
            best_category = max(scores, key=scores.get)
            max_possible_score = len(self.categories[best_category]['keywords']) + \
                               len(self.categories[best_category].get('patterns', [])) * 1.5
            confidence = min(scores[best_category] / max_possible_score, 1.0)
            
            # 신뢰도가 너무 낮으면 None 반환
            if confidence < 0.15:
                return None, 0.0
                
            return best_category, confidence
        
        return None, 0.0

# ===== 5. GPT 통합 검색 클래스 (새로운 기능) =====
class GPTIntegratedSearch:
    """GPT가 검색과 분석을 모두 담당하는 통합 검색"""
    
    def __init__(self, chunks: List[Dict]):
        self.chunks = chunks
        self.max_chunks_per_call = 50  # GPT 토큰 제한 고려
        
    def search_and_analyze(self, query: str, top_k: int = 5) -> Tuple[List[SearchResult], Dict]:
        """GPT가 검색과 분석을 통합적으로 수행"""
        start_time = time.time()
        
        # 1단계: GPT가 검색 전략 수립
        search_strategy = self._develop_search_strategy(query)
        
        # 2단계: 청크를 배치로 나누어 GPT 평가
        all_evaluations = []
        for i in range(0, len(self.chunks), self.max_chunks_per_call):
            batch = self.chunks[i:i + self.max_chunks_per_call]
            evaluations = self._evaluate_chunks_batch(query, batch, search_strategy)
            all_evaluations.extend(evaluations)
        
        # 3단계: 상위 결과 선택 및 재정렬
        all_evaluations.sort(key=lambda x: x['relevance_score'], reverse=True)
        top_results = all_evaluations[:top_k * 2]  # 여유있게 선택
        
        # 4단계: GPT가 최종 순위 결정
        final_results = self._finalize_ranking(query, top_results, top_k)
        
        # 통계 생성
        stats = {
            'method': 'gpt_integrated',
            'search_time': time.time() - start_time,
            'chunks_evaluated': len(self.chunks),
            'strategy': search_strategy,
            'estimated_cost': self._estimate_cost(len(self.chunks))
        }
        
        return final_results, stats
    
    def _develop_search_strategy(self, query: str) -> Dict:
        """GPT가 검색 전략을 수립"""
        prompt = f"""
        다음 법률 질문을 분석하여 검색 전략을 수립하세요:
        
        질문: {query}
        
        반드시 다음과 같은 JSON 형식으로 응답하세요:
        {{
            "key_concepts": ["핵심 개념1", "핵심 개념2"],
            "related_concepts": ["관련 개념1", "관련 개념2"],
            "legal_areas": ["관련 법률 영역1", "관련 법률 영역2"],
            "search_focus": "검색 초점 설명"
        }}
        """
        
        try:
            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            strategy = json.loads(response.choices[0].message.content)
            
            # 필수 필드 검증
            required_fields = ["key_concepts", "related_concepts", "legal_areas", "search_focus"]
            for field in required_fields:
                if field not in strategy:
                    strategy[field] = [] if field != "search_focus" else "일반 검색"
            
            return strategy
            
        except Exception as e:
            print(f"Error in _develop_search_strategy: {str(e)}")
            # 오류 발생 시 기본 전략 반환
            return {
                "key_concepts": [query],
                "related_concepts": [],
                "legal_areas": ["공정거래법"],
                "search_focus": "질문과 관련된 모든 문서 검색"
            }
    
    def _evaluate_chunks_batch(self, query: str, chunks: List[Dict], strategy: Dict) -> List[Dict]:
        """GPT가 청크 배치의 관련성을 평가"""
        # 청크 요약 생성
        chunks_summary = "\n".join([
            f"[청크 {i}] ({chunk['source']}, p.{chunk['page']}): {chunk['content'][:150]}..."
            for i, chunk in enumerate(chunks)
        ])
        
        prompt = f"""
        질문: {query}
        검색 전략: {json.dumps(strategy, ensure_ascii=False)}
        
        다음 문서 청크들의 관련성을 평가하세요.
        각 청크에 대해 0-10점의 관련성 점수와 이유를 제공하세요.
        
        {chunks_summary}
        
        반드시 다음과 같은 JSON 객체 형식으로 응답하세요:
        {{
            "evaluations": [
                {{"chunk_index": 0, "relevance_score": 8.5, "reason": "관련성 이유"}},
                {{"chunk_index": 1, "relevance_score": 6.0, "reason": "관련성 이유"}}
            ]
        }}
        """
        
        try:
            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=2000,
                response_format={"type": "json_object"}
            )
            
            # 응답 파싱
            response_content = response.choices[0].message.content
            evaluations = json.loads(response_content)
            
            # 응답 구조 검증
            if isinstance(evaluations, dict) and 'evaluations' in evaluations:
                eval_list = evaluations['evaluations']
            elif isinstance(evaluations, list):
                eval_list = evaluations
            else:
                # 예상치 못한 형식인 경우 빈 리스트 반환
                print(f"Unexpected GPT response format: {evaluations}")
                return []
            
            # 원본 청크 정보와 병합
            results = []
            for eval_item in eval_list:
                # 타입 검증
                if not isinstance(eval_item, dict):
                    continue
                    
                # 필수 필드 확인
                if 'chunk_index' not in eval_item:
                    continue
                    
                idx = eval_item.get('chunk_index', -1)
                if isinstance(idx, int) and 0 <= idx < len(chunks):
                    chunk = chunks[idx]
                    results.append({
                        'chunk': chunk,
                        'relevance_score': float(eval_item.get('relevance_score', 0)),
                        'reason': eval_item.get('reason', '')
                    })
            
            return results
            
        except Exception as e:
            print(f"Error in _evaluate_chunks_batch: {str(e)}")
            # 오류 발생 시 기본 점수로 모든 청크 반환
            return [{
                'chunk': chunk,
                'relevance_score': 5.0,
                'reason': 'GPT 평가 실패로 기본 점수 부여'
            } for chunk in chunks]
    
    def _finalize_ranking(self, query: str, candidates: List[Dict], top_k: int) -> List[SearchResult]:
        """GPT가 최종 순위를 결정"""
        # 후보 요약
        candidates_summary = "\n".join([
            f"[후보 {i}] (점수: {c['relevance_score']:.1f}) {c['chunk']['source']}: {c['chunk']['content'][:100]}..."
            for i, c in enumerate(candidates[:10])
        ])
        
        prompt = f"""
        질문: {query}
        
        다음 후보들 중에서 가장 관련성 높은 {top_k}개를 선택하고 순위를 매기세요.
        법적 정확성과 실무적 유용성을 모두 고려하세요.
        
        {candidates_summary}
        
        반드시 JSON 형식으로 응답하세요:
        {{
            "selected_indices": [0, 3, 1, 2, 4],
            "explanation": "선택 이유 설명"
        }}
        
        selected_indices는 선택한 후보의 인덱스를 순서대로 나열한 배열이어야 합니다.
        """
        
        try:
            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                response_format={"type": "json_object"}
            )
            
            # 응답 파싱
            result = json.loads(response.choices[0].message.content)
            
            # 인덱스 추출
            if isinstance(result, dict) and 'selected_indices' in result:
                indices = result['selected_indices']
            else:
                # 텍스트에서 숫자 추출 시도
                content = str(result)
                indices = re.findall(r'\d+', content)[:top_k]
            
            # SearchResult 객체 생성
            results = []
            for idx in indices[:top_k]:
                try:
                    idx_int = int(idx)
                    if 0 <= idx_int < len(candidates):
                        candidate = candidates[idx_int]
                        chunk = candidate['chunk']
                        results.append(SearchResult(
                            chunk_id=chunk.get('chunk_id', str(idx_int)),
                            content=chunk['content'],
                            score=candidate['relevance_score'],
                            source=chunk['source'],
                            page=chunk['page'],
                            chunk_type=chunk.get('chunk_type', 'unknown'),
                            metadata=json.loads(chunk.get('metadata', '{}'))
                        ))
                except (ValueError, IndexError):
                    continue
            
            # 결과가 부족하면 상위 후보로 채우기
            if len(results) < top_k:
                for candidate in candidates:
                    if len(results) >= top_k:
                        break
                    # 이미 추가된 청크인지 확인
                    chunk_id = candidate['chunk'].get('chunk_id', '')
                    if not any(r.chunk_id == chunk_id for r in results):
                        chunk = candidate['chunk']
                        results.append(SearchResult(
                            chunk_id=chunk_id or str(len(results)),
                            content=chunk['content'],
                            score=candidate['relevance_score'],
                            source=chunk['source'],
                            page=chunk['page'],
                            chunk_type=chunk.get('chunk_type', 'unknown'),
                            metadata=json.loads(chunk.get('metadata', '{}'))
                        ))
            
            return results
            
        except Exception as e:
            print(f"Error in _finalize_ranking: {str(e)}")
            # 오류 발생 시 점수 순으로 상위 k개 반환
            results = []
            for i, candidate in enumerate(candidates[:top_k]):
                chunk = candidate['chunk']
                results.append(SearchResult(
                    chunk_id=chunk.get('chunk_id', str(i)),
                    content=chunk['content'],
                    score=candidate.get('relevance_score', 5.0),
                    source=chunk['source'],
                    page=chunk['page'],
                    chunk_type=chunk.get('chunk_type', 'unknown'),
                    metadata=json.loads(chunk.get('metadata', '{}'))
                ))
            return results
    
    def _estimate_cost(self, num_chunks: int) -> float:
        """예상 비용 계산 (달러)"""
        # GPT-4o 가격 기준 (대략적)
        tokens_per_chunk = 200
        total_tokens = num_chunks * tokens_per_chunk
        price_per_1k_tokens = 0.01
        return (total_tokens / 1000) * price_per_1k_tokens

# ===== 6. 하이브리드 RAG 파이프라인 (핵심 통합) =====
class HybridRAGPipeline:
    """복잡도에 따라 전통적 방식과 GPT 통합 방식을 선택하는 하이브리드 파이프라인"""
    
    def __init__(self, embedding_model, reranker_model, index, chunks):
        self.embedding_model = embedding_model
        self.reranker_model = reranker_model
        self.index = index
        self.chunks = chunks
        
        # 컴포넌트 초기화
        self.classifier = QuestionClassifier()
        self.complexity_assessor = ComplexityAssessor()
        self.gpt_search = GPTIntegratedSearch(chunks)
        
        # 버전 관리 및 충돌 해결 시스템 초기화
        self.version_manager = DocumentVersionManager()
        self.conflict_resolver = ConflictResolver(self.version_manager)
        
        # 매뉴얼별 청크 인덱스 미리 구축
        self.manual_indices = self._build_manual_indices()
        
        # 검색 결과 캐시
        self.search_cache = {}
        self.cache_max_size = 100
        
        # 각 청크의 날짜 정보 추출 및 저장
        self._extract_chunk_dates()
        
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
    
    def process_query(self, query: str, top_k: int = 5) -> Tuple[List[SearchResult], Dict]:
        """질문 복잡도에 따라 최적의 처리 방식을 선택"""
        # 1. 복잡도 평가
        complexity, confidence, complexity_analysis = self.complexity_assessor.assess(query)
        
        # 2. 복잡도에 따른 처리
        if complexity == QueryComplexity.SIMPLE:
            # 단순 질문: 빠른 전통적 검색
            results, stats = self._fast_traditional_search(query, top_k)
            stats['processing_mode'] = 'fast_traditional'
            
        elif complexity == QueryComplexity.COMPLEX:
            # 복잡한 질문: GPT 통합 처리
            results, stats = self.gpt_search.search_and_analyze(query, top_k)
            stats['processing_mode'] = 'gpt_integrated'
            
        else:  # MEDIUM
            # 중간 복잡도: 하이브리드 접근
            # 먼저 빠른 검색으로 후보를 찾고, GPT로 정제
            initial_results, initial_stats = self._fast_traditional_search(query, top_k * 3)
            results, stats = self._gpt_enhance_results(query, initial_results, top_k)
            stats['processing_mode'] = 'hybrid'
            stats['initial_search_time'] = initial_stats['search_time']
        
        # 3. 최신성 검증 및 충돌 해결
        results = self.conflict_resolver.resolve_conflicts(results, query)
        
        # 4. 구버전 정보 경고 수집
        outdated_warnings = []
        for result in results:
            if result.metadata.get('has_outdated_info'):
                outdated_warnings.extend(result.metadata.get('warnings', []))
        
        # 5. 복잡도 정보 추가
        stats['complexity'] = complexity.value
        stats['complexity_confidence'] = confidence
        stats['complexity_analysis'] = complexity_analysis
        stats['outdated_warnings'] = outdated_warnings
        stats['has_version_conflicts'] = len(outdated_warnings) > 0
        
        return results, stats
    
    def _fast_traditional_search(self, query: str, top_k: int) -> Tuple[List[SearchResult], Dict]:
        """기존의 빠른 벡터 검색 방식"""
        start_time = time.time()
        
        # 캐시 확인
        cache_key = hashlib.md5(f"{query}_{top_k}_traditional".encode()).hexdigest()
        if cache_key in self.search_cache:
            cached = self.search_cache[cache_key]
            stats = cached['stats'].copy()
            stats['cache_hit'] = True
            return cached['results'], stats
        
        # 질문 분류
        category, cat_confidence = self.classifier.classify(query)
        
        # 검색 인덱스 결정
        if category and cat_confidence > 0.3:
            primary_indices = self.manual_indices.get(category, [])
            secondary_indices = []
            for cat, indices in self.manual_indices.items():
                if cat != category and cat != '기타':
                    secondary_indices.extend(indices)
        else:
            primary_indices = list(range(len(self.chunks)))
            secondary_indices = []
        
        # 벡터 검색
        query_vector = self.embedding_model.encode([query])
        query_vector = np.array(query_vector, dtype=np.float32)
        
        k_search = min(len(self.chunks), top_k * 10)
        scores, indices = self.index.search(query_vector, k_search)
        
        # 결과 수집
        results = []
        seen_chunks = set()
        
        # 우선순위 인덱스에서 결과 수집
        if primary_indices:
            primary_set = set(primary_indices)
            for idx, score in zip(indices[0], scores[0]):
                if idx in primary_set and idx not in seen_chunks:
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
        
        # 통계
        stats = {
            'search_time': time.time() - start_time,
            'category': category,
            'category_confidence': cat_confidence,
            'cache_hit': False
        }
        
        # 캐시 저장
        if stats['search_time'] < 1.0:
            self.search_cache[cache_key] = {
                'results': results,
                'stats': stats
            }
        
        return results, stats
    
    def _gpt_enhance_results(self, query: str, initial_results: List[SearchResult], 
                           top_k: int) -> Tuple[List[SearchResult], Dict]:
        """GPT로 검색 결과를 정제하고 개선"""
        start_time = time.time()
        
        # GPT에게 재정렬과 분석 요청
        results_summary = "\n".join([
            f"[결과 {i+1}] (점수: {r.score:.2f}) {r.source} p.{r.page}:\n{r.content[:200]}..."
            for i, r in enumerate(initial_results[:10])
        ])
        
        prompt = f"""
        사용자 질문: {query}
        
        다음은 초기 검색 결과입니다. 이 중에서 질문에 가장 관련성 높은 {top_k}개를 선택하고,
        각각이 왜 중요한지 설명해주세요.
        
        {results_summary}
        
        다음 형식으로 응답하세요:
        1. 선택한 결과 번호들: [1, 3, 2, ...]
        2. 각 결과가 중요한 이유
        """
        
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1000
        )
        
        # 응답 파싱
        content = response.choices[0].message.content
        selected_indices = []
        
        # 숫자 추출
        numbers = re.findall(r'\[([^\]]+)\]', content)
        if numbers:
            indices_str = numbers[0]
            selected_indices = [int(x.strip()) - 1 for x in indices_str.split(',') 
                              if x.strip().isdigit()]
        
        # 선택된 결과 반환
        enhanced_results = []
        for idx in selected_indices[:top_k]:
            if 0 <= idx < len(initial_results):
                enhanced_results.append(initial_results[idx])
        
        # 부족하면 원래 결과로 채우기
        if len(enhanced_results) < top_k:
            for result in initial_results:
                if result not in enhanced_results:
                    enhanced_results.append(result)
                    if len(enhanced_results) >= top_k:
                        break
        
        stats = {
            'enhancement_time': time.time() - start_time,
            'enhanced_count': len(selected_indices)
        }
        
        return enhanced_results, stats

# ===== 7. 동적 Temperature 답변 생성 (기존 코드 유지 + 개선) =====
def determine_temperature(query: str, complexity: QueryComplexity) -> float:
    """질문 유형과 복잡도에 따라 최적의 temperature 결정"""
    query_lower = query.lower()
    
    # 복잡도별 기본값
    base_temps = {
        QueryComplexity.SIMPLE: 0.1,
        QueryComplexity.MEDIUM: 0.3,
        QueryComplexity.COMPLEX: 0.5
    }
    
    temp = base_temps[complexity]
    
    # 질문 유형별 조정
    if any(keyword in query_lower for keyword in ['언제', '며칠', '기한', '날짜', '금액', '%']):
        temp = min(temp, 0.1)
    elif any(keyword in query_lower for keyword in ['전략', '대응', '리스크', '주의', '권장']):
        temp = max(temp, 0.7)
    
    return temp

def generate_answer(query: str, results: List[SearchResult], stats: Dict) -> str:
    """GPT-4o를 활용한 고품질 답변 생성 (최신 정보 우선)"""
    
    # 구버전 정보 경고 확인
    has_outdated = stats.get('has_version_conflicts', False)
    outdated_warnings = stats.get('outdated_warnings', [])
    
    # 컨텍스트 구성 (최신 정보 우선)
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
    
    # 최신 정보를 먼저, 구버전 정보는 나중에
    context_parts = latest_info_parts + outdated_info_parts
    context = "\n---\n".join(context_parts)
    
    # 중요 법규 변경사항 명시
    critical_updates = ""
    if has_outdated:
        critical_updates = "\n\n[중요 법규 변경사항]"
        for warning in outdated_warnings:
            if warning['severity'] == 'critical':
                critical_updates += f"\n- {warning['regulation']}: {warning['found']} → {warning['current']} (변경일: {warning['changed_date']})"
    
    # 복잡도 정보 활용
    complexity = QueryComplexity(stats.get('complexity', 'medium'))
    temperature = determine_temperature(query, complexity)
    
    # 처리 모드별 특별 지시
    mode_instructions = {
        'gpt_integrated': "GPT가 심층 분석한 결과를 바탕으로 종합적인 답변을 제공하세요.",
        'hybrid': "초기 검색 결과를 GPT가 정제한 내용을 바탕으로 답변하세요.",
        'fast_traditional': "제공된 참고 자료를 바탕으로 간결하고 정확한 답변을 제공하세요."
    }
    
    mode = stats.get('processing_mode', 'fast_traditional')
    extra_instruction = mode_instructions.get(mode, "")
    
    # 카테고리별 특화 지시사항
    category = stats.get('category')
    if category:
        category_instructions = {
            '대규모내부거래': "특히 이사회 의결 요건, 공시 기한, 면제 조건을 명확히 설명하세요. 금액 기준은 반드시 최신 기준(100억원 이상 또는 자본금 및 자본총계 중 큰 금액의 5% 이상)을 사용하세요.",
            '현황공시': "공시 주체, 시기, 제출 서류를 구체적으로 안내하세요.",
            '비상장사 중요사항': "공시 대상 거래, 기한, 제출 방법을 상세히 설명하세요."
        }
        extra_instruction += f"\n{category_instructions.get(category, '')}"
    
    # 시스템 프롬프트
    system_prompt = f"""당신은 한국 공정거래위원회 전문가입니다.
제공된 자료만을 근거로 정확하고 실무적인 답변을 제공하세요.

질문 복잡도: {complexity.value}
처리 방식: {mode}

중요: 법규가 변경된 경우 반드시 최신 정보를 기준으로 답변하세요. 
특히 대규모내부거래 금액 기준은 2023년부터 100억원 이상으로 변경되었습니다.

답변은 다음 구조를 따라주세요:
1. 핵심 답변 (1-2문장) - 최신 법규 기준
2. 상세 설명 (근거 조항 포함)
3. 주의사항 또는 예외사항 (있는 경우)
4. 법규 변경사항 (중요한 변경이 있었던 경우)

{extra_instruction}"""
    
    # GPT-4o 호출
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"""다음 자료를 바탕으로 질문에 답변해주세요.
{critical_updates}

[참고 자료]
{context}

[질문]
{query}

{"간결하고 명확하게" if complexity == QueryComplexity.SIMPLE else "상세하고 실무적으로"} 답변해주세요.
구버전 정보와 최신 정보가 상충하는 경우, 반드시 최신 정보를 기준으로 답변하세요."""}
    ]
    
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=temperature,
        max_tokens=1500
    )
    
    return response.choices[0].message.content

# ===== 8. 모델 및 데이터 로딩 =====
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
        
        # 데이터 로드
        with st.spinner("🤖 AI 시스템을 준비하는 중... (최초 1회만 수행됩니다)"):
            # 벡터 인덱스와 청크 데이터 로드
            index = faiss.read_index("manuals_vector_db.index")
            with open("all_manual_chunks.json", "r", encoding="utf-8") as f:
                chunks = json.load(f)
            
            # 임베딩 모델 로드
            try:
                embedding_model = SentenceTransformer('jhgan/ko-sroberta-multitask')
            except Exception as e:
                st.warning("한국어 모델 로드 실패. 대체 모델을 사용합니다.")
                embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
            
            # 재정렬 모델 로드 (선택적)
            try:
                reranker_model = CrossEncoder('Dongjin-kr/ko-reranker')
            except:
                reranker_model = None
        
        return embedding_model, reranker_model, index, chunks
        
    except Exception as e:
        st.error(f"시스템 초기화 실패: {str(e)}")
        return None, None, None, None

# ===== 9. 메인 UI (하이브리드 시스템 반영) =====
def main():
    # 헤더
    st.markdown("""
    <div class="main-header">
        <h1>⚖️ 전략기획부 AI 법률 보조원</h1>
        <p>공정거래위원회 규정 및 매뉴얼 기반 지능형 하이브리드 Q&A 시스템</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 모델 로드
    models = load_models_and_data()
    if not all(models):
        st.stop()
    
    embedding_model, reranker_model, index, chunks = models
    
    # 하이브리드 RAG 시스템 초기화
    rag = HybridRAGPipeline(embedding_model, reranker_model, index, chunks)
    
    # 세션 상태 초기화
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
                    # AI 응답 표시
                    if isinstance(message["content"], dict):
                        st.write(message["content"]["answer"])
                        
                        # 복잡도 표시
                        complexity = message["content"].get("complexity", "unknown")
                        complexity_html = f'<span class="complexity-indicator complexity-{complexity}">{complexity.upper()}</span>'
                        st.markdown(f"처리 복잡도: {complexity_html}", unsafe_allow_html=True)
                        
                        # 시간 정보 표시
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
        
        # 사용자 입력
        if prompt := st.chat_input("질문을 입력하세요 (예: 대규모내부거래 공시 기한은?)"):
            # 사용자 메시지 추가
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)
            
            # AI 응답 생성
            with st.chat_message("assistant"):
                # 전체 시간 측정 시작
                total_start_time = time.time()
                
                # 하이브리드 검색 수행
                search_start_time = time.time()
                with st.spinner("🔍 최적의 검색 방식을 선택하여 자료를 검색하는 중..."):
                    results, stats = rag.process_query(prompt, top_k=5)
                search_time = time.time() - search_start_time
                
                # 답변 생성
                generation_start_time = time.time()
                with st.spinner("💭 답변을 생성하는 중..."):
                    answer = generate_answer(prompt, results, stats)
                generation_time = time.time() - generation_start_time
                
                # 전체 시간 계산
                total_time = time.time() - total_start_time
                
                # 답변 표시
                st.write(answer)
                
                # 복잡도 표시
                complexity = stats.get('complexity', 'unknown')
                mode = stats.get('processing_mode', 'unknown')
                complexity_html = f'<span class="complexity-indicator complexity-{complexity}">{complexity.upper()}</span>'
                st.markdown(f"질문 복잡도: {complexity_html} | 처리 방식: **{mode}**", unsafe_allow_html=True)
                
                # 시간 정보 표시
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("🔍 검색", f"{search_time:.1f}초")
                with col2:
                    st.metric("✍️ 답변 생성", f"{generation_time:.1f}초")
                with col3:
                    st.metric("⏱️ 전체", f"{total_time:.1f}초")
                
                # 성능 분석 (접을 수 있게)
                with st.expander("🔍 상세 정보 보기"):
                    # 구버전 정보 경고 표시
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
                    
                    # 처리 방식 설명
                    mode_descriptions = {
                        'fast_traditional': "빠른 벡터 검색을 사용하여 신속하게 처리했습니다.",
                        'hybrid': "초기 검색 후 GPT로 결과를 정제하여 정확도를 높였습니다.",
                        'gpt_integrated': "GPT가 전체 과정을 담당하여 심층적으로 분석했습니다."
                    }
                    st.info(f"🎯 **처리 방식**: {mode_descriptions.get(mode, '알 수 없음')}")
                    
                    # 복잡도 분석
                    if 'complexity_analysis' in stats:
                        analysis = stats['complexity_analysis']
                        st.subheader("📊 복잡도 분석")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("단순 점수", f"{analysis['simple_score']:.1f}")
                        with col2:
                            st.metric("중간 점수", f"{analysis['medium_score']:.1f}")
                        with col3:
                            st.metric("복잡 점수", f"{analysis['complex_score']:.1f}")
                        
                        if mode == 'gpt_integrated':
                            st.warning(f"💰 예상 비용: 일반 검색의 약 {analysis['estimated_cost_multiplier']:.1f}배")
                    
                    # 검색 통계
                    if stats.get('category'):
                        st.info(f"📂 **{stats['category']}** 카테고리로 분류 (신뢰도: {stats.get('category_confidence', 0):.0%})")
                    
                    # 참고 자료
                    st.subheader("📚 참고 자료")
                    for i, result in enumerate(results[:3]):
                        # 구버전 정보 표시
                        version_indicator = ""
                        if result.metadata.get('has_outdated_info'):
                            version_indicator = " ⚠️ **[구버전 정보 포함]**"
                        
                        st.caption(f"**{result.source}** - 페이지 {result.page} (관련도: {result.score:.2f}){version_indicator}")
                        
                        # 문서 날짜 표시
                        if result.document_date:
                            st.caption(f"📅 문서 날짜: {result.document_date}")
                        
                        with st.container():
                            # 내용 표시 (구버전 정보는 취소선 처리)
                            content = result.content[:300] + "..." if len(result.content) > 300 else result.content
                            
                            # 50억원이나 30억원이 포함된 경우 하이라이트
                            if '50억원' in content or '30억원' in content:
                                content = re.sub(r'(50억원|30억원)', r'~~\1~~ → **100억원**', content)
                            
                            st.text(content)
                    
                    # 성능 평가
                    if total_time < 3:
                        st.success("⚡ 매우 빠른 응답 속도!")
                    elif total_time < 5:
                        st.info("✅ 적절한 응답 속도")
                    else:
                        st.warning("⏰ 응답 시간이 다소 길었습니다 (복잡한 질문으로 인한 정상적인 처리)")
                
                # 세션에 저장
                response_data = {
                    "answer": answer,
                    "search_time": search_time,
                    "generation_time": generation_time,
                    "total_time": total_time,
                    "complexity": complexity,
                    "processing_mode": mode
                }
                st.session_state.messages.append({"role": "assistant", "content": response_data})
    
    # 하단 안내
    st.divider()
    st.caption("⚠️ 본 답변은 AI가 생성한 참고자료입니다. 중요한 사항은 반드시 원문을 확인하시기 바랍니다.")
    
    # 사이드바 (예시 질문 - 복잡도별로 구성)
    with st.sidebar:
        st.header("💡 예시 질문")
        
        st.subheader("🟢 단순 질문 (빠른 검색)")
        if st.button("대규모내부거래 공시 기한은?"):
            st.session_state.new_question = "대규모내부거래 이사회 의결 후 공시 기한은 며칠인가요?"
            st.rerun()
        if st.button("이사회 의결 금액 기준은?"):
            st.session_state.new_question = "대규모내부거래에서 이사회 의결이 필요한 거래 금액은?"
            st.rerun()
            
        st.subheader("🟡 중간 복잡도 (하이브리드)")
        if st.button("계열사 거래 시 주의사항은?"):
            st.session_state.new_question = "계열사와 자금거래를 할 때 어떤 절차를 거쳐야 하고 주의할 점은 무엇인가요?"
            st.rerun()
        if st.button("비상장사 주식 양도 절차는?"):
            st.session_state.new_question = "비상장회사가 주식을 양도할 때 필요한 절차와 공시 의무는 어떻게 되나요?"
            st.rerun()
            
        st.subheader("🔴 복잡한 질문 (GPT 통합)")
        if st.button("복합 거래 분석"):
            st.session_state.new_question = "A회사가 B계열사에 자금을 대여하면서 동시에 C계열사의 주식을 취득하는 경우, 각각 어떤 규제가 적용되고 공시는 어떻게 해야 하나요?"
            st.rerun()
        if st.button("종합적 리스크 검토"):
            st.session_state.new_question = "우리 회사가 여러 계열사와 동시에 거래를 진행할 때 대규모내부거래 규제와 관련하여 종합적으로 검토해야 할 리스크와 대응 전략은?"
            st.rerun()
        
        st.divider()
        st.caption("💡 복잡한 질문일수록 더 정확한 답변을 제공하지만, 처리 시간과 비용이 증가합니다.")
    
    # 새 질문 처리
    if "new_question" in st.session_state:
        st.session_state.messages.append({"role": "user", "content": st.session_state.new_question})
        del st.session_state.new_question
        st.rerun()

if __name__ == "__main__":
    main()
