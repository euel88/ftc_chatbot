# 파일 이름: app_ftc.py (공정거래위원회 문서 특화 ChatGPT 수준 정확도 버전)

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
import pandas as pd

# --- 1. 페이지 설정 및 스타일링 ---
st.set_page_config(
    page_title="전략기획부 AI 법률 보조원", 
    page_icon="⚖️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# 커스텀 CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f4788;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-container {
        background: #f0f5ff;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f4788;
    }
    .source-tag {
        background: #e8f0fe;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-size: 0.9rem;
        color: #1967d2;
    }
    .importance-high {
        color: #d93025;
        font-weight: bold;
    }
    .importance-medium {
        color: #f9ab00;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# OpenAI API 설정
try:
    openai.api_key = st.secrets["OPENAI_API_KEY"]
except:
    st.error("⚠️ OpenAI API 키를 Streamlit Secrets에 등록해주세요.")
    st.stop()

# --- 2. 데이터 구조 정의 ---

@dataclass
class SearchResult:
    """검색 결과를 담는 데이터 클래스"""
    chunk_id: str
    content: str
    score: float
    source: str
    page: int
    chunk_type: str
    importance: float
    keywords: List[str]
    metadata: Dict

class FTCQueryAnalyzer:
    """공정거래 관련 쿼리 분석기"""
    
    def __init__(self):
        self.query_patterns = {
            'definition': re.compile(r'(정의|의미|뜻|개념|무엇|뭐|란)'),
            'requirement': re.compile(r'(요건|조건|기준|자격)'),
            'deadline': re.compile(r'(기한|기간|언제|까지|이내|마감)'),
            'procedure': re.compile(r'(절차|방법|어떻게|과정|단계)'),
            'penalty': re.compile(r'(벌칙|처벌|과태료|제재|벌금)'),
            'obligation': re.compile(r'(의무|필수|반드시|해야|하여야)'),
            'exception': re.compile(r'(예외|제외|면제|특례)'),
            'calculation': re.compile(r'(계산|산정|산출|비율|퍼센트)'),
            'reference': re.compile(r'(조항|조|항|호|별표|서식)')
        }
        
        self.entity_patterns = {
            'article': re.compile(r'제?\s*(\d+)조(?:의(\d+))?'),
            'percentage': re.compile(r'(\d+(?:\.\d+)?)\s*(%|퍼센트|프로)'),
            'amount': re.compile(r'(\d+(?:,\d{3})*)\s*(원|만원|억원)'),
            'days': re.compile(r'(\d+)\s*일'),
            'company_type': re.compile(r'(상장|비상장|계열사|자회사|손자회사|특수관계인)')
        }
    
    def analyze(self, query: str) -> Dict:
        """쿼리 분석 및 의도 파악"""
        analysis = {
            'query_type': self._identify_query_type(query),
            'entities': self._extract_entities(query),
            'key_terms': self._extract_key_terms(query),
            'priority_chunks': self._determine_priority_chunks(query)
        }
        return analysis
    
    def _identify_query_type(self, query: str) -> List[str]:
        """쿼리 타입 식별"""
        types = []
        for q_type, pattern in self.query_patterns.items():
            if pattern.search(query):
                types.append(q_type)
        return types if types else ['general']
    
    def _extract_entities(self, query: str) -> Dict:
        """엔티티 추출"""
        entities = {}
        for entity_type, pattern in self.entity_patterns.items():
            matches = pattern.findall(query)
            if matches:
                entities[entity_type] = matches
        return entities
    
    def _extract_key_terms(self, query: str) -> List[str]:
        """핵심 용어 추출"""
        # 공정거래 도메인 특화 용어
        ftc_terms = [
            '대규모내부거래', '공시', '이사회', '의결', '특수관계인',
            '지분율', '의결권', '상호출자', '순환출자', '기업집단',
            '동일인', '친족', '임원', '주주', '독립경영', '부당지원'
        ]
        
        key_terms = []
        query_lower = query.lower()
        
        for term in ftc_terms:
            if term in query_lower:
                key_terms.append(term)
        
        return key_terms
    
    def _determine_priority_chunks(self, query: str) -> List[str]:
        """우선순위 청크 타입 결정"""
        query_types = self._identify_query_type(query)
        
        priority_map = {
            'definition': ['definition', 'article'],
            'requirement': ['article', 'section'],
            'deadline': ['article', 'penalty', 'section'],
            'procedure': ['section', 'article'],
            'penalty': ['penalty', 'article'],
            'obligation': ['article', 'penalty'],
            'exception': ['article', 'section'],
            'calculation': ['article', 'section'],
            'reference': ['article']
        }
        
        priority_chunks = []
        for q_type in query_types:
            if q_type in priority_map:
                priority_chunks.extend(priority_map[q_type])
        
        return list(set(priority_chunks)) if priority_chunks else ['article', 'section']

# --- 3. 캐싱된 리소스 로딩 ---

@st.cache_resource(show_spinner=False)
def load_models_and_data():
    """모델과 데이터를 효율적으로 로드"""
    with st.spinner("🔧 AI 시스템을 초기화하는 중..."):
        try:
            # 환경 확인 (로컬인지 클라우드인지)
            is_cloud = os.environ.get('STREAMLIT_CLOUD', 'false').lower() == 'true'
            
            # 필수 파일 존재 여부 먼저 확인
            if not os.path.exists("manuals_vector_db.index"):
                st.error("❌ 벡터 데이터베이스 파일을 찾을 수 없습니다.")
                st.info("💡 먼저 prepare_pdfs_ftc.py를 실행하여 데이터를 준비하세요.")
                return None, None, None, None, None
                
            if not os.path.exists("all_manual_chunks.json"):
                st.error("❌ 청크 데이터 파일을 찾을 수 없습니다.")
                return None, None, None, None, None
            
            # 임베딩 모델 로드
            embedding_model = None
            
            # Streamlit Cloud 환경에서는 항상 온라인 모델 사용
            if is_cloud:
                st.info("☁️ 클라우드 환경에서 실행 중입니다. 온라인 모델을 로드합니다...")
                try:
                    # 주의: 이 모델은 prepare_pdfs_ftc.py에서 사용한 것과 동일해야 함
                    embedding_model = SentenceTransformer('jhgan/ko-sroberta-multitask')
                    st.success("✅ 한국어 임베딩 모델 로드 성공!")
                except Exception as e:
                    st.warning(f"⚠️ 한국어 모델 로드 실패: {str(e)}")
                    st.info("🔄 대체 다국어 모델을 사용합니다...")
                    try:
                        # 대체 모델 (prepare_pdfs에서도 같은 대체 모델을 사용해야 함)
                        embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
                        st.warning("⚠️ 주의: 대체 모델을 사용 중입니다. 검색 정확도가 낮을 수 있습니다.")
                    except Exception as e:
                        st.error(f"❌ 모든 임베딩 모델 로드 실패: {str(e)}")
                        return None, None, None, None, None
            
            # 로컬 환경에서는 로컬 모델 우선 시도
            else:
                # 로컬 모델 경로
                local_model_path = r"C:\Users\OK\Desktop\파이썬 코드 모음\챗봇_공정위 기업집단 관련\models\ko-sroberta-multitask"
                
                # 상대 경로로도 시도
                if not os.path.exists(local_model_path):
                    script_dir = os.path.dirname(os.path.abspath(__file__))
                    local_model_path = os.path.join(script_dir, "models", "ko-sroberta-multitask")
                
                # 로컬 모델 로드 시도
                if os.path.exists(local_model_path):
                    try:
                        st.info("💻 로컬 환경: 저장된 모델을 사용합니다...")
                        embedding_model = SentenceTransformer(local_model_path)
                        st.success("✅ 로컬 모델 로드 성공!")
                    except Exception as e:
                        st.warning(f"⚠️ 로컬 모델 로드 실패: {str(e)}")
                
                # 로컬 모델이 없으면 온라인 모델 시도
                if embedding_model is None:
                    try:
                        st.info("🌐 온라인에서 모델을 다운로드합니다...")
                        # SSL 오류 방지 (로컬 환경에서만)
                        import ssl
                        ssl._create_default_https_context = ssl._create_unverified_context
                        os.environ['HF_HUB_DISABLE_SSL_VERIFY'] = '1'
                        
                        embedding_model = SentenceTransformer('jhgan/ko-sroberta-multitask')
                        st.success("✅ 온라인 모델 로드 성공!")
                    except Exception as e:
                        st.error(f"❌ 임베딩 모델 로드 실패: {str(e)}")
                        return None, None, None, None, None
            
            # CrossEncoder 모델 로드 (재정렬용)
            try:
                reranker_model = CrossEncoder('Dongjin-kr/ko-reranker')
            except:
                st.warning("⚠️ 한국어 재정렬 모델 로드 실패. 기본 재정렬 모델을 사용합니다.")
                try:
                    # 대체 재정렬 모델
                    reranker_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
                except:
                    st.warning("⚠️ 재정렬 모델을 사용할 수 없습니다. 기본 검색만 수행합니다.")
                    reranker_model = None
            
            # 인덱스와 데이터 로드
            index = faiss.read_index("manuals_vector_db.index")
            
            with open("all_manual_chunks.json", "r", encoding="utf-8") as f:
                chunks_data = json.load(f)
            
            # 청크 타입별 인덱스 생성 (빠른 필터링용)
            chunk_type_index = defaultdict(list)
            for idx, chunk in enumerate(chunks_data):
                chunk_type_index[chunk.get('chunk_type', 'unknown')].append(idx)
            
            # 시스템 정보 표시
            st.success(f"✅ 시스템 준비 완료!")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("총 정보 단위", f"{len(chunks_data):,}개")
            with col2:
                st.metric("벡터 차원", f"{index.d}")
            with col3:
                env_type = "☁️ 클라우드" if is_cloud else "💻 로컬"
                st.metric("실행 환경", env_type)
            
            return embedding_model, reranker_model, index, chunks_data, chunk_type_index
            
        except Exception as e:
            st.error(f"❌ 시스템 초기화 실패: {str(e)}")
            st.info("💡 문제 해결 방법:")
            st.info("1. prepare_pdfs_ftc.py를 먼저 실행했는지 확인하세요")
            st.info("2. 생성된 파일들이 GitHub에 제대로 업로드되었는지 확인하세요")
            st.info("3. requirements.txt에 필요한 패키지가 모두 포함되었는지 확인하세요")
            
            # 디버깅 정보
            with st.expander("🔍 디버깅 정보"):
                st.write("현재 디렉토리:", os.getcwd())
                st.write("파일 목록:", os.listdir('.'))
                st.write("환경 변수 STREAMLIT_CLOUD:", os.environ.get('STREAMLIT_CLOUD', 'Not set'))
            
            return None, None, None, None, None

# --- 4. 고급 RAG 파이프라인 ---

class FTCAdvancedRAG:
    """공정거래 문서 특화 고급 RAG 시스템"""
    
    def __init__(self, embedding_model, reranker_model, index, chunks, chunk_type_index):
        self.embedding_model = embedding_model
        self.reranker_model = reranker_model
        self.index = index
        self.chunks = chunks
        self.chunk_type_index = chunk_type_index
        self.query_analyzer = FTCQueryAnalyzer()
        
    def search(self, query: str, top_k: int = 7) -> Tuple[List[SearchResult], Dict]:
        """통합 검색 파이프라인"""
        start_time = time.time()
        
        # 1. 쿼리 분석
        query_analysis = self.query_analyzer.analyze(query)
        
        # 2. 쿼리 확장
        expanded_queries = self._expand_query(query, query_analysis)
        
        # 3. 벡터 검색 + 필터링
        candidates = self._vector_search_with_filtering(
            expanded_queries, 
            query_analysis['priority_chunks'],
            k=50
        )
        
        # 4. 키워드 매칭 점수 추가
        candidates = self._add_keyword_scores(candidates, query, query_analysis)
        
        # 5. CrossEncoder 재정렬
        reranked = self._rerank_results(query, candidates, top_k=top_k*2)
        
        # 6. 컨텍스트 확장 (인접 청크 포함)
        final_results = self._expand_context(reranked[:top_k])
        
        # 통계 생성
        stats = {
            'query_analysis': query_analysis,
            'expanded_queries': expanded_queries,
            'initial_candidates': len(candidates),
            'after_rerank': len(reranked),
            'final_results': len(final_results),
            'search_time': time.time() - start_time
        }
        
        return final_results, stats
    
    def _expand_query(self, original_query: str, analysis: Dict) -> List[str]:
        """쿼리 확장 (공정거래 도메인 특화)"""
        queries = [original_query]
        
        # 1. 동의어/유사어 확장
        synonym_map = {
            '공시': ['공시의무', '공시사항', '신고', '보고'],
            '대규모내부거래': ['내부거래', '계열사거래', '특수관계인거래'],
            '이사회': ['이사회 의결', '이사회 결의', '이사회 승인'],
            '과태료': ['벌금', '제재금', '벌칙', '처벌'],
            '특수관계인': ['특관자', '특수관계자', '관계회사'],
            '기한': ['기간', '마감일', '제출일', '신고일']
        }
        
        query_lower = original_query.lower()
        for key, synonyms in synonym_map.items():
            if key in query_lower:
                for syn in synonyms:
                    queries.append(query_lower.replace(key, syn))
        
        # 2. 구조화된 쿼리 생성
        if analysis['entities'].get('article'):
            for article in analysis['entities']['article']:
                queries.append(f"제{article[0]}조 {' '.join(analysis['key_terms'])}")
        
        # 3. 질문 유형별 확장
        if 'definition' in analysis['query_type']:
            queries.append(f"{' '.join(analysis['key_terms'])} 정의")
            queries.append(f"{' '.join(analysis['key_terms'])}란")
        
        if 'deadline' in analysis['query_type']:
            queries.append(f"{' '.join(analysis['key_terms'])} 기한 일수")
            queries.append(f"{' '.join(analysis['key_terms'])} 제출 기간")
        
        # 중복 제거 및 반환
        return list(dict.fromkeys(queries))[:6]
    
    def _vector_search_with_filtering(self, queries: List[str], 
                                    priority_chunks: List[str], 
                                    k: int = 50) -> List[SearchResult]:
        """벡터 검색 및 청크 타입 필터링"""
        all_results = []
        seen_ids = set()
        
        for query in queries:
            # 벡터 검색
            query_vector = self.embedding_model.encode([query])
            scores, indices = self.index.search(
                np.array(query_vector, dtype=np.float32), 
                min(k, len(self.chunks))
            )
            
            for idx, score in zip(indices[0], scores[0]):
                chunk = self.chunks[idx]
                chunk_id = chunk.get('chunk_id', str(idx))
                
                if chunk_id not in seen_ids:
                    seen_ids.add(chunk_id)
                    
                    # 청크 타입에 따른 가중치
                    type_weight = 1.2 if chunk.get('chunk_type') in priority_chunks else 1.0
                    
                    # 중요도 가중치
                    importance_weight = chunk.get('importance', 0.5) + 0.5
                    
                    # 최종 점수
                    adjusted_score = score * type_weight * importance_weight
                    
                    result = SearchResult(
                        chunk_id=chunk_id,
                        content=chunk['content'],
                        score=adjusted_score,
                        source=chunk['source'],
                        page=chunk['page'],
                        chunk_type=chunk.get('chunk_type', 'unknown'),
                        importance=chunk.get('importance', 0.5),
                        keywords=chunk.get('keywords', []),
                        metadata=json.loads(chunk.get('metadata', '{}'))
                    )
                    all_results.append(result)
        
        # 점수순 정렬
        all_results.sort(key=lambda x: x.score, reverse=True)
        return all_results[:k]
    
    def _add_keyword_scores(self, results: List[SearchResult], 
                          query: str, analysis: Dict) -> List[SearchResult]:
        """키워드 매칭 점수 추가"""
        query_terms = set(query.lower().split()) | set(analysis['key_terms'])
        
        for result in results:
            content_lower = result.content.lower()
            
            # 정확한 매칭 점수
            exact_matches = sum(1 for term in query_terms if term in content_lower)
            
            # 키워드 매칭 점수
            keyword_matches = sum(1 for kw in result.keywords if kw.lower() in query_terms)
            
            # 엔티티 매칭 점수
            entity_score = 0
            for entity_type, entities in analysis['entities'].items():
                for entity in entities:
                    if str(entity) in content_lower:
                        entity_score += 1
            
            # 점수 조정
            keyword_boost = (exact_matches * 0.1 + 
                           keyword_matches * 0.05 + 
                           entity_score * 0.15)
            
            result.score = result.score * (1 + keyword_boost)
        
        return results
    
    def _rerank_results(self, query: str, candidates: List[SearchResult], 
                       top_k: int = 20) -> List[SearchResult]:
        """CrossEncoder를 사용한 정밀 재정렬"""
        if not candidates:
            return []
        
        # 재정렬을 위한 텍스트 쌍 생성
        pairs = []
        for candidate in candidates[:50]:  # 상위 50개만 재정렬
            # 메타데이터 정보 포함
            enhanced_text = f"{candidate.content}"
            if candidate.chunk_type == 'article':
                if 'article_number' in candidate.metadata:
                    enhanced_text = f"{candidate.metadata['article_number']} {enhanced_text}"
            pairs.append([query, enhanced_text])
        
        # CrossEncoder 점수 계산
        ce_scores = self.reranker_model.predict(pairs)
        
        # 기존 점수와 결합
        for i, (candidate, ce_score) in enumerate(zip(candidates[:len(pairs)], ce_scores)):
            # 벡터 점수와 CrossEncoder 점수 결합
            combined_score = candidate.score * 0.3 + float(ce_score) * 0.7
            candidate.score = combined_score
        
        # 재정렬
        candidates.sort(key=lambda x: x.score, reverse=True)
        return candidates[:top_k]
    
    def _expand_context(self, results: List[SearchResult]) -> List[SearchResult]:
        """선택된 청크의 컨텍스트 확장"""
        expanded_results = []
        added_ids = set()
        
        for result in results:
            # 현재 청크 추가
            if result.chunk_id not in added_ids:
                expanded_results.append(result)
                added_ids.add(result.chunk_id)
            
            # 같은 문서의 인접 청크 찾기 (조항 구조 고려)
            if result.chunk_type == 'article_paragraph':
                # 같은 조항의 다른 항 찾기
                article_num = result.metadata.get('article_number')
                if article_num:
                    for chunk in self.chunks:
                        if (chunk.get('metadata') and 
                            json.loads(chunk['metadata']).get('article_number') == article_num and
                            chunk['chunk_id'] != result.chunk_id and
                            chunk['chunk_id'] not in added_ids):
                            
                            context_result = SearchResult(
                                chunk_id=chunk['chunk_id'],
                                content=chunk['content'],
                                score=result.score * 0.8,  # 컨텍스트는 약간 낮은 점수
                                source=chunk['source'],
                                page=chunk['page'],
                                chunk_type=chunk.get('chunk_type', 'unknown'),
                                importance=chunk.get('importance', 0.5),
                                keywords=chunk.get('keywords', []),
                                metadata=json.loads(chunk.get('metadata', '{}'))
                            )
                            context_result.is_context = True
                            expanded_results.append(context_result)
                            added_ids.add(chunk['chunk_id'])
                            break
        
        return expanded_results

# --- 5. 답변 생성 엔진 ---

class FTCAnswerGenerator:
    """공정거래 문서 특화 답변 생성기"""
    
    def __init__(self):
        self.system_prompt = """당신은 한국 공정거래위원회 전략기획부의 전문 AI 법률 보조원입니다.

역할:
- 공정거래법, 관련 고시, 규정, 매뉴얼을 정확히 해석하여 실무에 적용 가능한 답변 제공
- 법률 용어를 명확히 설명하면서도 실무자가 이해하기 쉽게 전달
- 구체적인 조항, 기한, 요건 등을 정확히 인용

답변 원칙:
1. **정확성 최우선**: 제공된 자료의 내용만을 근거로 답변
2. **구조화된 설명**: 복잡한 내용은 단계별로 구분하여 설명
3. **실무 적용성**: 이론적 설명과 함께 실제 적용 방법 제시
4. **명확한 근거**: 모든 주장에 대해 조항이나 페이지 인용
5. **예외사항 명시**: 일반 원칙과 예외를 구분하여 설명

답변 구조:
- 핵심 답변 (1-2문장으로 간단명료하게)
- 상세 설명 (근거 조항과 함께)
- 주의사항 또는 예외사항
- 실무 적용 가이드 (필요시)"""
    
    def generate(self, query: str, search_results: List[SearchResult], 
                query_analysis: Dict) -> Tuple[str, str, Dict]:
        """고품질 답변 생성"""
        
        # 1. 컨텍스트 구성
        context = self._build_context(search_results, query_analysis)
        
        # 2. 프롬프트 구성
        messages = self._build_prompt(query, context, query_analysis)
        
        # 3. 답변 생성
        answer = self._generate_answer(messages)
        
        # 4. 답변 검증
        verification = self._verify_answer(answer, context)
        
        # 5. 메타데이터 생성
        metadata = {
            'sources': list(set(r.source for r in search_results if not hasattr(r, 'is_context'))),
            'primary_chunks': sum(1 for r in search_results if not hasattr(r, 'is_context')),
            'context_chunks': sum(1 for r in search_results if hasattr(r, 'is_context')),
            'query_type': query_analysis['query_type'],
            'confidence': self._calculate_confidence(search_results, query_analysis)
        }
        
        return answer, verification, metadata
    
    def _build_context(self, results: List[SearchResult], analysis: Dict) -> str:
        """구조화된 컨텍스트 생성"""
        context_parts = []
        
        # 청크 타입별로 그룹화
        grouped_results = defaultdict(list)
        for result in results:
            grouped_results[result.chunk_type].append(result)
        
        # 우선순위에 따라 컨텍스트 구성
        priority_order = ['definition', 'article', 'penalty', 'section', 'paragraph']
        
        for chunk_type in priority_order:
            if chunk_type in grouped_results:
                context_parts.append(f"\n[{self._get_type_label(chunk_type)}]")
                
                for i, result in enumerate(grouped_results[chunk_type]):
                    source_info = f"{result.source} (p.{result.page})"
                    
                    # 중요도 표시
                    importance_marker = ""
                    if result.importance > 0.8:
                        importance_marker = "⭐ [핵심] "
                    elif result.importance > 0.6:
                        importance_marker = "● [중요] "
                    
                    # 조항 정보 포함
                    article_info = ""
                    if 'article_number' in result.metadata:
                        article_info = f"{result.metadata['article_number']} "
                        if 'article_title' in result.metadata:
                            article_info += f"({result.metadata['article_title']}) "
                    
                    context_parts.append(
                        f"\n{importance_marker}{article_info}- 출처: {source_info}\n"
                        f"{result.content}\n"
                    )
        
        return "\n".join(context_parts)
    
    def _get_type_label(self, chunk_type: str) -> str:
        """청크 타입의 한글 레이블"""
        labels = {
            'definition': '용어 정의',
            'article': '관련 조항',
            'penalty': '벌칙/제재',
            'section': '세부 내용',
            'paragraph': '참고 내용',
            'article_paragraph': '조항 세부사항'
        }
        return labels.get(chunk_type, chunk_type)
    
    def _build_prompt(self, query: str, context: str, analysis: Dict) -> List[Dict]:
        """쿼리 타입에 따른 프롬프트 구성"""
        
        # 쿼리 타입별 추가 지시사항
        type_instructions = {
            'definition': "용어의 법적 정의를 명확히 설명하고, 관련 조항을 인용하세요.",
            'deadline': "구체적인 기한(일수)을 명시하고, 기산일과 마감일 계산 방법을 설명하세요.",
            'penalty': "위반 시 제재 내용을 구체적으로 설명하고, 과태료 금액이나 처벌 수준을 명시하세요.",
            'procedure': "절차를 단계별로 구분하여 설명하고, 각 단계의 주체와 기한을 명확히 하세요.",
            'requirement': "필요한 요건을 목록 형태로 정리하고, 각 요건의 충족 기준을 설명하세요."
        }
        
        additional_instruction = ""
        for q_type in analysis['query_type']:
            if q_type in type_instructions:
                additional_instruction += f"\n- {type_instructions[q_type]}"
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"""다음 자료를 바탕으로 질문에 답변해주세요.

[검색된 자료]
{context}

[질문]
{query}

[답변 시 유의사항]
1. 반드시 제공된 자료의 내용만을 근거로 답변하세요.
2. 조항 번호, 페이지 등 출처를 명확히 표시하세요.
3. 법률 용어는 정확히 사용하되, 필요시 쉬운 설명을 추가하세요.{additional_instruction}

답변 형식:
📌 **핵심 답변**
(질문에 대한 직접적인 답변을 1-2문장으로)

📋 **상세 설명**
(근거 조항과 구체적인 내용)

⚠️ **주의사항**
(있는 경우에만, 예외사항이나 특별히 유의할 점)"""}
        ]
        
        return messages
    
    def _generate_answer(self, messages: List[Dict]) -> str:
        """GPT-4를 사용한 답변 생성"""
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.1,
            max_tokens=2000
        )
        return response.choices[0].message.content
    
    def _verify_answer(self, answer: str, context: str) -> str:
        """답변 정확성 검증"""
        verification_prompt = f"""다음 AI 답변이 제공된 참고 자료에만 근거하여 작성되었는지 엄격히 검증하세요.

[AI 답변]
{answer}

[참고 자료]
{context}

검증 항목:
1. 답변의 모든 사실이 참고 자료에 명시되어 있는가?
2. 조항 번호나 인용이 정확한가?
3. 추측이나 일반화가 포함되지 않았는가?
4. 수치나 기한이 정확히 일치하는가?

형식: "✅ 검증 통과" 또는 "⚠️ 검증 주의" 또는 "❌ 검증 실패"
이유를 한 줄로 설명하세요."""

        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": verification_prompt}],
            temperature=0
        )
        
        return response.choices[0].message.content
    
    def _calculate_confidence(self, results: List[SearchResult], analysis: Dict) -> float:
        """답변 신뢰도 계산"""
        if not results:
            return 0.0
        
        # 요인별 점수
        factors = {
            'top_score': min(results[0].score / 1.0, 1.0) * 0.3,  # 최고 점수
            'avg_score': min(np.mean([r.score for r in results[:3]]) / 0.8, 1.0) * 0.2,  # 상위 평균
            'type_match': 0.2 if any(r.chunk_type in analysis['priority_chunks'] for r in results[:3]) else 0.0,
            'keyword_coverage': min(len([kw for kw in analysis['key_terms'] if any(kw in r.content.lower() for r in results[:3])]) / max(len(analysis['key_terms']), 1), 1.0) * 0.2,
            'source_diversity': min(len(set(r.source for r in results[:5])) / 3, 1.0) * 0.1
        }
        
        return sum(factors.values())

# --- 6. 메인 UI 구현 ---

def main():
    # 헤더
    st.markdown('<h1 class="main-header">⚖️ 전략기획부 AI 법률 보조원</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">공정거래위원회 규정 및 매뉴얼 기반 전문 Q&A 시스템</p>', unsafe_allow_html=True)
    
    # 모델 로드
    model, reranker, index, chunks, chunk_type_index = load_models_and_data()
    
    if model is None:
        st.error("필수 데이터를 로드할 수 없습니다. 파일을 확인해주세요.")
        return
    
    # RAG 시스템 초기화
    rag_system = FTCAdvancedRAG(model, reranker, index, chunks, chunk_type_index)
    answer_generator = FTCAnswerGenerator()
    
    # 세션 상태 초기화
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "search_history" not in st.session_state:
        st.session_state.search_history = []
    
    # 사이드바
    with st.sidebar:
        st.header("🔍 검색 옵션")
        
        # 검색 설정
        top_k = st.slider("검색 결과 수", min_value=3, max_value=10, value=7, 
                         help="더 많은 자료를 검토하면 정확도가 높아지지만 시간이 더 걸립니다.")
        
        use_context = st.checkbox("문맥 확장 사용", value=True,
                                help="관련 조항의 다른 항목도 함께 검토합니다.")
        
        st.divider()
        
        # 통계 정보
        if chunks:
            st.header("📊 시스템 정보")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("학습 문서", f"{len(set(c['source'] for c in chunks))}개")
                st.metric("총 정보 단위", f"{len(chunks):,}개")
            
            with col2:
                # 청크 타입별 통계
                type_counts = Counter(c.get('chunk_type', 'unknown') for c in chunks)
                st.metric("조항 정보", f"{type_counts.get('article', 0):,}개")
                st.metric("정의 정보", f"{type_counts.get('definition', 0):,}개")
        
        st.divider()
        
        # 자주 묻는 질문
        st.header("💡 예시 질문")
        example_questions = [
            "대규모내부거래 공시 기한은?",
            "특수관계인의 정의는?",
            "공시의무 위반 시 과태료는?",
            "이사회 의결 예외사항은?",
            "독립경영 인정 요건은?"
        ]
        
        for q in example_questions:
            if st.button(q, key=f"example_{q}"):
                st.session_state.pending_question = q
    
    # 메인 대화 영역
    # 이전 대화 표시
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.markdown(message["content"])
            else:
                st.markdown(message["content"]["answer"])
                
                # 메타데이터 표시
                if "metadata" in message["content"]:
                    with st.expander("📎 답변 상세 정보", expanded=False):
                        meta = message["content"]["metadata"]
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("참조 문서", f"{len(meta['sources'])}개")
                        with col2:
                            confidence_pct = meta['confidence'] * 100
                            st.metric("신뢰도", f"{confidence_pct:.0f}%")
                        with col3:
                            st.metric("검토 자료", f"{meta['primary_chunks']}개")
                        
                        # 출처 표시
                        st.subheader("📚 참조 문서")
                        for source in meta['sources']:
                            st.markdown(f'<span class="source-tag">{source}</span>', 
                                      unsafe_allow_html=True)
    
    # 사용자 입력 처리
    # 예시 질문 클릭 처리
    if hasattr(st.session_state, 'pending_question'):
        question = st.session_state.pending_question
        del st.session_state.pending_question
    else:
        question = st.chat_input("공정거래 관련 질문을 입력하세요...")
    
    if question:
        # 사용자 메시지 추가
        st.session_state.messages.append({"role": "user", "content": question})
        
        with st.chat_message("user"):
            st.markdown(question)
        
        # AI 응답 생성
        with st.chat_message("assistant"):
            # 진행 상황 표시
            progress_placeholder = st.empty()
            
            # 1. 검색 수행
            with st.spinner("🔍 관련 자료를 검색하는 중..."):
                search_results, search_stats = rag_system.search(question, top_k=top_k)
                progress_placeholder.info(f"✅ {len(search_results)}개의 관련 자료를 찾았습니다.")
            
            # 2. 답변 생성
            with st.spinner("💭 답변을 생성하는 중..."):
                answer, verification, metadata = answer_generator.generate(
                    question, 
                    search_results,
                    search_stats['query_analysis']
                )
                progress_placeholder.empty()
            
            # 답변 표시
            st.markdown(answer)
            
            # 검증 결과 표시
            if "✅" in verification:
                st.success(verification)
            elif "⚠️" in verification:
                st.warning(verification)
            else:
                st.error(verification)
            
            # 신뢰도 시각화
            confidence = metadata['confidence']
            confidence_color = "#4CAF50" if confidence > 0.7 else "#FF9800" if confidence > 0.4 else "#F44336"
            st.markdown(
                f"""
                <div style="background: linear-gradient(to right, {confidence_color} {confidence*100}%, #e0e0e0 {confidence*100}%); 
                     padding: 5px 15px; border-radius: 20px; text-align: center; margin: 20px 0;">
                    <strong>답변 신뢰도: {confidence*100:.0f}%</strong>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # 상세 정보 (접힌 상태)
            with st.expander("🔍 검색 및 분석 상세 정보"):
                tab1, tab2, tab3 = st.tabs(["검색 통계", "쿼리 분석", "참조 자료"])
                
                with tab1:
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("검색 시간", f"{search_stats['search_time']:.2f}초")
                    with col2:
                        st.metric("초기 후보", f"{search_stats['initial_candidates']}개")
                    with col3:
                        st.metric("재정렬 후", f"{search_stats['after_rerank']}개")
                    with col4:
                        st.metric("최종 사용", f"{search_stats['final_results']}개")
                
                with tab2:
                    st.json(search_stats['query_analysis'])
                
                with tab3:
                    for i, result in enumerate(search_results[:5]):
                        with st.container():
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.markdown(f"**{result.source}** - 페이지 {result.page}")
                            with col2:
                                importance_class = "importance-high" if result.importance > 0.7 else "importance-medium"
                                st.markdown(f'<span class="{importance_class}">중요도: {result.importance:.2f}</span>', 
                                          unsafe_allow_html=True)
                            
                            st.text(result.content[:300] + "..." if len(result.content) > 300 else result.content)
                            st.caption(f"점수: {result.score:.3f} | 타입: {result.chunk_type}")
                            st.divider()
            
            # 경고 메시지
            st.divider()
            st.caption("⚠️ 본 답변은 AI가 제공하는 참고용 정보입니다. 중요한 의사결정에는 반드시 원문을 확인하시기 바랍니다.")
            
            # 세션에 저장
            response_data = {
                "answer": answer,
                "verification": verification,
                "metadata": metadata,
                "search_stats": search_stats
            }
            st.session_state.messages.append({"role": "assistant", "content": response_data})
            st.session_state.search_history.append({
                "query": question,
                "timestamp": time.time(),
                "confidence": confidence
            })
    
    # 하단 정보
    with st.container():
        st.divider()
        col1, col2, col3 = st.columns(3)
        with col1:
            st.caption("🏢 전략기획부")
        with col2:
            st.caption("📅 2025년 최신 자료 기반")
        with col3:
            st.caption("🤖 Powered by GPT-4")

if __name__ == "__main__":
    main()
