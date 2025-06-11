# 파일 이름: app_manual.py (공정거래위원회 AI 법률 보조원 - 통합 개선 버전)

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

# ===== 3. 질문 분류기 (개선사항 1: 질문 유형별 맞춤 검색) =====
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

# ===== 4. 최적화된 RAG 파이프라인 (개선사항 3, 6, 7: 속도 개선 + 캐싱) =====
class OptimizedRAGPipeline:
    """속도와 정확도를 극대화한 RAG 파이프라인"""
    
    def __init__(self, embedding_model, reranker_model, index, chunks):
        self.embedding_model = embedding_model
        self.reranker_model = reranker_model
        self.index = index
        self.chunks = chunks
        self.classifier = QuestionClassifier()
        
        # 매뉴얼별 청크 인덱스 미리 구축 (빠른 필터링)
        self.manual_indices = self._build_manual_indices()
        
        # 검색 결과 캐시 (개선사항 7)
        self.search_cache = {}
        self.cache_max_size = 100
        
    def _build_manual_indices(self) -> Dict[str, List[int]]:
        """각 매뉴얼별로 청크 인덱스를 미리 구축"""
        indices = defaultdict(list)
        
        for idx, chunk in enumerate(self.chunks):
            source = chunk.get('source', '').lower()
            
            # 카테고리별 분류
            if '대규모내부거래' in source:
                indices['대규모내부거래'].append(idx)
            elif '현황공시' in source or '기업집단' in source:
                indices['현황공시'].append(idx)
            elif '비상장' in source:
                indices['비상장사 중요사항'].append(idx)
            else:
                indices['기타'].append(idx)
        
        return dict(indices)
    
    def search(self, query: str, top_k: int = 5) -> Tuple[List[SearchResult], Dict]:
        """통합 검색 파이프라인"""
        start_time = time.time()
        
        # 캐시 확인 (개선사항 7)
        cache_key = hashlib.md5(f"{query}_{top_k}".encode()).hexdigest()
        if cache_key in self.search_cache:
            cached = self.search_cache[cache_key]
            stats = cached['stats'].copy()
            stats['cache_hit'] = True
            stats['search_time'] = 0.001
            return cached['results'], stats
        
        # 1. 질문 분류 (개선사항 1)
        category, confidence = self.classifier.classify(query)
        
        # 2. 검색 전략 결정
        if category and confidence > 0.3:
            search_strategy = 'targeted'
            primary_indices = self.manual_indices.get(category, [])
            secondary_indices = []
            for cat, indices in self.manual_indices.items():
                if cat != category and cat != '기타':
                    secondary_indices.extend(indices)
        else:
            search_strategy = 'general'
            primary_indices = list(range(len(self.chunks)))
            secondary_indices = []
        
        # 3. 최적화된 벡터 검색 (개선사항 3)
        results = self._perform_optimized_search(
            query, primary_indices, secondary_indices, top_k
        )
        
        # 4. 통계 생성
        search_time = time.time() - start_time
        stats = {
            'category': category,
            'confidence': confidence,
            'strategy': search_strategy,
            'search_time': search_time,
            'primary_searched': len(primary_indices),
            'total_chunks': len(self.chunks),
            'cache_hit': False
        }
        
        # 5. 빠른 검색은 캐시에 저장
        if search_time < 2.0 and len(self.search_cache) < self.cache_max_size:
            self.search_cache[cache_key] = {
                'results': results,
                'stats': stats,
                'timestamp': time.time()
            }
        
        return results, stats
    
    def _perform_optimized_search(self, query: str, primary_indices: List[int], 
                                 secondary_indices: List[int], top_k: int) -> List[SearchResult]:
        """최적화된 FAISS 검색 (개선사항 3 핵심)"""
        # 쿼리 벡터 생성 (한 번만!)
        query_vector = self.embedding_model.encode([query])
        query_vector = np.array(query_vector, dtype=np.float32)
        
        # FAISS 인덱스 직접 활용
        k_search = min(len(self.chunks), top_k * 20)
        scores, indices = self.index.search(query_vector, k_search)
        
        results = []
        seen_chunks = set()
        
        # 우선순위 인덱스에서 먼저 결과 수집
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
        
        # 부족하면 보조 인덱스에서 추가
        if len(results) < top_k and secondary_indices:
            secondary_set = set(secondary_indices)
            for idx, score in zip(indices[0], scores[0]):
                if idx in secondary_set and idx not in seen_chunks:
                    seen_chunks.add(idx)
                    chunk = self.chunks[idx]
                    result = SearchResult(
                        chunk_id=chunk.get('chunk_id', str(idx)),
                        content=chunk['content'],
                        score=float(score) * 0.8,  # 보조 결과는 점수 감소
                        source=chunk['source'],
                        page=chunk['page'],
                        chunk_type=chunk.get('chunk_type', 'unknown'),
                        metadata=json.loads(chunk.get('metadata', '{}'))
                    )
                    results.append(result)
                    if len(results) >= top_k:
                        break
        
        return results

# ===== 5. 동적 Temperature 답변 생성 (개선사항 4, 5) =====
def determine_temperature(query: str) -> float:
    """질문 유형에 따라 최적의 temperature 결정"""
    query_lower = query.lower()
    
    # 단순 사실 확인 (낮은 temperature)
    if any(keyword in query_lower for keyword in ['언제', '며칠', '기한', '날짜', '금액', '%']):
        return 0.1
    
    # 정의나 범위 (중간 temperature)
    elif any(keyword in query_lower for keyword in ['정의', '범위', '포함', '해당', '의미']):
        return 0.3
    
    # 복잡한 판단 (높은 temperature)
    elif any(keyword in query_lower for keyword in ['어떻게', '경우', '만약', '예외', '가능']):
        return 0.5
    
    # 전략적 조언 (더 높은 temperature)
    elif any(keyword in query_lower for keyword in ['전략', '대응', '리스크', '주의', '권장']):
        return 0.7
    
    return 0.3  # 기본값

def generate_answer(query: str, results: List[SearchResult], category: str = None) -> str:
    """GPT-4o를 활용한 고품질 답변 생성"""
    
    # 컨텍스트 구성
    context_parts = []
    for i, result in enumerate(results[:5]):
        context_parts.append(f"""
[참고 {i+1}] {result.source} (페이지 {result.page})
{result.content}
""")
    
    context = "\n---\n".join(context_parts)
    
    # 동적 temperature 결정 (개선사항 5)
    temperature = determine_temperature(query)
    
    # 카테고리별 특화 지시사항
    category_instructions = {
        '대규모내부거래': "특히 이사회 의결 요건, 공시 기한, 면제 조건을 명확히 설명하세요.",
        '현황공시': "공시 주체, 시기, 제출 서류를 구체적으로 안내하세요.",
        '비상장사 중요사항': "공시 대상 거래, 기한, 제출 방법을 상세히 설명하세요."
    }
    
    extra_instruction = category_instructions.get(category, "") if category else ""
    
    # 시스템 프롬프트 구성
    system_prompt = """당신은 한국 공정거래위원회 전문가입니다.
제공된 자료만을 근거로 정확하고 실무적인 답변을 제공하세요.
답변은 다음 구조를 따라주세요:
1. 핵심 답변 (1-2문장)
2. 상세 설명 (근거 조항 포함)
3. 주의사항 또는 예외사항 (있는 경우)"""
    
    if temperature >= 0.5:
        system_prompt += "\n다양한 관점과 실무적 고려사항을 포함하여 종합적으로 분석해주세요."
    
    # GPT-4o 호출 (개선사항 4)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"""다음 자료를 바탕으로 질문에 답변해주세요.

[참고 자료]
{context}

[질문]
{query}

{extra_instruction}

{"간결하고 명확하게" if temperature < 0.3 else "상세하고 실무적으로"} 답변해주세요."""}
    ]
    
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=temperature,
        max_tokens=1500
    )
    
    return response.choices[0].message.content

# ===== 6. 모델 및 데이터 로딩 =====
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

# ===== 7. 메인 UI (개선사항 2: 깔끔한 UI) =====
def main():
    # 헤더
    st.markdown("""
    <div class="main-header">
        <h1>⚖️ 전략기획부 AI 법률 보조원</h1>
        <p>공정거래위원회 규정 및 매뉴얼 기반 전문 Q&A 시스템</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 모델 로드
    models = load_models_and_data()
    if not all(models):
        st.stop()
    
    embedding_model, reranker_model, index, chunks = models
    
    # RAG 시스템 초기화
    rag = OptimizedRAGPipeline(embedding_model, reranker_model, index, chunks)
    
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
                        
                        # 시간 정보 표시 (개선사항 6)
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
                # 전체 시간 측정 시작 (개선사항 6)
                total_start_time = time.time()
                
                # 검색 수행
                search_start_time = time.time()
                with st.spinner("🔍 관련 자료를 검색하는 중..."):
                    results, stats = rag.search(prompt, top_k=5)
                search_time = time.time() - search_start_time
                
                # 답변 생성
                generation_start_time = time.time()
                with st.spinner("💭 답변을 생성하는 중..."):
                    answer = generate_answer(prompt, results, stats.get('category'))
                generation_time = time.time() - generation_start_time
                
                # 전체 시간 계산
                total_time = time.time() - total_start_time
                
                # 답변 표시
                st.write(answer)
                
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
                    # 검색 통계
                    if stats.get('category'):
                        st.info(f"📂 **{stats['category']}** 카테고리로 분류 (신뢰도: {stats['confidence']:.0%})")
                        if stats.get('cache_hit'):
                            st.success("⚡ 캐시에서 즉시 응답!")
                        else:
                            st.info(f"🔍 {stats['primary_searched']}개 문서 우선 검색 (전체 {stats['total_chunks']}개 중)")
                    
                    # 참고 자료
                    st.subheader("📚 참고 자료")
                    for i, result in enumerate(results[:3]):
                        st.caption(f"**{result.source}** - 페이지 {result.page}")
                        with st.container():
                            st.text(result.content[:300] + "..." if len(result.content) > 300 else result.content)
                    
                    # 성능 평가
                    if total_time < 3:
                        st.success("⚡ 매우 빠른 응답 속도!")
                    elif total_time < 5:
                        st.info("✅ 적절한 응답 속도")
                    else:
                        st.warning("⏰ 응답 시간이 다소 길었습니다")
                
                # 세션에 저장
                response_data = {
                    "answer": answer,
                    "search_time": search_time,
                    "generation_time": generation_time,
                    "total_time": total_time
                }
                st.session_state.messages.append({"role": "assistant", "content": response_data})
    
    # 하단 안내
    st.divider()
    st.caption("⚠️ 본 답변은 AI가 생성한 참고자료입니다. 중요한 사항은 반드시 원문을 확인하시기 바랍니다.")
    
    # 사이드바 (예시 질문)
    with st.sidebar:
        st.header("💡 자주 묻는 질문")
        
        st.subheader("대규모내부거래")
        if st.button("이사회 의결이 필요한 거래 금액은?"):
            st.session_state.new_question = "대규모내부거래에서 이사회 의결이 필요한 거래 금액 기준은?"
            st.rerun()
        if st.button("공시 기한은 언제까지인가요?"):
            st.session_state.new_question = "대규모내부거래 이사회 의결 후 공시 기한은?"
            st.rerun()
            
        st.subheader("현황공시")
        if st.button("기업집단 현황공시 시기는?"):
            st.session_state.new_question = "기업집단 현황공시는 언제 해야 하나요?"
            st.rerun()
            
        st.subheader("비상장사 중요사항")
        if st.button("주식 양도 시 공시 의무는?"):
            st.session_state.new_question = "비상장회사 주식 양도 시 공시 의무가 있나요?"
            st.rerun()
    
    # 새 질문 처리
    if "new_question" in st.session_state:
        st.session_state.messages.append({"role": "user", "content": st.session_state.new_question})
        del st.session_state.new_question
        st.rerun()

if __name__ == "__main__":
    main()
