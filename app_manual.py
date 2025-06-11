# 파일 이름: app_manual.py (개선된 공정거래위원회 AI 법률 보조원)

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

# --- 1. 페이지 설정 ---
st.set_page_config(
    page_title="전략기획부 AI 법률 보조원", 
    page_icon="⚖️", 
    layout="wide",
    initial_sidebar_state="collapsed"  # 사이드바 기본 숨김
)

# 깔끔한 UI를 위한 CSS
st.markdown("""
<style>
    /* 불필요한 Streamlit 기본 요소 숨기기 */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* 메인 헤더 스타일 */
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #1f4788 0%, #2a5298 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .chat-container {
        max-width: 800px;
        margin: 0 auto;
    }
    
    /* 대화 메시지 스타일 개선 */
    .stChatMessage {
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    
    /* 로딩 애니메이션 개선 */
    .stSpinner > div {
        text-align: center;
        color: #1f4788;
    }
</style>
""", unsafe_allow_html=True)

# OpenAI API 설정
try:
    openai.api_key = st.secrets["OPENAI_API_KEY"]
except:
    st.error("⚠️ OpenAI API 키를 설정해주세요.")
    st.stop()

# --- 2. 질문 유형 분류기 ---
class QuestionClassifier:
    """질문을 분석하여 어떤 매뉴얼을 우선 검색할지 결정"""
    
    def __init__(self):
        # 각 카테고리별 핵심 키워드
        self.categories = {
            '대규모내부거래': {
                'keywords': ['대규모내부거래', '내부거래', '이사회 의결', '이사회', '의결', 
                           '계열사', '계열회사', '특수관계인', '자금', '대여', '차입', '보증'],
                'manual_pattern': '대규모내부거래.*매뉴얼',
                'priority': 1
            },
            '현황공시': {
                'keywords': ['현황공시', '기업집단', '소속회사', '동일인', '친족', 
                           '지분율', '임원', '순환출자', '상호출자'],
                'manual_pattern': '기업집단현황공시.*매뉴얼',
                'priority': 2
            },
            '비상장사 중요사항': {
                'keywords': ['비상장', '중요사항', '주식', '양도', '양수', '합병', 
                           '분할', '영업양도', '임원변경'],
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
            
            # 키워드 매칭
            for keyword in info['keywords']:
                if keyword in question_lower:
                    score += 1
                    matched_keywords.append(keyword)
            
            # 가중치 적용 (처음 3개 키워드는 더 높은 점수)
            if matched_keywords:
                primary_keywords = info['keywords'][:3]
                for kw in matched_keywords:
                    if kw in primary_keywords:
                        score += 0.5
            
            scores[category] = score
        
        # 가장 높은 점수의 카테고리 선택
        if scores:
            best_category = max(scores, key=scores.get)
            confidence = scores[best_category] / max(len(info['keywords']) for info in self.categories.values())
            
            # 신뢰도가 너무 낮으면 None 반환
            if confidence < 0.1:
                return None, 0.0
                
            return best_category, confidence
        
        return None, 0.0

# --- 3. 개선된 검색 시스템 ---
@dataclass
class SearchResult:
    chunk_id: str
    content: str
    score: float
    source: str
    page: int
    chunk_type: str
    metadata: Dict

class OptimizedRAGPipeline:
    """속도와 정확도를 개선한 RAG 파이프라인"""
    
    def __init__(self, embedding_model, reranker_model, index, chunks):
        self.embedding_model = embedding_model
        self.reranker_model = reranker_model
        self.index = index
        self.chunks = chunks
        self.classifier = QuestionClassifier()
        
        # 매뉴얼별 청크 인덱스 미리 생성 (빠른 필터링)
        self.manual_indices = self._build_manual_indices()
    
    def _build_manual_indices(self) -> Dict[str, List[int]]:
        """각 매뉴얼별로 청크 인덱스를 미리 구축"""
        indices = defaultdict(list)
        
        for idx, chunk in enumerate(self.chunks):
            source = chunk.get('source', '').lower()
            
            # 대규모내부거래 매뉴얼
            if '대규모내부거래' in source:
                indices['대규모내부거래'].append(idx)
            # 현황공시 매뉴얼  
            elif '현황공시' in source or '기업집단' in source:
                indices['현황공시'].append(idx)
            # 비상장사 매뉴얼
            elif '비상장' in source:
                indices['비상장사 중요사항'].append(idx)
            # 기타
            else:
                indices['기타'].append(idx)
        
        return dict(indices)
    
    def search(self, query: str, top_k: int = 5) -> Tuple[List[SearchResult], Dict]:
        """개선된 검색: 카테고리별 우선순위 적용"""
        start_time = time.time()
        
        # 1. 질문 분류
        category, confidence = self.classifier.classify(query)
        
        # 2. 검색 전략 결정
        if category and confidence > 0.3:
            # 해당 카테고리 매뉴얼 우선 검색
            search_strategy = 'targeted'
            primary_indices = self.manual_indices.get(category, [])
            secondary_indices = []
            
            # 나머지 인덱스도 포함 (낮은 우선순위)
            for cat, indices in self.manual_indices.items():
                if cat != category:
                    secondary_indices.extend(indices)
        else:
            # 전체 검색
            search_strategy = 'general'
            primary_indices = list(range(len(self.chunks)))
            secondary_indices = []
        
        # 3. 벡터 검색 수행
        results = self._perform_search(
            query, 
            primary_indices, 
            secondary_indices,
            top_k
        )
        
        # 4. 통계 생성
        stats = {
            'category': category,
            'confidence': confidence,
            'strategy': search_strategy,
            'search_time': time.time() - start_time,
            'primary_searched': len(primary_indices),
            'total_chunks': len(self.chunks)
        }
        
        return results, stats
    
    def _perform_search(self, query: str, primary_indices: List[int], 
                       secondary_indices: List[int], top_k: int) -> List[SearchResult]:
        """실제 검색 수행 (개선된 속도)"""
        # 쿼리 벡터 생성
        query_vector = self.embedding_model.encode([query])
        
        # 우선순위 청크만으로 제한된 검색
        if primary_indices:
            # FAISS 서브 인덱스 생성 (메모리 효율적)
            primary_chunks = [self.chunks[i] for i in primary_indices]
            primary_vectors = []
            
            for chunk in primary_chunks:
                # 간단한 텍스트로 벡터 재사용 (속도 개선)
                text = chunk['content'][:500]  # 처음 500자만 사용
                vec = self.embedding_model.encode([text])
                primary_vectors.append(vec[0])
            
            # 빠른 검색
            primary_vectors = np.array(primary_vectors, dtype=np.float32)
            distances = np.dot(query_vector[0], primary_vectors.T)
            top_indices = np.argsort(distances)[-top_k:][::-1]
            
            # 결과 생성
            results = []
            for idx in top_indices:
                original_idx = primary_indices[idx]
                chunk = self.chunks[original_idx]
                
                result = SearchResult(
                    chunk_id=chunk.get('chunk_id', str(original_idx)),
                    content=chunk['content'],
                    score=float(distances[idx]),
                    source=chunk['source'],
                    page=chunk['page'],
                    chunk_type=chunk.get('chunk_type', 'unknown'),
                    metadata=json.loads(chunk.get('metadata', '{}'))
                )
                results.append(result)
            
            return results
        
        # 폴백: 전체 검색
        return self._full_search(query_vector, top_k)
    
    def _full_search(self, query_vector, top_k: int) -> List[SearchResult]:
        """전체 인덱스 검색 (기존 방식)"""
        scores, indices = self.index.search(query_vector, top_k * 2)
        
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < len(self.chunks):
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
        
        return results[:top_k]

# --- 4. 모델 로딩 (간소화) ---
@st.cache_resource(show_spinner=False)
def load_models_and_data():
    """필요한 모델과 데이터 로드"""
    try:
        # 필수 파일 확인
        required_files = ["manuals_vector_db.index", "all_manual_chunks.json"]
        for file in required_files:
            if not os.path.exists(file):
                st.error(f"❌ 필수 파일이 없습니다: {file}")
                st.info("💡 prepare_pdfs_ftc.py를 먼저 실행하세요.")
                return None, None, None, None
        
        # 데이터 로드
        index = faiss.read_index("manuals_vector_db.index")
        with open("all_manual_chunks.json", "r", encoding="utf-8") as f:
            chunks = json.load(f)
        
        # 모델 로드 (심플하게)
        with st.spinner("AI 시스템을 준비하는 중... (최초 1회만 수행)"):
            # 임베딩 모델
            try:
                # 온라인 모델 시도
                embedding_model = SentenceTransformer('jhgan/ko-sroberta-multitask')
            except:
                # 대체 모델
                embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
            
            # 재정렬 모델
            try:
                reranker_model = CrossEncoder('Dongjin-kr/ko-reranker')
            except:
                reranker_model = None  # 재정렬 없이도 작동 가능
        
        return embedding_model, reranker_model, index, chunks
        
    except Exception as e:
        st.error(f"시스템 초기화 실패: {str(e)}")
        return None, None, None, None

# --- 5. 답변 생성 (간소화) ---
def generate_answer(query: str, results: List[SearchResult], category: str = None) -> str:
    """검색 결과를 바탕으로 답변 생성"""
    
    # 컨텍스트 구성
    context_parts = []
    for i, result in enumerate(results[:5]):  # 상위 5개만 사용
        context_parts.append(f"""
[참고 {i+1}] {result.source} (p.{result.page})
{result.content}
""")
    
    context = "\n---\n".join(context_parts)
    
    # 카테고리별 특화 지시사항
    category_instructions = {
        '대규모내부거래': "이사회 의결 요건, 공시 기한, 예외사항을 중심으로 설명하세요.",
        '현황공시': "공시 주체, 공시 시기, 제출 서류를 중심으로 설명하세요.",
        '비상장사 중요사항': "공시 대상 거래, 공시 기한, 제출 방법을 중심으로 설명하세요."
    }
    
    extra_instruction = category_instructions.get(category, "") if category else ""
    
    # 프롬프트
    messages = [
        {
            "role": "system", 
            "content": """당신은 공정거래위원회 전문가입니다. 
제공된 자료만을 근거로 정확하고 실무적인 답변을 제공하세요.
답변은 명확하고 구조적으로 작성하며, 근거 조항이나 페이지를 명시하세요."""
        },
        {
            "role": "user",
            "content": f"""다음 자료를 바탕으로 질문에 답변해주세요.

[참고 자료]
{context}

[질문]
{query}

{extra_instruction}

간결하고 명확하게 답변해주세요."""
        }
    ]
    
    # GPT 호출
    response = openai.chat.completions.create(
        model="gpt-4o",  # GPT-4o 모델 사용 (더 정확한 답변)
        messages=messages,
        temperature=0.1,
        max_tokens=1000
    )
    
    return response.choices[0].message.content

# --- 6. 메인 UI ---
def main():
    # 헤더 (간단하게)
    st.markdown("""
    <div class="main-header">
        <h1>⚖️ 전략기획부 AI 법률 보조원</h1>
        <p style="margin: 0; opacity: 0.9;">공정거래위원회 규정 및 매뉴얼 기반 Q&A 시스템</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 모델 로드
    models = load_models_and_data()
    if not all(models):
        st.stop()
    
    embedding_model, reranker_model, index, chunks = models
    
    # RAG 시스템 초기화
    rag = OptimizedRAGPipeline(embedding_model, reranker_model, index, chunks)
    
    # 세션 상태
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # 대화 영역
    chat_container = st.container()
    
    with chat_container:
        # 이전 대화 표시
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
        
        # 사용자 입력
        if prompt := st.chat_input("질문을 입력하세요 (예: 대규모내부거래 공시 기한은?)"):
            # 사용자 메시지 추가
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)
            
            # AI 응답
            with st.chat_message("assistant"):
                # 검색 수행
                with st.spinner("관련 자료를 검색하는 중..."):
                    results, stats = rag.search(prompt, top_k=5)
                
                # 답변 생성
                with st.spinner("답변을 생성하는 중..."):
                    answer = generate_answer(prompt, results, stats.get('category'))
                
                # 답변 표시
                st.write(answer)
                
                # 출처 정보 (접을 수 있게)
                with st.expander("📚 참고 자료 보기"):
                    for i, result in enumerate(results[:3]):
                        st.caption(f"**{result.source}** - 페이지 {result.page}")
                        st.text(result.content[:200] + "...")
                        st.divider()
                
                # 세션에 저장
                st.session_state.messages.append({"role": "assistant", "content": answer})
    
    # 하단 안내
    st.divider()
    st.caption("💡 본 답변은 AI가 생성한 참고자료입니다. 중요한 사항은 반드시 원문을 확인하세요.")
    
    # 예시 질문 (선택적)
    with st.sidebar:
        st.header("💡 예시 질문")
        examples = [
            "대규모내부거래 이사회 의결은 언제 필요한가요?",
            "비상장회사 주식 양도 시 공시 의무가 있나요?",
            "기업집단 현황공시는 언제 해야 하나요?",
            "특수관계인의 범위는 어떻게 되나요?",
            "공시 의무 위반 시 과태료는 얼마인가요?"
        ]
        
        for example in examples:
            if st.button(example, key=example):
                st.session_state.prompt_input = example
                st.rerun()

if __name__ == "__main__":
    main()
