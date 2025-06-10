# 파일 이름: app.py (Reranker 및 CoT 프롬프트 적용 최종 버전)

import streamlit as st
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder # CrossEncoder 추가
import json
import openai
import os

# --- 페이지 설정 (가장 먼저 실행) ---
st.set_page_config(page_title="전략기획부 AI 답변 챗봇", page_icon="🏢", layout="centered")

# --- API 키 설정 ---
try:
    openai.api_key = st.secrets["OPENAI_API_KEY"]
except Exception as e:
    st.error("OpenAI API 키가 설정되지 않았습니다. Streamlit Cloud의 Secrets에 키를 등록해주세요.")

# --- 모델 및 데이터 로딩 (캐싱으로 성능 최적화) ---
@st.cache_resource
def load_models_and_data():
    """사전에 준비된 데이터 파일과, 인터넷에서 다운로드한 모델을 로드합니다."""
    try:
        # 1단계 검색을 위한 Bi-Encoder 모델
        embedding_model = SentenceTransformer('jhgan/ko-sroberta-multitask')
        # 2단계 재정렬을 위한 Cross-Encoder 모델
        reranker_model = CrossEncoder('Dongjin-kr/ko-reranker')
        
        index = faiss.read_index("manuals_vector_db.index")
        with open("all_manual_chunks.json", "r", encoding="utf-8") as f:
            chunks_with_metadata = json.load(f)
        return embedding_model, reranker_model, index, chunks_with_metadata
    except FileNotFoundError:
        return None, None, None, None

# --- 핵심 기능 함수 ---
def get_relevant_manual_chunks(user_question, k=7, top_n=3):
    """2단계 검색(Retrieval & Reranking)을 통해 가장 관련성 높은 문서를 찾습니다."""
    # Stage 1: Bi-Encoder로 1차 후보군 검색 (넉넉하게 7개)
    question_vector = model.encode([user_question])
    distances, indices = index.search(np.array(question_vector, dtype=np.float32), k)
    initial_chunks = [chunks_with_metadata[i] for i in indices[0]]
    
    # Stage 2: Cross-Encoder(Reranker)로 최종 TOP 3 선정
    pairs = [[user_question, chunk['content']] for chunk in initial_chunks]
    scores = reranker.predict(pairs)
    
    # 점수가 높은 순으로 정렬하여 상위 top_n개만 선택
    ranked_chunks = [chunk for _, chunk in sorted(zip(scores, initial_chunks), key=lambda x: x[0], reverse=True)]
    return ranked_chunks[:top_n]

def generate_answer_with_llm(user_question, relevant_chunks):
    """검색된 매뉴얼 내용을 근거로 LLM(GPT)이 답변을 생성합니다."""
    context_str = "\n\n".join(
        [f"출처 파일명: {chunk['source']} (Page: {chunk.get('page', 'N/A')})\n내용: {chunk['content']}" for chunk in relevant_chunks]
    )
    
    # === 향상된 시스템 프롬프트 (Chain of Thought 포함) ===
    system_prompt = """
    # Role: 당신은 한국 공정거래법을 전문으로 다루는 '전략기획부 AI 법률 보조자'입니다.

    # Instructions:
    1.  **사고 과정(Chain of Thought):** 답변을 생성하기 전에 다음의 사고 과정을 반드시 거쳐야 합니다.
        (1) 사용자의 질문에서 핵심적인 법률적 쟁점을 정확히 파악한다.
        (2) 주어진 [관련 매뉴얼 정보]에서 해당 쟁점과 직접적으로 관련된 모든 조항과 사실들을 빠짐없이 추출한다.
        (3) 추출된 정보들을 종합하여 논리적인 결론을 도출한다.
        (4) 이 결론을 바탕으로, 아래 '답변 구조'에 맞춰 최종 답변을 작성한다.

    2.  **답변 구조 준수:** 모든 답변은 아래의 목차 구조를 반드시 따라야 합니다. 각 목차의 제목은 Markdown H3(###) 형식으로 작성하세요.
        - ### 1. 개요
        - ### 2. 주요 내용
        - ### 3. 출처
        - ### 4. 결론
        - ### 5. 추가 질문 제안

    3.  **답변 생성 규칙:**
        - 반드시 제공된 [관련 매뉴얼 정보]만을 근거로 답변해야 합니다. 당신의 사전 지식은 절대 사용하지 마세요.
        - 만약 [관련 매뉴얼 정보]에서 답변을 찾을 수 없다면, 다른 목차 없이 "죄송합니다, 제공된 자료 내에서는 문의하신 내용에 대한 답변을 찾을 수 없습니다."라고만 답변하세요.
        - 모든 답변은 간결하고 명확한 한국어로 작성합니다.
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"[관련 매뉴얼 정보]\n{context_str}\n---\n[질문]\n{user_question}"}
    ]
    
    try:
        response = openai.chat.completions.create(model="gpt-4o", messages=messages, temperature=0.1)
        return response.choices[0].message.content
    except Exception as e:
        return f"답변 생성 중 오류가 발생했습니다: {e}"

# --- 메인 UI 구성 ---
st.title("🏢 전략기획부 AI 답변 챗봇")
model, reranker, index, chunks_with_metadata = load_models_and_data()

if model is None or reranker is None:
    st.error("챗봇 데이터 파일 또는 모델을 로드할 수 없습니다. GitHub 저장소를 확인해주세요.")
else:
    st.success("AI 비서가 준비되었습니다.", icon="✅")
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"], unsafe_allow_html=True)

    if prompt := st.chat_input("공정거래법 공시 관련 질문을 입력해주세요..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("AI가 매뉴얼을 검토하고 답변을 생성하는 중입니다..."):
                relevant_chunks = get_relevant_manual_chunks(prompt)
                answer = generate_answer_with_llm(prompt, relevant_chunks)
                warning_message = "\n\n---\n◆본 답변은 전략기획부가 학습시킨 ChatGPT를 통해 제공하는 답변으로, 참고용으로만 활용하시기 바랍니다."
                full_answer = answer + warning_message
                
                st.markdown(full_answer, unsafe_allow_html=True)

                with st.expander("AI가 참고한 최종 매뉴얼 내용 보기 (재정렬 후)"):
                    for chunk in relevant_chunks:
                        st.info(f"**출처:** {chunk['source']}, **Page:** {chunk.get('page', 'N/A')}")
                        st.text(chunk['content'])
                        st.divider()

                st.session_state.messages.append({"role": "assistant", "content": full_answer})
