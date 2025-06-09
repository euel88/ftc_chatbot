# app_manual.py (최종 클라우드 배포 버전)

import streamlit as st
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import json
import openai
import os

st.set_page_config(page_title="AI 매뉴얼 전문가 챗봇", page_icon="📚", layout="centered")

# --- API 키 설정 (Streamlit의 Secrets 관리 기능 사용) ---
try:
    openai.api_key = st.secrets["OPENAI_API_KEY"]
except Exception as e:
    st.error("OpenAI API 키가 설정되지 않았습니다. Streamlit Cloud의 Secrets에 키를 등록해주세요.")


# --- 모델 및 데이터 로딩 (캐싱으로 성능 최적화) ---
@st.cache_resource
def load_models_and_data():
    """사전에 준비된 데이터 파일과, 인터넷에서 다운로드한 모델을 로드합니다."""
    try:
        # ======================================================================
        # ## 모델 로드 방식 변경 ##
        # 로컬 경로 대신, 허깅페이스 인터넷 주소를 직접 사용합니다.
        # Streamlit Cloud 서버가 이 모델을 직접 다운로드할 것입니다.
        model = SentenceTransformer('jhgan/ko-sroberta-multitask')
        # ======================================================================
        
        index = faiss.read_index("manuals_vector_db.index")
        with open("all_manual_chunks.json", "r", encoding="utf-8") as f:
            chunks_with_metadata = json.load(f)
        return model, index, chunks_with_metadata
    except FileNotFoundError:
        return None, None, None

def get_relevant_manual_chunks(user_question, k=5):
    question_vector = model.encode([user_question])
    distances, indices = index.search(np.array(question_vector, dtype=np.float32), k)
    relevant_chunks = [chunks_with_metadata[i] for i in indices[0]]
    return relevant_chunks

def generate_answer_with_llm(user_question, relevant_chunks):
    context_str = ""
    for chunk in relevant_chunks:
        context_str += f"출처: {chunk['source']}\n내용: {chunk['content']}\n\n"
    
    messages = [
        {"role": "system", "content": "당신은 제공된 사내 매뉴얼 전문가 AI입니다. 반드시 주어진 [관련 매뉴얼 정보] 내에서만 답변하고, 내용을 종합하여 명확하게 설명해주세요."},
        {"role": "user", "content": f"[관련 매뉴얼 정보]\n{context_str}\n---\n[질문]\n{user_question}"}
    ]
    
    try:
        response = openai.chat.completions.create(model="gpt-4o", messages=messages, temperature=0.7)
        return response.choices[0].message.content
    except Exception as e:
        return f"답변 생성 중 오류가 발생했습니다: {e}"

st.title("📚 AI 매뉴얼 전문가 챗봇")
model, index, chunks_with_metadata = load_models_and_data()

if model is None:
    st.error("챗봇 데이터 파일('manuals_vector_db.index' 또는 'all_manual_chunks.json')을 찾을 수 없습니다.")
else:
    st.success("모델과 데이터를 성공적으로 로드했습니다!", icon="✅")
    st.markdown("회사 내부 매뉴얼에 대해 궁금한 점을 질문해보세요.")
    user_question = st.text_input("질문 입력:", placeholder="예: 신규 입사자 노트북 신청 절차는 어떻게 되나요?")
    if st.button("질문하기", type="primary"):
        if user_question:
            with st.spinner("AI가 매뉴얼을 검토하고 답변을 생성하는 중입니다..."):
                relevant_chunks = get_relevant_manual_chunks(user_question)
                answer = generate_answer_with_llm(user_question, relevant_chunks)
                st.markdown("#### 💬 AI 답변")
                st.write(answer)
                st.markdown("---")
                with st.expander("AI가 참고한 매뉴얼 내용 보기"):
                    sources = sorted(list(set([chunk['source'] for chunk in relevant_chunks])))
                    st.markdown(f"**참고 매뉴얼:** {', '.join(sources)}")
                    st.json(relevant_chunks)
        else:
            st.warning("질문을 입력해주세요.")
    st.divider()
    st.caption("주의: 이 챗봇의 답변은 참고용이며, 최종 확인은 공식 문서를 통해 진행해주세요.")
