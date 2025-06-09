# 파일 이름: app_manual.py (최종 수정본)

import streamlit as st
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import json
import openai
import os

# 1. 페이지 제목 변경
st.set_page_config(page_title="전략기획부 AI 답변 챗봇", page_icon="🏢", layout="centered")

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
        model_path = "./models/ko-sroberta-multitask"
        model = SentenceTransformer(model_path)
        index = faiss.read_index("manuals_vector_db.index")
        with open("all_manual_chunks.json", "r", encoding="utf-8") as f:
            chunks_with_metadata = json.load(f)
        return model, index, chunks_with_metadata
    except FileNotFoundError:
        return None, None, None

def get_relevant_manual_chunks(user_question, k=5):
    """사용자 질문과 관련된 매뉴얼 문단 k개를 DB에서 찾습니다."""
    question_vector = model.encode([user_question])
    distances, indices = index.search(np.array(question_vector, dtype=np.float32), k)
    relevant_chunks = [chunks_with_metadata[i] for i in indices[0]]
    return relevant_chunks

def generate_answer_with_llm(user_question, relevant_chunks):
    """검색된 매뉴얼 내용을 근거로 LLM(GPT)이 답변을 생성합니다."""
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

# 1. 메인 타이틀 변경
st.title("🏢 전략기획부 AI 답변 챗봇")

# 데이터 로딩 시도
model, index, chunks_with_metadata = load_models_and_data()

if model is None:
    st.error("챗봇 데이터 파일('prepare_pdfs.py' 실행 필요)을 찾을 수 없습니다.")
else:
    st.success("모델과 데이터를 성공적으로 로드했습니다!", icon="✅")
    
    st.markdown("회사 내부 매뉴얼에 대해 궁금한 점을 질문해보세요.")
    user_question = st.text_input("질문 입력:", placeholder="예: 대규모내부거래 기준 금액은 어떻게 되나요?")

    if st.button("질문하기", type="primary"):
        if user_question:
            with st.spinner("AI가 매뉴얼을 검토하고 답변을 생성하는 중입니다..."):
                relevant_chunks = get_relevant_manual_chunks(user_question)
                answer = generate_answer_with_llm(user_question, relevant_chunks)
                
                # 3. 답변 마지막에 경고 문구 추가
                warning_message = "\n\n---\n◆본 답변은 전략기획부가 학습시킨 AI를 통해 제공하는 답변으로, 참고용으로만 활용하시기 바랍니다."
                answer += warning_message

                # 2. AI 답변 헤더 변경
                st.markdown("#### 💬 전략기획부 AI 답변")
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
