# 파일 이름: app.py (정확도 극대화 SOTA 버전)

import streamlit as st
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
import json
import openai
import numpy as np

# --- 1. 초기 설정: 페이지, API 키 ---
st.set_page_config(page_title="전략기획부 AI 법률 보조원", page_icon="⚖️", layout="wide")
try:
    openai.api_key = st.secrets["OPENAI_API_KEY"]
except:
    st.error("OpenAI API 키를 Streamlit Secrets에 등록해주세요.")

# --- 2. 모델 및 데이터 로딩 ---
@st.cache_resource
def load_models_and_data():
    try:
        embedding_model = SentenceTransformer('jhgan/ko-sroberta-multitask')
        reranker_model = CrossEncoder('Dongjin-kr/ko-reranker')
        index = faiss.read_index("manuals_vector_db.index")
        with open("all_manual_chunks.json", "r", encoding="utf-8") as f:
            chunks_with_metadata = json.load(f)
        return embedding_model, reranker_model, index, chunks_with_metadata
    except FileNotFoundError:
        return None, None, None, None

# --- 3. Advanced RAG 핵심 기능 함수들 ---
def transform_query(user_question):
    """LLM을 사용해 사용자의 질문을 여러 개의 구체적인 검색용 질문으로 변환"""
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that generates 3 varied versions of a user question to improve document retrieval. The queries should be specific and cover different aspects of the original question. Output a numbered list of questions."},
            {"role": "user", "content": f"Original question: {user_question}"}
        ],
        temperature=0.3,
    )
    # 변환된 질문들을 리스트로 파싱
    transformed_queries = [user_question] + [q.strip() for q in response.choices[0].message.content.split('\n') if q.strip() and q[0].isdigit()]
    return list(set(transformed_queries)) # 중복 제거

def get_relevant_manual_chunks(queries, k_initial=10, k_final=5):
    """쿼리 변환, 검색, 재정렬을 통해 가장 관련성 높은 문서를 찾습니다."""
    all_retrieved_chunks = []
    for query in queries:
        question_vector = model.encode([query])
        distances, indices = index.search(np.array(question_vector, dtype=np.float32), k_initial)
        all_retrieved_chunks.extend([chunks_with_metadata[i] for i in indices[0]])
    
    # 중복된 chunk 제거
    unique_chunks = [dict(t) for t in {tuple(d.items()) for d in all_retrieved_chunks}]
    
    # Reranker로 최종 순위 결정
    pairs = [[queries[0], chunk['content']] for chunk in unique_chunks]
    scores = reranker.predict(pairs)
    ranked_chunks = [chunk for _, chunk in sorted(zip(scores, unique_chunks), key=lambda x: x[0], reverse=True)]
    return ranked_chunks[:k_final]

def generate_answer(user_question, relevant_chunks):
    """검색된 내용을 바탕으로 구조화된 답변과 검증 결과를 생성"""
    context_str = "\n\n".join([f"참고 문서 {i+1} (출처: {chunk['source']}, Page: {chunk.get('page', 'N/A')}):\n{chunk['content']}" for i, chunk in enumerate(relevant_chunks)])
    
    # 답변 생성을 위한 시스템 프롬프트 (이전과 동일)
    generation_prompt = """
    # Role: 당신은 한국 공정거래법을 전문으로 다루는 '전략기획부 AI 법률 보조자'입니다... (이전 최종 프롬프트 내용)
    """
    
    # 답변 검증을 위한 프롬프트
    verification_prompt_template = """
    # Role: 당신은 AI 답변을 검증하는 엄격한 팩트체커입니다.
    # Task: 주어진 [생성된 답변]이, 오직 [참고 자료]에만 100% 근거하여 작성되었는지 확인하십시오.
    # Instructions:
    1. 답변의 모든 문장을 하나씩 읽으며, 각 문장이 [참고 자료]의 내용과 일치하는지 확인합니다.
    2. 만약 답변에 [참고 자료]에 없는 내용, 추측, 또는 과장이 포함되어 있다면 "부정확함"으로 판단합니다.
    3. 모든 내용이 완벽하게 일치할 경우에만 "정확함"으로 판단합니다.
    4. 최종 판단을 "판단: [정확함/부정확함]" 형식으로 가장 첫 줄에 출력하고, 그 이유를 다음 줄에 간결하게 설명하십시오.
    """
    
    # 1. 답변 생성
    messages_for_generation = [
        {"role": "system", "content": generation_prompt},
        {"role": "user", "content": f"[관련 매뉴얼 정보]\n{context_str}\n---\n[질문]\n{user_question}"}
    ]
    response = openai.chat.completions.create(model="gpt-4o", messages=messages_for_generation, temperature=0.1)
    generated_answer = response.choices[0].message.content

    # 2. 생성된 답변 검증
    messages_for_verification = [
        {"role": "system", "content": verification_prompt_template},
        {"role": "user", "content": f"[생성된 답변]\n{generated_answer}\n---\n[참고 자료]\n{context_str}"}
    ]
    verification_response = openai.chat.completions.create(model="gpt-4o", messages=messages_for_verification, temperature=0.0)
    verification_result = verification_response.choices[0].message.content
    
    return generated_answer, verification_result, relevant_chunks

# --- 4. 메인 UI 구성 ---
st.title("⚖️ 전략기획부 AI 법률 보조원 (SOTA Ver.)")
model, reranker, index, chunks_with_metadata = load_models_and_data()

if model is None:
    st.error("챗봇 핵심 데이터 파일을 로드할 수 없습니다.")
else:
    st.success("AI 법률 보조원이 준비되었습니다. 공정거래법 공시 의무에 대해 질문해주세요.")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if isinstance(message["content"], str):
                st.markdown(message["content"], unsafe_allow_html=True)
            else:
                st.write(message["content"])

    if prompt := st.chat_input("RCPS 전환권 행사에 따른 신주 취득 시 공시 의무가 있나요?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("질문 분석 및 변환 중..."):
                transformed_queries = transform_query(prompt)
            
            with st.spinner("관련 자료 검색 및 재정렬 중..."):
                relevant_chunks = get_relevant_manual_chunks(transformed_queries)

            with st.spinner("답변 생성 및 자체 검증 수행 중..."):
                answer, verification, sources = generate_answer(prompt, relevant_chunks)
            
            # 최종 답변 객체 생성
            final_response = {
                "answer": answer,
                "verification": verification,
            }
            
            # 답변 출력
            is_verified = "정확함" in verification.split('\n')[0]
            st.markdown(f"#### 💬 AI 법률 보조원 답변 {'✅' if is_verified else '⚠️'}")
            st.markdown(answer, unsafe_allow_html=True)
            
            # 검증 결과 및 참고자료 expander
            with st.expander("AI 답변 검증 결과 및 핵심 참고자료 보기"):
                if is_verified:
                    st.success(verification)
                else:
                    st.warning(verification)
                st.divider()
                st.markdown("**핵심 참고자료 (재정렬 후)**")
                for chunk in sources:
                    st.info(f"**출처:** {chunk['source']}, **Page:** {chunk.get('page', 'N/A')}")
                    st.text(chunk['content'])
            
            warning_message = "\n\n---\n◆본 답변은 전략기획부가 학습시킨 ChatGPT를 통해 제공하는 답변으로, 참고용으로만 활용하시기 바랍니다."
            st.markdown(warning_message)

            st.session_state.messages.append({"role": "assistant", "content": final_response})
