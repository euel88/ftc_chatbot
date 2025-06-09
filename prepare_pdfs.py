# 파일 이름: prepare_pdfs.py (Chunking 전략 개선 버전)

import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import json
import pypdf
# LangChain의 텍스트 분할기 import
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 사용자 요청에 따라 매뉴얼 파일이 있는 디렉토리 경로를 설정합니다.
MANUALS_DIR = r"H:\2. 기획 부서\1. 공정거래위원회\4. 공정위 자료_매뉴얼 등\25년"

TEXT_FILENAME = "all_manual_chunks.json"
DB_FILENAME = "manuals_vector_db.index"

def load_and_chunk_pdfs(directory):
    """지정된 디렉토리의 모든 .pdf 파일에서 텍스트를 추출하고, 정교하게 분할합니다."""
    all_chunks_with_metadata = []
    print(f"'{directory}' 폴더에서 PDF 매뉴얼 파일을 로드합니다...")
    if not os.path.exists(directory):
        raise FileNotFoundError(f"'{directory}' 폴더를 찾을 수 없습니다. 경로를 다시 확인해주세요.")

    # ======================================================================
    # ## 1. LangChain 텍스트 분할기 설정 ##
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,      # 각 chunk의 최대 글자 수
        chunk_overlap=200,    # chunk 간 겹치는 글자 수 (문맥 유지에 중요)
        length_function=len,
        is_separator_regex=False,
    )
    # ======================================================================

    for filename in os.listdir(directory):
        if filename.lower().endswith(".pdf"):
            filepath = os.path.join(directory, filename)
            print(f"- '{filepath}' 처리 중...")
            try:
                reader = pypdf.PdfReader(filepath)
                pdf_text = ""
                for i, page in enumerate(reader.pages):
                    page_content = page.extract_text()
                    if page_content:
                        # 페이지 번호 정보를 메타데이터로 추가하기 위해 페이지별로 처리
                        # LangChain 분할기를 사용하여 텍스트 나누기
                        chunks = text_splitter.split_text(page_content)
                        for chunk_text in chunks:
                            all_chunks_with_metadata.append({
                                "source": filename, 
                                "content": chunk_text,
                                "page": i + 1  # 페이지 번호 추가 (1부터 시작)
                            })
            except Exception as e:
                print(f"  [오류] '{filename}' 파일 처리 중 오류 발생: {e}")

    print(f"\n총 {len(all_chunks_with_metadata)}개의 문단(chunk)을 성공적으로 추출했습니다.")
    return all_chunks_with_metadata

def build_and_save_vector_db(chunks_with_metadata):
    """매뉴얼 문단들로 벡터 DB를 만들고 파일로 저장합니다."""
    if not chunks_with_metadata:
        print("벡터 DB를 생성할 내용이 없습니다.")
        return

    contents = [item['content'] for item in chunks_with_metadata]
    
    # ======================================================================
    # ## 2. 모델 로드 방식 통일 ##
    # 인터넷에서 직접 모델을 다운로드하도록 변경
    print("\n인터넷에서 임베딩 모델(ko-sroberta-multitask)을 로드합니다...")
    model = SentenceTransformer('jhgan/ko-sroberta-multitask')
    # ======================================================================
    
    print("\n매뉴얼 문단을 벡터로 변환합니다... (시간이 소요됩니다)")
    vectors = model.encode(contents, show_progress_bar=True)
    
    dimension = vectors.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(vectors, dtype=np.float32))
    
    print(f"\n벡터 데이터베이스를 '{DB_FILENAME}' 파일로 저장합니다.")
    faiss.write_index(index, DB_FILENAME)
    
    print(f"메타데이터를 '{TEXT_FILENAME}' 파일로 저장합니다.")
    with open(TEXT_FILENAME, 'w', encoding='utf-8') as f:
        json.dump(chunks_with_metadata, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    chunks = load_and_chunk_pdfs(MANUALS_DIR)
    
    if chunks:
        build_and_save_vector_db(chunks)
        print("\n--- ✅ 데이터 준비 완료 ---")
        print("이제 새로 생성된 'all_manual_chunks.json'과 'manuals_vector_db.index' 파일을 GitHub에 올리고 앱을 재부팅하세요.")
    else:
        print("처리할 PDF 파일이 없습니다. 지정된 폴더를 확인해주세요.")
