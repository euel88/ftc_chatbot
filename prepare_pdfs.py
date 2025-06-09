# 파일 이름: prepare_pdfs.py (최종 수정본)

import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import json
import pypdf

# 사용자 요청에 따라 매뉴얼 파일이 있는 디렉토리 경로를 설정합니다.
MANUALS_DIR = r"H:\2. 기획 부서\1. 공정거래위원회\4. 공정위 자료_매뉴얼 등\25년"

TEXT_FILENAME = "all_manual_chunks.json"
DB_FILENAME = "manuals_vector_db.index"

def load_and_chunk_pdfs(directory):
    # 이 함수는 이전과 동일합니다.
    all_chunks_with_metadata = []
    print(f"'{directory}' 폴더에서 PDF 매뉴얼 파일을 로드합니다...")
    if not os.path.exists(directory):
        raise FileNotFoundError(f"'{directory}' 폴더를 찾을 수 없습니다. 경로를 다시 확인해주세요.")
    for filename in os.listdir(directory):
        if filename.lower().endswith(".pdf"):
            filepath = os.path.join(directory, filename)
            print(f"- '{filepath}' 처리 중...")
            try:
                reader = pypdf.PdfReader(filepath)
                pdf_text = ""
                for page in reader.pages:
                    pdf_text += page.extract_text() + "\n\n"
                chunks = pdf_text.split('\n\n')
                for chunk_text in chunks:
                    if len(chunk_text.strip()) > 50:
                        all_chunks_with_metadata.append({"source": filename, "content": chunk_text.strip()})
            except Exception as e:
                print(f"  [오류] '{filename}' 파일 처리 중 오류 발생: {e}")
    print(f"총 {len(all_chunks_with_metadata)}개의 문단(chunk)을 성공적으로 추출했습니다.")
    return all_chunks_with_metadata

def build_and_save_vector_db(chunks_with_metadata):
    """매뉴얼 문단들로 벡터 DB를 만들고 파일로 저장합니다."""
    if not chunks_with_metadata:
        print("벡터 DB를 생성할 내용이 없습니다.")
        return

    contents = [item['content'] for item in chunks_with_metadata]
    
    # ======================================================================
    # ## 모델 로드 경로 수정 ##
    # 인터넷 주소 대신, 우리가 직접 다운로드한 모델의 로컬 폴더 경로를 지정합니다.
    model_path = "./models/ko-sroberta-multitask"
    print(f"로컬 경로에서 임베딩 모델을 로드합니다: {model_path}")
    model = SentenceTransformer(model_path)
    # ======================================================================
    
    print("매뉴얼 문단을 벡터로 변환합니다...")
    vectors = model.encode(contents, show_progress_bar=True)
    
    dimension = vectors.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(vectors, dtype=np.float32))
    
    print(f"벡터 데이터베이스를 '{DB_FILENAME}' 파일로 저장합니다.")
    faiss.write_index(index, DB_FILENAME)
    
    print(f"메타데이터를 '{TEXT_FILENAME}' 파일로 저장합니다.")
    with open(TEXT_FILENAME, 'w', encoding='utf-8') as f:
        json.dump(chunks_with_metadata, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    chunks = load_and_chunk_pdfs(MANUALS_DIR)
    
    if chunks:
        build_and_save_vector_db(chunks)
        print("\n--- ✅ 데이터 준비 완료 ---")
        print("이제 'app_manual.py'를 실행하여 챗봇을 시작할 수 있습니다.")
    else:
        print("처리할 PDF 파일이 없습니다. 지정된 폴더를 확인해주세요.")
