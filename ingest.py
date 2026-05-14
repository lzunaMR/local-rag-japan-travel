import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

def main():
    # 1. 定義資料夾路徑與模型名稱
    DATA_PATH = "data/"
    CHROMA_PATH = "chroma_db/"
    EMBED_MODEL = "all-MiniLM-L6-v2"

    print(f"🚀 開始處理資料夾: {DATA_PATH} 中的文件...")

    # 2. 建立支援多格式的讀取器
    # 這裡使用 DirectoryLoader 來自動辨別檔案類型
    text_loader_kwargs = {'encoding': 'utf8'}
    
    # 讀取 .txt 檔案
    txt_loader = DirectoryLoader(
        DATA_PATH, 
        glob="**/*.txt", 
        loader_cls=TextLoader, 
        loader_kwargs=text_loader_kwargs
    )
    
    # 讀取 .pdf 檔案
    pdf_loader = DirectoryLoader(
        DATA_PATH, 
        glob="**/*.pdf", 
        loader_cls=PyPDFLoader
    )

    # 執行載入動作
    docs = []
    docs.extend(txt_loader.load())
    docs.extend(pdf_loader.load())

    if not docs:
        print("⚠️ 找不到任何檔案，請確認 data/ 資料夾內是否有 .txt 或 .pdf 檔案。")
        return

    print(f"✅ 成功載入 {len(docs)} 份原始文件。")

    # 3. 切割文字 (Chunking)
    # RAG 的品質取決於切片的大小，這裡設為 800 字，並有 100 字重疊
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(docs)
    print(f"✂️ 文件切割完成，共產生 {len(chunks)} 個知識片段。")

    # 4. 初始化 Embedding 模型 (地端運行)
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    # 5. 存入或更新 ChromaDB 向量資料庫
    # 如果資料夾已存在，它會自動將新資料「增量」添加進去
    print("💎 正在進行向量化並存入地端資料庫 (ChromaDB)...")
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )

    print(f"🎉 大功告成！知識庫已更新，存放於: {CHROMA_PATH}")

if __name__ == "__main__":
    main()