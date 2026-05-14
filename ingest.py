import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# 1. 載入資料
print("正在載入旅遊文件...")
loader = TextLoader("data/japan_travel.txt", encoding="utf-8")
documents = loader.load()

# 2. 切割文字 (Chunking)
# 我們把文字每 500 字切一段，重疊 50 字是為了確保上下文不會被硬生生切斷
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, 
    chunk_overlap=50
)
chunks = text_splitter.split_documents(documents)
print(f"文件切割完成，共分成 {len(chunks)} 個段落。")

# 3. 建立向量 (Embedding)
# 使用 HuggingFace 的開源模型，這是在你電腦本地運行的
print("正在進行向量化並存入資料庫 (第一次運行會下載模型，請稍候)...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 4. 存入 ChromaDB 向量資料庫
# 會在你的資料夾下建立一個 'chroma_db' 的資料夾來存檔
vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

print("✅ 恭喜！你的旅遊知識庫已建立完成，存放在 ./chroma_db 資料夾中。")