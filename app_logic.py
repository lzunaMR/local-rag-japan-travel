import os
import warnings

# === 解決問題 3：強行關閉 Transformers 與相關警告雜訊 ===
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

try:
    from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
    from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
    HAS_ADVANCED = True
except ImportError:
    HAS_ADVANCED = False

def get_enhanced_rag_response(user_query, chat_history_list):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    
    # 1. 檢索資料
    if HAS_ADVANCED:
        base_retriever = vector_db.as_retriever(search_kwargs={"k": 8})
        compressor = FlashrankRerank()
        retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=base_retriever)
        docs = retriever.invoke(user_query)
    else:
        docs = vector_db.as_retriever(search_kwargs={"k": 3}).invoke(user_query)

    local_context = "\n".join([d.page_content for d in docs])

    # 2. 智慧聯網判斷 (修正：避免"預約"這種本地有的關鍵字誤觸上網)
    # 我們把時效性關鍵字縮小，若本地抓得到足夠字數，就不輕易上網
    search_keywords = ["天氣", "匯率", "最新", "現在", "今天", "即時資訊"]
    needs_web = len(local_context) < 150 or any(kw in user_query for kw in search_keywords)

    web_context = ""
    if needs_web:
        print(f"🌍 Agent 啟動：本地資料不足或涉及時效資訊，正在搜尋網路...")
        try:
            search = DuckDuckGoSearchRun()
            web_context = search.run(f"日本旅遊 {user_query} 2026")
        except Exception:
            web_context = "無法取得即時網路資訊。"

    # 3. 處理對話記憶
    history_str = "\n".join([f"{m['role']}: {m['content']}" for m in chat_history_list[-3:]])

    # 4. 最終生成 (=== 解決問題 1：加強繁體中文約束 ===)
    llm = OllamaLLM(model="llama3")
    
    prompt = ChatPromptTemplate.from_template("""
    【重要指令】：你必須完全使用「繁體中文」(Traditional Chinese) 回答問題！絕對不能使用英文或簡體中文回應！
    
    你是一位具備即時聯網能力的日本旅遊顧問。
    請結合【本地手冊】與【即時網路資訊】來精準回答使用者。
    
    【對話歷史】：{history}
    【本地手冊內容】：{local_context}
    【即時網路資訊】：{web_context}
    
    使用者問題：{question}
    
    請依照以下格式使用「繁體中文」回答：
    1. 針對問題給出詳細建議。
    2. 如果資料主要來自網路，請註明「(來源：即時網路搜尋)」。
    3. 如果是本地手冊內容，請註明「(來源：個人旅遊指南)」。
    
    【再次提醒】：回答全文字符必須是繁體中文！
    """)

    chain = prompt | llm | StrOutputParser()
    return chain.invoke({
        "history": history_str,
        "local_context": local_context,
        "web_context": web_context,
        "question": user_query
    })