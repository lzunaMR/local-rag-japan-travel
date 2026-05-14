from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 嘗試修正導入路徑
try:
    from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
    # 新版 Flashrank 路徑通常在這裡
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

    # 2. 智慧判斷：是否需要發動「上網」工具
    # 判斷條件：本地找不到關鍵內容，或是詢問具備時效性的問題
    search_keywords = ["天氣", "匯率", "最新", "現在", "今天", "預約", "門票價格"]
    needs_web = len(local_context) < 200 or any(kw in user_query for kw in search_keywords)

    web_context = ""
    if needs_web:
        print(f"🌍 Agent 啟動：本地資料不足或涉及時效資訊，正在搜尋網路...")
        search = DuckDuckGoSearchRun()
        # 加上 2026 確保搜尋到的是最新資訊
        web_context = search.run(f"日本旅遊 {user_query} 2026")

    # 3. 處理對話記憶
    history_str = "\n".join([f"{m['role']}: {m['content']}" for m in chat_history_list[-3:]])

    # 4. 最終生成 (LLM 負責把兩者揉合)
    llm = OllamaLLM(model="llama3")
    
    prompt = ChatPromptTemplate.from_template("""
    你是一位具備即時聯網能力的日本旅遊顧問。
    你的任務是結合【本地手冊】與【即時網路資訊】來精準回答使用者。
    
    【對話歷史】：{history}
    【本地手冊】：{local_context}
    【即時網路資訊】：{web_context}
    
    使用者問題：{question}
    
    請依照以下格式回答：
    1. 針對問題給出詳細建議。
    2. 如果資料來自網路，請註明「(來源：即時網路搜尋)」。
    3. 如果是本地手冊內容，請註明「(來源：個人旅遊指南)」。
    """)

    chain = prompt | llm | StrOutputParser()
    return chain.invoke({
        "history": history_str,
        "local_context": local_context,
        "web_context": web_context,
        "question": user_query
    })