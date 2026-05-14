from langchain_chroma import Chroma  # 換成新的
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM  # 換成新的
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def get_rag_response(user_query):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 使用新的 Chroma 類別
    vector_db = Chroma(
        persist_directory="./chroma_db", 
        embedding_function=embeddings
    )
    
    # 增加 k 值，讓 AI 看到更多相關片段
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})

    # 使用新的 OllamaLLM
    llm = OllamaLLM(model="llama3")

    template = """
    你是一位專業的日本旅遊助手。請嚴格根據以下提供的【參考資料】來回答問題。
    如果資料中沒有提到，請回答「在目前的旅遊手冊中找不到相關資訊」。
    回答請使用繁體中文。

    【參考資料】：
    {context}

    使用者問題：{question}
    
    專業回答："""
    
    prompt = ChatPromptTemplate.from_template(template)

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain.invoke(user_query)

if __name__ == "__main__":
    print("-" * 30)
    print(get_rag_response("去澀谷有什麼要注意的？"))