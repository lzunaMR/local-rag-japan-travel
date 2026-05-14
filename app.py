import streamlit as st
from app_logic import get_enhanced_rag_response

# 1. 網頁基礎設定
st.set_page_config(page_title="日本旅遊 RAG 助手", page_icon="🇯🇵")

st.title("🇯🇵 日本旅遊全能 AI 顧問")
st.caption("整合本地手冊、即時網頁搜尋與對話記憶的進階 RAG 系統")

# 2. 初始化聊天紀錄 (確保 session_state 存在)
if "messages" not in st.session_state:
    st.session_state.messages = []

# 3. 側邊欄：顯示系統狀態
with st.sidebar:
    st.title("系統資訊")
    st.info("模型：Llama 3 (Ollama)")
    st.success("模式：Agentic RAG + Web Search")
    if st.button("清除對話紀錄"):
        st.session_state.messages = []
        st.rerun()

# 4. 顯示歷史對話內容
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 5. 使用者輸入邏輯 (單一入口)
if prompt := st.chat_input("想問什麼？(例如：現在日幣匯率？或是東京行程？)"):
    # 顯示使用者的訊息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 生成回答
    with st.chat_message("assistant"):
        with st.spinner("正在調閱手冊並搜尋網頁..."):
            try:
                # 重點：傳入兩個參數 (問題, 歷史紀錄)
                response = get_enhanced_rag_response(prompt, st.session_state.messages)
                st.markdown(response)
                # 存入紀錄
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"發生錯誤：{str(e)}")
                st.info("請確保 Ollama 已啟動且模型 llama3 已下載。")