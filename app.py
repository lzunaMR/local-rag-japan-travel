import streamlit as st
from app_logic import get_rag_response

# 網頁設定
st.set_page_config(page_title="日本旅遊 RAG 助手", page_icon="🇯🇵")

st.title("🇯🇵 日本旅遊地端 AI 助手")
st.markdown("這是一個完全運行在你電腦上的 RAG 系統，不消耗任何 API 額度。")

# 初始化聊天紀錄
if "messages" not in st.session_state:
    st.session_state.messages = []

# 顯示歷史對話
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 使用者輸入
if prompt := st.chat_input("請輸入你的旅遊問題..."):
    # 存入紀錄並顯示
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 生成回答
    with st.chat_message("assistant"):
        with st.spinner("正在搜尋資料並思考中..."):
            response = get_rag_response(prompt)
            st.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})