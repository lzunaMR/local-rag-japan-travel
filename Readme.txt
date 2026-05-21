========================================================================
專案名稱：🇯🇵 日本旅遊全能地端 AI 顧問 (Agentic RAG System)
========================================================================

一、 專案簡介 (Project Overview)
------------------------------------------------------------------------
本專案是一款完全運行於地端（Local Environment）的智慧旅遊助理。
它結合了「基礎 RAG（檢索增強生成）」與「Agent（智慧代理人）」架構，能像人類
導遊一樣思考。當使用者提問時，系統會動態採取以下策略：
1. 翻閱私有知識庫：優先檢索本地的旅遊計畫（如 txt 或 pdf 手冊）。
2. 即時聯網搜尋：若本地資料不足，或問題涉及時效性（如天氣、當日最新匯率），
   系統會主動調用網路搜尋引擎（DuckDuckGo）抓取外部世界的新資訊。
3. 揉合對話記憶：結合前後文歷史紀錄，提供精準、有邏輯的繁體中文回應。


二、 核心功能 (Key Features)
------------------------------------------------------------------------
* 【多格式知識庫載入】: 自動讀取、切割並向量化 data/ 資料夾下的 .txt 與 .pdf 檔案。
* 【智慧重排 (Re-ranking)】: 預留 Flashrank 機制，自動對檢索結果進行二次排序，提升精準度。
* 【動態 Agent 聯網判斷】: 智慧識別關鍵字（天氣、匯率、今天等），自動發動網路搜尋。
* 【多輪連續對話】: 內建 Session State 記憶機制，能夠理解「那明天呢？」等前後文情境。
* 【直覺網頁介面】: 採用 Streamlit 打造極簡、流暢的對話視窗，並附帶側邊欄清除記憶功能。


三、 技術堆疊 (Technical Stack)
------------------------------------------------------------------------
本專案採用現代 AI 應用最核心的開源技術生態鏈：

1. 核心 AI 框架 (LangChain Ecosystem):
   - LangChain / LangChain Community: 串聯整套 RAG、工具（Tools）與鏈（Chains）的骨幹。
   - LCEL (LangChain Expression Language): 使用 `|` 管道符號實作高效能的組件串聯。

2. 大型語言模型驅動 (LLM Implementation):
   - Ollama: 在地端高效運行開源模型（專案預設使用 Llama 3）。
   - LangChain-Ollama: 負責與地端模型進行流式或標準化文字生成。

3. 向量檢索與嵌入 (Vector Search & Embeddings):
   - HuggingFace Embeddings (all-MiniLM-L6-v2): 將文本片段轉化為 384 維數學向量的模型。
   - ChromaDB (langchain-chroma): 地端輕量化向量資料庫，支援資料持久化（Persistent Storage）。
   - Flashrank (ContextualCompressionRetriever): 地端超輕量、高效能的二次重排（Re-rank）技術。

4. 外部工具調用 (Agent Tools):
   - DuckDuckGoSearchRun: 免 API Key 的聯網搜尋工具，賦予 LLM 獲取即時世界資訊的能力。

5. 前端網頁介面 (Frontend UI):
   - Streamlit: 基於 Python 的資料應用網頁框架，負責渲染對話框、側邊欄及狀態加載動畫。

6. 開發環境與版控 (Development Tools):
   - Python venv: 獨立的虛擬環境，防止套件版本衝突。
   - Git / GitHub: 分支管理（Branching）、變更追蹤與代碼庫儲存。


四、 專案檔案結構 (Project Structure)
------------------------------------------------------------------------
local-rag-project/
│
├── .gitignore              # 設定忽略提交的檔案（如 venv/、chroma_db/）
├── README.txt              # 本說明文件（專案導覽地圖）
│
├── ingest.py               # 資料優化腳本：負責讀取 data/、切割文字並建置 ChromaDB
├── app_logic.py            # 核心邏輯：檢索、Agent 聯網決策、LLM 鏈結的實作
├── app.py                  # 前端介面：Streamlit 網頁佈局、對話狀態維護
│
├── data/                   # [請自行建立] 存放個人私有旅遊手冊 (.txt, .pdf)
└── chroma_db/              # [執行後自動生成] 存放向量化後的本地資料庫


五、 快速啟動步驟 (Quick Start)
------------------------------------------------------------------------
1. 環境準備與套件安裝：
   確保已啟動虛擬環境 (venv)，並安裝所需套件（streamlit, langchain 等）。
   確保電腦已安裝 Ollama 並在背景執行，且已下載 llama3 模型 (`ollama run llama3`)。

2. 建立本地知識庫：
   將你的旅遊行程文字檔或 PDF 放進 `data/` 資料夾中。
   在終端機執行資料載入：
   > python ingest.py

3. 啟動 Streamlit 網頁應用：
   在終端機執行以下指令：
   > streamlit run app.py

4. 開始體驗：
   瀏覽器會自動開啟 `http://localhost:8501`，即可開始與你的全能導遊對話！
========================================================================