"""
streamlit_app.py — GradCopilot 前端主入口。

启动方式：
  streamlit run streamlit_app.py

环境变量：
  BACKEND_URL — FastAPI 后端地址（默认 http://localhost:8000）
"""
import streamlit as st

# ── 页面配置（必须是第一个 Streamlit 调用） ─────────────────
st.set_page_config(
    page_title="GradCopilot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 全局 Session State 初始化，包含当前会话 ID、聊天历史、搜索结果、已选论文、发送状态
if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "search_results" not in st.session_state:
    st.session_state.search_results = []

if "selected_paper_ids" not in st.session_state:
    st.session_state.selected_paper_ids = []

if "is_sending" not in st.session_state:
    st.session_state.is_sending = False

# ── 组件导入 ─────────────────────────────────────────────────
from components.sidebar import render_sidebar
from components.chat_panel import render_chat
from components.search_panel import render_search
from components.knowledge_panel import render_knowledge

# ── 左侧边栏 ─────────────────────────────────────────────────
with st.sidebar:
    render_sidebar()

# ── 主内容区（Tab 切换） ─────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["💬 聊天", "🔍 论文搜索", "📦 知识库"])

with tab1:
    render_chat()

with tab2:
    render_search()

with tab3:
    render_knowledge()
