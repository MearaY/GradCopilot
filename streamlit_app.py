"""
Paper Knowledge Base Q&A System
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional

import httpx
import streamlit as st

# ===================== Basic configuration =====================
BASE_URL = "http://localhost:8000"

# Initialize theme state
if "theme_mode" not in st.session_state:
    st.session_state.theme_mode = "light"

st.set_page_config(
    page_title="论文知识库问答系统",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ===================== Design Token System =====================
def get_design_tokens(mode: str = "dark"):
    """Get complete design token system for specified theme mode."""
    if mode == "light":
        return {
            "bg": "#f8fafc",
            "bg_secondary": "#f1f5f9",
            "panel": "#ffffff",
            "panel_hover": "#f8fafc",
            "border": "#e2e8f0",
            "border_strong": "#cbd5e1",
            "text": "#1e293b",
            "text_secondary": "#64748b",
            "text_muted": "#94a3b8",
            "accent": "#10b981",
            "accent_soft": "rgba(16, 185, 129, 0.10)",
            "accent_strong": "rgba(16, 185, 129, 0.15)",
            "danger": "#ef4444",
            "warning": "#f59e0b",
            "info": "#3b82f6",
            "shadow": "0 10px 40px rgba(0, 0, 0, 0.08)",
            "shadow_strong": "0 20px 60px rgba(0, 0, 0, 0.12)",
            "radius_xl": "24px",
            "radius_lg": "18px",
            "radius_md": "14px",
            "radius_sm": "10px",
            "gradient_brand": "linear-gradient(135deg, #10b981 0%, #059669 100%)",
            "gradient_panel": "linear-gradient(180deg, #ffffff 0%, #fafafa 100%)",
        }
    else:
        return {
            "bg": "#0b0f14",
            "bg_secondary": "#070b10",
            "panel": "#111827",
            "panel_hover": "#151e2d",
            "border": "rgba(148, 163, 184, 0.18)",
            "border_strong": "rgba(148, 163, 184, 0.28)",
            "text": "#ffffff",
            "text_secondary": "#cbd5e1",
            "text_muted": "#94a3b8",
            "accent": "#10b981",
            "accent_soft": "rgba(16, 185, 129, 0.12)",
            "accent_strong": "rgba(16, 185, 129, 0.18)",
            "danger": "#ef4444",
            "warning": "#f59e0b",
            "info": "#3b82f6",
            "shadow": "0 18px 50px rgba(0, 0, 0, 0.30)",
            "shadow_strong": "0 24px 70px rgba(0, 0, 0, 0.40)",
            "radius_xl": "24px",
            "radius_lg": "18px",
            "radius_md": "14px",
            "radius_sm": "10px",
            "gradient_brand": "linear-gradient(135deg, #10b981 0%, #059669 100%)",
            "gradient_panel": "linear-gradient(180deg, rgba(17, 24, 39, 0.96), rgba(15, 23, 42, 0.92))",
        }

tokens = get_design_tokens(st.session_state.theme_mode)

# ===================== Global style with theme support =====================
style_css = f"""
<style>
    :root {{
        --bg: {tokens["bg"]};
        --bg-secondary: {tokens["bg_secondary"]};
        --panel: {tokens["panel"]};
        --panel-hover: {tokens["panel_hover"]};
        --border: {tokens["border"]};
        --border-strong: {tokens["border_strong"]};
        --text: {tokens["text"]};
        --text-secondary: {tokens["text_secondary"]};
        --text-muted: {tokens["text_muted"]};
        --accent: {tokens["accent"]};
        --accent-soft: {tokens["accent_soft"]};
        --accent-strong: {tokens["accent_strong"]};
        --danger: {tokens["danger"]};
        --warning: {tokens["warning"]};
        --info: {tokens["info"]};
        --shadow: {tokens["shadow"]};
        --shadow-strong: {tokens["shadow_strong"]};
        --radius-xl: {tokens["radius_xl"]};
        --radius-lg: {tokens["radius_lg"]};
        --radius-md: {tokens["radius_md"]};
        --radius-sm: {tokens["radius_sm"]};
        --gradient-brand: {tokens["gradient_brand"]};
        --gradient-panel: {tokens["gradient_panel"]};
    }}

    html, body, [data-testid="stAppViewContainer"], [data-testid="stAppViewContainer"] > .main, .block-container {{
        height: 100%;
        background: var(--bg);
        color: var(--text);
    }}

    [data-testid="stHeader"], footer, #MainMenu {{
        visibility: hidden;
        height: 0;
    }}

    .block-container {{
        padding: 0.8rem 1rem 0.9rem 1rem;
        max-width: 100% !important;
    }}

    .topbar {{
        background: var(--gradient-panel);
        border: 1px solid var(--border);
        border-radius: 22px;
        box-shadow: var(--shadow);
        padding: 0.9rem 1rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
    }}

    .brand {{
        display: flex;
        flex-direction: column;
        gap: 0.15rem;
    }}

    .brand .title {{
        font-size: 1.2rem;
        font-weight: 700;
        color: var(--text);
        letter-spacing: 0.2px;
    }}

    .brand .subtitle {{
        font-size: 0.85rem;
        color: var(--text-muted);
    }}

    .status-pill {{
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        padding: 0.45rem 0.75rem;
        border-radius: 999px;
        border: 1px solid var(--border);
        background: var(--panel-hover);
        color: var(--text);
        font-size: 0.86rem;
        white-space: nowrap;
    }}

    .status-dot {{
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: var(--accent);
        box-shadow: 0 0 0 4px var(--accent-soft);
    }}

    .workspace {{
        flex: 1;
        min-height: 0;
        overflow: hidden;
    }}

    .panel-card {{
        background: var(--gradient-panel);
        border: 1px solid var(--border);
        border-radius: var(--radius-xl);
        box-shadow: var(--shadow);
        overflow: hidden;
        height: 100%;
        min-height: 0;
        display: flex;
        flex-direction: column;
    }}

    .panel-head {{
        padding: 0.9rem 1rem;
        border-bottom: 1px solid var(--border);
        background: var(--panel-hover);
        flex-shrink: 0;
    }}

    .panel-title {{
        font-size: 1rem;
        font-weight: 700;
        color: var(--text);
        margin: 0;
    }}

    .panel-desc {{
        color: var(--text-muted);
        font-size: 0.82rem;
        margin-top: 0.2rem;
    }}

    .panel-body {{
        padding: 0.9rem;
        flex: 1;
        overflow-y: auto;
        overflow-x: hidden;
        scrollbar-width: thin;
        scrollbar-color: rgba(148, 163, 184, 0.35) transparent;
        min-height: 0;
    }}

    .panel-body::-webkit-scrollbar {{
        width: 8px;
    }}

    .panel-body::-webkit-scrollbar-thumb {{
        background: rgba(148, 163, 184, 0.3);
        border-radius: 999px;
    }}

    .chat-messages-wrapper {{
        flex: 1;
        overflow-y: auto;
        overflow-x: hidden;
        display: flex;
        flex-direction: column;
        justify-content: flex-end;
        min-height: 0;
        padding-bottom: 0.5rem;
    }}

    .chat-messages-inner {{
        display: flex;
        flex-direction: column;
        gap: 0.75rem;
    }}

    .chat-input-wrapper {{
        flex-shrink: 0;
        border-top: 1px solid var(--border);
        background: var(--panel);
        padding: 0.75rem 0.9rem;
    }}

    .card {{
        background: var(--panel-hover);
        border: 1px solid var(--border);
        border-radius: var(--radius-lg);
        padding: 0.9rem;
        margin-bottom: 0.85rem;
    }}

    .stButton > button {{
        border-radius: var(--radius-md) !important;
        border: 1px solid var(--border) !important;
        background: var(--panel-hover) !important;
        color: var(--text) !important;
        transition: all 0.15s ease;
        box-shadow: none !important;
    }}

    .stButton > button:hover {{
        border-color: var(--accent) !important;
        background: var(--accent-soft) !important;
    }}

    .stButton > button[kind="primary"] {{
        background: var(--gradient-brand) !important;
        border-color: var(--accent) !important;
        color: white !important;
    }}

    .stTextInput input, .stTextArea textarea, .stNumberInput input, .stSelectbox div[data-baseweb="select"] > div, .stDateInput input {{
        background: var(--panel-hover) !important;
        color: var(--text) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius-md) !important;
    }}

    .stTextInput label, .stNumberInput label, .stSelectbox label, .stDateInput label {{
        color: var(--text-secondary) !important;
        font-weight: 600;
    }}

    .stExpander {{
        border: 1px solid var(--border) !important;
        border-radius: var(--radius-lg) !important;
        background: var(--panel-hover) !important;
    }}

    .stMetric {{
        background: var(--panel-hover);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 0.6rem 0.75rem;
    }}

    .helper-line {{
        color: var(--text-muted);
        font-size: 0.8rem;
        margin-top: 0.3rem;
    }}

    .stats-row {{
        display: flex;
        gap: 0.75rem;
        margin-bottom: 0.75rem;
    }}

    .stat-item {{
        flex: 1;
        padding: 0.6rem 0.75rem;
        background: var(--panel-hover);
        border: 1px solid var(--border);
        border-radius: var(--radius-md);
    }}

    .stat-label {{
        font-size: 0.75rem;
        color: var(--text-muted);
        margin-bottom: 0.15rem;
    }}

    .stat-value {{
        font-size: 1.1rem;
        font-weight: 700;
        color: var(--text);
    }}

    /* Chat message styles */
    [data-testid="stChatMessage"] {{
        border-radius: var(--radius-lg);
        margin-bottom: 0.75rem;
    }}

    /* Message content - using stable DOM structure */
    [data-testid="stChatMessage"] > div > div > div {{
        font-weight: 450 !important;
        line-height: 1.6 !important;
        font-size: 1rem !important;
    }}

    /* User message (based on avatar color - red) */
    [data-testid="stChatMessage"]:has([style*="background-color: rgb(239, 68, 68)"]) {{
        background: var(--panel) !important;
        border: 1px solid var(--border-strong) !important;
    }}

    /* User message text */
    [data-testid="stChatMessage"]:has([style*="background-color: rgb(239, 68, 68)"]) > div > div > div {{
        color: var(--text) !important;
        font-weight: 450 !important;
    }}

    /* Assistant message (based on avatar color - orange) */
    [data-testid="stChatMessage"]:has([style*="background-color: rgb(245, 158, 11)"]) {{
        background: var(--accent-strong) !important;
        border: 1px solid var(--accent) !important;
    }}

    /* Assistant message text */
    [data-testid="stChatMessage"]:has([style*="background-color: rgb(245, 158, 11)"]) > div > div > div {{
        color: #ffffff !important;
        font-weight: 550 !important;
        text-shadow: 0 0 2px rgba(255, 255, 255, 0.3) !important;
    }}

    /* Alternative: using aria-label for better stability */
    [data-testid="stChatMessage"]:has([aria-label="User"]) > div > div > div {{
        color: var(--text) !important;
    }}

    [data-testid="stChatMessage"]:has([aria-label="Assistant"]) > div > div > div {{
        color: #ffffff !important;
        font-weight: 550 !important;
        text-shadow: 0 0 2px rgba(255, 255, 255, 0.3) !important;
    }}

    /* Additional fallback: using direct child selectors */
    [data-testid="stChatMessage"] > div:nth-child(2) > div {{
        font-weight: 450 !important;
        line-height: 1.6 !important;
        font-size: 1rem !important;
    }}

    /* Chat input */
    [data-testid="stChatInput"] {{
        background: var(--panel) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius-lg) !important;
    }}

    [data-testid="stChatInput"] input {{
        background: var(--panel-hover) !important;
        color: var(--text) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius-md) !important;
    }}

    /* Assistant message - 高对比版本（最终方案） */
    [data-testid="stChatMessage"]:has([aria-label="Assistant"]) {{
        background: var(--accent) !important;
        border: 1px solid var(--accent) !important;
    }}

    [data-testid="stChatMessage"]:has([aria-label="Assistant"]) * {{
        color: #ffffff !important;
        font-weight: 500 !important;
    }}

    /* 兜底：防止 DOM 结构变化导致失效 */
    [data-testid="stChatMessage"][aria-label="assistant"] * {{
        color: #ffffff !important;
    }}

    [data-testid="stChatMessage"] div {{
        color: inherit !important;
    }}
&lt;/style&gt;
"""

st.markdown(style_css, unsafe_allow_html=True)

# ===================== Async client =====================
@st.cache_resource(show_spinner=False)
def get_async_client() -> httpx.AsyncClient:
    """Get async HTTP client for backend API calls."""
    return httpx.AsyncClient(base_url=BASE_URL, timeout=120.0)

client = get_async_client()


def run_async(coro):
    """Run async coroutine safely in Streamlit environment."""
    return asyncio.run(coro)


# ===================== Session state =====================
if "sessions" not in st.session_state:
    st.session_state.sessions = []
if "current_session" not in st.session_state:
    st.session_state.current_session = None
if "messages" not in st.session_state:
    st.session_state.messages = {}
if "search_results" not in st.session_state:
    st.session_state.search_results = []
if "selected_papers" not in st.session_state:
    st.session_state.selected_papers = []
if "sessions_loaded" not in st.session_state:
    st.session_state.sessions_loaded = False
if "confirm_clear" not in st.session_state:
    st.session_state.confirm_clear = False
if "renaming_session" not in st.session_state:
    st.session_state.renaming_session = None
if "show_new_session_dialog" not in st.session_state:
    st.session_state.show_new_session_dialog = False
if "new_session_name_input" not in st.session_state:
    st.session_state.new_session_name_input = ""

MODEL_OPTIONS = {
    "Qwen": "qwen-turbo-2025-04-28",
    "Gemini": "gemini-2.0-flash",
}

# ===================== Backend API functions =====================
async def check_server_connection() -> bool:
    """Check if backend server is connected and healthy."""
    try:
        response = await client.get("/health")
        return response.status_code == 200
    except Exception:
        return False


def create_new_session(custom_name: Optional[str] = None) -> str:
    """Create a new session with optional custom name.
    
    Args:
        custom_name: Optional custom name for the session
        
    Returns:
        Session ID string
    """
    if custom_name and custom_name.strip():
        session_id = custom_name.strip()
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_id = f"session_{timestamp}"

    if session_id not in st.session_state.sessions:
        st.session_state.sessions.append(session_id)
    st.session_state.messages.setdefault(session_id, [])
    return session_id


async def get_session_info(session_id: str) -> Optional[Dict[str, Any]]:
    """Get session information from backend.
    
    Args:
        session_id: ID of the session
        
    Returns:
        Session information dictionary or None
    """
    try:
        response = await client.post("/api/session/info", json={"session_id": session_id})
        return response.json()["data"] if response.status_code == 200 else None
    except Exception:
        return None


async def clear_session_and_remove(session_id: str) -> bool:
    """Clear session and remove it from the list.
    
    Args:
        session_id: ID of the session to clear
        
    Returns:
        True if successful, False otherwise
    """
    try:
        response = await client.post("/api/session/clear", json={"session_id": session_id})
        if response.status_code == 200:
            if session_id in st.session_state.sessions:
                st.session_state.sessions.remove(session_id)
            st.session_state.messages.pop(session_id, None)
            if st.session_state.current_session == session_id:
                st.session_state.current_session = None
            return True
        return False
    except Exception as e:
        st.error(f"清除会话失败: {e}")
        return False


async def search_papers(
    session_id: str,
    querys: List[str],
    max_results: int = 10,
    start_date=None,
    end_date=None,
):
    """Search papers with given queries.
    
    Args:
        session_id: ID of the current session
        querys: List of search keywords
        max_results: Maximum number of results to return
        start_date: Optional start date filter (YYYY-MM-DD)
        end_date: Optional end date filter (YYYY-MM-DD)
        
    Returns:
        List of paper dictionaries
    """
    try:
        payload = {"session_id": session_id, "querys": querys, "max_results": max_results}
        if start_date:
            payload["start_date"] = start_date
        if end_date:
            payload["end_date"] = end_date
        response = await client.post("/api/papers/search", json=payload)
        return response.json()["data"]["papers"] if response.status_code == 200 else []
    except Exception:
        return []


async def download_papers(session_id: str, paper_indices: List[int], search_results: List[Dict]):
    """Download selected papers.
    
    Args:
        session_id: ID of the current session
        paper_indices: List of indices of papers to download
        search_results: Complete list of search results
        
    Returns:
        True if successful, False otherwise
    """
    try:
        response = await client.post(
            "/api/papers/download",
            json={
                "session_id": session_id,
                "paper_indices": paper_indices,
                "search_results": search_results,
            },
        )
        return response.status_code == 200
    except Exception:
        return False


async def build_knowledge_base(session_id: str, clear_existing=False):
    """Build knowledge base from downloaded papers.
    
    Args:
        session_id: ID of the current session
        clear_existing: Whether to clear existing knowledge base
        
    Returns:
        True if successful, False otherwise
    """
    try:
        response = await client.post(
            "/api/knowledge/build",
            json={"session_id": session_id, "clear_existing": clear_existing},
        )
        return response.status_code == 200
    except Exception:
        return False


async def load_knowledge_base(session_id: str):
    """Load existing knowledge base.
    
    Args:
        session_id: ID of the current session
        
    Returns:
        True if successful, False otherwise
    """
    try:
        response = await client.post("/api/knowledge/load", json={"session_id": session_id})
        return response.status_code == 200
    except Exception:
        return False


async def query_knowledge_base(session_id: str, question: str):
    """Query knowledge base with a question.
    
    Args:
        session_id: ID of the current session
        question: Question to ask
        
    Returns:
        Query result dictionary or None
    """
    try:
        response = await client.post("/api/query", json={"session_id": session_id, "question": question})
        return response.json()["data"] if response.status_code == 200 else None
    except Exception:
        return None


async def get_message_history(session_id: str):
    """Get chat message history for a session.
    
    Args:
        session_id: ID of the current session
        
    Returns:
        List of message dictionaries
    """
    try:
        response = await client.get(f"/api/messages/{session_id}")
        return response.json()["data"]["messages"] if response.status_code == 200 else []
    except Exception:
        return []


async def get_session_model(session_id: str) -> Optional[str]:
    """Get current model for a session.
    
    Args:
        session_id: ID of the current session
        
    Returns:
        Model name string or None
    """
    try:
        response = await client.post("/api/session/model", json={"session_id": session_id})
        return response.json()["data"]["model_name"] if response.status_code == 200 else None
    except Exception:
        return None


async def set_session_model(session_id: str, model_name: str) -> bool:
    """Set model for a session.
    
    Args:
        session_id: ID of the current session
        model_name: Name of the model to set
        
    Returns:
        True if successful, False otherwise
    """
    try:
        response = await client.post(
            "/api/session/set_model",
            json={"session_id": session_id, "model_name": model_name},
        )
        return response.status_code == 200 and response.json().get("success")
    except Exception:
        return False


async def get_all_sessions():
    """Get all sessions from backend.
    
    Returns:
        List of session IDs
    """
    try:
        response = await client.get("/api/sessions")
        return response.json()["data"]["sessions"] if response.status_code == 200 else []
    except Exception:
        return []


async def create_session_on_backend(session_id: str) -> bool:
    """Create a session on backend.
    
    Args:
        session_id: ID of the session to create
        
    Returns:
        True if successful, False otherwise
    """
    try:
        response = await client.post("/api/session/create", json={"session_id": session_id})
        return response.status_code == 200
    except Exception:
        return False


async def rename_session(old_session_id: str, new_session_id: str) -> bool:
    """Rename a session.
    
    Args:
        old_session_id: Current session ID
        new_session_id: New session ID
        
    Returns:
        True if successful, False otherwise
    """
    try:
        response = await client.post(
            "/api/session/rename",
            json={"old_session_id": old_session_id, "new_session_id": new_session_id},
        )
        if response.status_code == 200 and response.json().get("success"):
            if old_session_id in st.session_state.sessions:
                idx = st.session_state.sessions.index(old_session_id)
                st.session_state.sessions[idx] = new_session_id
            if old_session_id in st.session_state.messages:
                st.session_state.messages[new_session_id] = st.session_state.messages[old_session_id]
                del st.session_state.messages[old_session_id]
            if st.session_state.current_session == old_session_id:
                st.session_state.current_session = new_session_id
            return True
        return False
    except Exception as e:
        st.error(f"重命名会话失败: {e}")
        return False


# ===================== Bootstrap backend state =====================
server_connected = run_async(check_server_connection())
if server_connected and not st.session_state.sessions_loaded:
    backend_sessions = run_async(get_all_sessions())
    for s in backend_sessions:
        if s not in st.session_state.sessions:
            st.session_state.sessions.append(s)
    st.session_state.sessions_loaded = True

# ===================== Header with theme toggle =====================
with st.container():
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        status_text = "后端连接正常" if server_connected else "后端未连接"
        st.markdown(
            f"""
            <div class="topbar" style="margin-bottom: 0.75rem;">
                <div class="brand">
                    <div class="title">📚 论文知识库问答系统</div>
                    <div class="subtitle">上传论文 → 构建知识库 → 精准问答，一站式科研辅助工具</div>
                </div>
                <div style="display: flex; gap: 0.75rem; align-items: center;">
                    <div class="status-pill">
                        <span class="status-dot" style="background:{'var(--accent)' if server_connected else 'var(--danger)'}"></span>
                        <span>{status_text}</span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        theme_icon = "☀️ 亮色" if st.session_state.theme_mode == "dark" else "🌙 暗色"
        if st.button(theme_icon, key="theme_toggle", use_container_width=True):
            st.session_state.theme_mode = "light" if st.session_state.theme_mode == "dark" else "dark"
            st.rerun()

# ===================== Ensure a current session if available =====================
if st.session_state.current_session is None and st.session_state.sessions:
    st.session_state.current_session = st.session_state.sessions[0]

# ===================== Main 3-panel workspace =====================
with st.container():
    st.markdown('<div class="workspace">', unsafe_allow_html=True)
    col_left, col_mid, col_right = st.columns([1.0, 1.15, 1.35], gap="small")
    panel_height = 740

    # -------- Left panel: sessions --------
    with col_left:
        left_panel = st.container(height=panel_height, border=False)
        with left_panel:
            st.markdown(
                """
                <div class="panel-card">
                    <div class="panel-head">
                        <div class="panel-title">会话管理</div>
                        <div class="panel-desc">一个会话 = 一个独立知识库，用于管理不同研究主题，数据完全隔离</div>
                    </div>
                    <div class="panel-body">
                """,
                unsafe_allow_html=True,
            )

            if st.button("➕ 新建研究会话", key="new_session_btn", use_container_width=True):
                st.session_state.show_new_session_dialog = True

            if st.session_state.show_new_session_dialog:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.write("请输入会话名称（建议使用研究主题命名）")
                st.session_state.new_session_name_input = st.text_input(
                    "会话名称",
                    value=st.session_state.new_session_name_input,
                    placeholder="例如：我的研究主题",
                    label_visibility="visible",
                    key="new_session_name_input_widget",
                )
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("✅ 确定创建", use_container_width=True, key="confirm_new_session"):
                        new_session = create_new_session(st.session_state.new_session_name_input)
                        with st.spinner("创建会话中..."):
                            success = run_async(create_session_on_backend(new_session))
                        if success:
                            st.session_state.current_session = new_session
                            st.session_state.show_new_session_dialog = False
                            st.session_state.new_session_name_input = ""
                            st.session_state.sessions_loaded = False
                            st.rerun()
                        else:
                            st.error("创建会话失败")
                with c2:
                    if st.button("❌ 取消", use_container_width=True, key="cancel_new_session"):
                        st.session_state.show_new_session_dialog = False
                        st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.write("当前会话")
            if st.session_state.current_session:
                st.success(st.session_state.current_session)
            else:
                st.info("尚未选择会话")
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.write("会话列表")
            if st.session_state.sessions:
                for session in st.session_state.sessions:
                    is_current = st.session_state.current_session == session
                    if st.session_state.renaming_session == session:
                        st.markdown(f"**✏️ 重命名：{session}**")
                        new_name = st.text_input(
                            "新名称",
                            value=session,
                            key=f"rename_input_{session}",
                            label_visibility="collapsed",
                        )
                        a, b = st.columns(2)
                        with a:
                            if st.button("✅ 确认", key=f"confirm_rename_{session}", use_container_width=True):
                                if new_name and new_name != session:
                                    success = run_async(rename_session(session, new_name))
                                    if success:
                                        st.session_state.renaming_session = None
                                        st.session_state.sessions_loaded = False
                                        st.rerun()
                                else:
                                    st.session_state.renaming_session = None
                                    st.rerun()
                        with b:
                            if st.button("❌ 取消", key=f"cancel_rename_{session}", use_container_width=True):
                                st.session_state.renaming_session = None
                                st.rerun()
                    else:
                        row = st.columns([6, 1, 1], vertical_alignment="center")
                        with row[0]:
                            btn_label = f"{'💬' if is_current else '🗂️'} {session}"
                            if st.button(
                                btn_label,
                                key=f"session_{session}",
                                use_container_width=True,
                                type="primary" if is_current else "secondary",
                            ):
                                st.session_state.current_session = session
                                if session not in st.session_state.messages:
                                    st.session_state.messages[session] = run_async(get_message_history(session))
                                st.rerun()
                        with row[1]:
                            if st.button("✏️", key=f"rename_{session}"):
                                st.session_state.renaming_session = session
                                st.rerun()
                        with row[2]:
                            if st.button("🗑️", key=f"delete_{session}"):
                                success = run_async(clear_session_and_remove(session))
                                if success:
                                    st.session_state.sessions_loaded = False
                                    st.rerun()
            else:
                st.info("暂无会话，点击上方创建你的第一个研究空间")
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.write("后端状态")
            hint_text = "✅ 后端已连接，所有功能可正常使用" if server_connected else "❌ 当前后端未连接，请先启动服务"
            st.markdown(
                f'<div class="hint">{hint_text}</div>',
                unsafe_allow_html=True,
            )
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('</div></div>', unsafe_allow_html=True)

    # -------- Middle panel: paper search / knowledge base --------
    with col_mid:
        mid_panel = st.container(height=panel_height, border=False)
        with mid_panel:
            st.markdown(
                """
                <div class="panel-card">
                    <div class="panel-head">
                        <div class="panel-title">论文检索与知识库</div>
                        <div class="panel-desc">论文搜索 · 下载 · 知识库构建 · 语义检索</div>
                    </div>
                """,
                unsafe_allow_html=True,
            )

            current_session = st.session_state.current_session
            if not current_session:
                st.markdown(
                    '<div class="card"><div class="hint">请先在左侧创建或选择一个会话，然后开始构建你的论文知识库</div></div>',
                    unsafe_allow_html=True,
                )
            else:
                session_info = run_async(get_session_info(current_session))

                st.markdown(
                """
                    <div class="panel-title" style="margin: 1rem 0;">知识库管理</div>
                """,
                unsafe_allow_html=True,
                )
                if session_info and session_info.get("papers"):
                    st.markdown("**已下载的论文**")
                    for paper in session_info["papers"]:
                        st.markdown(f"- 📄 {paper}")
                else:
                    st.warning("暂未下载论文，请先搜索并下载论文")

                b1, b2 = st.columns(2)
                with b1:
                    if st.button("🔨 构建知识库", use_container_width=True, key=f"build_kb_{current_session}"):
                        with st.spinner("正在解析论文并构建向量索引..."):
                            success = run_async(build_knowledge_base(current_session))
                        if success:
                            st.success("知识库构建完成，正在自动加载...")
                            with st.spinner("🚀 正在加载知识库.."):
                                load_success = run_async(load_knowledge_base(current_session))
                            if load_success:
                                st.success("✅ 知识库加载完成，可以开始问答")
                                st.rerun()
                            else:
                                st.error("❌ 加载失败")
                        else:
                            st.error("❌ 构建失败，请检查论文文件")
                with b2:
                    if st.button("⚡ 加载知识库", use_container_width=True, key=f"load_kb_{current_session}"):
                        with st.spinner("🚀 正在加载知识库.."):
                            success = run_async(load_knowledge_base(current_session))
                        if success:
                            st.success("✅ 知识库加载完成，可以开始问答")
                            st.rerun()
                        else:
                            st.error("❌ 加载失败")

                st.markdown(
                """
                    <div class="panel-title" style="margin: 1rem 0;">论文搜索</div>
                """,
                unsafe_allow_html=True,
                )

                q1, q2 = st.columns([2, 1])
                search_query = q1.text_input(
                    "搜索关键词（请使用英文，多个用逗号分隔）例如: large language model, RAG, diffusion model",
                    value="",
                    key=f"search_query_{current_session}",
                )
                max_results = q2.number_input(
                    "最大结果数",
                    min_value=5,
                    max_value=100,
                    value=10,
                    key=f"max_results_{current_session}",
                )
                q3, q4 = st.columns(2)
                start_date = q3.date_input(
                    "开始时间（可选）",
                    value=None,
                    key=f"start_date_{current_session}",
                )
                end_date = q4.date_input(
                    "结束时间（可选）",
                    value=None,
                    key=f"end_date_{current_session}",
                )

                if st.button("🔍 搜索论文", key=f"search_btn_{current_session}", use_container_width=True):
                    with st.spinner("正在检索论文..."):
                        querys = [q.strip() for q in search_query.split(",") if q.strip()]
                        start_date_str = start_date.strftime("%Y-%m-%d") if start_date else None
                        end_date_str = end_date.strftime("%Y-%m-%d") if end_date else None
                        st.session_state.search_results = run_async(
                            search_papers(
                                current_session,
                                querys,
                                int(max_results),
                                start_date_str,
                                end_date_str,
                            )
                        )
                        st.session_state.selected_papers = []

                if st.session_state.search_results:
                    st.success(f"找到 {len(st.session_state.search_results)} 篇论文")
                    for idx, paper in enumerate(st.session_state.search_results):
                        col_title, col_checkbox = st.columns([4, 1])
                        with col_title:
                            expander = st.expander(f"📄 {paper['title']}", expanded=False)
                        with col_checkbox:
                            is_selected = idx in st.session_state.selected_papers
                            checked = st.checkbox(
                                "",
                                key=f"paper_{current_session}_{idx}",
                                value=is_selected,
                                help="选择此论文"
                            )
                            if checked and idx not in st.session_state.selected_papers:
                                st.session_state.selected_papers.append(idx)
                            if not checked and idx in st.session_state.selected_papers:
                                st.session_state.selected_papers.remove(idx)
                        with expander:
                            st.markdown(f"**作者**：{', '.join(paper.get('authors', [])[:3])}")
                            st.markdown(f"**年份**：{paper.get('published', '')}")
                            summary = paper.get('summary', '')[:250]
                            st.markdown(f"**摘要**：{summary}{'...' if len(paper.get('summary', '')) > 250 else ''}")
                            st.markdown(f"[链接]({paper.get('url', '#')}) | [PDF]({paper.get('pdf_url', '#')})")

                    if st.session_state.selected_papers:
                        if st.button(
                            f"📥 下载选中的 {len(st.session_state.selected_papers)} 篇论文",
                            key=f"download_selected_{current_session}",
                            use_container_width=True,
                        ):
                            with st.spinner("下载中..."):
                                success = run_async(
                                    download_papers(
                                        current_session,
                                        st.session_state.selected_papers,
                                        st.session_state.search_results,
                                    )
                                )
                            if success:
                                st.success("下载完成！")
                                st.rerun()
                            else:
                                st.error("下载失败")
                else:
                    st.info("搜索结果会显示在这里")

                st.markdown("---")
                if st.button("🗑️ 清空当前会话（论文 + 知识库 + 对话）", type="secondary", use_container_width=True, key=f"clear_all_{current_session}"):
                    st.session_state.confirm_clear = True

                if st.session_state.confirm_clear:
                    st.warning("⚠️ 确定要清除当前会话的所有资源吗？此操作不可恢复！")
                    c1, c2 = st.columns(2)
                    with c1:
                        if st.button("✅ 确认清除", type="primary", use_container_width=True, key=f"confirm_clear_{current_session}"):
                            success = run_async(clear_session_and_remove(current_session))
                            if success:
                                st.session_state.confirm_clear = False
                                st.session_state.sessions_loaded = False
                                st.rerun()
                    with c2:
                        if st.button("❌ 取消", use_container_width=True, key=f"cancel_clear_{current_session}"):
                            st.session_state.confirm_clear = False
                            st.rerun()

    # -------- Right panel: chat with bottom-fixed input --------
    with col_right:
        right_panel = st.container(height=panel_height, border=False)
        with right_panel:
            st.markdown(
                """
                <div class="panel-card" style="display: flex; flex-direction: column; height: 100%;">
                    <div class="panel-head" style="flex-shrink: 0;">
                        <div class="panel-title">智能问答</div>
                        <div class="panel-desc">基于论文知识库的精准问答，仅使用已导入论文内容进行对话</div>
                    </div>
                """,
                unsafe_allow_html=True,
            )

            current_session = st.session_state.current_session
            
            if not current_session:
                st.markdown(
                    '<div class="panel-body" style="flex: 1;"><div class="card"><div class="hint">请选择一个会话后开始问答</div></div></div>',
                    unsafe_allow_html=True,
                )
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                session_info = run_async(get_session_info(current_session))
                
                top_fixed = st.container()
                with top_fixed:
                    if session_info:
                        kb_status = "已构建" if session_info["has_knowledge"] else "未构建"
                        paper_count = len(session_info["papers"])
                        message_count = session_info["message_count"]
                        kb_color = "var(--accent)" if kb_status == "已构建" else "var(--warning)"
                        st.markdown(
                            f'''
                            <div style="display: flex; gap: 1rem; margin: 0.5rem 0;">
                                <div style="flex: 1; padding: 0.5rem; color: {kb_color}; font-weight: 600;">
                                    知识库状态: {kb_status}
                                </div>
                                <div style="flex: 1; padding: 0.5rem; color: var(--info); font-weight: 600;">
                                    已导入论文: {paper_count}
                                </div>
                                <div style="flex: 1; padding: 0.5rem; color: var(--text); font-weight: 600;">
                                    历史对话: {message_count}
                                </div>
                            </div>
                            ''',
                            unsafe_allow_html=True,
                        )

                    st.markdown(
                        f'<div class="panel-title" style="margin: 1rem 0;">当前会话(知识库)：{current_session}</div>',
                        unsafe_allow_html=True,
                    )
                    current_model_name = session_info.get("model_name", "") if session_info else ""
                    model_display = next((k for k, v in MODEL_OPTIONS.items() if v == current_model_name), "Qwen")
                    new_model = st.selectbox(
                        "选择当前聊天模型",
                        options=list(MODEL_OPTIONS.keys()),
                        index=list(MODEL_OPTIONS.keys()).index(model_display) if model_display in MODEL_OPTIONS else 0,
                        key=f"model_select_{current_session}",
                    )
                    if new_model != model_display:
                        with st.spinner("切换模型中..."):
                            success = run_async(set_session_model(current_session, MODEL_OPTIONS[new_model]))
                        if success:
                            st.success(f"✅ 模型切换成功, 已切换到 {new_model}")
                            st.rerun()
                        else:
                            st.error("切换模型失败")

                chat_scroll = st.container(height=320, border=False)
                with chat_scroll:
                    if current_session not in st.session_state.messages:
                        st.session_state.messages[current_session] = run_async(get_message_history(current_session))

                    for msg in st.session_state.messages[current_session]:
                        with st.chat_message(msg["role"]):
                            st.markdown(msg["content"])

                bottom_fixed = st.container()
                with bottom_fixed:
                    prompt = st.chat_input("请输入你的问题（系统仅基于论文内容回答）...", key=f"chat_input_{current_session}")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                if prompt:
                    if current_session not in st.session_state.messages:
                        st.session_state.messages[current_session] = []

                    st.session_state.messages[current_session].append({"role": "user", "content": prompt})
                    with st.spinner("思考中..."):
                        result = run_async(query_knowledge_base(current_session, prompt))

                    if result:
                        answer = result.get("answer", "")
                        st.session_state.messages[current_session].append({"role": "assistant", "content": answer})
                        st.session_state.confirm_clear = False
                        st.rerun()
                    else:
                        st.error("❌ 查询失败，请确认知识库已构建")

            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
