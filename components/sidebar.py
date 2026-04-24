"""
components/sidebar.py — 左侧边栏（会话管理）。
"""
import requests
import streamlit as st
from components import api_client
from components.api_client import APIError


# ── 模块级缓存（TTL 5s），避免每次 Streamlit rerun 都请求后端 ──
@st.cache_data(ttl=5, show_spinner=False)
def _fetch_sessions() -> list[dict]:
    """
    GET /api/sessions — 带 5s 缓存，防止 Streamlit 频繁轮询阻塞。
    """
    data = api_client.get_sessions()
    return data.get("sessions", [])


def render_sidebar():
    """
    渲染左侧边栏：会话列表 + 新建 + 删除。
    """
    st.markdown("## 📚 GradCopilot")
    st.markdown("---")

    # ── 新建会话 ─────────────────────────────────────────────
    with st.expander("➕ 新建会话", expanded=False):
        new_name = st.text_input(
            "会话名称",
            value="",
            placeholder="输入会话名称（可留空）",
            key="sidebar_new_session_name",
        )
        if st.button("创建", key="sidebar_create_btn", use_container_width=True):
            try:
                result = api_client.create_session(name=new_name.strip() or "新会话")
                st.session_state.current_session_id = result["session_id"]
                st.session_state.chat_history = []
                st.session_state.search_results = []
                _fetch_sessions.clear()   # 清除缓存，使列表立即刷新
                st.success(f"✅ 已创建：{result['name']}")
                st.rerun()
            except APIError as e:
                st.error(f"创建失败：{e.message}")
            except requests.exceptions.Timeout:
                st.warning("⏳ 后端响应超时，请稍后重试。")
            except Exception as e:
                st.warning(f"网络错误：{e}")

    st.markdown("---")
    st.markdown("**会话列表**")

    # ── 会话列表（模块级缓存 5s）────────────────────────────────
    try:
        sessions = _fetch_sessions()
    except APIError as e:
        st.error(f"加载失败：{e.message}")
        sessions = []
    except requests.exceptions.Timeout:
        st.warning("⏳ 后端连接超时，请确认服务已启动并刷新页面。")
        sessions = []
    except requests.exceptions.ConnectionError:
        st.warning("⚠️ 无法连接后端（localhost:8000），请确认 uvicorn 已运行。")
        sessions = []
    except Exception as e:
        st.warning(f"连接异常：{e}")
        sessions = []

    if not sessions:
        st.caption("暂无会话，请先新建一个。")
        return

    current_id = st.session_state.get("current_session_id")

    for s in sessions:
        sid = s["session_id"]
        name = s.get("name") or sid
        count = s.get("message_count", 0)

        col_btn, col_del = st.columns([5, 1])

        is_active = sid == current_id
        label = f"{'▶ ' if is_active else ''}{name}"
        help_text = f"ID: {sid} · {count} 条消息"

        with col_btn:
            if st.button(
                label,
                key=f"sess_btn_{sid}",
                help=help_text,
                use_container_width=True,
                type="primary" if is_active else "secondary",
            ):
                st.session_state.current_session_id = sid
                st.session_state.chat_history = []
                st.rerun()

        with col_del:
            if st.button("🗑", key=f"sess_del_{sid}", help="删除会话"):
                try:
                    api_client.delete_session(sid)
                    if st.session_state.get("current_session_id") == sid:
                        st.session_state.current_session_id = None
                        st.session_state.chat_history = []
                    _fetch_sessions.clear()   # 清除缓存
                    st.success("已删除")
                    st.rerun()
                except APIError as e:
                    st.error(f"删除失败：{e.message}")
                except Exception as e:
                    st.warning(f"网络错误：{e}")
