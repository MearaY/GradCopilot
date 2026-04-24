"""
components/chat_panel.py — 主聊天面板。
"""
import streamlit as st
from components import api_client
from components.api_client import APIError


def render_chat():
    """
    渲染聊天面板：历史消息 + 输入框 + 发送。
    """
    session_id = st.session_state.get("current_session_id")

    if not session_id:
        st.info("👈 请先在左侧**创建或选择一个会话**后开始聊天。")
        return

    # ── 加载历史（首次进入或切换会话） ──────────────────────────
    if not st.session_state.get("chat_history"):
        try:
            data = api_client.get_history(session_id, limit=20)
            st.session_state.chat_history = data.get("history", [])
        except Exception:
            st.session_state.chat_history = []

    # ── 渲染历史消息 ──────────────────────────────────────────
    for msg in st.session_state.chat_history:
        role = msg.get("role", "assistant")
        content = msg.get("content", "")
        with st.chat_message(role):
            st.markdown(content)

    # ── 输入框 ────────────────────────────────────────────────
    user_input = st.chat_input(
        "输入你的问题…",
        disabled=st.session_state.get("is_sending", False),
    )

    if not user_input:
        return

    # ── 发送消息 ──────────────────────────────────────────────
    st.session_state.is_sending = True

    # 立即显示用户消息
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("思考中…"):
            try:
                result = api_client.chat(session_id, user_input)
                response_text = result.get("response", "（无回答）")
                sources = result.get("sources", [])

                st.markdown(response_text)

                # 元信息折叠显示 [REF: frontend.md#3.2]
                with st.expander("📊 调用详情", expanded=False):
                    col1, col2, col3 = st.columns(3)
                    col1.metric("意图", result.get("intent", "-"))
                    col2.metric("路由", result.get("route", "-"))
                    col3.metric("模型", result.get("model_used", "-"))
                    if sources:
                        st.markdown("**来源**")
                        for s in sources:
                            st.caption(s)
                    tokens = result.get("tokens_used", 0)
                    st.caption(f"Token 消耗：{tokens}")

                # 追加到本地缓存
                st.session_state.chat_history.append(
                    {"role": "user", "content": user_input}
                )
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": response_text}
                )

            except APIError as e:
                st.error(f"❌ {e.code}：{e.message}")
            except Exception as e:
                st.warning(f"⚠️ 请求失败：{e}")

    st.session_state.is_sending = False
