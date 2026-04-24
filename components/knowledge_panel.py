"""
components/knowledge_panel.py — 知识库管理面板。
"""
from pathlib import Path

import streamlit as st
from components import api_client
from components.api_client import APIError

_PAPERS_ROOT = Path("papers")


def render_knowledge():
    """
    渲染知识库管理面板：论文列表 + 构建按钮 + 构建结果。
    """
    session_id = st.session_state.get("current_session_id")

    if not session_id:
        st.info("👈 请先在左侧选择一个会话。")
        return

    st.subheader("📦 知识库管理")

    # ── 已下载论文列表 ────────────────────────────────────────
    paper_dir = _PAPERS_ROOT / session_id
    if paper_dir.exists():
        pdf_files = list(paper_dir.glob("*.pdf"))
    else:
        pdf_files = []

    if pdf_files:
        st.markdown(f"**已下载论文（{len(pdf_files)} 篇）**")
        for f in pdf_files:
            size_kb = f.stat().st_size / 1024
            st.markdown(f"- `{f.name}` ({size_kb:.1f} KB)")
    else:
        st.caption("当前会话下暂无已下载的论文。请先在「论文搜索」页面搜索并下载论文。")

    st.markdown("---")

    # ── 构建按钮 ─────────────────────────────────────────────
    build_disabled = len(pdf_files) == 0

    help_text = (
        "将已下载的 PDF 解析、切片、生成向量索引写入知识库。"
        if not build_disabled else
        "请先下载论文后再构建知识库。"
    )

    if st.button(
        "🔨 构建向量知识库",
        disabled=build_disabled,
        help=help_text,
        use_container_width=True,
        type="primary",
    ):
        with st.spinner("解析 PDF、生成向量中，请稍候…"):
            try:
                result = api_client.build_knowledge(session_id)
                status = result.get("status", "unknown")
                chunks = result.get("chunks_indexed", 0)

                if status == "success":
                    st.success(f"✅ 知识库构建成功！写入 {chunks} 个 chunks。")
                elif status == "partial":
                    st.warning(
                        f"⚠️ 部分成功：写入 {chunks} 个 chunks，"
                        f"部分文件解析失败（请查看后端日志）。"
                    )
                elif status == "no_papers":
                    st.info("当前会话下没有可用的 PDF，请先下载论文。")
                else:
                    st.error(f"未知状态：{status}")

            except APIError as e:
                st.error(f"❌ {e.code}：{e.message}")
            except Exception as e:
                st.warning(f"请求失败：{e}")

    # ── 状态提示 ─────────────────────────────────────────────
    st.markdown("---")
    st.caption(
        "💡 构建知识库后，在「聊天」面板中提问时系统将自动从知识库检索相关论文片段作为上下文。"
    )
