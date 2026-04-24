"""
components/search_panel.py — 论文搜索面板。
"""
import streamlit as st
from components import api_client
from components.api_client import APIError


def render_search():
    """
    渲染论文搜索面板：搜索表单 + 结果列表 + 下载选中。
    """
    session_id = st.session_state.get("current_session_id")

    if not session_id:
        st.info("👈 请先在左侧选择一个会话。")
        return

    st.subheader("🔍 arXiv 论文搜索")

    # ── 搜索表单 ──────────────────────────────────────────────
    with st.form("search_form"):
        query = st.text_input(
            "搜索关键词",
            placeholder="例如：transformer attention mechanism",
        )

        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            max_results = st.slider(
                "最大结果数",
                min_value=1,
                max_value=50,
                value=10,
                step=1,
            )
        with col2:
            start_date = st.date_input("开始日期", value=None, format="YYYY-MM-DD")
        with col3:
            end_date = st.date_input("结束日期", value=None, format="YYYY-MM-DD")

        submitted = st.form_submit_button("🔍 搜索", use_container_width=True)

    if submitted:
        if not query.strip():
            st.warning("请输入搜索关键词。")
            return

        start_str = start_date.strftime("%Y-%m-%d") if start_date else None
        end_str = end_date.strftime("%Y-%m-%d") if end_date else None

        with st.spinner("搜索中…"):
            try:
                result = api_client.search_papers(
                    session_id=session_id,
                    query=query.strip(),
                    max_results=max_results,
                    start_date=start_str,
                    end_date=end_str,
                )
                papers = result.get("papers", [])
                st.session_state.search_results = papers
                st.session_state.selected_paper_ids = []
                st.success(f"找到 {result.get('total', len(papers))} 篇论文")
            except APIError as e:
                st.error(f"搜索失败：{e.message}")
                return
            except Exception as e:
                st.warning(f"请求失败：{e}")
                return

    # ── 搜索结果 ──────────────────────────────────────────────
    papers = st.session_state.get("search_results", [])
    selected_ids = st.session_state.get("selected_paper_ids", [])

    if not papers:
        return

    st.markdown(f"**共 {len(papers)} 篇结果**（勾选后点击「下载选中」）")

    new_selected = []
    for i, p in enumerate(papers):
        paper_id = p.get("paper_id", "")
        title = p.get("title", "（无标题）")
        authors = p.get("authors", [])
        author_str = "、".join(authors[:3]) + ("…" if len(authors) > 3 else "")
        published = p.get("published_date", "")
        abstract = p.get("abstract") or "（无摘要）"
        pdf_url = p.get("pdf_url", "")
        arxiv_url = p.get("arxiv_url", "")

        # 勾选框
        checked = st.checkbox(
            f"**{title}**",
            value=(paper_id in selected_ids),
            key=f"paper_check_{i}_{paper_id}",
        )
        if checked and paper_id:
            new_selected.append(paper_id)

        with st.expander(f"📄 {author_str} · {published}", expanded=False):
            st.markdown(f"**摘要**\n\n{abstract[:500]}{'…' if len(abstract) > 500 else ''}")
            col_a, col_b = st.columns(2)
            if arxiv_url:
                col_a.markdown(f"[🔗 arXiv 主页]({arxiv_url})")
            if pdf_url:
                col_b.markdown(f"[📥 PDF 下载]({pdf_url})")

    st.session_state.selected_paper_ids = new_selected

    # ── 下载按钮 ──────────────────────────────────────────────
    st.markdown("---")
    col_dl, col_info = st.columns([2, 3])

    with col_dl:
        dl_disabled = len(new_selected) == 0
        if st.button(
            f"📥 下载选中（{len(new_selected)} 篇）",
            disabled=dl_disabled,
            use_container_width=True,
            type="primary",
        ):
            with st.spinner(f"下载 {len(new_selected)} 篇论文…"):
                try:
                    result = api_client.download_papers(
                        session_id=session_id,
                        paper_ids=new_selected,
                    )
                    downloaded = result.get("downloaded", [])
                    failed = result.get("failed", [])

                    if downloaded:
                        st.success(f"✅ 下载成功 {len(downloaded)} 篇")
                    if failed:
                        st.warning(f"⚠️ 下载失败 {len(failed)} 篇：{', '.join(failed)}")

                except APIError as e:
                    st.error(f"下载失败：{e.message}")
                except Exception as e:
                    st.warning(f"请求失败：{e}")

    with col_info:
        if new_selected:
            st.caption(f"已选：{', '.join(new_selected[:5])}" +
                       ("…" if len(new_selected) > 5 else ""))
