"""
M-04：工具执行调度器。

根据 route 和 tool_name 分发到对应工具执行。
每个工具独立实例化，不共享状态。
支持路由类型：rag / tool / llm。
"""
import logging
import re
import json

from src.modules.llm_client import call_llm

logger = logging.getLogger(__name__)

# 工具超时（秒），超过此时间则降级返回错误提示
_TOOL_TIMEOUT = 30


def execute(
    session_id: str,
    route: str,
    tool_name: str | None,
    cleaned_input: str,
) -> dict:
    """
    根据路由决策执行对应工具或直接进入 LLM 路径。

    Args:
        session_id:    会话 ID
        route:         路由类型："rag" / "tool" / "llm"
        tool_name:     工具名称（route=llm 时为 None）
        cleaned_input: 预处理后的用户输入

    Returns:
        dict: {
            "tool_result": dict | None,
            "raw_response": str,
            "sources": list[str],
            "tool_used": str | None
        }

    Refs:
        返回字段说明：tool_result=工具执行结果字典或 None，
        raw_response=工具文本摘要，sources=来源引用列表，tool_used=工具名。
    """
    logger.info(
        f"AgentExecutor: route={route} tool={tool_name} session={session_id}"
    )

    if route == "llm":
        # 通用对话路径，不调用工具，直接进入 LLM 生成阶段
        return {
            "tool_result": None,
            "raw_response": "",
            "sources": [],
            "tool_used": None,
        }

    if route == "rag":
        return _execute_rag(session_id, cleaned_input)

    if route == "tool":
        return _execute_tool(session_id, tool_name, cleaned_input)

    # 未知 route，安全降级
    logger.warning(f"AgentExecutor: 未知 route='{route}'，按 llm 处理")
    return {
        "tool_result": None,
        "raw_response": "",
        "sources": [],
        "tool_used": None,
    }


_SUMMARY_RE = re.compile(
    r"(这篇|本篇|这个|本文|this\s+paper|the\s+paper).*(什么|讲|介绍|讨论|关于|主要|内容|贡献|总结|概括)"
    r"|(摘要|abstract|introduction|overview|main\s*contribution|summary)",
    re.IGNORECASE,
)


def _is_summary_query(text: str) -> bool:
    """判断是否为论文摘要/概述类查询，用于 query 增强。"""
    return bool(_SUMMARY_RE.search(text))


def _execute_rag(session_id: str, cleaned_input: str) -> dict:
    """
    执行 RAG 检索路径。从情景上下文读取当前操作论文，限定检索范围。
    """
    from src.tools.rag_tool import run as rag_run
    from src.modules.session_memory import get_context

    # 从情景上下文读取最近操作的论文 ID，限定 RAG 检索范围
    ctx = get_context(session_id)
    paper_ids = ctx.get("downloaded_papers") or None
    if paper_ids:
        logger.info(f"AgentExecutor: RAG 限定论文范围 paper_ids={paper_ids} session={session_id}")

    # 摘要类查询：增强 query 引导命中摘要/引言章节，扩大 top_k
    if paper_ids and _is_summary_query(cleaned_input):
        query_for_rag = "abstract introduction main contribution methodology overview " + cleaned_input
        top_k_for_rag = 8
        logger.info(f"AgentExecutor: 摘要类查询，query 增强 top_k=8 session={session_id}")
    else:
        query_for_rag = cleaned_input
        top_k_for_rag = 5

    try:
        result = rag_run(query=query_for_rag, session_id=session_id, top_k=top_k_for_rag, paper_ids=paper_ids)
        contexts = result.get("contexts", [])

        if not contexts:
            # RAG 返回空结果时，不继续生成，直接返回无内容提示
            logger.info(f"AgentExecutor: RAG 返回空结果 session={session_id}")
            return {
                "tool_result": result,
                "raw_response": "未找到相关内容",
                "sources": [],
                "tool_used": "rag_retrieval_tool",
            }

        sources = result.get("sources", [])
        return {
            "tool_result": result,
            "raw_response": "",  # 由 ResponseGenerator 根据 contexts 生成
            "sources": sources,
            "tool_used": "rag_retrieval_tool",
        }
    except Exception as e:
        logger.error(f"AgentExecutor: RAG 执行失败 session={session_id} error={e}")
        return {
            "tool_result": None,
            "raw_response": f"知识库检索失败：{e}",
            "sources": [],
            "tool_used": "rag_retrieval_tool",
        }



def _extract_search_params(text: str) -> dict:
    """
    使用 LLM 从自然语言输入中智能提取 arXiv 搜索结构化参数。
    """
    import json
    from src.modules.llm_client import call_llm

    prompt = f"""
请从以下用户输入中提取 arXiv 论文搜索的参数，并以纯 JSON 格式返回。

需要提取的字段：
- max_results: 整数，提取用户要求的论文数量（例如"5篇"提取为 5），如果没有指定则默认为 10。最大不超过 50。
- start_date: 字符串，起始日期（如 "YYYY-MM-DD"）。如果用户只指定了年份（例如"2026年"），请将其转换为该年的 1 月 1 日，即 "2026-01-01"。如果没有指定则为 null。
- end_date: 字符串，结束日期（如 "YYYY-MM-DD"）。如果用户只指定了年份（例如"2026年"），请将其转换为该年的 12 月 31 日，即 "2026-12-31"。如果没有指定则为 null。
- query: 字符串，去除了数量、年份、指示性动词（如"搜索"、"查找"、"帮我找"、"论文"等）后的核心检索词。例如输入"检索2026年2篇agent论文"，query 应为 "agent"。

用户输入：
"{text}"

只返回一段有效的 JSON，不要输出任何额外的说明文字或 Markdown 标记（不要用 ```json 包裹）。例如：
{{
  "max_results": 2,
  "start_date": "2026-01-01",
  "end_date": "2026-12-31",
  "query": "agent"
}}
"""
    try:
        resp = call_llm(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            event="extract_search_params"
        )
        content = resp["content"].strip()
        
        # 移除 Markdown 代码块标记（防御性处理）
        if content.startswith("```"):
            lines = content.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            content = "\n".join(lines).strip()
            
        params = json.loads(content)
        
        max_results = max(1, min(int(params.get("max_results") or 10), 50))
        start_date = params.get("start_date")
        end_date = params.get("end_date")
        query = (params.get("query") or text).strip()
        if not query:
            query = text

        logger.info(
            f"_extract_search_params (LLM): query={query!r} max={max_results} "
            f"start={start_date} end={end_date}"
        )
        return {
            "query": query,
            "max_results": max_results,
            "start_date": start_date,
            "end_date": end_date,
        }
    except Exception as e:
        logger.error(f"_extract_search_params LLM 提取失败: {e}，回退到原始文本")
        return {
            "query": text,
            "max_results": 10,
            "start_date": None,
            "end_date": None,
        }

def _extract_download_params(text: str, history: list[dict]) -> str:
    """
    提取用户想要下载的 arXiv 论文 ID。
    """
    prompt = f"""
你是一个参数提取器。你需要从用户的输入以及对话历史中，提取出用户想要下载的 arXiv 论文的 ID（paper_id）。
如果有多个 ID，请用逗号分隔。不要输出任何额外的说明或标记，只需返回 ID 即可。

对话历史概要：
{json.dumps(history[-3:] if history else [], ensure_ascii=False)}

用户当前输入：
"{text}"

只返回论文 ID，例如：2601.17768, 2604.03350
"""
    try:
        resp = call_llm(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            event="extract_download_params"
        )
        return resp["content"].strip()
    except Exception as e:
        logger.error(f"_extract_download_params LLM 提取失败: {e}，回退到原始文本")
        return text



def _execute_tool(session_id: str, tool_name: str, cleaned_input: str) -> dict:
    """
    执行普通工具（搜索/下载/构建知识库）。
    工具列表：arxiv_search_tool / paper_download_tool / build_knowledge_tool。
    """
    try:
        if tool_name == "arxiv_search_tool":
            params = _extract_search_params(cleaned_input)
            try:
                from src.tools.mcp_arxiv_tool import search as mcp_search
                result = mcp_search(
                    query=params["query"],
                    session_id=session_id,
                    max_results=params["max_results"],
                    start_date=params["start_date"],
                    end_date=params["end_date"],
                )
                logger.info(f"AgentExecutor: MCP arxiv search 成功 session={session_id}")
            except Exception as mcp_err:
                logger.warning(
                    f"AgentExecutor: MCP search 失败，fallback 到旧工具 "
                    f"session={session_id} error={mcp_err}"
                )
                from src.tools.search_tool import run as search_run
                result = search_run(
                    query=params["query"],
                    session_id=session_id,
                    max_results=params["max_results"],
                    start_date=params["start_date"],
                    end_date=params["end_date"],
                )
            papers = result.get('papers', [])
            raw = f"找到 {len(papers)} 篇论文"
            if papers:
                details = [f"- {p.get('title', '未知')} (ID: {p.get('paper_id', '未知')})" for p in papers[:10]]
                raw += "：\n" + "\n".join(details)
            return {
                "tool_result": result,
                "raw_response": raw,
                "sources": [],
                "tool_used": tool_name,
            }

        if tool_name == "paper_download_tool":
            from src.modules.session_memory import set_context, get_history, get_context
            history = get_history(session_id)
            query = _extract_download_params(cleaned_input, history)
            try:
                from src.tools.mcp_arxiv_tool import download as mcp_download
                result = mcp_download(query=query, session_id=session_id)
                logger.info(f"AgentExecutor: MCP arxiv download 成功 session={session_id}")
            except Exception as mcp_err:
                logger.warning(
                    f"AgentExecutor: MCP download 失败，fallback 到旧工具 "
                    f"session={session_id} error={mcp_err}"
                )
                from src.tools.download_tool import run as download_run
                result = download_run(query=query, session_id=session_id)
            downloaded = result.get("downloaded", [])
            failed = result.get("failed", [])

            # 写入情景上下文：记录本次下载的论文 ID，供后续 RAG 限定检索范围
            if downloaded:
                set_context(session_id, {
                    "last_action": "paper_download",
                    "downloaded_papers": downloaded,
                })

            # 构建下载阶段摘要
            raw = f"📥 论文下载完成：成功 {len(downloaded)} 篇，失败 {len(failed)} 篇。"
            if downloaded:
                raw += f"\n  ✅ 已下载：{', '.join(downloaded)}"
            if failed:
                raw += f"\n  ❌ 下载失败：{', '.join(failed)}"

            # 下载成功后自动构建/更新向量知识库
            if downloaded:
                logger.info(
                    f"AgentExecutor: 下载完成，自动触发知识库构建 session={session_id}"
                )
                raw += "\n\n🔨 正在自动构建向量知识库，请稍候..."
                try:
                    from src.tools.build_knowledge_tool import run as build_run
                    build_result = build_run(session_id=session_id)
                    chunks = build_result.get("chunks_indexed", 0)
                    # 保留 downloaded_papers 上下文，更新 last_action
                    prev_ctx = get_context(session_id)
                    new_ctx = {
                        "last_action": "build_knowledge",
                        "chunks_indexed": chunks,
                    }
                    if prev_ctx.get("downloaded_papers"):
                        new_ctx["downloaded_papers"] = prev_ctx["downloaded_papers"]
                    set_context(session_id, new_ctx)
                    raw += f"\n✅ 知识库构建完成，已索引 {chunks} 个片段。现在可以直接提问了！"
                    logger.info(
                        f"AgentExecutor: 知识库自动构建完成 chunks={chunks} session={session_id}"
                    )
                except Exception as build_err:
                    logger.error(
                        f"AgentExecutor: 知识库自动构建失败 session={session_id} error={build_err}"
                    )
                    raw += f"\n⚠️ 知识库构建失败：{build_err}。请手动执行 /build 命令。"

            return {
                "tool_result": result,
                "raw_response": raw,
                "sources": [],
                "tool_used": tool_name,
            }

        if tool_name == "build_knowledge_tool":
            from src.tools.build_knowledge_tool import run as build_run
            from src.modules.session_memory import set_context, get_context
            # 构建前读取已有上下文，保留 downloaded_papers 不被覆盖
            prev_ctx = get_context(session_id)
            result = build_run(session_id=session_id)
            chunks = result.get('chunks_indexed', 0)
            new_ctx = {
                "last_action": "build_knowledge",
                "chunks_indexed": chunks,
            }
            # 保留上一步下载的论文 ID，供后续 'rag_query' 限定检索范围
            if prev_ctx.get("downloaded_papers"):
                new_ctx["downloaded_papers"] = prev_ctx["downloaded_papers"]
            set_context(session_id, new_ctx)
            raw = f"知识库构建完成，索引 {chunks} 个片段"
            return {
                "tool_result": result,
                "raw_response": raw,
                "sources": [],
                "tool_used": tool_name,
            }

        logger.warning(f"AgentExecutor: 未知 tool_name='{tool_name}'")
        return {
            "tool_result": None,
            "raw_response": f"未找到工具：{tool_name}",
            "sources": [],
            "tool_used": tool_name,
        }

    except TimeoutError:
        logger.error(f"AgentExecutor: 工具超时 tool={tool_name} session={session_id}")
        return {
            "tool_result": None,
            "raw_response": "操作超时，请稍后重试",
            "sources": [],
            "tool_used": tool_name,
        }
    except Exception as e:
        logger.error(
            f"AgentExecutor: 工具执行失败 tool={tool_name} session={session_id} error={e}"
        )
        return {
            "tool_result": None,
            "raw_response": f"工具执行失败：{e}",
            "sources": [],
            "tool_used": tool_name,
        }
