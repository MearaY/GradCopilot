"""
M-04：工具执行调度器。

根据 route 和 tool_name 分发到对应工具执行。
每个工具独立实例化，不共享状态。
支持路由类型：rag / tool / llm。
"""
import logging
import re

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
    从自然语言输入中提取 arXiv 搜索结构化参数。

    提取规则：
      - max_results：匹配 "N篇" / "N papers"，默认 10
      - start_date/end_date：匹配 20xx 年份，设为全年范围
      - query：去掉数量、年份、中文指令词后的剩余关键词
    """
    # 数量提取
    max_results = 10
    num_match = re.search(r'(\d+)\s*篇', text)
    if not num_match:
        num_match = re.search(r'(\d+)\s*papers?', text, re.IGNORECASE)
    if num_match:
        max_results = max(1, min(int(num_match.group(1)), 50))

    # 年份提取：匹配 2000-2099
    start_date, end_date = None, None
    year_match = re.search(r'\b(20\d{2})\b', text)
    if year_match:
        year = year_match.group(1)
        start_date = f"{year}-01-01"
        end_date = f"{year}-12-31"

    # 关键词提取：剔除数量、年份、中文指令词
    query = text
    query = re.sub(r'\d+\s*篇', '', query)
    query = re.sub(r'\b20\d{2}\b\s*年?', '', query)
    query = re.sub(
        r'(检索|搜索|查找|找|关于|的|论文|帮我|请|需要|想要|paper|papers?)',
        ' ', query, flags=re.IGNORECASE
    )
    query = ' '.join(query.split()).strip()
    if not query:
        query = text  # 兼底：还原原始输入

    logger.info(
        f"_extract_search_params: query={query!r} max={max_results} "
        f"start={start_date} end={end_date}"
    )
    return {
        "query": query,
        "max_results": max_results,
        "start_date": start_date,
        "end_date": end_date,
    }


def _execute_tool(session_id: str, tool_name: str, cleaned_input: str) -> dict:
    """
    执行普通工具（搜索/下载/构建知识库）。
    工具列表：arxiv_search_tool / paper_download_tool / build_knowledge_tool。
    """
    try:
        if tool_name == "arxiv_search_tool":
            from src.tools.search_tool import run as search_run
            params = _extract_search_params(cleaned_input)
            result = search_run(
                query=params["query"],
                session_id=session_id,
                max_results=params["max_results"],
                start_date=params["start_date"],
                end_date=params["end_date"],
            )
            raw = f"找到 {len(result.get('papers', []))} 篇论文"
            return {
                "tool_result": result,
                "raw_response": raw,
                "sources": [],
                "tool_used": tool_name,
            }

        if tool_name == "paper_download_tool":
            from src.tools.download_tool import run as download_run
            from src.modules.session_memory import set_context
            result = download_run(query=cleaned_input, session_id=session_id)
            downloaded = result.get("downloaded", [])
            failed = result.get("failed", [])
            # 写入情景上下文：记录本次下载的论文 ID，供后续 RAG 限定检索范围
            if downloaded:
                set_context(session_id, {
                    "last_action": "paper_download",
                    "downloaded_papers": downloaded,
                })
            raw = f"下载成功 {len(downloaded)} 篇，失败 {len(failed)} 篇"
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
