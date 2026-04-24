"""
M-03：RAG 路由决策模块。

根据意图枚举，通过固定路由表映射到 route 和 tool_name。
路由表禁止修改，返回剖本防止外部修改路由表。
"""
import logging

logger = logging.getLogger(__name__)

# 固定路由表，禁止修改，返回副本防止外部窟改
ROUTE_MAP: dict[str, dict] = {
    "rag_query":       {"route": "rag",  "tool_name": "rag_retrieval_tool"},
    "paper_search":    {"route": "tool", "tool_name": "arxiv_search_tool"},
    "paper_download":  {"route": "tool", "tool_name": "paper_download_tool"},
    "build_knowledge": {"route": "tool", "tool_name": "build_knowledge_tool"},
    "general_chat":    {"route": "llm",  "tool_name": None},
}


def route(intent: str, confidence: float) -> dict:
    """
    根据意图返回路由决策。

    Args:
        intent:     意图枚举字符串
        confidence: 置信度（0.0-1.0），仅用于日志，路由决策不依赖置信度

    Returns:
        dict: {
            "route": str,            # "rag" / "tool" / "llm"
            "tool_name": str | None  # 工具名，route=llm 时为 None
        }

    Refs:
        返回字段：route="rag"/"tool"/"llm"，tool_name=工具名或 None。
        返回副本，修改返回値不影响 ROUTE_MAP。
    """
    result = ROUTE_MAP.get(intent)

    if result is None:
        # intent 不在枚举范围：降级到 llm 处理
        logger.warning(
            f"RAGRouter: 未知 intent='{intent}'，降级到 general_chat。"
            f" confidence={confidence}"
        )
        result = ROUTE_MAP["general_chat"]
    else:
        logger.info(
            f"RAGRouter: intent={intent} → route={result['route']}"
            f" tool={result['tool_name']} confidence={confidence}"
        )

    return dict(result)  # 返回副本，防止外部修改路由表
