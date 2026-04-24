"""
M-05：结果生成与优化模块。

根据 route 选择 prompt 模板，调用 LLM 生成最终回答，并写入会话历史。
支持三种路由：rag（基于检索片段）/ tool（工具执行结果）/ llm（纯对话）。
"""
import json
import logging
import os

from src.modules.llm_client import call_llm
from src.modules.session_memory import append_history, get_history, get_context
from src.prompts.response_templates import (
    RAG_RESPONSE_PROMPT,
    TOOL_RESPONSE_PROMPT,
    LLM_RESPONSE_PROMPT,
)

logger = logging.getLogger(__name__)

# Fallback 文本，当 LLM 调用失败时返回
_FALLBACK_RESPONSE = "抱歉，系统暂时无法处理您的请求，请稍后重试"

# 传入 LLM 的最近对话轮数
_HISTORY_LIMIT = 5

# RAG 上下文最大片段数
_MAX_RAG_CONTEXTS = 5


def _format_history(history: list[dict]) -> str:
    """将历史列表格式化为可读字符串。"""
    if not history:
        return "（无历史）"
    lines = []
    for msg in history:
        role_label = "用户" if msg.get("role") == "user" else "助手"
        lines.append(f"{role_label}：{msg.get('content', '')}")
    return "\n".join(lines)


def _format_rag_contexts(tool_result: dict | None) -> str:
    """将 RAG 检索结果格式化为 prompt 中的上下文字符串。"""
    if not tool_result:
        return "（无检索内容）"
    contexts = tool_result.get("contexts", [])[:_MAX_RAG_CONTEXTS]
    if not contexts:
        return "（无检索内容）"
    parts = []
    for i, ctx in enumerate(contexts, 1):
        paper_id = ctx.get("paper_id", "unknown")
        page = ctx.get("page_number", "?")
        content = ctx.get("content", "")
        parts.append(f"[{i}] 论文 {paper_id}（第 {page} 页）：\n{content}")
    return "\n\n".join(parts)


def _format_context(context: dict) -> str:
    """将情景上下文格式化为 prompt 可读字符串。

    context 来自会话情景记忆，为回答生成提供操作上下文。
    """
    if not context:
        return "（无记录）"
    action = context.get("last_action")
    if action == "paper_download":
        papers = context.get("downloaded_papers", [])
        return f"最近操作：已下载论文 {papers}"
    if action == "build_knowledge":
        chunks = context.get("chunks_indexed", 0)
        return f"最近操作：已构建知识库，索引 {chunks} 个片段"
    return "（无记录）"


def generate(
    session_id: str,
    cleaned_input: str,
    route: str,
    tool_result: dict | None,
    raw_response: str,
    sources: list[str],
    tool_used: str | None,
) -> dict:
    """
    根据路由类型选择模板，调用 LLM 生成最终回答，写入会话历史。

    Args:
        session_id:    会话 ID
        cleaned_input: 用户输入
        route:         路由类型："rag" / "tool" / "llm"
        tool_result:   工具执行结果（dict 或 None）
        raw_response:  工具执行的文本摘要（tool 路径）
        sources:       来源引用列表（RAG 路径）
        tool_used:     实际调用的工具名

    Returns:
        dict: {
            "response": str,
            "sources": list[str],
            "model_used": str,
            "tokens_used": int
        }

    Refs:
        返回字段：response=生成回答，sources=来源引用，model_used=模型名，tokens_used=Token 消耗。
    """
    # 步骤1：读取最近历史，为生成提供对话上下文
    history = get_history(session_id, limit=_HISTORY_LIMIT)
    history_str = _format_history(history)

    # 读取情景上下文，为回答提供最近操作记录
    ctx = get_context(session_id)
    context_str = _format_context(ctx)

    # 步骤2：RAG 返回空结果时跳过生成，直接返回无内容提示
    if route == "rag" and raw_response == "未找到相关内容":
        _write_history(session_id, cleaned_input, raw_response)
        return {
            "response": raw_response,
            "sources": [],
            "model_used": os.environ.get("MODEL_NAME", "gpt-5-nano"),
            "tokens_used": 0,
        }

    # 步骤3：按 route 选择对应模板，注入上下文和检索结果
    if route == "rag":
        contexts_str = _format_rag_contexts(tool_result)
        prompt_text = RAG_RESPONSE_PROMPT.format(
            context=context_str,
            contexts=contexts_str,
            history=history_str,
            user_input=cleaned_input,
        )
    elif route == "tool":
        tool_result_str = (
            raw_response
            or json.dumps(tool_result, ensure_ascii=False)
            if tool_result
            else raw_response
        )
        prompt_text = TOOL_RESPONSE_PROMPT.format(
            tool_result=tool_result_str,
            user_input=cleaned_input,
        )
    else:  # route == "llm"
        prompt_text = LLM_RESPONSE_PROMPT.format(
            history=history_str,
            user_input=cleaned_input,
        )

    messages = [{"role": "user", "content": prompt_text}]

    # 步骤4：调用 LLM 生成最终回答
    try:
        llm_result = call_llm(
            messages=messages,
            temperature=0.7,
            session_id=session_id,
            event="response_generate",
        )
        response_text = llm_result.get("content", "").strip()
        model_used = llm_result.get("model", os.environ.get("MODEL_NAME", "gpt-5-nano"))
        tokens_used = llm_result.get("usage", {}).get("total_tokens", 0)

        if not response_text:
            response_text = _FALLBACK_RESPONSE

    except (TimeoutError, RuntimeError) as e:
        logger.error(f"ResponseGenerator: LLM 失败 session={session_id} error={e}")
        response_text = _FALLBACK_RESPONSE
        model_used = os.environ.get("MODEL_NAME", "gpt-5-nano")
        tokens_used = 0

    # 步骤5：将本轮对话写入会话历史
    _write_history(session_id, cleaned_input, response_text)

    return {
        "response": response_text,
        "sources": sources,
        "model_used": model_used,
        "tokens_used": tokens_used,
    }


def _write_history(session_id: str, user_input: str, assistant_response: str) -> None:
    """将本轮对话写入 Redis 会话历史。"""
    append_history(session_id, "user", user_input)
    append_history(session_id, "assistant", assistant_response)
