"""
M-02：用户意图分类模块。

使用 LLM 进行零样本意图分类，置信度 < 0.7 时强制降级到 general_chat。
分类结果包含 intent、confidence、reasoning 三个字段。
"""
import json
import logging
import re

from src.modules.llm_client import call_llm
from src.modules.session_memory import get_history, get_context
from src.prompts.intent_classify import INTENT_CLASSIFY_PROMPT, VALID_INTENTS

logger = logging.getLogger(__name__)

# 置信度阈值，低于此值强制降级到 general_chat
CONFIDENCE_THRESHOLD = 0.7

# LLM 调用失败时的 fallback 返回值
_FALLBACK_RESULT = {
    "intent": "general_chat",
    "confidence": 0.0,
    "reasoning": "LLM 调用失败，已降级到通用对话模式",
}


def _format_history(history: list[dict]) -> str:
    """将历史列表格式化为 prompt 中可读的字符串。"""
    if not history:
        return "（无历史）"
    lines = []
    for msg in history:
        role_label = "用户" if msg.get("role") == "user" else "助手"
        lines.append(f"{role_label}：{msg.get('content', '')}")
    return "\n".join(lines)


def _format_context(context: dict) -> str:
    """将情景上下文格式化为 prompt 可读字符串。

    context 来自会话情景记忆，为意图分类提供操作上下文提示。
    """
    if not context:
        return "（无记录）"
    action = context.get("last_action")
    if action == "paper_download":
        papers = context.get("downloaded_papers", [])
        return f"最近操作：已下载论文 {papers}，用户提到'这篇'时优先指代这些论文"
    if action == "build_knowledge":
        chunks = context.get("chunks_indexed", 0)
        return f"最近操作：已构建知识库，索引 {chunks} 个片段，可对已下载论文进行 RAG 检索"
    return "（无记录）"


def _extract_intent_json(raw: str) -> dict | None:
    """
    从 LLM 原始输出中提取 JSON 意图结果。
    先尝试直接 parse，失败则用正则提取。
    """
    # 尝试直接解析
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        pass

    # 正则提取 JSON 对象 [REF: product.md#F-02 异常处理]
    pattern = r'\{[^{}]*"intent"\s*:\s*"[^"]+[^{}]*\}'
    match = re.search(pattern, raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return None


def classify_intent(
    session_id: str,
    cleaned_input: str,
    history: list[dict] | None = None,
) -> dict:
    """
    使用 LLM 对用户意图进行零样本分类。

    Args:
        session_id:    会话 ID
        cleaned_input: 预处理后的用户输入
        history:       可选传入历史（None 时从 Redis 自动读取最近 5 条）

    Returns:
        dict: {
            "intent": str,       # 来自 VALID_INTENTS 枚举
            "confidence": float, # 0.0 ~ 1.0
            "reasoning": str
        }
    """
    # 步骤1：读取最近 5 条历史，为分类提供对话上下文
    if history is None:
        history = get_history(session_id, limit=5)

    history_str = _format_history(history)

    # 步骤2：读取情景上下文，包含最近操作记录
    ctx = get_context(session_id)
    context_str = _format_context(ctx)

    # 步骤3：注入历史和情景上下文，构造分类 prompt
    prompt_text = INTENT_CLASSIFY_PROMPT.format(
        context=context_str,
        history=history_str,
        user_input=cleaned_input,
    )
    messages = [{"role": "user", "content": prompt_text}]

    # 步骤3：调用 LLM，使用低温度保证分类稳定性
    try:
        llm_result = call_llm(
            messages=messages,
            temperature=0.1,  # 分类任务用低温度
            session_id=session_id,
            event="intent_classify",
        )
        raw_output = llm_result.get("content", "")
    except (TimeoutError, RuntimeError) as e:
        logger.error(f"IntentClassifier: LLM 调用失败 session={session_id} error={e}")
        return _FALLBACK_RESULT.copy()

    # 步骤4：解析 LLM 返回的 JSON 意图结果
    parsed = _extract_intent_json(raw_output)
    if parsed is None:
        logger.warning(
            f"IntentClassifier: LLM 输出无法解析为 JSON，降级。"
            f" session={session_id} raw={raw_output!r}"
        )
        return _FALLBACK_RESULT.copy()

    intent = str(parsed.get("intent", "general_chat"))
    confidence = float(parsed.get("confidence", 0.0))
    reasoning = str(parsed.get("reasoning", ""))

    # 步骤5：intent 不在枚举范围时降级到 general_chat
    if intent not in VALID_INTENTS:
        logger.warning(
            f"IntentClassifier: 返回未知 intent='{intent}'，降级到 general_chat。"
            f" session={session_id}"
        )
        intent = "general_chat"
        confidence = 0.0

    # 步骤6：置信度过低时强制降级到 general_chat（阈値：0.7）
    if confidence < CONFIDENCE_THRESHOLD:
        logger.info(
            f"IntentClassifier: confidence={confidence} < {CONFIDENCE_THRESHOLD}，"
            f"原 intent={intent} 降级到 general_chat。 session={session_id}"
        )
        intent = "general_chat"

    return {
        "intent": intent,
        "confidence": confidence,
        "reasoning": reasoning,
    }
