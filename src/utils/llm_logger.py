"""
M-13：LLM 调用追踪日志记录器。

所有 LLM 调用日志必须通过此模块写入，禁止其他模块直接写 logs/llm.jsonl。
日志格式为 JSONL（每行一条），每条包含 event/model/input/output/tokens/latency_ms/session_id/timestamp 字段。
"""
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import TypedDict

logger = logging.getLogger(__name__)

# LLM 日志文件路径，输出到 logs/llm.jsonl(JSONL 格式）
_LOG_DIR = Path(os.environ.get("LOG_DIR", "logs"))
_LLM_LOG_FILE = _LOG_DIR / "llm.jsonl"


class LLMLogEntry(TypedDict):
    """
    LLM 调用日志条目结构。
    字段：event/model/input/output/tokens/latency_ms/session_id/timestamp。
    """

    event: str       # 事件标识，如 "intent_classify", "response_generate"
    model: str       # 模型名称
    input: str       # LLM 输入的完整 prompt（JSON 序列化的 messages）
    output: str      # LLM 返回的原始内容
    tokens: int      # 总 token 数
    latency_ms: int  # 调用耗时（毫秒）
    session_id: str  # 关联的会话 ID
    timestamp: str   # ISO 8601 格式时间戳


def log_llm_call(
    event: str,
    model: str,
    messages: list[dict],
    output: str,
    tokens: int,
    latency_ms: int,
    session_id: str,
) -> None:
    """
    记录一次 LLM 调用到 logs/llm.jsonl（JSONL 格式，每行一条）。

    Args:
        event:      事件标识，如 "intent_classify"
        model:      实际使用的模型名称
        messages:   传入 LLM 的 messages 列表
        output:     LLM 返回的原始字符串内容
        tokens:     本次调用消耗的总 token 数
        latency_ms: 调用耗时（毫秒）
        session_id: 关联的会话 ID

    Refs:
        日志条目字段：event/model/input/output/tokens/latency_ms/session_id/timestamp。
        日志格式：JSONL，每调用追加一行。
    """
    _LOG_DIR.mkdir(parents=True, exist_ok=True)

    entry: LLMLogEntry = {
        "event": event,
        "model": model,
        "input": json.dumps(messages, ensure_ascii=False),
        "output": output,
        "tokens": tokens,
        "latency_ms": latency_ms,
        "session_id": session_id,
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
    }

    try:
        with open(_LLM_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError as e:
        # 日志写入失败不应中断主流程，仅记录 WARNING
        logger.warning(f"LLM 日志写入失败: {e}")
