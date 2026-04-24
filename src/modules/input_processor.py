"""
M-01：用户输入预处理。

执行去空白、空检查、超长截断（上限 2000 字符）三个步骤。
"""
import logging

logger = logging.getLogger(__name__)

# 单条消息最大输入长度（超过此长度截断）
MAX_INPUT_LENGTH = 2000


def process_input(session_id: str, raw_input: str) -> dict:
    """
    预处理用户输入：去空白、空检查、截断。

    Args:
        session_id: 会话 ID（透传，用于日志）
        raw_input:  用户原始输入字符串

    Returns:
        dict: {
            "session_id": str,
            "cleaned_input": str,
            "input_length": int,
            "is_empty": bool
        }

    Refs:
        返回字段：session_id/cleaned_input/input_length/is_empty。
    """
    # 步骤1：去除首尾空白
    cleaned = raw_input.strip()

    # 步骤2：空检查，纯空白输入直接返回 is_empty=True
    if not cleaned:
        return {
            "session_id": session_id,
            "cleaned_input": "",
            "input_length": 0,
            "is_empty": True,
        }

    # 步骤3：超长截断，保留前 2000 字符
    if len(cleaned) > MAX_INPUT_LENGTH:
        logger.warning(
            f"InputProcessor: 输入超过 {MAX_INPUT_LENGTH} 字符，已截断。"
            f" session={session_id} original_len={len(cleaned)}"
        )
        cleaned = cleaned[:MAX_INPUT_LENGTH]

    return {
        "session_id": session_id,
        "cleaned_input": cleaned,
        "input_length": len(cleaned),
        "is_empty": False,
    }
