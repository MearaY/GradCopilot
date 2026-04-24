"""
M-06：短期会话记忆管理（Redis）。

会话历史（Conversation History）：
  Key 格式：session:{session_id}:history
  类型：Redis List，每个元素为 JSON 序列化的消息对象
  字段：role（"user" | "assistant"）、content、timestamp
  上限：最多保留 20 条，FIFO 淘汰
  TTL：24 小时自动过期

情景上下文（Episodic Context）：
  Key 格式：session:{session_id}:context
  类型：Redis String，JSON 序列化的操作元数据
  用途：记录最近一次工具操作（下载/构建知识库）的结构化信息，
        供意图分类和回答生成模块感知当前会话状态。
"""
import json
import logging
from datetime import datetime, timezone

from src.db.redis_client import get_redis

logger = logging.getLogger(__name__)

# 常量定义：单会话历史上限与过期时间
MAX_HISTORY_LEN = 20         # 每个 session 最多保留 20 条，FIFO 淘汰最旧记录
HISTORY_TTL_SECONDS = 86400  # 24 小时 TTL，无活动后自动清理


def _make_key(session_id: str) -> str:
    """生成会话历史 Redis Key，格式：session:{session_id}:history。"""
    return f"session:{session_id}:history"


def get_history(session_id: str, limit: int = 5) -> list[dict]:
    """
    读取会话历史，返回最新的 limit 条记录。

    Args:
        session_id: 会话 ID
        limit:      最多返回条数，范围 1-20

    Returns:
        list[dict]: 每条消息包含 role, content, timestamp 字段
    """
    limit = max(1, min(limit, MAX_HISTORY_LEN))
    key = _make_key(session_id)
    try:
        raw_items = get_redis().lrange(key, -limit, -1)
        history = []
        for raw in raw_items:
            try:
                history.append(json.loads(raw))
            except json.JSONDecodeError:
                logger.warning(f"SessionMemory: 解析历史条目失败，跳过: {raw!r}")
        return history
    except Exception as e:
        logger.error(f"SessionMemory.get_history 失败: session={session_id} error={e}")
        return []


def append_history(session_id: str, role: str, content: str) -> bool:
    """
    向会话历史追加一条消息，维护 MAX_HISTORY_LEN 上限，刷新 TTL。

    Args:
        session_id: 会话 ID
        role:       消息角色，枚举："user" 或 "assistant"
        content:    消息内容

    Returns:
        bool: True 表示写入成功
    """
    entry = {
        "role": role,
        "content": content,
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
    }
    key = _make_key(session_id)
    try:
        r = get_redis()
        pipe = r.pipeline()
        pipe.rpush(key, json.dumps(entry, ensure_ascii=False))
        pipe.ltrim(key, -MAX_HISTORY_LEN, -1)  # 保留最后 MAX_HISTORY_LEN 条
        pipe.expire(key, HISTORY_TTL_SECONDS)   # 刷新 TTL
        pipe.execute()
        return True
    except Exception as e:
        logger.error(f"SessionMemory.append_history 失败: session={session_id} error={e}")
        return False


def clear_history(session_id: str) -> bool:
    """
    清空指定会话的历史记录（删除 Redis Key）。

    Args:
        session_id: 会话 ID

    Returns:
        bool: True 表示清空成功

    """
    key = _make_key(session_id)
    try:
        get_redis().delete(key)
        return True
    except Exception as e:
        logger.error(f"SessionMemory.clear_history 失败: session={session_id} error={e}")
        return False


def count_history(session_id: str) -> int:
    """
    返回指定会话的历史消息总数。

    Args:
        session_id: 会话 ID

    Returns:
        int: 消息条数

    """
    key = _make_key(session_id)
    try:
        return get_redis().llen(key)
    except Exception as e:
        logger.error(f"SessionMemory.count_history 失败: session={session_id} error={e}")
        return 0


def set_context(session_id: str, context: dict) -> bool:
    """
    写入会话情景上下文（最近一次工具操作的结构化元数据）。

    Args:
        session_id: 会话 ID
        context:    操作元数据，例如：
                    {"last_action": "paper_download", "downloaded_papers": ["2401.xxxx"]}
                    {"last_action": "build_knowledge", "chunks_indexed": 647}

    Returns:
        bool: True 表示写入成功

    """
    key = f"session:{session_id}:context"
    try:
        get_redis().setex(
            key,
            HISTORY_TTL_SECONDS,
            json.dumps(context, ensure_ascii=False),
        )
        logger.info(f"SessionMemory.set_context: session={session_id} action={context.get('last_action')}")
        return True
    except Exception as e:
        logger.error(f"SessionMemory.set_context 失败: session={session_id} error={e}")
        return False


def get_context(session_id: str) -> dict:
    """
    读取会话情景上下文。

    Args:
        session_id: 会话 ID

    Returns:
        dict: 情景元数据，读取失败或无记录时返回空 dict

    """
    key = f"session:{session_id}:context"
    try:
        raw = get_redis().get(key)
        if raw is None:
            return {}
        return json.loads(raw)
    except Exception as e:
        logger.error(f"SessionMemory.get_context 失败: session={session_id} error={e}")
        return {}
