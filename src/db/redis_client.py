"""
Redis 连接管理（单例模式）。

Key 格式约定：session:{session_id}:history（历史）、session:{session_id}:context（情景上下文）。
REDIS_URL 必须配置，未配置时启动即报错。
"""
import os

import redis

_redis_client: redis.Redis | None = None


def get_redis() -> redis.Redis:
    """
    返回 Redis 单例连接。

    Key 格式约定：session:{session_id}:history（会话历史）
    """
    global _redis_client
    if _redis_client is None:
        # 必填环境变量，未配置时报 KeyError
        redis_url = os.environ["REDIS_URL"]
        _redis_client = redis.from_url(redis_url, decode_responses=True)
    return _redis_client


def check_connection() -> bool:
    """
    验证 Redis 连接是否正常，用于 /api/health。

    Returns:
        bool: True 表示连接正常

    Refs:
        用于 /api/health 健康检查接口。
    """
    try:
        get_redis().ping()
        return True
    except Exception:
        return False
