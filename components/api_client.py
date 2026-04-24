"""
components/api_client.py — 唯一 API 调用入口。
"""
import os
from typing import Optional

import requests

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")


class APIError(Exception):
    """后端返回的业务错误（HTTP 4xx/5xx）。"""
    def __init__(self, status_code: int, code: str, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


def _handle(resp: requests.Response) -> dict:
    """
    统一解析响应，非 2xx 时抛出 APIError。
    错误响应格式：{"detail": {"error": {"code": ..., "message": ...}}}
    """
    try:
        data = resp.json()
    except Exception:
        data = {}

    if not resp.ok:
        # api_spec.md 规范错误结构：{"detail": {"error": {"code": ..., "message": ...}}}
        detail = data.get("detail", {})
        if isinstance(detail, dict):
            err = detail.get("error", {})
            code = err.get("code", "INTERNAL_ERROR")
            msg = err.get("message", resp.text)
        else:
            code = "INTERNAL_ERROR"
            msg = str(detail) or resp.text
        raise APIError(resp.status_code, code, msg)

    return data


# ─────────────────────────────────────────────────────────────
# 会话接口
# ─────────────────────────────────────────────────────────────

def get_sessions() -> dict:
    """GET /api/sessions — 获取所有会话列表。"""
    resp = requests.get(f"{BACKEND_URL}/api/sessions", timeout=15)
    return _handle(resp)


def create_session(name: str = "新会话") -> dict:
    """POST /api/sessions/create — 创建新会话。"""
    resp = requests.post(
        f"{BACKEND_URL}/api/sessions/create",
        json={"name": name},
        timeout=10,
    )
    return _handle(resp)


def delete_session(session_id: str) -> dict:
    """DELETE /api/sessions/{session_id} — 删除会话及历史。"""
    resp = requests.delete(
        f"{BACKEND_URL}/api/sessions/{session_id}",
        timeout=10,
    )
    return _handle(resp)


def get_history(session_id: str, limit: int = 20) -> dict:
    """GET /api/sessions/{session_id}/history — 获取对话历史。"""
    resp = requests.get(
        f"{BACKEND_URL}/api/sessions/{session_id}/history",
        params={"limit": limit},
        timeout=10,
    )
    return _handle(resp)


# ─────────────────────────────────────────────────────────────
# 对话接口
# ─────────────────────────────────────────────────────────────

def chat(session_id: str, message: str) -> dict:
    """POST /api/agent/chat — 主对话接口。"""
    resp = requests.post(
        f"{BACKEND_URL}/api/agent/chat",
        json={"session_id": session_id, "message": message},
        timeout=120,  # LLM 推理可能耗时
    )
    return _handle(resp)


# ─────────────────────────────────────────────────────────────
# 论文接口
# ─────────────────────────────────────────────────────────────

def search_papers(
    session_id: str,
    query: str,
    max_results: int = 10,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> dict:
    """POST /api/papers/search — 搜索 arXiv 论文。"""
    resp = requests.post(
        f"{BACKEND_URL}/api/papers/search",
        json={
            "session_id": session_id,
            "query": query,
            "max_results": max_results,
            "start_date": start_date,
            "end_date": end_date,
        },
        timeout=30,
    )
    return _handle(resp)


def download_papers(session_id: str, paper_ids: list[str]) -> dict:
    """POST /api/papers/download — 下载论文 PDF。"""
    resp = requests.post(
        f"{BACKEND_URL}/api/papers/download",
        json={"session_id": session_id, "paper_ids": paper_ids},
        timeout=120,
    )
    return _handle(resp)


# ─────────────────────────────────────────────────────────────
# 知识库接口
# ─────────────────────────────────────────────────────────────

def build_knowledge(session_id: str) -> dict:
    """POST /api/knowledge/build — 构建向量知识库。"""
    resp = requests.post(
        f"{BACKEND_URL}/api/knowledge/build",
        json={"session_id": session_id},
        timeout=600,  # 解析 PDF + embedding 可能较慢
    )
    return _handle(resp)


# ─────────────────────────────────────────────────────────────
# 健康检查
# ─────────────────────────────────────────────────────────────

def health_check() -> dict:
    """GET /api/health — 健康检查接口。"""
    resp = requests.get(f"{BACKEND_URL}/api/health", timeout=60)
    return _handle(resp)
