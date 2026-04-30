"""
GradCopilot FastAPI 主入口 — Phase 2 Agentic RAG

接口列表（共 12 个）：
  1.  POST   /api/agent/chat
  2.  GET    /api/sessions
  3.  POST   /api/sessions/create
  4.  DELETE /api/sessions/{session_id}
  5.  GET    /api/sessions/{session_id}/history
  6.  POST   /api/papers/search
  7.  POST   /api/papers/download
  8.  POST   /api/knowledge/build
  9.  GET    /api/health
  10. GET    /api/config
  11. PUT    /api/config
  12. DELETE /api/config
"""
import asyncio
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import text

# ── 环境变量加载（最优先）─────────────────────────────────────
load_dotenv(Path(__file__).parent.parent / ".env")

# ──# 应用日志初始化，必须在模块导入前调用───────
from src.utils.logging_config import setup_logging
setup_logging()

import logging
logger = logging.getLogger(__name__)

# DB 连接初始化，依赖 POSTGRES_URL / REDIS_URL 环境变量───
from src.db.postgres import engine, check_connection as pg_check
from src.db.redis_client import get_redis, check_connection as redis_check

# ── 业务模块 ──────────────────────────────────────────────────
from src.modules.input_processor import process_input
from src.modules.intent_classifier import classify_intent
from src.modules.rag_router import route as rag_route
from src.modules.agent_executor import execute as agent_execute
from src.modules.response_generator import generate as response_generate
from src.modules.session_memory import get_history, clear_history, count_history
from src.utils.errors import ErrorCode
from src.utils.config_loader import (
    get as config_get,
    get_masked,
    set as config_set,
    reset as config_reset,
)

# ── FastAPI 实例 ──────────────────────────────────────────────
app = FastAPI(
    title="GradCopilot API",
    description="Agentic RAG 学术研究助手 — Phase 2",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Pydantic 请求/响应模型
# ============================================================

class ChatRequest(BaseModel):
    """POST /api/agent/chat 请求体。"""
    session_id: str = Field(..., min_length=1, max_length=64, description="会话 ID")
    message: str = Field(..., min_length=1, max_length=2000, description="用户消息")


class CreateSessionRequest(BaseModel):
    """POST /api/sessions/create 请求体。"""
    name: str = Field(default="新会话", max_length=100, description="会话名称")


class SearchRequest(BaseModel):
    """POST /api/papers/search 请求体。"""
    session_id: str = Field(..., min_length=1, description="会话 ID")
    query: str = Field(..., min_length=1, max_length=200, description="搜索关键词")
    max_results: int = Field(default=10, ge=1, le=50, description="最大返回数量")
    start_date: Optional[str] = Field(default=None, description="起始日期 YYYY-MM-DD")
    end_date: Optional[str] = Field(default=None, description="截止日期 YYYY-MM-DD")


class DownloadRequest(BaseModel):
    """POST /api/papers/download 请求体。"""
    session_id: str = Field(..., min_length=1, description="会话 ID")
    paper_ids: list[str] = Field(..., min_length=1, max_length=20, description="论文 ID 列表")


class BuildKnowledgeRequest(BaseModel):
    """POST /api/knowledge/build 请求体。"""
    session_id: str = Field(..., min_length=1, description="会话 ID")


class ConfigUpdateRequest(BaseModel):
    """PUT /api/config 请求体，所有字段可选"""
    api_key: Optional[str] = Field(default=None, max_length=512)
    base_url: Optional[str] = Field(default=None)
    model_name: Optional[str] = Field(default=None, max_length=128)
    embedding_model: Optional[str] = Field(default=None, max_length=128)
    ollama_base_url: Optional[str] = Field(default=None)
    ollama_model: Optional[str] = Field(default=None, max_length=128)

    @field_validator("base_url", "ollama_base_url", mode="before")
    @classmethod
    def validate_url(cls, v):
        if v is not None and not (v.startswith("http://") or v.startswith("https://")):
            raise ValueError("URL 必须以 http:// 或 https:// 开头")
        return v


# ============================================================
# 工具函数
# ============================================================

def _error_response(code: str, message: str, status_code: int):
    """统一错误响应格式。"""
    raise HTTPException(
        status_code=status_code,
        detail={"error": {"code": code, "message": message}},
    )


def _session_exists(session_id: str) -> bool:
    """检查 sessions 表是否存在该 session_id。"""
    try:
        with engine.connect() as conn:
            row = conn.execute(
                text("SELECT 1 FROM sessions WHERE id = :sid"),
                {"sid": session_id},
            ).fetchone()
        return row is not None
    except Exception:
        return False


def _create_session_in_db(session_id: str, name: str) -> dict:
    """在 sessions 表中创建新会话并返回记录。"""
    now = datetime.now(tz=timezone.utc)
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO sessions (id, name, created_at, updated_at, message_count) "
                "VALUES (:id, :name, :created_at, :updated_at, 0)"
            ),
            {"id": session_id, "name": name, "created_at": now, "updated_at": now},
        )
    return {
        "session_id": session_id,
        "name": name,
        "created_at": now.isoformat(),
    }


def _delete_session_in_db(session_id: str) -> None:
    """删除 sessions 表记录（paper_chunks 由外键级联删除）。"""
    with engine.begin() as conn:
        conn.execute(
            text("DELETE FROM sessions WHERE id = :sid"),
            {"sid": session_id},
        )


def _get_all_sessions() -> list[dict]:
    """读取全部会话，附带 message_count。"""
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT id, name, created_at, message_count "
                "FROM sessions ORDER BY created_at DESC"
            )
        ).fetchall()
    return [
        {
            "session_id": r.id,
            "name": r.name,
            "created_at": r.created_at.isoformat() if r.created_at else None,
            "message_count": r.message_count,
        }
        for r in rows
    ]


# ============================================================
# 1. POST /api/agent/chat — 主对话入口
# ============================================================

@app.post("/api/agent/chat")
async def agent_chat(req: ChatRequest):
    """
    主对话接口：InputProcessor → IntentClassifier → RAGRouter →
    AgentExecutor → ResponseGenerator
    """
    # 步骤1：输入预处理（去空白、空检查、超长截断）
    input_result = process_input(req.session_id, req.message)
    if input_result["is_empty"]:
        _error_response(ErrorCode.VALIDATION_ERROR, "消息内容不能为空", 422)

    cleaned = input_result["cleaned_input"]

    # 步骤2：意图识别（LLM 分类，置信度 < 0.7 降级 general_chat）
    try:
        intent_result = classify_intent(req.session_id, cleaned)
    except Exception as e:
        logger.error(f"IntentClassifier 异常: {e}")
        _error_response(ErrorCode.LLM_ERROR, f"意图识别失败: {e}", 500)

    intent = intent_result["intent"]
    confidence = intent_result["confidence"]

    # 步骤3：路由决策（固定路由表映射 intent 到 route/tool_name）
    route_result = rag_route(intent, confidence)
    route = route_result["route"]
    tool_name = route_result["tool_name"]

    # 步骤4：工具执行（RAG检索/arXiv搜索/下载/构建知识库/纯对话）
    try:
        exec_result = agent_execute(
            session_id=req.session_id,
            route=route,
            tool_name=tool_name,
            cleaned_input=cleaned,
        )
    except Exception as e:
        logger.error(f"AgentExecutor 异常: {e}")
        _error_response(ErrorCode.TOOL_ERROR, f"工具执行失败: {e}", 500)

    # 步骤5：生成回答，写入会话历史
    try:
        gen_result = response_generate(
            session_id=req.session_id,
            cleaned_input=cleaned,
            route=route,
            tool_result=exec_result.get("tool_result"),
            raw_response=exec_result.get("raw_response", ""),
            sources=exec_result.get("sources", []),
            tool_used=exec_result.get("tool_used"),
        )
    except Exception as e:
        logger.error(f"ResponseGenerator 异常: {e}")
        _error_response(ErrorCode.LLM_ERROR, f"回答生成失败: {e}", 500)

    # 更新 sessions 表的 message_count
    try:
        with engine.begin() as conn:
            conn.execute(
                text(
                    "UPDATE sessions SET message_count = message_count + 2, "
                    "updated_at = NOW() WHERE id = :sid"
                ),
                {"sid": req.session_id},
            )
    except Exception:
        pass  # message_count 更新失败不影响主流程

    return {
        "session_id": req.session_id,
        "response": gen_result["response"],
        "intent": intent,
        "route": route,
        "tool_used": exec_result.get("tool_used"),
        "sources": gen_result.get("sources", []),
        "model_used": gen_result.get("model_used", config_get("model_name")),
        "tokens_used": gen_result.get("tokens_used", 0),
    }


# ============================================================
# 2. GET /api/sessions — 获取所有会话列表
# ============================================================

@app.get("/api/sessions")
async def list_sessions():
    """获取所有会话列表。"""
    try:
        sessions = await asyncio.to_thread(_get_all_sessions)
        return {"sessions": sessions}
    except Exception as e:
        logger.error(f"list_sessions 失败: {e}")
        _error_response(ErrorCode.INTERNAL_ERROR, str(e), 500)


# ============================================================
# 3. POST /api/sessions/create — 创建会话
# ============================================================

@app.post("/api/sessions/create")
async def create_session(req: CreateSessionRequest):
    """创建新会话。"""
    session_id = str(uuid.uuid4()).replace("-", "")[:16]
    try:
        result = await asyncio.to_thread(_create_session_in_db, session_id, req.name)
        return result
    except Exception as e:
        logger.error(f"create_session 失败: {e}")
        _error_response(ErrorCode.INTERNAL_ERROR, str(e), 500)


# ============================================================
# 4. DELETE /api/sessions/{session_id} — 删除会话
# ============================================================

@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    """删除指定会话及所有历史记录。"""
    exists = await asyncio.to_thread(_session_exists, session_id)
    if not exists:
        _error_response(ErrorCode.SESSION_NOT_FOUND, f"会话 {session_id} 不存在", 404)

    try:
        clear_history(session_id)
        await asyncio.to_thread(_delete_session_in_db, session_id)
        return {"success": True, "deleted_session_id": session_id}
    except Exception as e:
        logger.error(f"delete_session 失败: {e}")
        _error_response(ErrorCode.INTERNAL_ERROR, str(e), 500)


# ============================================================
# 5. GET /api/sessions/{session_id}/history — 获取对话历史
# ============================================================

@app.get("/api/sessions/{session_id}/history")
async def get_session_history(session_id: str, limit: int = 20):
    """获取指定会话的对话历史。"""
    exists = await asyncio.to_thread(_session_exists, session_id)
    if not exists:
        _error_response(ErrorCode.SESSION_NOT_FOUND, f"会话 {session_id} 不存在", 404)

    limit = max(1, min(limit, 20))

    try:
        history = get_history(session_id, limit=limit)
        return {"session_id": session_id, "history": history}
    except Exception as e:
        logger.error(f"get_history 失败: {e}")
        _error_response(ErrorCode.INTERNAL_ERROR, str(e), 500)


# ============================================================
# 6. POST /api/papers/search — 搜索 arXiv 论文
# ============================================================

@app.post("/api/papers/search")
async def search_papers(req: SearchRequest):
    """搜索 arXiv 论文（优先走 MCP，失败 fallback 到旧工具）。"""
    try:
        tool_source = "mcp"
        try:
            from src.tools.mcp_arxiv_tool import search as mcp_search
            result = await asyncio.to_thread(
                mcp_search,
                req.query,
                req.session_id,
                req.max_results,
                req.start_date,
                req.end_date,
            )
            logger.info(f"API /papers/search: MCP 成功 session={req.session_id}")
        except Exception as mcp_err:
            tool_source = "legacy"
            logger.warning(f"API /papers/search: MCP 失败，fallback session={req.session_id} error={mcp_err}")
            from src.tools.search_tool import run as search_run
            result = await asyncio.to_thread(
                search_run,
                req.query,
                req.session_id,
                req.max_results,
                req.start_date,
                req.end_date,
            )
        return {**result, "tool_source": tool_source}
    except Exception as e:
        logger.error(f"search_papers 失败: {e}")
        _error_response(ErrorCode.TOOL_ERROR, str(e), 500)


# ============================================================
# 7. POST /api/papers/download — 下载论文
# ============================================================

@app.post("/api/papers/download")
async def download_papers(req: DownloadRequest):
    """下载论文 PDF（优先走 MCP，失败 fallback 到旧工具）。"""
    try:
        from src.modules.session_memory import set_context
        query = " ".join(req.paper_ids)
        tool_source = "mcp"
        try:
            from src.tools.mcp_arxiv_tool import download as mcp_download
            result = await asyncio.to_thread(mcp_download, query, req.session_id)
            logger.info(f"API /papers/download: MCP 成功 session={req.session_id}")
        except Exception as mcp_err:
            tool_source = "legacy"
            logger.warning(f"API /papers/download: MCP 失败，fallback session={req.session_id} error={mcp_err}")
            from src.tools.download_tool import download_by_ids
            result = await asyncio.to_thread(download_by_ids, req.paper_ids, req.session_id)
        # 写入情景上下文：记录本次下载的论文 ID，供后续 RAG 限定检索范围
        downloaded = result.get("downloaded", [])
        if downloaded:
            set_context(req.session_id, {
                "last_action": "paper_download",
                "downloaded_papers": downloaded,
            })
        return {**result, "tool_source": tool_source}
    except Exception as e:
        logger.error(f"download_papers 失败: {e}")
        _error_response(ErrorCode.TOOL_ERROR, str(e), 500)


# ============================================================
# 8. POST /api/knowledge/build — 构建知识库
# ============================================================

@app.post("/api/knowledge/build")
async def build_knowledge(req: BuildKnowledgeRequest):
    """构建向量知识库。"""
    try:
        from src.tools.build_knowledge_tool import run as build_run
        from src.modules.session_memory import set_context, get_context
        # 构建前读取已有上下文，保留 downloaded_papers 不被覆盖
        prev_ctx = get_context(req.session_id)
        result = build_run(session_id=req.session_id)
        chunks = result["chunks_indexed"]
        # 写入情景上下文，合并保留上一步下载的论文 ID
        new_ctx = {
            "last_action": "build_knowledge",
            "chunks_indexed": chunks,
        }
        if prev_ctx.get("downloaded_papers"):
            new_ctx["downloaded_papers"] = prev_ctx["downloaded_papers"]
        set_context(req.session_id, new_ctx)
        return {
            "session_id": req.session_id,
            "chunks_indexed": chunks,
            "status": result["status"],
        }
    except RuntimeError as e:
        logger.error(f"build_knowledge 失败: {e}")
        _error_response(ErrorCode.TOOL_ERROR, str(e), 500)
    except Exception as e:
        logger.error(f"build_knowledge 异常: {e}")
        _error_response(ErrorCode.INTERNAL_ERROR, str(e), 500)

# ============================================================
# 10-12. /api/config — 运行时配置管理
# ============================================================

@app.get("/api/config")
async def get_config():
    """GET /api/config — 读取当前脱敏配置（三层优先级解析后的最终值）。"""
    return get_masked()


@app.put("/api/config")
async def update_config(req: ConfigUpdateRequest):
    """PUT /api/config — 局部更新 config.local.json。"""
    updates = req.model_dump(exclude_unset=True)
    config_set(updates)
    return {"success": True, "updated_fields": list(updates.keys())}


@app.delete("/api/config")
async def reset_config():
    """DELETE /api/config — 清空 config.local.json，回退到 .env 和默认值。"""
    config_reset()
    return {"success": True, "message": "配置已重置为默认值"}


# ============================================================
# 9. GET /api/health — 健康检查
# ============================================================

@app.get("/api/health")
async def health_check():
    """
    健康检查：验证 Redis / PostgreSQL / LLM API 三组件状态。
    三组件并发检测，总耗时 ≤ 单组件最大超时，不串行阻塞。
    """
    redis_ok, pg_ok, llm_ok = await asyncio.gather(
        asyncio.to_thread(redis_check),
        asyncio.to_thread(pg_check),
        _check_llm_api(),
        return_exceptions=True,
    )

    redis_status = "ok" if redis_ok is True else "error"
    pg_status = "ok" if pg_ok is True else "error"
    llm_status = "ok" if llm_ok == "ok" else "error"

    overall = "ok" if all(s == "ok" for s in [redis_status, pg_status, llm_status]) else "degraded"

    return {
        "status": overall,
        "redis": redis_status,
        "postgres": pg_status,
        "llm_api": llm_status,
    }


async def _check_llm_api() -> str:
    """
    异步验证 LLM API 可用性（通过 models 端点，不消耗 token）。
    设置独立 2s 超时，超时直接返回 error，不阻塞 health 接口。
    """
    def _sync_check() -> str:
        try:
            from openai import OpenAI
            client = OpenAI(
                api_key=config_get("api_key"),
                base_url=config_get("base_url"),
                timeout=3,
            )
            client.models.list()
            return "ok"
        except Exception:
            return "error"

    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(_sync_check),
            timeout=2.0,  # 独立 2s 超时，防止拖慢 health 接口
        )
        return result
    except asyncio.TimeoutError:
        return "error"
