"""
M-12: RAG 检索工具。

将查询转为 embedding，在 pgvector 中检索最相关的论文片段。
支持可选的 paper_ids 过滤，将检索范围限定到情景上下文中指定的论文。
"""
import logging

from openai import OpenAI

from src.modules.vector_store import search
from src.utils.config_loader import get as config_get

logger = logging.getLogger(__name__)


def run(query: str, session_id: str, top_k: int = 5, paper_ids: list[str] | None = None) -> dict:
    """
    M-12 RAG 检索入口，供 AgentExecutor 调用。

    将查询文本转为 embedding，在 pgvector 中进行余弦相似度检索。

    Args:
        query:      用户查询文本
        session_id: 会话 ID（检索范围限定在此 session）
        top_k:      返回最相关片段数量，默认 5
        paper_ids:  可选，限定检索的论文 ID 列表（来自会话情景上下文）

    Returns:
        dict: {
            "contexts": list[dict],  # 检索到的片段，含 paper_id/chunk_index/content/page_number/distance
            "sources": list[str]     # 来源引用字符串列表
        }
    """
    if not query.strip():
        return {"contexts": [], "sources": []}

    # 步骤1：生成查询 embedding，维度 1536（text-embedding-3-small）
    try:
        query_embedding = _embed_query(query)
    except Exception as e:
        logger.error(f"RAGRetrievalTool: embedding 失败 session={session_id} error={e}")
        raise RuntimeError(f"RAG 检索失败（embedding）: {e}") from e

    # 步骤2：pgvector 余弦相似度检索，可选通过 paper_ids 限定论文范围
    try:
        results = search(
            query_embedding=query_embedding,
            session_id=session_id,
            top_k=top_k,
            paper_ids=paper_ids,
        )
    except Exception as e:
        logger.error(f"RAGRetrievalTool: 检索失败 session={session_id} error={e}")
        raise RuntimeError(f"RAG 检索失败（向量搜索）: {e}") from e

    # 步骤3：构造来源引用列表，格式：papers/{session_id}/{paper_id}.pdf#page={page}
    sources = [
        f"papers/{session_id}/{r['paper_id']}.pdf#page={r.get('page_number', '?')}"
        for r in results
    ]

    logger.info(
        f"RAGRetrievalTool: session={session_id} top_k={top_k} "
        f"paper_ids={paper_ids} found={len(results)}"
    )
    return {"contexts": results, "sources": sources}


def _embed_query(text: str) -> list[float]:
    """
    将单条查询文本转为 embedding 向量（维度 1536）。

    Refs:
        向量维度 1536，与 paper_chunks 表的 embedding VECTOR(1536) 字段匹配。
    """
    client = OpenAI(
        api_key=config_get("api_key"),
        base_url=config_get("base_url"),
    )
    resp = client.embeddings.create(model=config_get("embedding_model"), input=[text])
    return resp.data[0].embedding
