"""
M-07：向量存储模块（pgvector）。

唯一负责 paper_chunks 表的读写，禁止其他模块直接操作 pgvector。
表结构：paper_id / session_id / chunk_index / content / page_number / embedding VECTOR(1536)。
"""
import logging

from sqlalchemy import text

from src.db.postgres import engine

logger = logging.getLogger(__name__)

# 检索返回字段
_SELECT_FIELDS = "paper_id, chunk_index, content, page_number"


def insert_chunks(
    session_id: str,
    chunks: list[dict],
    embeddings: list[list[float]],
) -> int:
    """
    批量写入论文切片和对应向量到 paper_chunks 表。

    Args:
        session_id:  会话 ID
        chunks:      切片列表，每条包含 paper_id, chunk_index, content, page_number
        embeddings:  与 chunks 一一对应的向量列表（维度 1536）

    Returns:
        int: 成功写入的条数

    Raises:
        ValueError: chunks 与 embeddings 长度不匹配
        RuntimeError: 数据库写入失败

    Refs:
        写入时将 embedding list 转为 pgvector 字符串格式 '[x,y,z,...]'。
        session_id 用于数据隔离，禁止跨 session 检索。
    """
    if len(chunks) != len(embeddings):
        raise ValueError(
            f"chunks({len(chunks)}) 与 embeddings({len(embeddings)}) 数量不匹配"
        )
    if not chunks:
        return 0

    rows = []
    for chunk, emb in zip(chunks, embeddings):
        # 将向量列表转为 pgvector 字符串格式 '[x,y,z,...]'
        vec_str = "[" + ",".join(str(v) for v in emb) + "]"
        rows.append(
            {
                "paper_id": chunk["paper_id"],
                "session_id": session_id,
                "chunk_index": chunk["chunk_index"],
                "content": chunk["content"],
                "page_number": chunk.get("page_number"),
                "embedding": vec_str,
            }
        )

    sql = text(
        """
        INSERT INTO paper_chunks (paper_id, session_id, chunk_index, content, page_number, embedding)
        VALUES (:paper_id, :session_id, :chunk_index, :content, :page_number, CAST(:embedding AS vector))
        """
    )

    try:
        with engine.begin() as conn:
            conn.execute(sql, rows)
        logger.info(f"VectorStore.insert_chunks: session={session_id} count={len(rows)}")
        return len(rows)
    except Exception as e:
        logger.error(f"VectorStore.insert_chunks 失败: session={session_id} error={e}")
        raise RuntimeError(f"向量写入失败: {e}") from e


def search(
    query_embedding: list[float],
    session_id: str,
    top_k: int = 5,
    paper_ids: list[str] | None = None,
) -> list[dict]:
    """
    余弦相似度检索，在指定 session 中找最相近的 top_k 片段。

    Args:
        query_embedding: 查询向量（维度 1536）
        session_id:      会话 ID，进行 session 隔离，禁止跨 session 检索
        top_k:           返回结果数量，默认 5
        paper_ids:       可选，限定检索的论文 ID 列表（来自情景上下文）

    Returns:
        list[dict]: 每条包含 paper_id, chunk_index, content, page_number, distance

    Refs:
        使用 pgvector 余弦距离（<=>）进行最近邻质检索。
        session_id 必须作为过滤条件，禁止跨 session 返回数据。
    """
    vec_str = "[" + ",".join(str(v) for v in query_embedding) + "]"

    # 构造可选的 paper_id 过滤子句（情景上下文限定论文范围）
    if paper_ids:
        pid_clause = "AND paper_id IN ({})".format(
            ", ".join(f":pid_{i}" for i in range(len(paper_ids)))
        )
        pid_params = {f"pid_{i}": pid for i, pid in enumerate(paper_ids)}
    else:
        pid_clause = ""
        pid_params = {}

    sql = text(
        f"""
        SELECT {_SELECT_FIELDS},
               embedding <=> CAST(:query_vec AS vector) AS distance
        FROM paper_chunks
        WHERE session_id = :session_id
          {pid_clause}
        ORDER BY embedding <=> CAST(:query_vec AS vector)
        LIMIT :top_k
        """
    )

    try:
        with engine.connect() as conn:
            params = {"query_vec": vec_str, "session_id": session_id, "top_k": top_k}
            params.update(pid_params)
            rows = conn.execute(sql, params).fetchall()

        results = [
            {
                "paper_id": row.paper_id,
                "chunk_index": row.chunk_index,
                "content": row.content,
                "page_number": row.page_number,
                "distance": float(row.distance),
            }
            for row in rows
        ]
        logger.info(
            f"VectorStore.search: session={session_id} top_k={top_k} found={len(results)}"
        )
        return results
    except Exception as e:
        logger.error(f"VectorStore.search 失败: session={session_id} error={e}")
        raise RuntimeError(f"向量检索失败: {e}") from e


def delete_by_session(session_id: str) -> int:
    """
    删除指定 session 的所有向量数据（级联删除时由 DB 外键处理，此函数供显式调用）。

    Args:
        session_id: 会话 ID

    Returns:
        int: 删除的行数

    Refs:
        级联删除时由 DB 外键处理，此函数供显式调用。
    """
    sql = text("DELETE FROM paper_chunks WHERE session_id = :session_id")
    try:
        with engine.begin() as conn:
            result = conn.execute(sql, {"session_id": session_id})
        count = result.rowcount
        logger.info(f"VectorStore.delete_by_session: session={session_id} deleted={count}")
        return count
    except Exception as e:
        logger.error(f"VectorStore.delete_by_session 失败: session={session_id} error={e}")
        raise RuntimeError(f"向量删除失败: {e}") from e
