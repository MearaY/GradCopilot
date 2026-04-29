"""
M-11: 知识库构建工具。

读取 papers/{session_id}/ 下的 PDF，解析、切片、生成 embedding、写入 pgvector。
支持按文件逐一处理，分批 embedding 调用（每批 100 条），降低内存压力和 API 请求次数。
"""
import logging
from pathlib import Path

from src.utils.config_loader import get as config_get

import fitz  # PyMuPDF，用于 PDF 文本提取
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI

from src.modules.vector_store import insert_chunks

logger = logging.getLogger(__name__)

# 论文存储根目录，与 download_tool 保持一致
_PAPERS_ROOT = Path("papers")

# 文本切分参数（chunk_size=500 字符，overlap=50 字符）
_CHUNK_SIZE = 500
_CHUNK_OVERLAP = 50

# 每批 embedding 请求的最大文本数
_EMBED_BATCH_SIZE = 100


def run(session_id: str) -> dict:
    """
    M-11 构建知识库入口。

    扫描 papers/{session_id}/ 下的所有 PDF，解析文本、切片、
    调用 Embedding API 生成向量、批量写入 pgvector。

    Args:
        session_id: 会话 ID

    Returns:
        dict: {"chunks_indexed": int, "status": str}
        status 枚举：success / no_papers / partial
    """
    paper_dir = _PAPERS_ROOT / session_id
    if not paper_dir.exists():
        logger.warning(f"BuildKnowledgeTool: 目录不存在 {paper_dir}")
        return {"chunks_indexed": 0, "status": "no_papers"}

    pdf_files = list(paper_dir.glob("*.pdf"))
    if not pdf_files:
        logger.warning(f"BuildKnowledgeTool: 无 PDF 文件 session={session_id}")
        return {"chunks_indexed": 0, "status": "no_papers"}

    logger.info(f"BuildKnowledgeTool: 开始构建 session={session_id} files={len(pdf_files)}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=_CHUNK_SIZE,
        chunk_overlap=_CHUNK_OVERLAP,
    )

    all_chunks: list[dict] = []
    failed_files: list[str] = []

    for pdf_path in pdf_files:
        paper_id = pdf_path.stem.replace("_", "/")
        try:
            chunks = _parse_and_split(pdf_path, paper_id, splitter)
            all_chunks.extend(chunks)
            logger.info(
                f"BuildKnowledgeTool: 解析 {pdf_path.name} → {len(chunks)} 个 chunks"
            )
        except Exception as e:
            logger.error(f"BuildKnowledgeTool: 解析失败 file={pdf_path.name} error={e}")
            failed_files.append(pdf_path.name)

    if not all_chunks:
        return {"chunks_indexed": 0, "status": "no_papers"}

    # 生成 embedding 并写入 pgvector
    total_indexed = 0
    try:
        embeddings = _generate_embeddings([c["content"] for c in all_chunks])
        total_indexed = insert_chunks(session_id, all_chunks, embeddings)
        logger.info(
            f"BuildKnowledgeTool: 写入完成 session={session_id} indexed={total_indexed}"
        )
    except Exception as e:
        logger.error(f"BuildKnowledgeTool: 写入向量库失败 session={session_id} error={e}")
        raise RuntimeError(f"知识库构建失败: {e}") from e

    status = "partial" if failed_files else "success"
    return {"chunks_indexed": total_indexed, "status": status}


def _parse_and_split(
    pdf_path: Path,
    paper_id: str,
    splitter: RecursiveCharacterTextSplitter,
) -> list[dict]:
    """
    解析 PDF 并切片，返回 chunk dict 列表。

    每个 chunk 包含：paper_id, chunk_index, content, page_number。
    """
    doc = fitz.open(str(pdf_path))
    chunk_list = []
    global_index = 0

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text").strip()
        if not text:
            continue

        page_chunks = splitter.split_text(text)
        for chunk_text in page_chunks:
            if not chunk_text.strip():
                continue
            # 过滤 NUL 字节（\x00），psycopg2 不允许字符串包含 NUL
            clean_chunk = chunk_text.replace("\x00", "")
            if not clean_chunk.strip():
                continue
            chunk_list.append(
                {
                    "paper_id": paper_id,
                    "chunk_index": global_index,
                    "content": clean_chunk,
                    "page_number": page_num + 1,  # 从 1 开始 [REF: schema.md#1.3]
                }
            )
            global_index += 1

    doc.close()
    return chunk_list


def _generate_embeddings(texts: list[str]) -> list[list[float]]:
    """
    调用 OpenAI Embedding API 生成向量，分批处理。

    Args:
        texts: 文本列表

    Returns:
        list[list[float]]: 与 texts 一一对应的向量列表（维度 1536，text-embedding-3-small）
    """
    client = OpenAI(
        api_key=config_get("api_key"),
        base_url=config_get("base_url"),
    )
    embedding_model = config_get("embedding_model")

    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), _EMBED_BATCH_SIZE):
        batch = texts[i: i + _EMBED_BATCH_SIZE]
        resp = client.embeddings.create(model=embedding_model, input=batch)
        batch_embeddings = [item.embedding for item in resp.data]
        all_embeddings.extend(batch_embeddings)
        logger.info(
            f"BuildKnowledgeTool: embedding batch {i // _EMBED_BATCH_SIZE + 1} "
            f"({len(batch)} texts)"
        )

    return all_embeddings
