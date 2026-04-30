"""
M-09-MCP：arXiv MCP 工具封装。

使用官方 arxiv-mcp-server（通过 uvx 启动）替代原生 arxiv 库。
对外暴露与旧工具相同签名的同步接口：
  - search(query, session_id, max_results, start_date, end_date) -> dict
  - download(query, session_id) -> dict

额外备用接口（当前不接入路由，供未来扩展）：
  - read_paper(paper_id) -> str
  - list_papers() -> list[str]

降级策略：
  MCP 调用失败时抛出异常，由 agent_executor.py 捕获后 fallback 到旧工具。

MCP 返回格式（JSON）：
  search_papers → {"total_results": N, "papers": [{"id": "...", "title": "...",
                    "authors": [...], "abstract": "..."}]}
  download_paper → 论文全文文本（字符串），无异常即视为下载成功
  list_papers    → 每行一个 paper_id 的文本
  read_paper     → 论文全文文本
"""
import asyncio
import json
import re
import time
from pathlib import Path

import requests
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from src.utils.log_utils import setup_logger

logger = setup_logger(__name__)

# MCP server 本地论文存储路径（与旧工具 papers/ 目录保持一致）
_STORAGE_PATH = Path("papers").resolve()


def _get_server_params() -> StdioServerParameters:
    """构造 MCP Server 启动参数（stdio 模式，使用 uvx）。"""
    return StdioServerParameters(
        command="uvx",
        args=["arxiv-mcp-server", "--storage-path", str(_STORAGE_PATH)],
    )


async def _call_mcp_tool(tool_name: str, arguments: dict) -> str:
    """
    启动 MCP 子进程，调用指定工具，返回原始文本结果。

    每次调用独立启动子进程（无状态，简单可靠）。
    注意：子进程启动 + arXiv API 调用耗时约 30-90 秒，属正常范围。
    """
    server_params = _get_server_params()
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, arguments=arguments)
            if result.content:
                return result.content[0].text
            return ""


def _run_async(coro) -> str:
    """在同步上下文中运行 async 协程（桥接层）。"""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # 在已有事件循环中（如 Streamlit）：使用新线程避免嵌套
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(asyncio.run, coro)
                return future.result(timeout=120)
        else:
            return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


def _parse_search_results(raw_text: str) -> list[dict]:
    """
    将 MCP search_papers 返回的 JSON 文本解析为 PaperMeta 列表。

    MCP 实际返回格式（JSON）：
    {
      "total_results": 2,
      "papers": [
        {
          "id": "2201.00978v1",
          "title": "...",
          "authors": ["Author A", "Author B"],
          "abstract": "..."
        }
      ]
    }
    """
    if not raw_text or not raw_text.strip():
        return []

    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as e:
        logger.error(f"MCP search: JSON 解析失败，可能遇到限流或服务异常。原始内容前 200 字: {raw_text[:200]!r}")
        raise RuntimeError(f"MCP 返回非预期格式 (可能被限流): {raw_text[:200]}") from e

    raw_papers = data.get("papers", [])
    papers = []
    for p in raw_papers:
        try:
            paper = _map_paper(p)
            if paper:
                papers.append(paper)
        except Exception as e:
            logger.warning(f"MCP search: 单篇论文映射失败 error={e} paper={str(p)[:100]}")

    return papers


def _map_paper(p: dict) -> dict | None:
    """
    将 MCP 返回的单篇论文字典映射为系统内部 PaperMeta 格式。

    字段映射：
      id         → paper_id  （同时去掉版本号 v1 等后缀，取基础 ID）
      title      → title
      authors    → authors（已是列表）
      abstract   → abstract
      （构造）   → pdf_url, arxiv_url
    """
    raw_id = p.get("id", "").strip()
    if not raw_id:
        return None

    # 去除版本号后缀（如 2201.00978v1 → 2201.00978）
    clean_id = re.sub(r"v\d+$", "", raw_id)

    authors = p.get("authors", [])
    if isinstance(authors, str):
        authors = [a.strip() for a in authors.split(",") if a.strip()]

    published_date = p.get("published", p.get("published_date", None))

    return {
        "paper_id": clean_id,
        "title": p.get("title", ""),
        "authors": authors,
        "abstract": p.get("abstract") or None,
        "published_date": published_date or None,
        "pdf_url": f"https://arxiv.org/pdf/{clean_id}.pdf",
        "arxiv_url": f"https://arxiv.org/abs/{clean_id}",
    }


# ──────────────────────────────────────────
# 内部辅助：PDF 会话目录备份
# ──────────────────────────────────────────

def _ensure_pdf_in_session(paper_id: str, session_id: str) -> None:
    """
    确保论文 PDF 存储在 papers/{session_id}/ 目录下。

    MCP server 把文件存到 _STORAGE_PATH（项目根 papers/），
    但 RAG 管线（build_knowledge_tool）扫描的是 papers/{session_id}/*.pdf。
    此函数保证 PDF 同时存在于会话目录，不影响 MCP 自身存储。
    """
    session_dir = _STORAGE_PATH / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    filepath = session_dir / f"{paper_id.replace('/', '_')}.pdf"

    if filepath.exists():
        logger.info(f"PDF 已存在于会话目录，跳过: {filepath}")
        return

    pdf_url = f"https://arxiv.org/pdf/{paper_id}.pdf"
    logger.info(f"PDF 备份存储到会话目录中: {filepath}")
    try:
        resp = requests.get(pdf_url, stream=True, timeout=30)
        resp.raise_for_status()
        with open(filepath, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        logger.info(f"PDF 备份完成: {filepath} ({filepath.stat().st_size // 1024} KB)")
    except Exception as e:
        logger.warning(f"PDF 备份失败（不影响 MCP 主流程）: paper_id={paper_id} error={e}")


# ──────────────────────────────────────────
# 公开接口（签名与旧工具完全一致）
# ──────────────────────────────────────────

def search(
    query: str,
    session_id: str,
    max_results: int = 10,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict:
    """
    使用 arxiv-mcp-server 搜索论文。

    返回格式与 search_tool.run() 完全一致：
    {"papers": [PaperMeta...], "total": int}

    注意：由于 MCP 每次启动独立子进程并调用 arXiv API，耗时约 30-90 秒，属正常现象。
    """
    arguments: dict = {
        "query": query,
        "max_results": max_results,
        "sort_by": "relevance",
    }
    if start_date:
        arguments["date_from"] = start_date

    logger.info(
        f"MCP arxiv 搜索中... session={session_id} query={query!r} "
        f"max={max_results} start={start_date}（预计耗时 30-90 秒，请稍候）"
    )

    raw_text = _run_async(_call_mcp_tool("search_papers", arguments))
    papers = _parse_search_results(raw_text)

    logger.info(f"MCP arxiv 搜索完成：找到 {len(papers)} 篇论文 session={session_id}")
    return {"papers": papers, "total": len(papers)}


def download(query: str, session_id: str) -> dict:
    """
    使用 arxiv-mcp-server 下载论文。

    从 query 中提取 paper_id 列表，逐个调用 MCP download_paper。
    返回格式与 download_tool.run() 完全一致：
    {"downloaded": [...], "failed": [...]}

    注意：MCP download_paper 会自动优先下载 HTML 版本，失败则 fallback 到 PDF。
    """
    paper_ids = [p.strip() for p in re.split(r"[,\s]+", query) if p.strip()]

    downloaded: list[str] = []
    failed: list[str] = []

    for paper_id in paper_ids:
        try:
            logger.info(
                f"MCP arxiv 下载中... paper_id={paper_id} session={session_id}"
                f"（预计耗时 30-90 秒，请稍候）"
            )
            raw = _run_async(_call_mcp_tool("download_paper", {"paper_id": paper_id}))

            # 判断 MCP 返回是否为成功
            # MCP 可能返回 JSON（如缓存命中）或纯文本（论文全文）
            is_success = False
            try:
                resp_json = json.loads(raw)
                status = resp_json.get("status", "").lower()
                if status == "success":
                    is_success = True
                    logger.info(
                        f"MCP arxiv 缓存命中：{paper_id} message={resp_json.get('message', '')} session={session_id}"
                    )
                else:
                    # JSON 但 status 不是 success，视为失败
                    raise RuntimeError(f"MCP 返回失败状态: {raw[:200]}")
            except (json.JSONDecodeError, AttributeError):
                # 非 JSON：论文全文文本，视为下载成功
                if raw and not ("Error" in raw or "Failed" in raw or "Exception" in raw):
                    is_success = True
                else:
                    raise RuntimeError(f"MCP 返回错误: {raw[:200]}")

            if is_success:
                # 备份 PDF 到 papers/{session_id}/（RAG 管线需要）
                _ensure_pdf_in_session(paper_id, session_id)
                downloaded.append(paper_id)
                logger.info(f"MCP arxiv 下载完成：{paper_id} session={session_id}")
        except Exception as e:
            logger.error(f"MCP arxiv 下载失败: paper_id={paper_id} session={session_id} error={e}")
            failed.append(paper_id)

    logger.info(
        f"MCP arxiv 下载汇总: session={session_id} "
        f"成功={len(downloaded)} 失败={len(failed)}"
    )
    return {"downloaded": downloaded, "failed": failed}


# ──────────────────────────────────────────
# 备用接口（当前不接入路由，供未来扩展）
# ──────────────────────────────────────────

def read_paper(paper_id: str) -> str:
    """
    读取本地已下载论文的全文（Markdown 格式）。
    需要先调用 download() 确保论文已缓存。
    """
    logger.info(f"MCP arxiv read_paper: paper_id={paper_id}")
    return _run_async(_call_mcp_tool("read_paper", {"paper_id": paper_id}))


def list_papers() -> list[str]:
    """
    列出本地已缓存的所有论文 ID（来自 MCP server 的 storage-path）。
    """
    logger.info("MCP arxiv list_papers")
    raw = _run_async(_call_mcp_tool("list_papers", {}))
    # MCP list_papers 按行返回每篇论文 ID
    ids = [line.strip() for line in raw.splitlines() if line.strip()]
    return ids
