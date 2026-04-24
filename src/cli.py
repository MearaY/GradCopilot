"""
src/cli.py — GradCopilot Phase 2 CLI 模式。

通过 REST API 驱动后端（与 Streamlit 共享同一套接口），
无需直接操作数据库或 LangChain。

使用方式：
    python src/cli.py                         # 交互模式
    python src/cli.py --session <session_id>  # 使用指定会话
    python src/cli.py --base-url http://localhost:8000

命令（进入后输入 /help 查看）：
    /help           帮助
    /new [名称]     新建会话
    /sessions       列出所有会话
    /switch <id>    切换会话
    /search <词>    搜索 arXiv 论文
    /download <n>   下载搜索结果第 n 篇（可逗号分隔多个）
    /build          构建向量知识库
    /history        查看当前会话历史
    /delete         删除当前会话
    /quit           退出
    <直接输入>      发送消息到 Agent
"""
import argparse
import json
import os
import sys
from typing import Optional

import requests
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).parent.parent / ".env")

# ── 配置 ──────────────────────────────────────────────────────
DEFAULT_BASE_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")
CHAT_TIMEOUT = 120
DEFAULT_TIMEOUT = 15
SEARCH_TIMEOUT = 30

# ── ANSI 颜色 ──────────────────────────────────────────────────
_C = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "cyan": "\033[96m",
    "green": "\033[92m",
    "yellow": "\033[93m",
    "red": "\033[91m",
    "blue": "\033[94m",
    "gray": "\033[90m",
}


def c(color: str, text: str) -> str:
    return f"{_C.get(color, '')}{text}{_C['reset']}"


def _hr(char: str = "─", width: int = 60) -> str:
    return char * width


# ── HTTP 工具 ──────────────────────────────────────────────────

class APIError(Exception):
    def __init__(self, status: int, code: str, msg: str):
        super().__init__(msg)
        self.status = status
        self.code = code
        self.msg = msg


def _call(method: str, url: str, timeout: int = DEFAULT_TIMEOUT, **kwargs) -> dict:
    try:
        resp = requests.request(method, url, timeout=timeout, **kwargs)
    except requests.exceptions.ConnectionError:
        print(c("red", f"\n✗ 无法连接后端 {url.split('/api')[0]}，请确认 uvicorn 已运行。"))
        sys.exit(1)
    except requests.exceptions.Timeout:
        raise TimeoutError(f"请求超时（>{timeout}s）")

    try:
        data = resp.json()
    except Exception:
        data = {}

    if not resp.ok:
        detail = data.get("detail", {})
        if isinstance(detail, dict):
            err = detail.get("error", {})
            raise APIError(resp.status_code, err.get("code", "ERR"), err.get("message", str(detail)))
        raise APIError(resp.status_code, "ERR", str(detail) or resp.text)

    return data


# ── CLI 主类 ────────────────────────────────────────────────────

class GradCopilotCLI:
    def __init__(self, base_url: str, initial_session: Optional[str] = None):
        self.base = base_url.rstrip("/")
        self.session_id: Optional[str] = initial_session
        self.session_name: str = ""
        self.search_results: list[dict] = []   # 最近一次搜索结果缓存

    # ── 会话管理 ─────────────────────────────────────────────

    def _ensure_session(self) -> bool:
        if not self.session_id:
            print(c("yellow", "⚠  请先新建或选择一个会话（/new 或 /sessions）"))
            return False
        return True

    def cmd_new_session(self, name: str = ""):
        """新建会话并自动切换进去。"""
        name = name.strip() or "新会话"
        data = _call("POST", f"{self.base}/api/sessions/create",
                     json={"name": name})
        self.session_id = data["session_id"]
        self.session_name = data.get("name", name)
        print(c("green", f"✓ 会话已创建：{self.session_name}  (id={self.session_id})"))

    def cmd_list_sessions(self):
        """列出所有会话。"""
        data = _call("GET", f"{self.base}/api/sessions")
        sessions = data.get("sessions", [])
        if not sessions:
            print(c("gray", "  暂无会话"))
            return
        print(f"\n{'ID':20} {'名称':20} {'消息数':>6}  创建时间")
        print(_hr())
        for s in sessions:
            mark = " ←" if s["session_id"] == self.session_id else ""
            print(
                f"{s['session_id']:20} {(s.get('name') or '-'):20} "
                f"{s.get('message_count', 0):>6}  "
                f"{(s.get('created_at') or '')[:19]}{c('cyan', mark)}"
            )
        print()

    def cmd_switch_session(self, sid: str):
        """切换到已有会话。"""
        data = _call("GET", f"{self.base}/api/sessions")
        sessions = {s["session_id"]: s for s in data.get("sessions", [])}
        if sid not in sessions:
            print(c("red", f"✗ 会话 {sid} 不存在"))
            return
        self.session_id = sid
        self.session_name = sessions[sid].get("name", sid)
        print(c("green", f"✓ 已切换到：{self.session_name}  (id={sid})"))

    def cmd_delete_session(self):
        """删除当前会话。"""
        if not self._ensure_session():
            return
        confirm = input(c("yellow", f"  确认删除会话 [{self.session_id}]？(y/N) ")).strip().lower()
        if confirm != "y":
            print("  已取消")
            return
        _call("DELETE", f"{self.base}/api/sessions/{self.session_id}")
        print(c("green", f"✓ 会话 {self.session_id} 已删除"))
        self.session_id = None
        self.session_name = ""

    def cmd_history(self):
        """显示当前会话历史。"""
        if not self._ensure_session():
            return
        data = _call("GET", f"{self.base}/api/sessions/{self.session_id}/history",
                     params={"limit": 20})
        history = data.get("history", [])
        if not history:
            print(c("gray", "  （暂无历史记录）"))
            return
        print()
        for msg in history:
            role = msg.get("role", "?")
            content = msg.get("content", "")
            prefix = c("cyan", "You") if role == "user" else c("green", "AI ")
            print(f"{prefix}  {content[:200]}")
        print()

    # ── 论文工具 ─────────────────────────────────────────────

    def cmd_search(self, query: str):
        """搜索 arXiv 论文。"""
        if not self._ensure_session():
            return
        if not query.strip():
            query = input("  搜索关键词：").strip()
        if not query:
            return

        max_r_str = input("  最大结果数（默认 10）：").strip()
        max_r = int(max_r_str) if max_r_str.isdigit() else 10

        start_date_str = input("  开始年份或日期（如 2026 或 2026-01-01，回车跳过）：").strip() or None
        end_date_str   = input("  结束年份或日期（如 2026 或 2026-12-31，回车跳过）：").strip() or None

        # 将纯 4 位年份补全为 YYYY-MM-DD 格式
        def _norm_date(d: str | None, is_start: bool) -> str | None:
            if not d:
                return None
            if len(d) == 4 and d.isdigit():
                return f"{d}-01-01" if is_start else f"{d}-12-31"
            return d

        start_date = _norm_date(start_date_str, is_start=True)
        end_date   = _norm_date(end_date_str,   is_start=False)

        print(c("gray", "  搜索中…"))
        try:
            data = _call(
                "POST", f"{self.base}/api/papers/search",
                timeout=SEARCH_TIMEOUT,
                json={
                    "session_id": self.session_id,
                    "query": query,
                    "max_results": max_r,
                    "start_date": start_date,
                    "end_date": end_date,
                },
            )
        except TimeoutError:
            print(c("yellow", "  搜索超时，arXiv 网络较慢，请稍后重试"))
            return

        papers = data.get("papers", [])
        self.search_results = papers
        total = data.get("total", len(papers))

        print(f"\n{c('bold', f'找到 {total} 篇论文（显示前 {len(papers)} 篇）')}")
        print(_hr())
        for i, p in enumerate(papers, 1):
            title = p.get("title", "（无标题）")
            authors = p.get("authors", [])
            author_str = ", ".join(authors[:2]) + ("…" if len(authors) > 2 else "")
            date = p.get("published_date", "")
            print(f"  {c('cyan', f'[{i}]')} {title}")
            print(f"      {c('gray', author_str + '  ' + date)}")
        print()

    def cmd_download(self, nums_str: str):
        """下载搜索结果中指定序号的论文。"""
        if not self._ensure_session():
            return
        if not self.search_results:
            print(c("yellow", "  请先执行 /search 搜索论文"))
            return

        if not nums_str.strip():
            nums_str = input("  输入序号（逗号分隔，如 1,2,3）：").strip()
        if not nums_str:
            return

        indices = []
        for n in nums_str.split(","):
            try:
                idx = int(n.strip())
                if 1 <= idx <= len(self.search_results):
                    indices.append(idx - 1)
                else:
                    print(c("yellow", f"  序号 {idx} 超出范围，已跳过"))
            except ValueError:
                pass

        if not indices:
            print(c("red", "  没有有效序号"))
            return

        paper_ids = [self.search_results[i]["paper_id"] for i in indices]
        titles = [self.search_results[i].get("title", paper_ids[i])[:50] for i in indices]
        print(c("gray", f"  下载 {len(paper_ids)} 篇：{', '.join(titles)}"))
        print(c("gray", "  下载中（可能需要 1-3 分钟）…"))

        try:
            data = _call(
                "POST", f"{self.base}/api/papers/download",
                timeout=180,
                json={"session_id": self.session_id, "paper_ids": paper_ids},
            )
        except TimeoutError:
            print(c("yellow", "  下载超时，请检查网络"))
            return

        downloaded = data.get("downloaded", [])
        failed = data.get("failed", [])
        if downloaded:
            print(c("green", f"  ✓ 成功 {len(downloaded)} 篇"))
        if failed:
            print(c("yellow", f"  ⚠ 失败 {len(failed)} 篇：{failed}"))

    def cmd_build(self):
        """构建当前会话的向量知识库。"""
        if not self._ensure_session():
            return
        print(c("gray", "  解析 PDF + 生成向量中，请稍候…"))
        try:
            data = _call(
                "POST", f"{self.base}/api/knowledge/build",
                timeout=600,
                json={"session_id": self.session_id},
            )
        except TimeoutError:
            print(c("yellow", "  构建超时，PDF 较多时请耐心等待"))
            return

        status = data.get("status", "?")
        chunks = data.get("chunks_indexed", 0)
        if status == "success":
            print(c("green", f"  ✓ 知识库构建成功，写入 {chunks} 个 chunks"))
        elif status == "partial":
            print(c("yellow", f"  ⚠ 部分成功，写入 {chunks} 个 chunks，部分文件失败"))
        elif status == "no_papers":
            print(c("yellow", "  当前会话下没有已下载的 PDF，请先 /search + /download"))
        else:
            print(c("red", f"  ✗ 状态未知：{status}"))

    # ── 对话 ─────────────────────────────────────────────────

    def cmd_chat(self, message: str):
        """发送消息给 Agent，显示回答和元信息。"""
        if not self._ensure_session():
            return
        print(c("gray", "  思考中…"))
        try:
            data = _call(
                "POST", f"{self.base}/api/agent/chat",
                timeout=CHAT_TIMEOUT,
                json={"session_id": self.session_id, "message": message},
            )
        except TimeoutError:
            print(c("yellow", "  LLM 响应超时，请重试"))
            return
        except APIError as e:
            print(c("red", f"  ✗ {e.code}：{e.msg}"))
            return

        response = data.get("response", "（无回答）")
        intent = data.get("intent", "-")
        route = data.get("route", "-")
        tool = data.get("tool_used") or "-"
        sources = data.get("sources", [])
        model = data.get("model_used", "-")
        tokens = data.get("tokens_used", 0)

        print(f"\n{c('green', '◆ AI')}")
        print(f"{response}")
        print()

        # 元信息（折叠风格）
        meta = f"intent={intent}  route={route}  tool={tool}  model={model}  tokens={tokens}"
        print(c("gray", f"  [{meta}]"))
        if sources:
            print(c("gray", f"  来源：{' | '.join(sources[:3])}"))
        print()

    # ── 帮助 ─────────────────────────────────────────────────

    def cmd_help(self):
        print(f"""
{c('bold', 'GradCopilot Phase 2 CLI — 命令列表')}
{_hr()}
  /new [名称]         新建会话（自动切换进去）
  /sessions           列出所有会话
  /switch <id>        切换到已有会话
  /delete             删除当前会话
  /history            查看当前会话对话历史

  /search [关键词]    搜索 arXiv 论文
  /download [序号]    下载搜索结果（序号逗号分隔，如 1,2,3）
  /build              构建/更新向量知识库

  /help               显示帮助
  /quit  /exit        退出

  {c('cyan', '直接输入')} 即为发送消息给 Agent
{_hr()}
""")

    # ── REPL 主循环 ──────────────────────────────────────────

    def _prompt(self) -> str:
        sid_short = f"{self.session_id[:8]}…" if self.session_id else "无会话"
        name = f"[{self.session_name}]" if self.session_name else ""
        return f"{c('cyan', 'GradCopilot')} {c('gray', sid_short)}{c('cyan', name)} > "

    def run(self):
        print(f"""
{c('bold', '=' * 60)}
{c('bold', '  GradCopilot Phase 2 CLI')}
{c('gray', '  Agentic RAG 学术研究助手')}
{c('gray', f'  后端：{self.base}')}
{c('bold', '=' * 60)}
输入 /help 查看命令，直接输入问题与 Agent 对话。
""")
        # 验证后端可达
        try:
            _call("GET", f"{self.base}/api/health", timeout=8)
            print(c("green", "✓ 后端连接正常\n"))
        except SystemExit:
            raise
        except Exception as e:
            print(c("yellow", f"⚠ 后端健康检查警告：{e}\n"))

        # 如果指定了 session_id，尝试加载
        if self.session_id:
            try:
                data = _call("GET", f"{self.base}/api/sessions")
                sessions = {s["session_id"]: s for s in data.get("sessions", [])}
                if self.session_id in sessions:
                    self.session_name = sessions[self.session_id].get("name", "")
                    print(c("green", f"✓ 已加载会话：{self.session_name or self.session_id}\n"))
                else:
                    print(c("yellow", f"⚠ 会话 {self.session_id} 不存在，将从空状态开始\n"))
                    self.session_id = None
            except Exception:
                pass

        while True:
            try:
                line = input(self._prompt()).strip()
            except (EOFError, KeyboardInterrupt):
                print("\n再见！")
                break

            if not line:
                continue

            if line.startswith("/"):
                parts = line[1:].split(maxsplit=1)
                cmd = parts[0].lower()
                arg = parts[1] if len(parts) > 1 else ""

                dispatch = {
                    "help": lambda: self.cmd_help(),
                    "new": lambda: self.cmd_new_session(arg),
                    "sessions": lambda: self.cmd_list_sessions(),
                    "switch": lambda: self.cmd_switch_session(arg),
                    "delete": lambda: self.cmd_delete_session(),
                    "history": lambda: self.cmd_history(),
                    "search": lambda: self.cmd_search(arg),
                    "download": lambda: self.cmd_download(arg),
                    "build": lambda: self.cmd_build(),
                    "quit": None,
                    "exit": None,
                }

                if cmd in ("quit", "exit"):
                    print("再见！")
                    break
                elif cmd in dispatch:
                    try:
                        dispatch[cmd]()
                    except APIError as e:
                        print(c("red", f"  ✗ API错误 {e.code}：{e.msg}"))
                    except Exception as e:
                        print(c("red", f"  ✗ 错误：{e}"))
                else:
                    print(c("yellow", f"  未知命令 /{cmd}，输入 /help 查看帮助"))
            else:
                # 直接对话
                try:
                    self.cmd_chat(line)
                except APIError as e:
                    print(c("red", f"  ✗ {e.code}：{e.msg}"))
                except Exception as e:
                    print(c("red", f"  ✗ {e}"))


# ── 入口 ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="GradCopilot Phase 2 CLI — Agentic RAG 学术研究助手"
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"后端地址（默认 {DEFAULT_BASE_URL}）",
    )
    parser.add_argument(
        "--session",
        default=None,
        help="启动时加载的 session_id（可选）",
    )
    args = parser.parse_args()

    cli = GradCopilotCLI(base_url=args.base_url, initial_session=args.session)
    cli.run()


if __name__ == "__main__":
    main()
