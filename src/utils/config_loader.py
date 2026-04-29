"""
src/utils/config_loader.py — 运行时配置读取模块。

三层优先级（从高到低）：
  1. config.local.json（用户运行时配置）
  2. .env / os.environ（模板默认值）
  3. 硬编码默认值（最后兜底）

公开接口：
  get(key)       -> str        读取单个配置值
  get_all()      -> dict       读取全部配置（AppConfig 字典）
  set(updates)   -> None       局部更新 config.local.json
  reset()        -> None       清空 config.local.json
  get_masked()   -> dict       脱敏配置（供 CLI / API 展示）
"""
import json
import os
from pathlib import Path

# ── 路径 ──────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIG_LOCAL_PATH = _PROJECT_ROOT / "config.local.json"
_LOCK_PATH = str(CONFIG_LOCAL_PATH) + ".lock"

# ── 可配置 Key 枚举 ────────────────────────────────────────────
CONFIG_KEYS: list[str] = [
    "api_key",
    "base_url",
    "model_name",
    "embedding_model",
    "ollama_base_url",
    "ollama_model",
]

# ── ENV 变量映射 ───────────────────────────────────────────────
ENV_KEY_MAP: dict[str, str] = {
    "api_key":         "API_KEY",
    "base_url":        "BASE_URL",
    "model_name":      "MODEL_NAME",
    "embedding_model": "EMBEDDING_MODEL",
    "ollama_base_url": "OLLAMA_BASE_URL",
    "ollama_model":    "OLLAMA_MODEL",
}

# ── 兜底默认值 ─────────────────────────────────────────────────
HARDCODED_DEFAULTS: dict[str, str] = {
    "api_key":         "",
    "base_url":        "https://api.openai.com/v1",
    "model_name":      "gpt-5-nano",
    "embedding_model": "text-embedding-3-small",
    "ollama_base_url": "http://localhost:11434",
    "ollama_model":    "llama3:latest",
}

# ── filelock 降级处理 ──────────────────────────────────────────
try:
    from filelock import FileLock
    _USE_FILELOCK = True
except ImportError:
    _USE_FILELOCK = False


def _read_local() -> dict:
    """读取 config.local.json，文件不存在或解析失败均返回 {}。"""
    if not CONFIG_LOCAL_PATH.exists():
        return {}
    try:
        return json.loads(CONFIG_LOCAL_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def get(key: str) -> str:
    """
    三层优先级读取单个配置值。

    优先级：config.local.json > os.getenv(ENV_KEY_MAP[key]) > HARDCODED_DEFAULTS[key]

    Args:
        key: 必须在 CONFIG_KEYS 中，否则 raise ValueError

    Returns:
        str: 已解析的配置值，保证非 None
    """
    if key not in CONFIG_KEYS:
        raise ValueError(f"未知配置项 '{key}'，合法值：{CONFIG_KEYS}")

    # 层 1：config.local.json
    local = _read_local()
    if local.get(key):          # None / "" 均视为未设置
        return str(local[key])

    # 层 2：环境变量
    env_val = os.environ.get(ENV_KEY_MAP[key], "")
    if env_val:
        return env_val

    # 层 3：兜底默认值
    return HARDCODED_DEFAULTS[key]


def get_all() -> dict:
    """
    返回所有配置的 AppConfig 字典（所有值已解析，保证非 None）。
    """
    return {k: get(k) for k in CONFIG_KEYS}


def set(updates: dict) -> None:
    """
    局部更新 config.local.json。

    - updates 只包含需要修改的字段
    - value 为 None 或空字符串时，从文件中删除该字段（等效于清除，回退到 .env）
    - 使用原子写入防止文件损坏
    - 若 filelock 可用则加锁，否则降级为无锁写入

    Args:
        updates: {key: value} 字典，只传需要修改的字段
    """
    def _do_write():
        current = _read_local()
        for k, v in updates.items():
            if v is None or v == "":
                current.pop(k, None)    # 清除字段，回退到 .env
            else:
                current[k] = str(v)

        tmp_path = CONFIG_LOCAL_PATH.with_suffix(".json.tmp")
        tmp_path.write_text(
            json.dumps(current, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        os.replace(tmp_path, CONFIG_LOCAL_PATH)  # 原子替换

    if _USE_FILELOCK:
        with FileLock(_LOCK_PATH):
            _do_write()
    else:
        _do_write()


def reset() -> None:
    """
    清空 config.local.json（写入空对象 {}）。
    文件不存在时无操作。
    """
    if not CONFIG_LOCAL_PATH.exists():
        return

    def _do_reset():
        CONFIG_LOCAL_PATH.write_text(
            json.dumps({}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    if _USE_FILELOCK:
        with FileLock(_LOCK_PATH):
            _do_reset()
    else:
        _do_reset()


def get_masked() -> dict:
    """
    返回脱敏配置，用于 CLI show 和 API GET /api/config 展示。

    所有字段进行匿名化处理，主要显示后面位置。
    """
    local = _read_local()

    sources: dict[str, str] = {}
    result: dict = {}

    config = get_all()

    for key in CONFIG_KEYS:
        if local.get(key):
            sources[key] = "local"
        elif os.environ.get(ENV_KEY_MAP[key], ""):
            sources[key] = "env"
        else:
            sources[key] = "default"

    def _mask_val(v: str) -> str:
        if not v:
            return ""
        if len(v) <= 6:
            return "***" + v[-2:] if len(v) > 2 else "***" + v[-1:]
        return "***" + v[-8:]

    # api_key 特殊处理
    api_key_val = config["api_key"]
    if not api_key_val:
        result["api_key_preview"] = "(未设置)"
        result["api_key_set"] = False
    else:
        result["api_key_preview"] = _mask_val(api_key_val)
        result["api_key_set"] = True

    # 其余字段直接处理
    for key in CONFIG_KEYS:
        if key == "api_key":
            continue
        val = config[key]
        result[key] = _mask_val(val) if val else "(未设置)"

    result["source"] = sources
    return result
