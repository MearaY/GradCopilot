"""
components/settings_panel.py — 运行时配置设置面板。

通过后端 /api/config 接口读写配置，不直接操作 config.local.json。
"""
import os

import requests
import streamlit as st

# 后端地址与 api_client 保持一致
_BACKEND = os.environ.get("BACKEND_URL", "http://localhost:8000").rstrip("/")
_TIMEOUT = 10


def _get_config() -> dict | None:
    """调用 GET /api/config，失败返回 None。"""
    try:
        resp = requests.get(f"{_BACKEND}/api/config", timeout=_TIMEOUT)
        if resp.ok:
            return resp.json()
    except Exception:
        pass
    return None


def _update_config(updates: dict) -> tuple[bool, str]:
    """调用 PUT /api/config，返回 (success, message)。"""
    try:
        resp = requests.put(f"{_BACKEND}/api/config", json=updates, timeout=_TIMEOUT)
        if resp.ok:
            data = resp.json()
            fields = data.get("updated_fields", [])
            return True, f"已更新：{', '.join(fields)}"
        return False, f"请求失败 ({resp.status_code})"
    except Exception as e:
        return False, f"连接后端失败：{e}"


def _reset_config() -> tuple[bool, str]:
    """调用 DELETE /api/config，返回 (success, message)。"""
    try:
        resp = requests.delete(f"{_BACKEND}/api/config", timeout=_TIMEOUT)
        if resp.ok:
            return True, resp.json().get("message", "已重置")
        return False, f"请求失败 ({resp.status_code})"
    except Exception as e:
        return False, f"连接后端失败：{e}"


def _check_health() -> str:
    """调用 GET /api/health，返回状态字符串。"""
    try:
        resp = requests.get(f"{_BACKEND}/api/health", timeout=5)
        if resp.ok:
            data = resp.json()
            overall = data.get("status", "unknown")
            llm = data.get("llm_api", "?")
            pg  = data.get("postgres", "?")
            return f"整体: {overall} | LLM: {llm} | PostgreSQL: {pg}"
        return f"健康检查失败 ({resp.status_code})"
    except Exception as e:
        return f"连接失败：{e}"


_SOURCE_BADGE = {
    "local":   "🟢 local",
    "env":     "🔵 env",
    "default": "⚪ default",
}


def render_settings():
    """渲染「⚙️ 设置」面板。"""
    st.header("⚙️ 运行时配置")
    st.caption("修改配置后无需重启服务，下一次请求立即生效。")

    # ── 读取当前配置 ─────────────────────────────────────────
    cfg = _get_config()
    if cfg is None:
        st.error("❌ 无法连接后端，请确认 FastAPI 服务已启动。")
        return

    source = cfg.get("source", {})

    # ── 当前配置展示 ─────────────────────────────────────────
    st.subheader("当前生效配置")

    col1, col2 = st.columns([3, 1])
    with col1:
        api_status = "✅ 已设置" if cfg.get("api_key_set") else "⚠️ 未设置"
        st.metric("API Key", cfg.get("api_key_preview", "(未设置)"), api_status)
    with col2:
        st.caption(f"来源：{_SOURCE_BADGE.get(source.get('api_key', 'default'), '')}")

    rows = [
        ("LLM Base URL",    "base_url"),
        ("LLM Model",       "model_name"),
        ("Embedding Model", "embedding_model"),
        ("Ollama URL",      "ollama_base_url"),
        ("Ollama Model",    "ollama_model"),
    ]
    for label, key in rows:
        c1, c2 = st.columns([3, 1])
        with c1:
            st.text_input(label, value=cfg.get(key, ""), disabled=True, key=f"_disp_{key}")
        with c2:
            st.caption(f"来源：{_SOURCE_BADGE.get(source.get(key, 'default'), '')}")

    st.divider()

    # ── 编辑区 ───────────────────────────────────────────────
    st.subheader("修改配置")
    st.info("留空表示不修改该项；填入内容将覆盖当前设置。", icon="ℹ️")

    with st.form("config_form"):
        new_api_key    = st.text_input("API Key",          type="password", placeholder="sk-...")
        new_base_url   = st.text_input("LLM Base URL",     placeholder="https://api.openai.com/v1")
        new_model      = st.text_input("LLM Model",        placeholder="gpt-5-nano")
        new_emb_model  = st.text_input("Embedding Model",  placeholder="text-embedding-3-small")
        new_ollama_url = st.text_input("Ollama URL",       placeholder="http://localhost:11434")
        new_ollama_mod = st.text_input("Ollama Model",     placeholder="llama3:latest")

        submitted = st.form_submit_button("💾 保存配置", type="primary")

    if submitted:
        updates = {}
        if new_api_key:    updates["api_key"]         = new_api_key
        if new_base_url:   updates["base_url"]        = new_base_url
        if new_model:      updates["model_name"]      = new_model
        if new_emb_model:  updates["embedding_model"] = new_emb_model
        if new_ollama_url: updates["ollama_base_url"] = new_ollama_url
        if new_ollama_mod: updates["ollama_model"]    = new_ollama_mod

        if updates:
            ok, msg = _update_config(updates)
            if ok:
                st.success(f"✅ {msg}")
                st.rerun()
            else:
                st.error(f"❌ {msg}")
        else:
            st.warning("未填写任何内容")

    st.divider()

    # ── 操作按钮 ─────────────────────────────────────────────
    col_reset, col_check = st.columns(2)

    with col_reset:
        if st.button("🔄 重置为 .env 默认值", use_container_width=True):
            ok, msg = _reset_config()
            if ok:
                st.success(f"✅ {msg}")
                st.rerun()
            else:
                st.error(f"❌ {msg}")

    with col_check:
        if st.button("🩺 验证当前配置可用性", use_container_width=True):
            result = _check_health()
            if "ok" in result:
                st.success(result)
            else:
                st.warning(result)
