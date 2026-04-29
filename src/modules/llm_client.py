"""
M-08：统一 LLM 调用客户端。

支持两种后端：
  - OpenAI 兼容接口（BASE_URL + API_KEY）
  - Ollama 本地模型（model 名以 "ollama:" 前缀标识）

所有 LLM 调用必须通过此模块，禁止在其他模块直接实例化 LLM。
"""
import logging
import time

from src.utils.config_loader import get as config_get

from openai import OpenAI, APITimeoutError, APIConnectionError, APIStatusError

from src.utils.llm_logger import log_llm_call

logger = logging.getLogger(__name__)

# 超时设置（秒），超时抛出 TimeoutError
_TIMEOUT_SECONDS = 60


def _get_openai_client() -> OpenAI:
    """构造 OpenAI 兼容客户端，使用 BASE_URL + API_KEY。"""
    return OpenAI(
        api_key=config_get("api_key"),
        base_url=config_get("base_url"),
        timeout=_TIMEOUT_SECONDS,
    )


def _get_ollama_client() -> OpenAI:
    """
    构造 Ollama 客户端（使用 OpenAI 兼容接口）。
    Ollama 无需真实 API Key，使用占位符字符串。
    """
    ollama_base = config_get("ollama_base_url")
    return OpenAI(
        api_key="ollama",  # Ollama 不需要真实 key
        base_url=f"{ollama_base}/v1",
        timeout=_TIMEOUT_SECONDS,
    )


def call_llm(
    messages: list[dict],
    model: str | None = None,
    temperature: float = 0.7,
    session_id: str = "",
    event: str = "llm_call",
) -> dict:
    """
    统一 LLM 调用接口，支持 API 模型和 Ollama 本地模型。

    Args:
        messages:    OpenAI 格式的 messages 列表
        model:       模型名称；以 "ollama:" 开头时使用 Ollama（如 "ollama:llama3:latest"）
        temperature: 生成温度
        session_id:  关联的会话 ID（用于日志）
        event:       日志事件标识（用于 llm.jsonl）

    Returns:
        dict: {
            "content": str,       # LLM 返回的文本内容
            "model": str,         # 实际使用的模型名
            "usage": {
                "prompt_tokens": int,
                "completion_tokens": int,
                "total_tokens": int
            }
        }

    Raises:
        TimeoutError:  超过 30 秒未响应
        RuntimeError:  LLM 调用失败（含错误信息）

    Refs:
        返回字段：content/model/usage(prompt_tokens/completion_tokens/total_tokens)。
        崧异常：TimeoutError(超时)/RuntimeError(调用失败)。
    """
    resolved_model = model or config_get("model_name")

    # 判断后端类型：Ollama model 以 "ollama:" 开头
    if resolved_model.startswith("ollama:"):
        actual_model = resolved_model[len("ollama:"):]
        client = _get_ollama_client()
    else:
        actual_model = resolved_model
        client = _get_openai_client()

    start_ts = time.monotonic()
    content = ""
    total_tokens = 0

    try:
        logger.info(f"LLM call start: event={event} model={actual_model} session={session_id}")
        resp = client.chat.completions.create(
            model=actual_model,
            messages=messages,
            temperature=temperature,
        )
        content = resp.choices[0].message.content or ""
        usage = resp.usage
        total_tokens = usage.total_tokens if usage else 0
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0

    except APITimeoutError as e:
        latency_ms = int((time.monotonic() - start_ts) * 1000)
        logger.error(f"LLM timeout: event={event} model={actual_model} latency={latency_ms}ms")
        raise TimeoutError(f"LLM call timed out after {_TIMEOUT_SECONDS}s") from e

    except (APIConnectionError, APIStatusError) as e:
        latency_ms = int((time.monotonic() - start_ts) * 1000)
        logger.error(f"LLM error: event={event} model={actual_model} error={e}")
        raise RuntimeError(f"LLM call failed: {e}") from e

    finally:
        latency_ms = int((time.monotonic() - start_ts) * 1000)
        # 不论成功失败都记录日志（content/tokens 在失败时为空/0）
        log_llm_call(
            event=event,
            model=actual_model,
            messages=messages,
            output=content,
            tokens=total_tokens,
            latency_ms=latency_ms,
            session_id=session_id,
        )

    logger.info(
        f"LLM call done: event={event} model={actual_model} "
        f"tokens={total_tokens} latency={latency_ms}ms"
    )

    return {
        "content": content,
        "model": actual_model,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        },
    }
