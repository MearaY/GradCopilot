"""
错误码枚举常量。

定义所有 API 错误响应的 code 字段枚举，禁止在 ErrorCode 类外自定义错误码。
"""


class ErrorCode:
    """所有 API 错误响应的 code 字段枚举。禁止在此类外自定义错误码。"""

    VALIDATION_ERROR = "VALIDATION_ERROR"   # HTTP 422：请求参数校验失败
    SESSION_NOT_FOUND = "SESSION_NOT_FOUND" # HTTP 404：会话不存在
    LLM_ERROR = "LLM_ERROR"                # HTTP 500：LLM 调用失败
    TOOL_ERROR = "TOOL_ERROR"              # HTTP 500：工具执行失败
    TIMEOUT_ERROR = "TIMEOUT_ERROR"        # HTTP 504：操作超时
    INTERNAL_ERROR = "INTERNAL_ERROR"      # HTTP 500：未分类内部错误
