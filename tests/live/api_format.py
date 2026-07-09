"""Shared ``LLM_API_FORMAT`` resolution for live tests.

One vocabulary across every live file: canonical :class:`ApiFormat` values
(``openai_chat_completions``, ``openai_responses``, ``anthropic_messages``)
plus the documented shorthand aliases (``openai``, ``chat``, ``responses``,
``anthropic``). Callers decide how to react to an unrecognized value
(``None``) — typically ``pytest.skip`` with the raw value in the reason.
"""

import os

from ecs_agent.providers.config import ApiFormat

_ALIASES: dict[str, ApiFormat] = {
    "openai": ApiFormat.OPENAI_CHAT_COMPLETIONS,
    "chat": ApiFormat.OPENAI_CHAT_COMPLETIONS,
    "openai_chat_completions": ApiFormat.OPENAI_CHAT_COMPLETIONS,
    "responses": ApiFormat.OPENAI_RESPONSES,
    "openai_responses": ApiFormat.OPENAI_RESPONSES,
    "anthropic": ApiFormat.ANTHROPIC_MESSAGES,
    "anthropic_messages": ApiFormat.ANTHROPIC_MESSAGES,
}


def resolve_live_api_format(
    default: ApiFormat = ApiFormat.OPENAI_CHAT_COMPLETIONS,
) -> ApiFormat | None:
    """Resolve ``LLM_API_FORMAT`` to an :class:`ApiFormat`.

    Returns ``default`` when the variable is unset or blank, and ``None``
    when it holds an unrecognized value.
    """
    raw = os.getenv("LLM_API_FORMAT")
    if raw is None or not raw.strip():
        return default
    return _ALIASES.get(raw.strip().lower())


_DASHSCOPE_CHAT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
_DASHSCOPE_RESPONSES_BASE_URL = (
    "https://dashscope.aliyuncs.com/api/v2/apps/protocols/compatible-mode/v1"
)


def live_openai_base_url(
    api_format: ApiFormat = ApiFormat.OPENAI_CHAT_COMPLETIONS,
) -> str:
    """Base URL for an OpenAI-family live test, from env with DashScope defaults.

    Precedence, most specific first: the historical ``ALIYUN_LIVE_*_BASE_URL``
    vars, then the shared ``LLM_RESPONSES_BASE_URL``/``LLM_BASE_URL``, then the
    DashScope default. This lets one ``LLM_BASE_URL`` drive every OpenAI-format
    live test against an aggregator gateway (chat -> ``/chat/completions``,
    responses -> ``/responses`` are appended by the provider), while leaving the
    historical DashScope endpoints in place when nothing is set.
    """
    if api_format is ApiFormat.OPENAI_RESPONSES:
        return (
            os.getenv("ALIYUN_LIVE_RESPONSES_BASE_URL")
            or os.getenv("LLM_RESPONSES_BASE_URL")
            or os.getenv("LLM_BASE_URL")
            or _DASHSCOPE_RESPONSES_BASE_URL
        )
    return (
        os.getenv("ALIYUN_LIVE_CHAT_BASE_URL")
        or os.getenv("LLM_BASE_URL")
        or _DASHSCOPE_CHAT_BASE_URL
    )


def live_openai_model(default: str = "qwen3.5-flash") -> str:
    """OpenAI-family live model id: ``ALIYUN_LIVE_MODEL`` -> ``LLM_MODEL`` -> default."""
    return os.getenv("ALIYUN_LIVE_MODEL") or os.getenv("LLM_MODEL") or default
