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
