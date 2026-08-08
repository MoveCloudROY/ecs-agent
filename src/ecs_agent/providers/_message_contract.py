"""Provider-agnostic contract checks and vocabularies for outbound requests.

Adapters translate a ``list[Message]`` into their wire format. Translation must
stay faithful: a message the framework cannot express in the target protocol is
a defect in the history, not something an adapter may quietly drop. Dropping it
breaks the invariant ``repair_dangling_tool_spans`` establishes upstream (every
assistant ``tool_calls`` entry is answered) and leaves a dangling call the
provider either rejects or answers by hallucinating.

The finish-reason maps live here for the same reason: ``FinishReason`` is a
cross-provider vocabulary, so the dialect translations belong next to the
contract rather than inside one adapter.
"""

from __future__ import annotations

from ecs_agent.logging import get_logger
from ecs_agent.types import FinishReason, Message

logger = get_logger(__name__)


# OpenAI Chat Completions / LiteLLM dialect. "function_call" is the pre-tools
# legacy value some gateways still emit.
_OPENAI_FINISH_REASONS: dict[str, FinishReason] = {
    "stop": FinishReason.STOP,
    "length": FinishReason.LENGTH,
    "max_tokens": FinishReason.LENGTH,
    "tool_calls": FinishReason.TOOL_CALLS,
    "function_call": FinishReason.TOOL_CALLS,
    "content_filter": FinishReason.CONTENT_FILTER,
    "cancelled": FinishReason.CANCELLED,
}

# Anthropic Messages dialect (``stop_reason`` on message_delta / responses).
_ANTHROPIC_STOP_REASONS: dict[str, FinishReason] = {
    "end_turn": FinishReason.STOP,
    "stop_sequence": FinishReason.STOP,
    "max_tokens": FinishReason.LENGTH,
    "tool_use": FinishReason.TOOL_CALLS,
    "pause_turn": FinishReason.STOP,
    "refusal": FinishReason.CONTENT_FILTER,
}


def _normalize(
    raw: object, table: dict[str, FinishReason], dialect: str
) -> FinishReason | None:
    if raw is None:
        return None
    if not isinstance(raw, str) or not raw:
        return FinishReason.UNKNOWN
    mapped = table.get(raw)
    if mapped is not None:
        return mapped
    # An unrecognised value still means the provider ended the turn for a
    # reason it chose to name — report UNKNOWN rather than pretending it
    # stopped normally, and log so the vocabulary can be extended.
    logger.warning("unmapped_finish_reason", dialect=dialect, raw_value=raw)
    return FinishReason.UNKNOWN


def normalize_openai_finish_reason(raw: object) -> FinishReason | None:
    """Map an OpenAI-dialect finish_reason onto :class:`FinishReason`.

    Returns ``None`` when the provider sent no value at all (mid-stream chunks),
    which callers must distinguish from a stated reason.
    """
    return _normalize(raw, _OPENAI_FINISH_REASONS, "openai")


def normalize_anthropic_stop_reason(raw: object) -> FinishReason | None:
    """Map an Anthropic-dialect stop_reason onto :class:`FinishReason`."""
    return _normalize(raw, _ANTHROPIC_STOP_REASONS, "anthropic")


def validate_tool_spans(messages: list[Message]) -> None:
    """Reject histories an adapter cannot translate faithfully.

    A ``role="tool"`` message carries a result that only means something when
    paired with the call it answers, so a missing ``tool_call_id`` cannot be
    rendered in any provider's format. Adapters call this before building a
    request body: failing here names the offending message, whereas silently
    skipping it strands the matching call and dropping the whole span would
    hide a corrupted history.

    Args:
        messages: The outbound message list, already repaired and assembled.

    Raises:
        ValueError: If a tool message has no ``tool_call_id``.
    """
    for index, message in enumerate(messages):
        if message.role != "tool" or message.tool_call_id is not None:
            continue
        logger.error(
            "tool_message_missing_tool_call_id",
            message_index=index,
            content_length=len(message.content),
        )
        raise ValueError(
            f"Tool message requires tool_call_id (messages[{index}]); "
            "a tool result cannot be paired with its call without one."
        )
