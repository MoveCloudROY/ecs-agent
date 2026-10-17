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


def tool_span_error(
    messages: list[Message], *, allow_external_results: bool = False,
) -> str | None:
    """Check call/result identity, order and closure before full-history replay.

    Explicit Responses incremental input may begin with results for calls in
    its referenced remote response. All locally declared calls still have to
    be answered exactly once, in their declared order.
    """
    pending: list[str] = []
    calls_seen: set[str] = set()
    results_seen: set[str] = set()
    external_prefix = allow_external_results
    for index, message in enumerate(messages):
        if message.role == "system":
            continue
        if message.role == "tool":
            call_id = message.tool_call_id
            if not call_id:
                return f"Tool message requires tool_call_id (messages[{index}])"
            if call_id in results_seen:
                return f"Duplicate tool result for {call_id} (messages[{index}])"
            if pending:
                if call_id != pending[0]:
                    return f"Out-of-order or unknown tool result {call_id} (messages[{index}])"
                pending.pop(0)
            elif not external_prefix:
                return f"Orphan tool result {call_id} (messages[{index}])"
            results_seen.add(call_id)
            continue
        external_prefix = False
        if pending:
            return f"Unanswered tool calls before messages[{index}]: {pending}"
        for call in message.tool_calls or []:
            if message.role != "assistant" or not call.id or not call.name:
                return f"Invalid tool call identity (messages[{index}])"
            if call.id in calls_seen or call.id in results_seen:
                return f"Duplicate tool call {call.id} (messages[{index}])"
            calls_seen.add(call.id)
            pending.append(call.id)
    if pending:
        return f"Unanswered tool calls at end of history: {pending}"
    return None


def validate_tool_spans(
    messages: list[Message], *, allow_external_results: bool = False,
) -> None:
    """Raise ValueError for an unpaired, duplicate, or misordered tool span.

    ``allow_external_results`` is reserved for explicit incremental Responses
    input whose previous_response_id supplies the missing remote call prefix.
    """
    error = tool_span_error(messages, allow_external_results=allow_external_results)
    if error is not None:
        logger.error("invalid_tool_history", error=error)
        raise ValueError(error)
