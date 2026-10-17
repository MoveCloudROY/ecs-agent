"""Canonical Responses state identities and portable message projections."""

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

from ecs_agent.types import Message, ProviderOutput


@dataclass(slots=True)
class ResponseContext:
    """Actual outbound protocol prefix and its current message boundaries."""

    origin: str
    items: list[dict[str, Any]]
    message_ends: list[int]
    automatic_chaining: bool

    def covered_messages(self, item_count: int) -> int | None:
        """Find a current message boundary for a verified protocol prefix."""
        for index, end in enumerate(self.message_ends):
            if end == item_count:
                return index + 1
        return None


def response_context_for(model: object, messages: list[Message]) -> ResponseContext | None:
    """Resolve optional Responses capabilities through retry wrappers."""
    visited: set[int] = set()
    while id(model) not in visited:
        visited.add(id(model))
        method = getattr(model, "get_response_context", None)
        if callable(method):
            context = method(messages)
            return context if isinstance(context, ResponseContext) else None
        wrapped = getattr(model, "_model", None)
        if wrapped is None or isinstance(wrapped, str):
            break
        model = wrapped
    return None


def canonical_digest(value: Any) -> str:
    """Hash JSON structure without changing array order or opaque strings."""
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def previous_response_error(body: str) -> bool:
    """Identify errors attributed to a response reference, not echoed input."""
    def mentions_reference(text: str) -> bool:
        text = text.lower()
        return "previous_response" in text or "previous response" in text

    try:
        payload = json.loads(body)
    except ValueError:
        return mentions_reference(body)
    error = payload.get("error") if isinstance(payload, dict) else None
    if not isinstance(error, dict):
        return False
    return error.get("param") == "previous_response_id" or any(
        isinstance(error.get(key), str) and mentions_reference(error[key])
        for key in ("message", "code")
    )


def response_origin(
    *, provider_id: str, base_url: str, model: str, headers: dict[str, str]
) -> str:
    """Bind state to endpoint/model/auth scope without persisting credentials."""
    return canonical_digest({
        "provider": provider_id, "base_url": base_url.rstrip("/"), "model": model,
        "headers": {key.lower(): value for key, value in headers.items()},
    })


def message_view_hash(message: Message) -> str:
    """Detect edits to the model-visible projection of a raw output unit."""
    return canonical_digest({
        "role": message.role, "content": message.content,
        "parts": [asdict(part) for part in message.parts] if message.parts else None,
        "tool_calls": [asdict(call) for call in message.tool_calls] if message.tool_calls else None,
        "tool_call_id": message.tool_call_id,
        "reasoning_content": message.reasoning_content,
        "reasoning_signature": message.reasoning_signature,
    })


def bind_provider_output(message: Message, output: ProviderOutput) -> Message:
    """Associate a display/execution projection with its original output items."""
    output.view_hash = message_view_hash(message)
    message.provider_output = output
    return message
