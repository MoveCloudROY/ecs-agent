"""Helpers for publishing low-cardinality LLM accounting events."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from typing import Any, cast

from ecs_agent.accounting.models import (
    LLMInvocationEvent,
    LLMInvocationStatus,
    StreamCompleteness,
    UsageRecord,
)
from ecs_agent.core.event_bus import EventBus
from ecs_agent.types import CompletionResult, EntityId, Message, StreamDelta, ToolSchema, Usage


def resolve_provider_id(model: object) -> str:
    """Resolve a bounded provider label from a model or wrapper."""
    provider_id = getattr(model, "provider_id", None)
    if isinstance(provider_id, str) and provider_id:
        return provider_id
    wrapped = getattr(model, "_model", None)
    if wrapped is not None:
        return resolve_provider_id(wrapped)
    return type(model).__name__


def resolve_model_id(model: object) -> str:
    """Resolve a bounded model label without assuming protocol conformance."""
    model_id = getattr(model, "model_id", None)
    if isinstance(model_id, str) and model_id:
        return model_id
    wrapped = getattr(model, "_model", None)
    if wrapped is not None:
        return resolve_model_id(wrapped)
    return type(model).__name__


def attach_retry_event_bus(model: object, event_bus: EventBus) -> None:
    """Give retry-capable wrappers access to the world's EventBus if supported."""
    setter = getattr(model, "set_event_bus", None)
    if callable(setter):
        setter(event_bus)


def usage_to_usage_record(
    *,
    usage: Usage | None,
    provider_id: str,
    model: str,
    stream_completeness: StreamCompleteness,
) -> UsageRecord:
    """Normalize provider usage into the canonical accounting shape."""
    if usage is None:
        return UsageRecord(
            provider_id=provider_id,
            model=model,
            stream_completeness=stream_completeness,
        )

    return UsageRecord(
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        total_tokens=usage.total_tokens,
        cached_input_tokens=usage.cached_input_tokens,
        cache_creation_tokens=usage.cache_creation_tokens,
        cache_read_tokens=usage.cache_read_tokens,
        image_count=usage.image_count,
        audio_seconds=usage.audio_seconds,
        provider_id=provider_id,
        model=model,
        stream_completeness=stream_completeness,
    )


async def publish_llm_invocation_event(
    *,
    event_bus: EventBus,
    entity_id: EntityId | int,
    provider_id: str,
    model: str,
    usage: Usage | None,
    stream_completeness: StreamCompleteness,
    request_id: str | None,
    operation: str,
    status: str,
    streaming: bool,
    duration_seconds: float | None,
) -> None:
    """Publish one logical LLM invocation event with normalized usage."""
    await event_bus.publish(
        LLMInvocationEvent(
            entity_id=int(entity_id),
            provider_id=provider_id,
            model=model,
            usage=usage_to_usage_record(
                usage=usage,
                provider_id=provider_id,
                model=model,
                stream_completeness=stream_completeness,
            ),
            cost=None,
            request_id=request_id,
            operation=operation,
            status=cast(LLMInvocationStatus, status),
            streaming=streaming,
            duration_seconds=duration_seconds,
        )
    )


async def complete_with_llm_invocation_event(
    *,
    event_bus: EventBus,
    entity_id: EntityId,
    model: Any,
    messages: list[Message],
    operation: str,
    tools: list[ToolSchema] | None = None,
    stream: bool = False,
    response_format: dict[str, Any] | None = None,
    extra_kwargs: dict[str, Any] | None = None,
) -> CompletionResult | AsyncIterator[StreamDelta]:
    """Call a model and publish one logical invocation terminal event."""
    attach_retry_event_bus(model, event_bus)
    provider_id = resolve_provider_id(model)
    model_id = resolve_model_id(model)
    kwargs: dict[str, Any] = {
        "messages": messages,
        "tools": tools,
    }
    if stream:
        kwargs["stream"] = stream
    if response_format is not None:
        kwargs["response_format"] = response_format
    if extra_kwargs:
        kwargs.update(extra_kwargs)

    start_time = time.monotonic()
    try:
        result = cast(
            CompletionResult | AsyncIterator[StreamDelta],
            await model.complete(**kwargs),
        )
    except asyncio.CancelledError:
        await publish_llm_invocation_event(
            event_bus=event_bus,
            entity_id=entity_id,
            provider_id=provider_id,
            model=model_id,
            usage=None,
            stream_completeness=StreamCompleteness.UNKNOWN,
            request_id=None,
            operation=operation,
            status="cancelled",
            streaming=stream,
            duration_seconds=time.monotonic() - start_time,
        )
        raise
    except Exception:
        await publish_llm_invocation_event(
            event_bus=event_bus,
            entity_id=entity_id,
            provider_id=provider_id,
            model=model_id,
            usage=None,
            stream_completeness=StreamCompleteness.UNKNOWN,
            request_id=None,
            operation=operation,
            status="error",
            streaming=stream,
            duration_seconds=time.monotonic() - start_time,
        )
        raise

    usage = result.usage if isinstance(result, CompletionResult) else None
    request_id = result.response_id if isinstance(result, CompletionResult) else None
    await publish_llm_invocation_event(
        event_bus=event_bus,
        entity_id=entity_id,
        provider_id=provider_id,
        model=model_id,
        usage=usage,
        stream_completeness=StreamCompleteness.COMPLETE,
        request_id=request_id,
        operation=operation,
        status="success",
        streaming=stream,
        duration_seconds=time.monotonic() - start_time,
    )
    return result


__all__ = [
    "attach_retry_event_bus",
    "complete_with_llm_invocation_event",
    "publish_llm_invocation_event",
    "resolve_model_id",
    "resolve_provider_id",
    "usage_to_usage_record",
]
