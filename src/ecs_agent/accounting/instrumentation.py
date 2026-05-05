"""Helpers for publishing low-cardinality LLM accounting events."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from typing import Any, cast

from ecs_agent.accounting.models import (
    LLMInvocationEvent,
    LLMInvocationStatus,
    StreamCompleteness,
    UsageRecord,
)
from ecs_agent.core.event_bus import EventBus
from ecs_agent.observability.events import (
    LLMObservationCompletedEvent,
    LLMObservationStartedEvent,
    LLMObservationStatus,
)
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


async def publish_llm_observation_started_event(
    *,
    event_bus: EventBus,
    entity_id: EntityId | int,
    provider_id: str,
    model: str,
    operation: str,
    messages: list[Message],
    tools: list[ToolSchema] | None,
    streaming: bool,
    model_parameters: dict[str, Any] | None = None,
) -> None:
    """Publish a raw LLM generation observation start event."""
    await event_bus.publish(
        LLMObservationStartedEvent(
            entity_id=entity_id,
            provider_id=provider_id,
            model=model,
            operation=operation,
            messages=list(messages),
            tools=list(tools) if tools is not None else None,
            streaming=streaming,
            model_parameters=model_parameters,
        )
    )


async def publish_llm_observation_completed_event(
    *,
    event_bus: EventBus,
    entity_id: EntityId | int,
    provider_id: str,
    model: str,
    operation: str,
    messages: list[Message],
    tools: list[ToolSchema] | None,
    streaming: bool,
    model_parameters: dict[str, Any] | None = None,
    response_message: Message | None = None,
    reasoning_content: str | None = None,
    usage: Usage | None = None,
    response_id: str | None = None,
    status: str = "success",
    error: str | None = None,
    duration_seconds: float | None = None,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    cost_details: dict[str, Any] | None = None,
) -> None:
    """Publish a raw LLM generation observation completion event."""
    await event_bus.publish(
        LLMObservationCompletedEvent(
            entity_id=entity_id,
            provider_id=provider_id,
            model=model,
            operation=operation,
            messages=list(messages),
            tools=list(tools) if tools is not None else None,
            streaming=streaming,
            model_parameters=model_parameters,
            response_message=response_message,
            reasoning_content=reasoning_content,
            usage=usage,
            response_id=response_id,
            status=cast(LLMObservationStatus, status),
            error=error,
            duration_seconds=duration_seconds,
            start_time=start_time,
            end_time=end_time,
            cost_details=cost_details or {},
        )
    )


async def _observe_stream_result(
    *,
    stream: AsyncIterator[StreamDelta],
    event_bus: EventBus,
    entity_id: EntityId,
    provider_id: str,
    model: str,
    operation: str,
    messages: list[Message],
    tools: list[ToolSchema] | None,
    model_parameters: dict[str, Any] | None,
    start_time: float,
    started_at: datetime,
) -> AsyncIterator[StreamDelta]:
    content_chunks: list[str] = []
    reasoning_chunks: list[str] = []
    usage: Usage | None = None
    response_id: str | None = None
    status = "success"
    error: str | None = None

    try:
        async for delta in stream:
            if delta.content is not None:
                content_chunks.append(delta.content)
            if delta.reasoning_content is not None:
                reasoning_chunks.append(delta.reasoning_content)
            if delta.usage is not None:
                usage = delta.usage
            if delta.response_id is not None:
                response_id = delta.response_id
            yield delta
    except asyncio.CancelledError:
        status = "cancelled"
        raise
    except Exception as exc:
        status = "error"
        error = str(exc)
        raise
    finally:
        ended_at = datetime.now(timezone.utc)
        await publish_llm_observation_completed_event(
            event_bus=event_bus,
            entity_id=entity_id,
            provider_id=provider_id,
            model=model,
            operation=operation,
            messages=messages,
            tools=tools,
            streaming=True,
            model_parameters=model_parameters,
            response_message=Message(role="assistant", content="".join(content_chunks)),
            reasoning_content="".join(reasoning_chunks) or None,
            usage=usage,
            response_id=response_id,
            status=status,
            error=error,
            duration_seconds=time.monotonic() - start_time,
            start_time=started_at,
            end_time=ended_at,
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
    model_parameters: dict[str, Any] = {}
    if response_format is not None:
        model_parameters["response_format"] = response_format
    if extra_kwargs:
        model_parameters.update(extra_kwargs)

    start_time = time.monotonic()
    started_at = datetime.now(timezone.utc)
    await publish_llm_observation_started_event(
        event_bus=event_bus,
        entity_id=entity_id,
        provider_id=provider_id,
        model=model_id,
        operation=operation,
        messages=messages,
        tools=tools,
        streaming=stream,
        model_parameters=model_parameters or None,
    )
    try:
        result = cast(
            CompletionResult | AsyncIterator[StreamDelta],
            await model.complete(**kwargs),
        )
    except asyncio.CancelledError:
        duration_seconds = time.monotonic() - start_time
        ended_at = datetime.now(timezone.utc)
        await publish_llm_observation_completed_event(
            event_bus=event_bus,
            entity_id=entity_id,
            provider_id=provider_id,
            model=model_id,
            operation=operation,
            messages=messages,
            tools=tools,
            streaming=stream,
            model_parameters=model_parameters or None,
            status="cancelled",
            duration_seconds=duration_seconds,
            start_time=started_at,
            end_time=ended_at,
        )
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
            duration_seconds=duration_seconds,
        )
        raise
    except Exception as exc:
        duration_seconds = time.monotonic() - start_time
        ended_at = datetime.now(timezone.utc)
        await publish_llm_observation_completed_event(
            event_bus=event_bus,
            entity_id=entity_id,
            provider_id=provider_id,
            model=model_id,
            operation=operation,
            messages=messages,
            tools=tools,
            streaming=stream,
            model_parameters=model_parameters or None,
            status="error",
            error=str(exc),
            duration_seconds=duration_seconds,
            start_time=started_at,
            end_time=ended_at,
        )
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
            duration_seconds=duration_seconds,
        )
        raise

    usage = result.usage if isinstance(result, CompletionResult) else None
    request_id = result.response_id if isinstance(result, CompletionResult) else None
    duration_seconds = time.monotonic() - start_time
    ended_at = datetime.now(timezone.utc)
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
        duration_seconds=duration_seconds,
    )
    if isinstance(result, CompletionResult):
        await publish_llm_observation_completed_event(
            event_bus=event_bus,
            entity_id=entity_id,
            provider_id=provider_id,
            model=model_id,
            operation=operation,
            messages=messages,
            tools=tools,
            streaming=stream,
            model_parameters=model_parameters or None,
            response_message=result.message,
            reasoning_content=result.reasoning_content,
            usage=result.usage,
            response_id=result.response_id,
            status="success",
            duration_seconds=duration_seconds,
            start_time=started_at,
            end_time=ended_at,
        )
        return result

    return _observe_stream_result(
        stream=result,
        event_bus=event_bus,
        entity_id=entity_id,
        provider_id=provider_id,
        model=model_id,
        operation=operation,
        messages=messages,
        tools=tools,
        model_parameters=model_parameters or None,
        start_time=start_time,
        started_at=started_at,
    )


__all__ = [
    "attach_retry_event_bus",
    "complete_with_llm_invocation_event",
    "publish_llm_observation_completed_event",
    "publish_llm_observation_started_event",
    "publish_llm_invocation_event",
    "resolve_model_id",
    "resolve_provider_id",
    "usage_to_usage_record",
]
