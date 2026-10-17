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
from ecs_agent.logging import get_logger
from ecs_agent.tools.validation import tool_call_error
from ecs_agent.tools.streaming import merge_stream_tool_calls, finalize_tool_calls
from ecs_agent.outcomes import GenerationStatus, classify_generation
from ecs_agent.observability.events import (
    LLMObservationCompletedEvent,
    LLMObservationStartedEvent,
    LLMObservationStatus,
)
from ecs_agent.types import CompletionResult, EntityId, FinishReason, Message, StreamDelta, ToolSchema, Usage


logger = get_logger(__name__)

def _wrapped_model(model: object) -> object | None:
    """Return the model wrapped by ``model``, if ``_model`` holds one.

    Providers reuse the ``_model`` attribute name for the model *name string*
    (e.g. ``OpenAIModel._model == "gpt-4o"``), so only object-like values are
    treated as a wrapper chain — descending into the string would resolve
    every unlabeled provider to ``"str"``.
    """
    wrapped: object | None = getattr(model, "_model", None)
    if wrapped is None or isinstance(wrapped, (str, bytes)):
        return None
    return wrapped


def resolve_provider_id(model: object) -> str:
    """Resolve a bounded provider label from a model or wrapper."""
    provider_id = getattr(model, "provider_id", None)
    if isinstance(provider_id, str) and provider_id:
        return provider_id
    wrapped = _wrapped_model(model)
    if wrapped is not None:
        return resolve_provider_id(wrapped)
    return type(model).__name__


def resolve_model_id(model: object) -> str:
    """Resolve a bounded model label without assuming protocol conformance."""
    model_id = getattr(model, "model_id", None)
    if isinstance(model_id, str) and model_id:
        return model_id
    wrapped = _wrapped_model(model)
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
    generation_status: GenerationStatus | None = None,
    finish_reason: str | None = None,
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
            generation_status=generation_status,
            finish_reason=finish_reason,
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
    generation_status: GenerationStatus | None = None,
    finish_reason: str | None = None,
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
            generation_status=generation_status,
            finish_reason=finish_reason,
        )
    )


class _ObservedStream(AsyncIterator[StreamDelta]):
    """Own a stream and publish exactly one terminal, including early close."""

    def __init__(
        self, *, stream: AsyncIterator[StreamDelta], event_bus: EventBus,
        entity_id: EntityId, provider_id: str, model: str, operation: str,
        messages: list[Message], tools: list[ToolSchema] | None,
        model_parameters: dict[str, Any] | None, start_time: float,
        started_at: datetime,
    ) -> None:
        self.stream = stream
        self.event_bus = event_bus
        self.entity_id = entity_id
        self.provider_id = provider_id
        self.model = model
        self.operation = operation
        self.messages = messages
        self.tools = tools
        self.model_parameters = model_parameters
        self.start_time = start_time
        self.started_at = started_at
        self.message = Message(role="assistant", content="")
        self.usage: Usage | None = None
        self.response_id: str | None = None
        self.finish_reason: str | None = None
        self.error: str | None = None
        self.closed = False
        self.tool_buffers: dict[str, dict[str, Any]] = {}

    def __aiter__(self) -> AsyncIterator[StreamDelta]:
        return self

    async def __anext__(self) -> StreamDelta:
        if self.closed:
            raise StopAsyncIteration
        try:
            delta = await anext(self.stream)
        except StopAsyncIteration:
            await self._finish(classify_generation(
                self.finish_reason, has_tool_calls=bool(self.message.tool_calls),
            ))
            raise
        except asyncio.CancelledError:
            self.finish_reason = FinishReason.CANCELLED
            await self._finish(GenerationStatus.CANCELLED)
            raise
        except Exception as exc:
            self.error = str(exc)
            self.finish_reason = FinishReason.ERROR
            await self._finish(GenerationStatus.ERROR)
            raise
        if delta.content is not None:
            self.message.content += delta.content
        if delta.reasoning_content is not None:
            self.message.reasoning_content = (
                self.message.reasoning_content or ""
            ) + delta.reasoning_content
        if delta.tool_calls:
            merge_stream_tool_calls(self.tool_buffers, delta.tool_calls)
            self.message.tool_calls = finalize_tool_calls(self.tool_buffers)
        if delta.provider_output is not None:
            self.message.provider_output = delta.provider_output
        if delta.message is not None:
            self.message = delta.message
        if delta.usage is not None:
            self.usage = delta.usage
        if delta.response_id is not None:
            self.response_id = delta.response_id
        if delta.finish_reason is not None:
            self.finish_reason = delta.finish_reason
        if delta.error is not None:
            self.error = delta.error
        return delta

    async def aclose(self) -> None:
        """Release the underlying connection even if iteration never started."""
        generation = classify_generation(
            self.finish_reason, has_tool_calls=bool(self.message.tool_calls),
        )
        if self.finish_reason is None:
            generation = GenerationStatus.CANCELLED
            self.finish_reason = FinishReason.CANCELLED
        await self._finish(generation)

    async def _finish(self, generation: GenerationStatus) -> None:
        if self.closed:
            return
        self.closed = True
        close = getattr(self.stream, "aclose", None)
        if close is not None:
            try:
                await close()
            except Exception as exc:
                logger.error("llm_stream_close_error", exception=str(exc))
                self.error = self.error or str(exc)
                generation = GenerationStatus.ERROR
        if generation is GenerationStatus.TOOL_CALLS:
            invalid = tool_call_error(self.message.tool_calls or [], self.tools)
            self.error = self.error or invalid
        if self.error is not None:
            generation = GenerationStatus.ERROR
        self.message.generation_status = generation
        self.message.finish_reason = (
            FinishReason(self.finish_reason) if self.finish_reason in set(FinishReason)
            else FinishReason.UNKNOWN
        )
        ended_at = datetime.now(timezone.utc)
        duration = time.monotonic() - self.start_time
        await publish_llm_invocation_event(
            event_bus=self.event_bus, entity_id=self.entity_id,
            provider_id=self.provider_id, model=self.model, usage=self.usage,
            stream_completeness=(self.usage.stream_completeness if self.usage is not None
                                 else StreamCompleteness.UNKNOWN),
            request_id=self.response_id, operation=self.operation,
            status=generation.invocation_status, streaming=True,
            duration_seconds=duration, generation_status=generation,
            finish_reason=self.finish_reason,
        )
        await publish_llm_observation_completed_event(
            event_bus=self.event_bus, entity_id=self.entity_id,
            provider_id=self.provider_id, model=self.model, operation=self.operation,
            messages=self.messages, tools=self.tools, streaming=True,
            model_parameters=self.model_parameters, response_message=self.message,
            reasoning_content=self.message.reasoning_content,
            usage=self.usage, response_id=self.response_id,
            status=generation.invocation_status, error=self.error,
            duration_seconds=duration, start_time=self.started_at, end_time=ended_at,
            generation_status=generation, finish_reason=self.finish_reason,
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
            generation_status=GenerationStatus.CANCELLED,
            finish_reason=FinishReason.CANCELLED,
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
            generation_status=GenerationStatus.CANCELLED,
            finish_reason=FinishReason.CANCELLED,
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
            generation_status=GenerationStatus.ERROR,
            finish_reason=FinishReason.ERROR,
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
            generation_status=GenerationStatus.ERROR,
            finish_reason=FinishReason.ERROR,
            streaming=stream,
            duration_seconds=duration_seconds,
        )
        raise

    if not isinstance(result, CompletionResult):
        return _ObservedStream(
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

    usage = result.usage
    request_id = result.response_id
    generation_status = classify_generation(
        result.finish_reason,
        has_tool_calls=bool(result.message.tool_calls),
        legacy_complete=True,
    )
    if generation_status is GenerationStatus.TOOL_CALLS:
        invalid = tool_call_error(result.message.tool_calls or [], tools)
        if invalid is not None:
            generation_status = GenerationStatus.ERROR
            result.error = invalid
    if result.error is not None:
        generation_status = GenerationStatus.ERROR
    result.message.generation_status = generation_status
    result.message.finish_reason = result.finish_reason
    duration_seconds = time.monotonic() - start_time
    ended_at = datetime.now(timezone.utc)
    await publish_llm_invocation_event(
        event_bus=event_bus,
        entity_id=entity_id,
        provider_id=provider_id,
        model=model_id,
        usage=usage,
        stream_completeness=(
            usage.stream_completeness if usage is not None else StreamCompleteness.UNKNOWN
        ),
        request_id=request_id,
        operation=operation,
        status=generation_status.invocation_status,
        generation_status=generation_status,
        finish_reason=result.finish_reason,
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
            status=generation_status.invocation_status,
            generation_status=generation_status,
            finish_reason=result.finish_reason,
            error=result.error,
            duration_seconds=duration_seconds,
            start_time=started_at,
            end_time=ended_at,
        )
        return result

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
