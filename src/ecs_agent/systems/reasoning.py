"""LLM inference system."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from ecs_agent.accounting.models import (
    StreamCompleteness,
    UsageRecord,
)
from ecs_agent.accounting.instrumentation import (
    attach_retry_event_bus,
    publish_llm_observation_completed_event,
    publish_llm_observation_started_event,
    publish_llm_invocation_event,
    resolve_provider_id,
)
from ecs_agent.logging import get_logger
from ecs_agent.components import (
    ChildStubComponent,
    ConversationComponent,
    ConversationTreeComponent,
    ContextTrimConfig,
    ContextEntry,
    ErrorComponent,
    InterruptionComponent,
    LLMComponent,
    PendingToolCallsComponent,
    PromptContextQueueComponent,
    PromptContextReservationComponent,
    ResponsesAPIStateComponent,
    RunnerStateComponent,
    StreamingComponent,
    TerminalComponent,
    TokenUsageComponent,
    ToolRegistryComponent,
)
from ecs_agent.core.world import World
from ecs_agent.providers.protocol import LLMModel
from ecs_agent.providers.openai_model import OpenAIModel
from ecs_agent.prompts.message_assembly import (
    commit_prompt_context_reservation,
    prepare_outbound_messages,
    resolve_system_prompt_parts,
)
from ecs_agent.types import (
    StreamContentStartEvent,
    CompletionResult,
    Message,
    StreamContentDeltaEvent,
    StreamEndEvent,
    StreamReasoningDeltaEvent,
    StreamReasoningEndEvent,
    StreamStartEvent,
    EntityId,
    ToolCall,
    ToolSchema,
    Usage,
    InterruptionReason,
    ReasoningCompleteEvent,
    ResponsesAPICallEvent,
)

logger = get_logger(__name__)


class ReasoningSystem:
    def __init__(self, priority: int = 0) -> None:
        self.priority = priority

    async def process(self, world: World) -> None:
        for entity_id, (llm_component,) in world.query(LLMComponent):
            if world.has_component(entity_id, InterruptionComponent):
                continue
            if world.has_component(entity_id, PendingToolCallsComponent):
                continue
            # Skip parent-world delegation stubs — they are tracked records,
            # not runnable agents. ReasoningSystem must not infer over them.
            if world.has_component(entity_id, ChildStubComponent):
                continue
            # Skip entities that have a persistent terminal condition set earlier
            # in this tick (e.g. user typed "exit").  "reasoning_complete" is
            # excluded because it is always transient and removed by
            # TerminalCleanupSystem within the same tick.
            _terminal = world.get_component(entity_id, TerminalComponent)
            if _terminal is not None and _terminal.reason != "reasoning_complete":
                continue

            start_time = time.time()
            assert isinstance(llm_component, LLMComponent)

            # Sample model at request start for in-flight stability
            active_model = llm_component.pending_model or llm_component.model
            provider_id = self._resolve_provider_id(active_model)
            responses_api_state = world.get_component(
                entity_id, ResponsesAPIStateComponent
            )
            previous_response_id = (
                responses_api_state.previous_response_id
                if responses_api_state is not None
                else None
            )

            # Check for interruption
            if world.has_component(entity_id, InterruptionComponent):
                logger.info("reasoning_interrupted", entity_id=int(entity_id))
                continue

            # Handle model switching
            if llm_component.pending_model is not None:
                llm_component.model = llm_component.pending_model
                llm_component.pending_model = None

            # Split the rendered prompt into a cache-stable prefix and a volatile
            # tail (ISSUE-6). Components predating the split fall back to full text.
            system_prompt_text, system_volatile_suffix = resolve_system_prompt_parts(
                world, entity_id
            )
            conversation = world.get_component(entity_id, ConversationComponent)
            context_queue = world.get_component(entity_id, PromptContextQueueComponent)
            if (
                conversation is None
                and world.get_component(entity_id, ConversationTreeComponent) is None
            ):
                # Skip entities without either tree or flat conversation
                continue

            runner_state = world.get_component(entity_id, RunnerStateComponent)
            current_tick = runner_state.current_tick if runner_state is not None else 0
            messages, context_reservation = prepare_outbound_messages(
                world,
                entity_id,
                system_prompt=system_prompt_text,
                system_volatile_suffix=system_volatile_suffix,
                current_tick=current_tick,
            )
            if not any(message.role != "system" for message in messages):
                continue
            if context_reservation is not None:
                world.add_component(entity_id, context_reservation)

            tools: list[ToolSchema] | None = None
            tool_registry = world.get_component(entity_id, ToolRegistryComponent)
            if tool_registry is not None and tool_registry.tools:
                tools = list(tool_registry.tools.values())

            streaming_component = world.get_component(entity_id, StreamingComponent)
            streaming_enabled = (
                streaming_component is not None and streaming_component.enabled
            )
            non_blocking_delta_publish = (
                streaming_component is not None
                and streaming_component.non_blocking_delta_publish
            )

            logger.info(
                "reasoning_start",
                entity_id=int(entity_id),
                model=active_model.model_id,
                streaming=streaming_enabled,
                non_blocking_delta_publish=non_blocking_delta_publish,
                system="ReasoningSystem",
            )

            invocation_event_published = False
            observation_event_published = False
            invocation_started_at = time.monotonic()
            invocation_start_time = datetime.now(timezone.utc)
            model_parameters = (
                {"thread_response_id": previous_response_id}
                if previous_response_id is not None
                else None
            )
            await publish_llm_observation_started_event(
                event_bus=world.event_bus,
                entity_id=entity_id,
                provider_id=provider_id,
                model=active_model.model_id,
                operation="reasoning",
                messages=messages,
                tools=tools,
                streaming=streaming_enabled,
                model_parameters=model_parameters,
            )
            try:
                attach_retry_event_bus(active_model, world.event_bus)
                if streaming_enabled:
                    result = await self._process_streaming(
                        world,
                        entity_id,
                        active_model,
                        provider_id,
                        conversation,
                        messages,
                        tools,
                        non_blocking_delta_publish,
                        previous_response_id,
                    )
                else:
                    if isinstance(active_model, OpenAIModel):
                        non_stream_result = await active_model.complete(
                            messages,
                            tools=tools,
                            thread_response_id=previous_response_id,
                        )
                    else:
                        non_stream_result = await active_model.complete(
                            messages,
                            tools=tools,
                        )
                    if not isinstance(non_stream_result, CompletionResult):
                        raise RuntimeError(
                            "Model returned stream iterator in non-streaming mode"
                        )
                    result = non_stream_result

                self._queue_reasoning_context_if_configured(
                    world=world,
                    entity_id=entity_id,
                    reasoning_content=result.reasoning_content,
                )

                invocation_duration_seconds = time.monotonic() - invocation_started_at
                invocation_end_time = datetime.now(timezone.utc)
                if result.response_id is not None and isinstance(active_model, OpenAIModel):
                    await world.event_bus.publish(
                        ResponsesAPICallEvent(
                            entity_id=entity_id,
                            response_id=result.response_id,
                            model=active_model.model_id,
                        )
                    )
                await publish_llm_observation_completed_event(
                    event_bus=world.event_bus,
                    entity_id=entity_id,
                    provider_id=provider_id,
                    model=active_model.model_id,
                    operation="reasoning",
                    messages=messages,
                    tools=tools,
                    streaming=streaming_enabled,
                    model_parameters=model_parameters,
                    response_message=result.message,
                    reasoning_content=result.reasoning_content,
                    usage=result.usage,
                    response_id=result.response_id,
                    status="success",
                    duration_seconds=invocation_duration_seconds,
                    start_time=invocation_start_time,
                    end_time=invocation_end_time,
                )
                observation_event_published = True

                await self._publish_llm_invocation_event(
                    world=world,
                    entity_id=entity_id,
                    provider_id=provider_id,
                    model=active_model.model_id,
                    usage=result.usage,
                    stream_completeness=StreamCompleteness.COMPLETE,
                    request_id=result.response_id,
                    operation="reasoning",
                    status="success",
                    streaming=streaming_enabled,
                    duration_seconds=invocation_duration_seconds,
                )
                invocation_event_published = True

                if context_queue is not None and context_reservation is not None:
                    commit_prompt_context_reservation(
                        queue=context_queue,
                        reservation=context_reservation,
                    )
                    world.remove_component(entity_id, PromptContextReservationComponent)

                # Append result to conversation (tree not yet supported for writing)
                if conversation is not None:
                    conversation.messages.append(result.message)

                if result.response_id is not None:
                    world.add_component(
                        entity_id,
                        ResponsesAPIStateComponent(
                            previous_response_id=result.response_id,
                        ),
                    )

                duration_ms = (time.time() - start_time) * 1000
                tool_call_names = (
                    [tc.name for tc in result.message.tool_calls]
                    if result.message.tool_calls
                    else []
                )
                logger.info(
                    "reasoning_complete",
                    entity_id=int(entity_id),
                    model=active_model.model_id,
                    duration_ms=round(duration_ms, 2),
                    tool_call_count=len(tool_call_names),
                    tool_call_names=tool_call_names,
                    system="ReasoningSystem",
                )
                if result.message.tool_calls:
                    world.add_component(
                        entity_id,
                        PendingToolCallsComponent(tool_calls=result.message.tool_calls),
                    )
                else:
                    await world.event_bus.publish(
                        ReasoningCompleteEvent(
                            entity_id=entity_id,
                            model=active_model.model_id,
                            duration_ms=round(duration_ms, 2),
                        )
                    )
                    world.add_component(
                        entity_id,
                        TerminalComponent(reason="reasoning_complete"),
                    )
            except (IndexError, StopIteration):
                if not observation_event_published:
                    invocation_end_time = datetime.now(timezone.utc)
                    await publish_llm_observation_completed_event(
                        event_bus=world.event_bus,
                        entity_id=entity_id,
                        provider_id=provider_id,
                        model=active_model.model_id,
                        operation="reasoning",
                        messages=messages,
                        tools=tools,
                        streaming=streaming_enabled,
                        model_parameters=model_parameters,
                        status="error",
                        error="provider exhausted",
                        duration_seconds=time.monotonic() - invocation_started_at,
                        start_time=invocation_start_time,
                        end_time=invocation_end_time,
                    )
                    observation_event_published = True
                if not invocation_event_published:
                    await self._publish_llm_invocation_event(
                        world=world,
                        entity_id=entity_id,
                        provider_id=provider_id,
                        model=active_model.model_id,
                        usage=None,
                        stream_completeness=StreamCompleteness.UNKNOWN,
                        request_id=None,
                        operation="reasoning",
                        status="error",
                        streaming=streaming_enabled,
                        duration_seconds=time.monotonic() - invocation_started_at,
                    )
                    invocation_event_published = True
                world.add_component(
                    entity_id,
                    TerminalComponent(reason="provider_exhausted"),
                )
            except asyncio.CancelledError:
                if not observation_event_published:
                    invocation_end_time = datetime.now(timezone.utc)
                    await publish_llm_observation_completed_event(
                        event_bus=world.event_bus,
                        entity_id=entity_id,
                        provider_id=provider_id,
                        model=active_model.model_id,
                        operation="reasoning",
                        messages=messages,
                        tools=tools,
                        streaming=streaming_enabled,
                        model_parameters=model_parameters,
                        status="cancelled",
                        duration_seconds=time.monotonic() - invocation_started_at,
                        start_time=invocation_start_time,
                        end_time=invocation_end_time,
                    )
                    observation_event_published = True
                if not invocation_event_published:
                    stream_completeness = StreamCompleteness.UNKNOWN
                    interruption = world.get_component(entity_id, InterruptionComponent)
                    if interruption is not None:
                        partial_chunks = interruption.metadata.get("partial_chunks")
                        if isinstance(partial_chunks, int) and partial_chunks > 0:
                            stream_completeness = StreamCompleteness.PARTIAL
                    await self._publish_llm_invocation_event(
                        world=world,
                        entity_id=entity_id,
                        provider_id=provider_id,
                        model=active_model.model_id,
                        usage=None,
                        stream_completeness=stream_completeness,
                        request_id=None,
                        operation="reasoning",
                        status="cancelled",
                        streaming=streaming_enabled,
                        duration_seconds=time.monotonic() - invocation_started_at,
                    )
                    invocation_event_published = True
                if world.get_component(entity_id, InterruptionComponent) is None:
                    world.add_component(
                        entity_id,
                        InterruptionComponent(
                            reason=InterruptionReason.USER_REQUESTED,
                            message="reasoning_cancelled",
                            metadata={"phase": "reasoning_process"},
                        ),
                    )
                raise
            except Exception as exc:
                # httpx timeout exceptions stringify to ""; keep at least the type name.
                error_text = str(exc) or type(exc).__name__
                if not observation_event_published:
                    invocation_end_time = datetime.now(timezone.utc)
                    await publish_llm_observation_completed_event(
                        event_bus=world.event_bus,
                        entity_id=entity_id,
                        provider_id=provider_id,
                        model=active_model.model_id,
                        operation="reasoning",
                        messages=messages,
                        tools=tools,
                        streaming=streaming_enabled,
                        model_parameters=model_parameters,
                        status="error",
                        error=error_text,
                        duration_seconds=time.monotonic() - invocation_started_at,
                        start_time=invocation_start_time,
                        end_time=invocation_end_time,
                    )
                    observation_event_published = True
                if not invocation_event_published:
                    stream_completeness = (
                        StreamCompleteness.PARTIAL
                        if streaming_enabled
                        else StreamCompleteness.UNKNOWN
                    )
                    await self._publish_llm_invocation_event(
                        world=world,
                        entity_id=entity_id,
                        provider_id=provider_id,
                        model=active_model.model_id,
                        usage=None,
                        stream_completeness=stream_completeness,
                        request_id=None,
                        operation="reasoning",
                        status="error",
                        streaming=streaming_enabled,
                        duration_seconds=time.monotonic() - invocation_started_at,
                    )
                    invocation_event_published = True
                logger.error(
                    "reasoning_error",
                    entity_id=int(entity_id),
                    system="ReasoningSystem",
                    reason=error_text,
                )
                world.add_component(
                    entity_id,
                    ErrorComponent(
                        error=error_text,
                        system_name="ReasoningSystem",
                        timestamp=time.time(),
                    ),
                )

    def _resolve_provider_id(self, active_model: LLMModel) -> str:
        return resolve_provider_id(active_model)

    def _usage_to_usage_record(
        self,
        usage: Usage | None,
        provider_id: str,
        model: str,
        stream_completeness: StreamCompleteness,
    ) -> UsageRecord:
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

    async def _publish_llm_invocation_event(
        self,
        world: World,
        entity_id: EntityId,
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
        await publish_llm_invocation_event(
            event_bus=world.event_bus,
            entity_id=entity_id,
            provider_id=provider_id,
            model=model,
            usage=usage,
            stream_completeness=stream_completeness,
            request_id=request_id,
            operation=operation,
            status=status,
            streaming=streaming,
            duration_seconds=duration_seconds,
        )
        self._record_token_usage(world, entity_id, usage)

    @staticmethod
    def _record_token_usage(
        world: World, entity_id: EntityId, usage: Usage | None
    ) -> None:
        """Persist the API-reported token usage on the entity (ground truth).

        No-op when the provider returned no usage (e.g. error/aborted calls)."""
        if usage is None:
            return
        component = world.get_component(entity_id, TokenUsageComponent)
        if component is None:
            component = TokenUsageComponent()
            world.add_component(entity_id, component)

        prompt = usage.prompt_tokens or 0
        completion = usage.completion_tokens or 0
        total = usage.total_tokens or (prompt + completion)
        # Anthropic reports cache reads in cache_read_tokens; OpenAI-compatible
        # providers report them in cached_input_tokens (same canonical meaning,
        # mirroring AccountingSubscriber).
        cache_read = usage.cache_read_tokens or usage.cached_input_tokens or 0
        cache_creation = usage.cache_creation_tokens or 0

        component.last_prompt_tokens = prompt
        component.last_completion_tokens = completion
        component.last_total_tokens = total
        component.last_cache_read_tokens = cache_read
        component.last_cache_creation_tokens = cache_creation

        component.total_prompt_tokens += prompt
        component.total_completion_tokens += completion
        component.total_tokens += total
        component.total_cache_read_tokens += cache_read
        component.total_cache_creation_tokens += cache_creation
        component.call_count += 1

        # Anchor for compaction calibration: how many conversation messages were
        # the input basis for this call. Recorded before the response is appended
        # (see process()), so it counts exactly what was sent.
        conversation = world.get_component(entity_id, ConversationComponent)
        component.last_prompt_message_count = (
            len(conversation.messages) if conversation is not None else -1
        )

    async def _process_streaming(
        self,
        world: World,
        entity_id: EntityId,
        active_model: LLMModel,
        provider_id: str,
        conversation: ConversationComponent | None,
        messages: list[Message],
        tools: list[ToolSchema] | None,
        non_blocking_delta_publish: bool,
        previous_response_id: str | None,
    ) -> CompletionResult:
        if isinstance(active_model, OpenAIModel):
            stream_result = await active_model.complete(
                messages,
                tools=tools,
                stream=True,
                thread_response_id=previous_response_id,
            )
        else:
            stream_result = await active_model.complete(
                messages,
                tools=tools,
                stream=True,
            )
        if isinstance(stream_result, CompletionResult):
            return stream_result

        stream = stream_result
        content_chunks: list[str] = []
        reasoning_chunks: list[str] = []
        tool_call_buffers: dict[str, dict[str, Any]] = {}
        usage = None
        stream_started_at = time.perf_counter()
        first_sse_seen = False
        first_content_delta_seen = False
        first_delta_seconds: float | None = None
        reasoning_phase_active = False
        reasoning_end_emitted = False
        response_id: str | None = None
        stream_status = "success"

        await world.event_bus.publish(
            StreamStartEvent(
                entity_id=entity_id,
                timestamp=time.time(),
                provider_id=provider_id,
                model=active_model.model_id,
                operation="reasoning",
            )
        )
        stream_start_published_at = time.perf_counter()

        try:
            async for delta in stream:
                if world.get_component(entity_id, InterruptionComponent) is not None:
                    raise asyncio.CancelledError()

                if not first_sse_seen:
                    first_sse_seen = True
                    first_sse_at = time.perf_counter()
                    logger.info(
                        "reasoning_stream_first_sse_event",
                        entity_id=int(entity_id),
                        model=active_model.model_id,
                        stream_setup_ms=round(
                            (stream_start_published_at - stream_started_at) * 1000,
                            2,
                        ),
                        time_to_first_sse_event_ms=round(
                            (first_sse_at - stream_start_published_at) * 1000,
                            2,
                        ),
                        total_to_first_sse_event_ms=round(
                            (first_sse_at - stream_started_at) * 1000,
                            2,
                        ),
                    )

                if delta.reasoning_content is not None:
                    delta_first_delta_seconds: float | None = None
                    if first_delta_seconds is None:
                        first_delta_seconds = time.perf_counter() - stream_start_published_at
                        delta_first_delta_seconds = first_delta_seconds
                    reasoning_chunks.append(delta.reasoning_content)
                    reasoning_phase_active = True
                    if non_blocking_delta_publish:
                        self._publish_stream_reasoning_delta_non_blocking(
                            world=world,
                            entity_id=entity_id,
                            reasoning_delta=delta.reasoning_content,
                            provider_id=provider_id,
                            model=active_model.model_id,
                            operation="reasoning",
                            first_delta_seconds=delta_first_delta_seconds,
                        )
                    else:
                        await world.event_bus.publish(
                            StreamReasoningDeltaEvent(
                                entity_id=entity_id,
                                reasoning_delta=delta.reasoning_content,
                                provider_id=provider_id,
                                model=active_model.model_id,
                                operation="reasoning",
                                first_delta_seconds=delta_first_delta_seconds,
                            )
                        )

                if delta.content is not None:
                    if reasoning_phase_active and not reasoning_end_emitted:
                        await world.event_bus.publish(
                            StreamReasoningEndEvent(entity_id=entity_id)
                        )
                        reasoning_end_emitted = True
                        reasoning_phase_active = False

                    if not first_content_delta_seen:
                        await world.event_bus.publish(
                            StreamContentStartEvent(entity_id=entity_id)
                        )

                    content_chunks.append(delta.content)
                    delta_first_delta_seconds = None
                    if not first_content_delta_seen:
                        first_content_delta_seen = True
                        first_content_delta_at = time.perf_counter()
                        if first_delta_seconds is None:
                            first_delta_seconds = (
                                first_content_delta_at - stream_start_published_at
                            )
                            delta_first_delta_seconds = first_delta_seconds
                        logger.info(
                            "reasoning_stream_first_content_delta",
                            entity_id=int(entity_id),
                            model=active_model.model_id,
                            stream_setup_ms=round(
                                (stream_start_published_at - stream_started_at) * 1000,
                                2,
                            ),
                            time_to_first_content_delta_ms=round(
                                (first_content_delta_at - stream_start_published_at)
                                * 1000,
                                2,
                            ),
                            total_to_first_content_delta_ms=round(
                                (first_content_delta_at - stream_started_at) * 1000,
                                2,
                            ),
                            start_to_first_delta_ms=round(
                                (first_content_delta_at - stream_start_published_at)
                                * 1000,
                                2,
                            ),
                            total_to_first_delta_ms=round(
                                (first_content_delta_at - stream_started_at) * 1000,
                                2,
                            ),
                        )

                    if non_blocking_delta_publish:
                        self._publish_stream_delta_non_blocking(
                            world=world,
                            entity_id=entity_id,
                            delta_content=delta.content,
                            provider_id=provider_id,
                            model=active_model.model_id,
                            operation="reasoning",
                            first_delta_seconds=delta_first_delta_seconds,
                        )
                    else:
                        await world.event_bus.publish(
                            StreamContentDeltaEvent(
                                entity_id=entity_id,
                                delta=delta.content,
                                provider_id=provider_id,
                                model=active_model.model_id,
                                operation="reasoning",
                                first_delta_seconds=delta_first_delta_seconds,
                            )
                        )

                self._merge_stream_tool_calls(tool_call_buffers, delta.tool_calls)

                if delta.usage is not None:
                    usage = delta.usage

                if delta.response_id is not None:
                    response_id = delta.response_id

                if world.get_component(entity_id, InterruptionComponent) is not None:
                    raise asyncio.CancelledError()
        except asyncio.CancelledError:
            stream_status = "cancelled"
            partial_message = Message(
                role="assistant",
                content="".join(content_chunks),
                tool_calls=self._finalize_tool_calls(tool_call_buffers),
            )
            if partial_message.content or partial_message.tool_calls:
                if conversation is not None:
                    conversation.messages.append(partial_message)

            interruption = world.get_component(entity_id, InterruptionComponent)
            partial_metadata = {
                "partial_chunks": len(content_chunks),
                "partial_content": partial_message.content,
                "partial_content_length": len(partial_message.content),
            }

            if interruption is None:
                world.add_component(
                    entity_id,
                    InterruptionComponent(
                        reason=InterruptionReason.USER_REQUESTED,
                        message="stream_cancelled",
                        metadata=partial_metadata,
                    ),
                )
            else:
                interruption.metadata.update(partial_metadata)
            raise
        except Exception:
            stream_status = "error"
            partial_message = Message(
                role="assistant",
                content="".join(content_chunks),
                tool_calls=self._finalize_tool_calls(tool_call_buffers),
            )
            if partial_message.content or partial_message.tool_calls:
                if conversation is not None:
                    conversation.messages.append(partial_message)
            raise
        finally:
            await world.event_bus.publish(
                StreamEndEvent(
                    entity_id=entity_id,
                    timestamp=time.time(),
                    provider_id=provider_id,
                    model=active_model.model_id,
                    operation="reasoning",
                    status=stream_status,
                    duration_seconds=time.perf_counter() - stream_start_published_at,
                    first_delta_seconds=first_delta_seconds,
                )
            )

        return CompletionResult(
            message=Message(
                role="assistant",
                content="".join(content_chunks),
                tool_calls=self._finalize_tool_calls(tool_call_buffers),
            ),
            usage=usage,
            response_id=response_id,
            reasoning_content="".join(reasoning_chunks) or None,
        )

    def _queue_reasoning_context_if_configured(
        self,
        *,
        world: World,
        entity_id: EntityId,
        reasoning_content: str | None,
    ) -> None:
        if not reasoning_content:
            return

        budget_config = world.get_component(entity_id, ContextTrimConfig)
        if budget_config is None or not budget_config.trim_reasoning:
            return

        conversation = world.get_component(entity_id, ConversationComponent)
        if conversation is None:
            return

        queue = world.get_component(entity_id, PromptContextQueueComponent)
        if queue is None:
            return

        next_registration_order = (
            max((entry.registration_order for entry in queue.entries), default=-1) + 1
        )
        queue.entries.append(
            ContextEntry(
                entry_id=uuid.uuid4().hex,
                priority=0,
                source_label="reasoning",
                content=reasoning_content,
                registration_order=next_registration_order,
                droppable_kind="reasoning",
            )
        )

    def _publish_stream_delta_non_blocking(
        self,
        world: World,
        entity_id: EntityId,
        delta_content: str,
        provider_id: str,
        model: str,
        operation: str,
        first_delta_seconds: float | None,
    ) -> None:
        task = asyncio.create_task(
            world.event_bus.publish(
                StreamContentDeltaEvent(
                    entity_id=entity_id,
                    delta=delta_content,
                    provider_id=provider_id,
                    model=model,
                    operation=operation,
                    first_delta_seconds=first_delta_seconds,
                )
            )
        )
        task.add_done_callback(
            lambda publish_task: self._log_non_blocking_delta_error(
                publish_task=publish_task,
                entity_id=entity_id,
            )
        )

    def _log_non_blocking_delta_error(
        self,
        publish_task: asyncio.Task[None],
        entity_id: EntityId,
    ) -> None:
        if publish_task.cancelled():
            return

        error = publish_task.exception()
        if error is None:
            return

        logger.error(
            "reasoning_stream_delta_publish_error",
            entity_id=int(entity_id),
            error=str(error),
            system="ReasoningSystem",
        )

    def _publish_stream_reasoning_delta_non_blocking(
        self,
        world: World,
        entity_id: EntityId,
        reasoning_delta: str,
        provider_id: str,
        model: str,
        operation: str,
        first_delta_seconds: float | None,
    ) -> None:
        task = asyncio.create_task(
            world.event_bus.publish(
                StreamReasoningDeltaEvent(
                    entity_id=entity_id,
                    reasoning_delta=reasoning_delta,
                    provider_id=provider_id,
                    model=model,
                    operation=operation,
                    first_delta_seconds=first_delta_seconds,
                )
            )
        )
        task.add_done_callback(
            lambda publish_task: self._log_non_blocking_reasoning_delta_error(
                publish_task=publish_task,
                entity_id=entity_id,
            )
        )

    def _log_non_blocking_reasoning_delta_error(
        self,
        publish_task: asyncio.Task[None],
        entity_id: EntityId,
    ) -> None:
        if publish_task.cancelled():
            return

        error = publish_task.exception()
        if error is None:
            return

        logger.error(
            "reasoning_stream_reasoning_delta_publish_error",
            entity_id=int(entity_id),
            error=str(error),
            system="ReasoningSystem",
        )

    def _merge_stream_tool_calls(
        self,
        buffers: dict[str, dict[str, Any]],
        delta_tool_calls: list[ToolCall] | None,
    ) -> None:
        if not delta_tool_calls:
            return

        for tool_call in delta_tool_calls:
            tool_call_id = tool_call.id or f"tool_call_{len(buffers)}"
            current = buffers.setdefault(
                tool_call_id,
                {
                    "id": tool_call_id,
                    "name": "",
                    "arguments_buffer": "",
                    "arguments": None,
                },
            )

            if tool_call.name:
                current["name"] = tool_call.name

            partial = tool_call.arguments.get("_partial")
            if isinstance(partial, str):
                current["arguments_buffer"] += partial
            elif tool_call.arguments:
                current["arguments"] = tool_call.arguments

    def _finalize_tool_calls(
        self, buffers: dict[str, dict[str, Any]]
    ) -> list[ToolCall] | None:
        if not buffers:
            return None

        completed: list[ToolCall] = []
        for buffered in buffers.values():
            parsed_arguments: dict[str, Any]
            arguments_buffer = buffered["arguments_buffer"]
            arguments = buffered["arguments"]

            if arguments_buffer:
                try:
                    loaded_arguments = json.loads(arguments_buffer)
                except json.JSONDecodeError:
                    loaded_arguments = {"_partial": arguments_buffer}

                if isinstance(loaded_arguments, dict):
                    parsed_arguments = loaded_arguments
                else:
                    parsed_arguments = {"_partial": arguments_buffer}
            elif isinstance(arguments, dict):
                parsed_arguments = arguments
            else:
                parsed_arguments = {}

            completed.append(
                ToolCall(
                    id=buffered["id"],
                    name=buffered["name"],
                    arguments=parsed_arguments,
                )
            )

        return completed
