from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Any

from ecs_agent.accounting.models import (
    LLMInvocationEvent,
    StreamCompleteness,
    UsageRecord,
)
from ecs_agent.logging import get_logger
from ecs_agent.components import (
    ChildStubComponent,
    ConversationComponent,
    ConversationTreeComponent,
    ContextBudgetConfig,
    ContextEntry,
    ErrorComponent,
    InterruptionComponent,
    LLMComponent,
    PendingToolCallsComponent,
    PromptContextQueueComponent,
    PromptContextReservationComponent,
    RenderedSystemPromptComponent,
    ResponsesAPIStateComponent,
    RunnerStateComponent,
    StreamingComponent,
    SystemPromptComponent,
    TerminalComponent,
    ToolRegistryComponent,
)
from ecs_agent.core.world import World
from ecs_agent.providers.protocol import LLMModel
from ecs_agent.providers.openai_model import OpenAIModel
from ecs_agent.prompts.message_assembly import (
    commit_prompt_context_reservation,
    prepare_outbound_messages,
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
)

logger = get_logger(__name__)


class ReasoningSystem:
    def __init__(self, priority: int = 0) -> None:
        self.priority = priority

    async def process(self, world: World) -> None:
        for entity_id, (llm_component,) in world.query(LLMComponent):
            if world.has_component(entity_id, InterruptionComponent):
                continue
            # Skip parent-world delegation stubs — they are tracked records,
            # not runnable agents. ReasoningSystem must not infer over them.
            if world.has_component(entity_id, ChildStubComponent):
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

            rendered_system_prompt = world.get_component(
                entity_id, RenderedSystemPromptComponent
            )
            system_prompt = world.get_component(entity_id, SystemPromptComponent)
            system_prompt_text = (
                rendered_system_prompt.text
                if rendered_system_prompt is not None
                else (
                    system_prompt.content
                    if system_prompt is not None
                    else (llm_component.system_prompt or None)
                )
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
                current_tick=current_tick,
            )
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
            try:
                if streaming_enabled:
                    result = await self._process_streaming(
                        world,
                        entity_id,
                        active_model,
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

                await self._publish_llm_invocation_event(
                    world=world,
                    entity_id=entity_id,
                    provider_id=provider_id,
                    model=active_model.model_id,
                    usage=result.usage,
                    stream_completeness=StreamCompleteness.COMPLETE,
                    request_id=result.response_id,
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
                world.add_component(
                    entity_id,
                    TerminalComponent(reason="provider_exhausted"),
                )
            except asyncio.CancelledError:
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
                    )
                    invocation_event_published = True
                logger.error(
                    "reasoning_error",
                    entity_id=int(entity_id),
                    system="ReasoningSystem",
                    reason=str(exc),
                )
                world.add_component(
                    entity_id,
                    ErrorComponent(
                        error=str(exc),
                        system_name="ReasoningSystem",
                        timestamp=time.time(),
                    ),
                )

    def _resolve_provider_id(self, active_model: LLMModel) -> str:
        provider_id = getattr(active_model, "provider_id", None)
        if isinstance(provider_id, str) and provider_id:
            return provider_id
        return type(active_model).__name__

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
    ) -> None:
        usage_record = self._usage_to_usage_record(
            usage=usage,
            provider_id=provider_id,
            model=model,
            stream_completeness=stream_completeness,
        )
        await world.event_bus.publish(
            LLMInvocationEvent(
                entity_id=int(entity_id),
                provider_id=provider_id,
                model=model,
                usage=usage_record,
                cost=None,
                request_id=request_id,
            )
        )

    async def _process_streaming(
        self,
        world: World,
        entity_id: EntityId,
        active_model: LLMModel,
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
        reasoning_phase_active = False
        reasoning_end_emitted = False
        response_id: str | None = None

        await world.event_bus.publish(
            StreamStartEvent(entity_id=entity_id, timestamp=time.time())
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
                    reasoning_chunks.append(delta.reasoning_content)
                    reasoning_phase_active = True
                    if non_blocking_delta_publish:
                        self._publish_stream_reasoning_delta_non_blocking(
                            world=world,
                            entity_id=entity_id,
                            reasoning_delta=delta.reasoning_content,
                        )
                    else:
                        await world.event_bus.publish(
                            StreamReasoningDeltaEvent(
                                entity_id=entity_id,
                                reasoning_delta=delta.reasoning_content,
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
                    if not first_content_delta_seen:
                        first_content_delta_seen = True
                        first_content_delta_at = time.perf_counter()
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
                        )
                    else:
                        await world.event_bus.publish(
                            StreamContentDeltaEvent(
                                entity_id=entity_id,
                                delta=delta.content,
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
                StreamEndEvent(entity_id=entity_id, timestamp=time.time())
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

        budget_config = world.get_component(entity_id, ContextBudgetConfig)
        if budget_config is None or not budget_config.prune_reasoning:
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
    ) -> None:
        task = asyncio.create_task(
            world.event_bus.publish(
                StreamContentDeltaEvent(entity_id=entity_id, delta=delta_content)
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
    ) -> None:
        task = asyncio.create_task(
            world.event_bus.publish(
                StreamReasoningDeltaEvent(
                    entity_id=entity_id,
                    reasoning_delta=reasoning_delta,
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
