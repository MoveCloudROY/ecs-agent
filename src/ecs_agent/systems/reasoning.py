from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from ecs_agent.logging import get_logger
from ecs_agent.components import (
    ConversationComponent,
    ConversationTreeComponent,
    ErrorComponent,
    InterruptionComponent,
    LLMComponent,
    PendingToolCallsComponent,
    PromptContextQueueComponent,
    PromptContextReservationComponent,
    UserPromptConfigComponent,
    RenderedSystemPromptComponent,
    RenderedUserPromptComponent,
    RunnerStateComponent,
    StreamingComponent,
    SystemPromptComponent,
    TerminalComponent,
    ToolRegistryComponent,
)
from ecs_agent.core.world import World
from ecs_agent.conversation_tree import get_active_leaf, linearize
from ecs_agent.providers.protocol import LLMProvider
from ecs_agent.prompts.message_assembly import (
    assemble_messages,
    build_keyword_registry,
    build_trigger_specs,
    commit_prompt_context_reservation,
    collect_active_events,
    reserve_prompt_context_reservation,
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
    InterruptionReason,
)

logger = get_logger(__name__)


def _substitute_last_user_message(
    messages: list[Message], new_text: str
) -> list[Message]:
    if not messages:
        return [Message(role="user", content=new_text)]

    result = list(messages)
    for index in range(len(result) - 1, -1, -1):
        if result[index].role == "user":
            result[index] = Message(role="user", content=new_text)
            return result

    return [*result, Message(role="user", content=new_text)]


class ReasoningSystem:
    def __init__(self, priority: int = 0) -> None:
        self.priority = priority

    async def process(self, world: World) -> None:
        for entity_id, (llm_component,) in world.query(LLMComponent):
            if world.has_component(entity_id, InterruptionComponent):
                continue

            start_time = time.time()
            assert isinstance(llm_component, LLMComponent)

            # Sample provider and model at request start for in-flight stability
            active_provider = llm_component.pending_provider or llm_component.provider
            active_model = llm_component.pending_model or llm_component.model

            # Check for interruption
            if world.has_component(entity_id, InterruptionComponent):
                logger.info("reasoning_interrupted", entity_id=int(entity_id))
                continue

            # Handle model switching
            if hasattr(llm_component, "pending_model") and llm_component.pending_model:
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
            # Check for tree first, fallback to flat conversation
            tree = world.get_component(entity_id, ConversationTreeComponent)
            conversation = world.get_component(entity_id, ConversationComponent)
            rendered_user_prompt = world.get_component(
                entity_id, RenderedUserPromptComponent
            )
            prompt_config = world.get_component(entity_id, UserPromptConfigComponent)
            context_queue = world.get_component(entity_id, PromptContextQueueComponent)
            context_reservation = world.get_component(
                entity_id, PromptContextReservationComponent
            )
            use_rendered_user_prompt = rendered_user_prompt is not None
            keyword_registry = None
            trigger_specs = None
            context_pool_enabled = False
            if not use_rendered_user_prompt:
                keyword_registry = (
                    build_keyword_registry(prompt_config.triggers)
                    if prompt_config is not None and prompt_config.triggers
                    else None
                )
                trigger_specs = (
                    build_trigger_specs(prompt_config.triggers)
                    if prompt_config is not None and prompt_config.triggers
                    else None
                )

                context_pool_enabled = (
                    prompt_config.enable_context_pool
                    if prompt_config is not None
                    else False
                )
            reserved_context_pool_items = None
            if context_pool_enabled and context_queue is not None:
                runner_state = world.get_component(entity_id, RunnerStateComponent)
                current_tick = (
                    runner_state.current_tick if runner_state is not None else 0
                )
                context_reservation = reserve_prompt_context_reservation(
                    queue=context_queue,
                    reservation=context_reservation,
                    current_tick=current_tick,
                )
                if not world.has_component(
                    entity_id, PromptContextReservationComponent
                ):
                    world.add_component(entity_id, context_reservation)
                reserved_context_pool_items = context_reservation.reserved_entries
            active_events = collect_active_events(reserved_context_pool_items)

            conversation_messages: list[Message] = []

            if tree is not None:
                # Use tree-based conversation if available
                active_leaf_id = get_active_leaf(tree)
                if active_leaf_id is not None:
                    tree_messages = linearize(tree, active_leaf_id)
                    conversation_messages.extend(tree_messages)
            elif conversation is not None:
                # Fallback to flat conversation (backward compatibility)
                conversation_messages.extend(conversation.messages)
            else:
                # Skip entities without either tree or flat conversation
                continue

            if rendered_user_prompt is not None:
                messages = assemble_messages(
                    system_prompt=system_prompt_text,
                    conversation_messages=_substitute_last_user_message(
                        conversation_messages,
                        rendered_user_prompt.text,
                    ),
                    enable_context_pool=False,
                )
            else:
                messages = assemble_messages(
                    system_prompt=system_prompt_text,
                    conversation_messages=conversation_messages,
                    enable_context_pool=context_pool_enabled,
                    context_pool_items=reserved_context_pool_items,
                    keyword_registry=keyword_registry,
                    trigger_specs=trigger_specs,
                    active_events=active_events,
                )

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
                model=active_model,
                streaming=streaming_enabled,
                non_blocking_delta_publish=non_blocking_delta_publish,
                system="ReasoningSystem",
            )

            try:
                if streaming_enabled:
                    result = await self._process_streaming(
                        world,
                        entity_id,
                        active_provider,
                        active_model,
                        conversation,
                        messages,
                        tools,
                        non_blocking_delta_publish,
                    )
                else:
                    non_stream_result = await active_provider.complete(
                        messages, tools=tools
                    )
                    if not isinstance(non_stream_result, CompletionResult):
                        raise RuntimeError(
                            "Provider returned stream iterator in non-streaming mode"
                        )
                    result = non_stream_result

                if (
                    context_pool_enabled
                    and context_queue is not None
                    and context_reservation is not None
                ):
                    commit_prompt_context_reservation(
                        queue=context_queue,
                        reservation=context_reservation,
                    )
                    world.remove_component(entity_id, PromptContextReservationComponent)

                # Append result to conversation (tree not yet supported for writing)
                if conversation is not None:
                    conversation.messages.append(result.message)

                if result.message.tool_calls:
                    world.add_component(
                        entity_id,
                        PendingToolCallsComponent(tool_calls=result.message.tool_calls),
                    )
                else:
                    duration_ms = (time.time() - start_time) * 1000
                    logger.info(
                        "reasoning_complete",
                        entity_id=int(entity_id),
                        model=active_model,
                        duration_ms=round(duration_ms, 2),
                        system="ReasoningSystem",
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

    async def _process_streaming(
        self,
        world: World,
        entity_id: EntityId,
        active_provider: LLMProvider,
        active_model: str,
        conversation: ConversationComponent | None,
        messages: list[Message],
        tools: list[ToolSchema] | None,
        non_blocking_delta_publish: bool,
    ) -> CompletionResult:
        stream_result = await active_provider.complete(
            messages,
            tools=tools,
            stream=True,
        )
        if isinstance(stream_result, CompletionResult):
            return stream_result

        stream = stream_result
        content_chunks: list[str] = []
        tool_call_buffers: dict[str, dict[str, Any]] = {}
        usage = None
        stream_started_at = time.perf_counter()
        first_sse_seen = False
        first_content_delta_seen = False
        reasoning_phase_active = False
        reasoning_end_emitted = False

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
                        model=active_model,
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
                            model=active_model,
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
