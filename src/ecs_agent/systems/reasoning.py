from __future__ import annotations

import json
import time
from typing import Any

from ecs_agent.components import (
    ConversationComponent,
    ErrorComponent,
    LLMComponent,
    PendingToolCallsComponent,
    StreamingComponent,
    SystemPromptComponent,
    TerminalComponent,
    ToolRegistryComponent,
)
from ecs_agent.core.world import World
from ecs_agent.types import (
    CompletionResult,
    Message,
    StreamDeltaEvent,
    StreamEndEvent,
    StreamStartEvent,
    EntityId,
    ToolCall,
    ToolSchema,
)


class ReasoningSystem:
    def __init__(self, priority: int = 0) -> None:
        self.priority = priority

    async def process(self, world: World) -> None:
        for entity_id, components in world.query(LLMComponent, ConversationComponent):
            llm_component, conversation = components
            assert isinstance(llm_component, LLMComponent)
            assert isinstance(conversation, ConversationComponent)

            messages: list[Message] = []

            system_prompt = world.get_component(entity_id, SystemPromptComponent)
            if system_prompt is not None:
                messages.append(Message(role="system", content=system_prompt.content))

            messages.extend(conversation.messages)

            tools: list[ToolSchema] | None = None
            tool_registry = world.get_component(entity_id, ToolRegistryComponent)
            if tool_registry is not None and tool_registry.tools:
                tools = list(tool_registry.tools.values())

            streaming_component = world.get_component(entity_id, StreamingComponent)
            streaming_enabled = (
                streaming_component is not None and streaming_component.enabled
            )

            try:
                if streaming_enabled:
                    result = await self._process_streaming(
                        world,
                        entity_id,
                        llm_component,
                        conversation,
                        messages,
                        tools,
                    )
                else:
                    non_stream_result = await llm_component.provider.complete(
                        messages, tools=tools
                    )
                    if not isinstance(non_stream_result, CompletionResult):
                        raise RuntimeError(
                            "Provider returned stream iterator in non-streaming mode"
                        )
                    result = non_stream_result

                conversation.messages.append(result.message)

                if result.message.tool_calls:
                    world.add_component(
                        entity_id,
                        PendingToolCallsComponent(tool_calls=result.message.tool_calls),
                    )
            except (IndexError, StopIteration):
                world.add_component(
                    entity_id,
                    TerminalComponent(reason="provider_exhausted"),
                )
            except Exception as exc:
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
        llm_component: LLMComponent,
        conversation: ConversationComponent,
        messages: list[Message],
        tools: list[ToolSchema] | None,
    ) -> CompletionResult:
        stream_result = await llm_component.provider.complete(
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

        await world.event_bus.publish(
            StreamStartEvent(entity_id=entity_id, timestamp=time.time())
        )

        try:
            async for delta in stream:
                if delta.content is not None:
                    content_chunks.append(delta.content)
                    await world.event_bus.publish(
                        StreamDeltaEvent(entity_id=entity_id, delta=delta.content)
                    )

                self._merge_stream_tool_calls(tool_call_buffers, delta.tool_calls)

                if delta.usage is not None:
                    usage = delta.usage
        except Exception:
            partial_message = Message(
                role="assistant",
                content="".join(content_chunks),
                tool_calls=self._finalize_tool_calls(tool_call_buffers),
            )
            if partial_message.content or partial_message.tool_calls:
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
