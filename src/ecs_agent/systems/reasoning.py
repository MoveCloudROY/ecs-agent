from __future__ import annotations

import time

from ecs_agent.components import (
    ConversationComponent,
    ErrorComponent,
    LLMComponent,
    PendingToolCallsComponent,
    SystemPromptComponent,
    TerminalComponent,
    ToolRegistryComponent,
)
from ecs_agent.core.world import World
from ecs_agent.types import Message, ToolSchema


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

            try:
                result = await llm_component.provider.complete(messages, tools=tools)
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
