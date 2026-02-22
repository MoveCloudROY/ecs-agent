from __future__ import annotations

import json
from typing import Awaitable, Callable

from ecs_agent.components import (
    ConversationComponent,
    PendingToolCallsComponent,
    ToolRegistryComponent,
    ToolResultsComponent,
)
from ecs_agent.core.world import World
from ecs_agent.types import Message, ToolCall


class ToolExecutionSystem:
    def __init__(self, priority: int = 0) -> None:
        self.priority = priority

    async def process(self, world: World) -> None:
        for entity_id, components in world.query(
            PendingToolCallsComponent,
            ToolRegistryComponent,
            ConversationComponent,
        ):
            pending, registry, conversation = components
            assert isinstance(pending, PendingToolCallsComponent)
            assert isinstance(registry, ToolRegistryComponent)
            assert isinstance(conversation, ConversationComponent)

            results: dict[str, str] = {}
            for tool_call in pending.tool_calls:
                result = await self._execute_tool_call(tool_call, registry.handlers)
                results[tool_call.id] = result
                conversation.messages.append(
                    Message(role="tool", content=result, tool_call_id=tool_call.id)
                )

            world.remove_component(entity_id, PendingToolCallsComponent)
            if results:
                world.add_component(entity_id, ToolResultsComponent(results=results))

    async def _execute_tool_call(
        self,
        tool_call: ToolCall,
        handlers: dict[str, Callable[..., Awaitable[str]]],
    ) -> str:
        handler = handlers.get(tool_call.name)
        if handler is None:
            return f"Error: unknown tool '{tool_call.name}'"

        try:
            arguments = json.loads(tool_call.arguments)
            if not isinstance(arguments, dict):
                raise TypeError("tool arguments must decode to an object")
            result = await handler(**arguments)
            return str(result)
        except Exception as exc:
            return f"Error executing tool '{tool_call.name}': {exc}"
