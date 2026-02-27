from __future__ import annotations

from typing import Awaitable, Callable

from ecs_agent.components import (
    ConversationComponent,
    PendingToolCallsComponent,
    SandboxConfigComponent,
    ToolRegistryComponent,
    ToolResultsComponent,
)
from ecs_agent.core.world import World
from ecs_agent.tools.sandbox import sandboxed_execute
from ecs_agent.types import (
    EntityId,
    Message,
    ToolCall,
    ToolExecutionCompletedEvent,
    ToolExecutionStartedEvent,
)


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
                # Publish ToolExecutionStartedEvent
                await world.event_bus.publish(
                    ToolExecutionStartedEvent(
                        entity_id=entity_id,
                        tool_call=tool_call,
                    )
                )

                # Execute the tool call
                result = await self._execute_tool_call(
                    entity_id,
                    world,
                    tool_call,
                    registry.handlers,
                )

                # Publish ToolExecutionCompletedEvent
                success = not result.startswith("Error")
                await world.event_bus.publish(
                    ToolExecutionCompletedEvent(
                        entity_id=entity_id,
                        tool_call_id=tool_call.id,
                        result=result,
                        success=success,
                    )
                )

                results[tool_call.id] = result
                conversation.messages.append(
                    Message(role="tool", content=result, tool_call_id=tool_call.id)
                )

            world.remove_component(entity_id, PendingToolCallsComponent)
            if results:
                world.add_component(entity_id, ToolResultsComponent(results=results))

    async def _execute_tool_call(
        self,
        entity_id: EntityId,
        world: World,
        tool_call: ToolCall,
        handlers: dict[str, Callable[..., Awaitable[str]]],
    ) -> str:
        handler = handlers.get(tool_call.name)
        if handler is None:
            return f"Error: unknown tool '{tool_call.name}'"

        try:
            arguments = tool_call.arguments
            sandbox_config = world.get_component(entity_id, SandboxConfigComponent)
            if sandbox_config is None:
                result = await handler(**arguments)
            else:
                result = await sandboxed_execute(
                    handler,
                    arguments,
                    timeout=sandbox_config.timeout,
                    max_output_size=sandbox_config.max_output_size,
                )
            return str(result)
        except Exception as exc:
            return f"Error executing tool '{tool_call.name}': {exc}"
