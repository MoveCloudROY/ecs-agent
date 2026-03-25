from __future__ import annotations

import time
from typing import Awaitable, Callable

from ecs_agent.components import (
    ConversationComponent,
    PendingToolCallsComponent,
    SandboxConfigComponent,
    ToolRegistryComponent,
    ToolResultsComponent,
)
from ecs_agent.core.world import World
from ecs_agent.logging import get_logger
from ecs_agent.scratchbook import ScratchbookService, ToolResultsSink
from ecs_agent.tools.sandbox import sandboxed_execute
from ecs_agent.types import (
    EntityId,
    Message,
    ToolCall,
    ToolExecutionCompletedEvent,
    ToolExecutionStartedEvent,
)

logger = get_logger(__name__)


class ToolExecutionSystem:
    def __init__(self, priority: int = 0, scratchbook_service: ScratchbookService | None = None) -> None:
        self.priority = priority
        self.scratchbook_service = scratchbook_service
        self.tool_sink = ToolResultsSink(scratchbook_service) if scratchbook_service else None

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

                # Persist to scratchbook and store ref
                if self.tool_sink is not None:
                    artifact_ref = self.tool_sink.persist_tool_result(
                        tool_call_id=tool_call.id,
                        tool_name=tool_call.name,
                        result=result,
                        arguments=tool_call.arguments,
                    )
                    results[tool_call.id] = artifact_ref
                    # Add artifact ref to conversation, not full result
                    conversation.messages.append(
                        Message(role="tool", content=artifact_ref, tool_call_id=tool_call.id)
                    )
                else:
                    # Fallback: store full result if no scratchbook service
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
        # Log tool invocation
        logger.info("tool_called", tool_name=tool_call.name, arguments=tool_call.arguments)

        handler = handlers.get(tool_call.name)
        if handler is None:
            reason = f"Error: unknown tool '{tool_call.name}'"
            logger.error("tool_failed", tool_name=tool_call.name, reason=reason)
            return reason

        start_time = time.monotonic()
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
            
            duration_ms = (time.monotonic() - start_time) * 1000
            result_str = str(result)
            result_tail = "\n".join(result_str.splitlines()[-10:])
            logger.debug(
                "tool_result",
                tool_name=tool_call.name,
                success=True,
                duration_ms=duration_ms,
                result_tail=result_tail,
            )
            return result_str
        except Exception as exc:
            duration_ms = (time.monotonic() - start_time) * 1000
            reason = f"Error executing tool '{tool_call.name}': {exc}"
            logger.error(
                "tool_failed",
                tool_name=tool_call.name,
                reason=str(exc),
                duration_ms=duration_ms,
            )
            return reason
