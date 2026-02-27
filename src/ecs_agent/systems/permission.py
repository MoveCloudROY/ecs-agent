"""Permission filtering for pending tool calls."""

from __future__ import annotations

from ecs_agent.components import (
    ConversationComponent,
    PendingToolCallsComponent,
    PermissionComponent,
)
from ecs_agent.core.world import World
from ecs_agent.types import Message, ToolDeniedEvent


class PermissionSystem:
    def __init__(self, priority: int = -10) -> None:
        self.priority = priority

    async def process(self, world: World) -> None:
        for entity_id, components in world.query(
            PendingToolCallsComponent,
            PermissionComponent,
        ):
            pending, permissions = components
            assert isinstance(pending, PendingToolCallsComponent)
            assert isinstance(permissions, PermissionComponent)

            denied = set(permissions.denied_tools)
            allowed = set(permissions.allowed_tools)
            conversation = world.get_component(entity_id, ConversationComponent)

            allowed_calls = []
            for tool_call in pending.tool_calls:
                is_denied = tool_call.name in denied
                is_allowed = not allowed or tool_call.name in allowed

                if is_denied or not is_allowed:
                    denied_message = (
                        f"Error: tool '{tool_call.name}' denied by permission policy"
                    )
                    if conversation is not None:
                        conversation.messages.append(
                            Message(
                                role="tool",
                                content=denied_message,
                                tool_call_id=tool_call.id,
                            )
                        )
                    await world.event_bus.publish(
                        ToolDeniedEvent(
                            entity_id=entity_id,
                            tool_call_id=tool_call.id,
                            reason=denied_message,
                        )
                    )
                    continue

                allowed_calls.append(tool_call)

            if allowed_calls:
                pending.tool_calls = allowed_calls
            else:
                world.remove_component(entity_id, PendingToolCallsComponent)


__all__ = ["PermissionSystem"]
