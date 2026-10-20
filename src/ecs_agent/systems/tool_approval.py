"""Optional authorization preflight; execution owns queue consumption and results."""

from ecs_agent.components import PendingToolCallsComponent, ToolApprovalComponent
from ecs_agent.core.world import World
from ecs_agent.tools.authorization import authorize_tool_call


class ToolApprovalSystem:
    """Precompute shared decisions without consuming calls or writing results."""

    def __init__(self, priority: int = 2) -> None:
        self.priority = priority

    async def process(self, world: World) -> None:
        for entity, (pending, _) in world.query_active(PendingToolCallsComponent, ToolApprovalComponent):
            for call in pending.tool_calls:
                await authorize_tool_call(world, entity, pending, call)


__all__ = ["ToolApprovalSystem"]
