"""ErrorHandlingSystem for ECS-based LLM Agent."""

from ecs_agent.components.definitions import ErrorComponent
from ecs_agent.core.world import World
from ecs_agent.types import ErrorOccurredEvent


class ErrorHandlingSystem:
    """System that handles error cleanup and logging."""

    def __init__(self, priority: int = 99) -> None:
        """Initialize ErrorHandlingSystem with priority.

        Args:
            priority: System execution priority (default 99 - runs last)
        """
        self.priority = priority

    async def process(self, world: World) -> None:
        """Process all entities with ErrorComponent.

        Logs error information, publishes ErrorOccurredEvent,
        and removes ErrorComponent from entity.

        Args:
            world: World instance to query and modify
        """
        for entity_id, (error_comp,) in world.query(ErrorComponent):
            print(
                f"[ERROR] Entity {entity_id} | System: {error_comp.system_name} | Error: {error_comp.error}"
            )

            await world.event_bus.publish(
                ErrorOccurredEvent(
                    entity_id=entity_id,
                    error=error_comp.error,
                    system_name=error_comp.system_name,
                )
            )

            world.remove_component(entity_id, ErrorComponent)
