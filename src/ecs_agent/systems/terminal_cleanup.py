"""System for clearing selected terminal reasons after processing."""

from ecs_agent.components import OwnerComponent, TerminalComponent
from ecs_agent.core.world import World


class TerminalCleanupSystem:
    """Clears configured TerminalComponent reasons from matching entities."""

    def __init__(
        self,
        priority: int = 1,
        clear_reasons: tuple[str, ...] = ("reasoning_complete",),
        include_owned_entities: bool = False,
    ) -> None:
        self.priority = priority
        self.clear_reasons = clear_reasons
        self.include_owned_entities = include_owned_entities

    async def process(self, world: World) -> None:
        for entity_id, (terminal,) in world.query(TerminalComponent):
            assert isinstance(terminal, TerminalComponent)

            if not self.include_owned_entities and world.has_component(
                entity_id, OwnerComponent
            ):
                continue

            if terminal.reason in self.clear_reasons:
                world.remove_component(entity_id, TerminalComponent)


__all__ = ["TerminalCleanupSystem"]
