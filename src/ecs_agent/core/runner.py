"""Runner for ECS-based LLM Agent."""

from ecs_agent.components.definitions import TerminalComponent
from ecs_agent.core.world import World


class Runner:
    """Orchestrates the main execution loop."""

    async def run(self, world: World, max_ticks: int = 100) -> None:
        """Run the main execution loop until terminal condition.

        Executes world.process() repeatedly until either:
        1. A TerminalComponent is found on any entity
        2. max_ticks iterations are reached

        If max_ticks is reached, adds TerminalComponent(reason='max_ticks')
        to the first available entity.

        Args:
            world: World instance to process
            max_ticks: Maximum number of ticks to run (default 100)
        """
        for tick in range(max_ticks):
            await world.process()

            has_terminal = any(
                world.has_component(eid, TerminalComponent)
                for eid, _ in world.query(TerminalComponent)
            )
            if has_terminal:
                return

        entity_id = world.create_entity()
        world.add_component(entity_id, TerminalComponent(reason="max_ticks"))
