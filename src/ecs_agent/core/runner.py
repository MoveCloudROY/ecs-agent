"""Runner for ECS-based LLM Agent with checkpoint resume support."""

import json
from pathlib import Path
from typing import Any
from ecs_agent.components.definitions import RunnerStateComponent, TerminalComponent
from ecs_agent.core.world import World
from ecs_agent.serialization import WorldSerializer


class Runner:
    """Orchestrates the main execution loop."""

    async def run(self, world: World, max_ticks: int | None = 100, start_tick: int = 0) -> None:
        """Run the main execution loop until terminal condition.

        Executes world.process() repeatedly until either:
        1. A TerminalComponent is found on any entity
        2. max_ticks iterations are reached (from start_tick)

        When max_ticks is None the loop runs indefinitely until a
        TerminalComponent appears (useful for interactive / chat agents).

        If max_ticks is reached, adds TerminalComponent(reason='max_ticks')
        to a newly created entity.

        Args:
            world: World instance to process
            max_ticks: Maximum number of ticks to run (default 100).
                       Pass None for unlimited execution.
            start_tick: Starting tick count for resume (default 0)
        """
        # Create or update RunnerStateComponent
        runner_state_entities = list(world.query(RunnerStateComponent))
        if runner_state_entities:
            runner_state_entity, (runner_state,) = runner_state_entities[0]
        else:
            runner_state_entity = world.create_entity()
            runner_state = RunnerStateComponent(current_tick=start_tick)
            world.add_component(runner_state_entity, runner_state)

        tick = start_tick
        while True:
            if max_ticks is not None and tick >= max_ticks:
                entity_id = world.create_entity()
                world.add_component(entity_id, TerminalComponent(reason="max_ticks"))
                return

            await world.process()
            tick += 1
            runner_state.current_tick = tick

            has_terminal = any(
                world.has_component(eid, TerminalComponent)
                for eid, _ in world.query(TerminalComponent)
            )
            if has_terminal:
                return

    def save_checkpoint(self, world: World, path: str | Path) -> None:
        """Save world state and runner state to checkpoint file.

        TerminalComponent is excluded from checkpoints to allow resuming.

        Args:
            world: World instance to serialize
            path: Filesystem path for checkpoint JSON file
        """
        checkpoint_path = Path(path)

        # Remove TerminalComponent before serializing (resume-friendly)
        terminal_entities = [eid for eid, _ in world.query(TerminalComponent)]
        for eid in terminal_entities:
            world.remove_component(eid, TerminalComponent)

        world_data = WorldSerializer.to_dict(world)

        # Extract runner state from world
        runner_state_entities = list(world.query(RunnerStateComponent))
        if runner_state_entities:
            _, (runner_state,) = runner_state_entities[0]
            current_tick = runner_state.current_tick
        else:
            current_tick = 0

        # Combine world data and runner state
        checkpoint_data = {
            **world_data,
            "runner_state": {
                "current_tick": current_tick,
            },
        }

        checkpoint_path.write_text(
            json.dumps(checkpoint_data, indent=2), encoding="utf-8"
        )

    @classmethod
    def load_checkpoint(
        cls,
        path: str | Path,
        providers: dict[str, Any],
        tool_handlers: dict[str, Any],
    ) -> tuple[World, int]:
        """Load world state and runner state from checkpoint file.

        Args:
            path: Filesystem path to checkpoint JSON file
            providers: Provider instances keyed by model name
            tool_handlers: Tool handler functions keyed by tool name

        Returns:
            Tuple of (restored World, current_tick)

        Raises:
            FileNotFoundError: If checkpoint file does not exist
        """
        checkpoint_path = Path(path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        checkpoint_data = json.loads(checkpoint_path.read_text(encoding="utf-8"))

        # Extract runner state
        runner_state_data = checkpoint_data.get("runner_state", {})
        current_tick = runner_state_data.get("current_tick", 0)

        # Deserialize world (runner state is already in components)
        world = WorldSerializer.from_dict(
            checkpoint_data, providers=providers, tool_handlers=tool_handlers
        )

        return world, current_tick