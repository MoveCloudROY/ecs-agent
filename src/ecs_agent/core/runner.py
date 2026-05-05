"""Runner for ECS-based LLM Agent with checkpoint resume support."""

import asyncio
import json
import time
import uuid
from pathlib import Path
from typing import Any

from ecs_agent.components.definitions import (
    InterruptionComponent,
    RunnerStateComponent,
    TerminalComponent,
)
from ecs_agent.types import (
    RunCompletedEvent,
    RunStartedEvent,
    RunnerLifecycleStatus,
    RunnerTickCompletedEvent,
    RunnerTickStartedEvent,
)
from ecs_agent.core.world import World
from ecs_agent.logging import STANDARD_EVENT_NAMES, get_logger
from ecs_agent.observability.context import reset_run_context, set_run_context
from ecs_agent.serialization import WorldSerializer

logger = get_logger(__name__)


class Runner:
    """Orchestrates the main execution loop."""

    async def run(
        self, world: World, max_ticks: int | None = 100, start_tick: int = 0
    ) -> None:
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
        context_token = set_run_context(trace_id=uuid.uuid4().hex, run_id=str(uuid.uuid4()))
        run_start_time = time.monotonic()
        tick = start_tick
        completion_published = False
        try:
            logger.info(
                STANDARD_EVENT_NAMES["RUN_START"],
                max_ticks=max_ticks,
                start_tick=start_tick,
                world_name=world.name,
            )
            await world.event_bus.publish(
                RunStartedEvent(
                    max_ticks=max_ticks,
                    start_tick=start_tick,
                    active_entities=self._active_entity_count(world),
                )
            )

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
                    logger.info(STANDARD_EVENT_NAMES["RUN_COMPLETE"], reason="max_ticks", world_name=world.name)
                    await self._publish_run_completed(
                        world=world,
                        status="max_ticks",
                        reason="max_ticks",
                        start_time=run_start_time,
                        ticks=tick - start_tick,
                    )
                    completion_published = True
                    return

                logger.debug(STANDARD_EVENT_NAMES["TICK_START"], tick=tick, world_name=world.name)
                tick_start_time = time.monotonic()
                await world.event_bus.publish(
                    RunnerTickStartedEvent(
                        tick=tick,
                        active_entities=self._active_entity_count(world),
                    )
                )
                interrupted_before_tick = self._has_top_level_component(
                    world, InterruptionComponent
                )

                try:
                    world.apply_pending_system_operations()
                    await world.process()
                except asyncio.CancelledError:
                    if interrupted_before_tick:
                        # Pre-existing interruption — intentional stop, swallow CancelledError
                        logger.info(
                            STANDARD_EVENT_NAMES["RUN_COMPLETE"],
                            reason="interruption_component",
                            world_name=world.name,
                        )
                        await self._publish_run_completed(
                            world=world,
                            status="interruption_component",
                            reason="interruption_component",
                            start_time=run_start_time,
                            ticks=tick - start_tick,
                        )
                        completion_published = True
                        return
                    # CancelledError arrived from outside (e.g. asyncio.wait_for timeout)
                    # — re-raise so the caller's timeout handler fires correctly
                    logger.info(
                        STANDARD_EVENT_NAMES["RUN_COMPLETE"],
                        reason="external_cancellation",
                        world_name=world.name,
                    )
                    await self._publish_run_completed(
                        world=world,
                        status="cancelled",
                        reason="external_cancellation",
                        start_time=run_start_time,
                        ticks=tick - start_tick,
                    )
                    completion_published = True
                    raise
                except Exception:
                    tick_duration_ms = (time.monotonic() - tick_start_time) * 1000
                    await world.event_bus.publish(
                        RunnerTickCompletedEvent(
                            tick=tick,
                            status="error",
                            duration_seconds=tick_duration_ms / 1000,
                            active_entities=self._active_entity_count(world),
                        )
                    )
                    await self._publish_run_completed(
                        world=world,
                        status="error",
                        reason="exception",
                        start_time=run_start_time,
                        ticks=tick - start_tick,
                    )
                    completion_published = True
                    raise

                interrupted_after_tick = self._has_top_level_component(
                    world, InterruptionComponent
                )
                if interrupted_after_tick and not interrupted_before_tick:
                    logger.debug("interruption_detected", tick=tick, world_name=world.name)

                tick_duration_ms = (time.monotonic() - tick_start_time) * 1000
                logger.debug(
                    STANDARD_EVENT_NAMES["TICK_COMPLETE"],
                    tick=tick,
                    duration_ms=tick_duration_ms,
                    world_name=world.name,
                )
                has_terminal = self._has_top_level_component(world, TerminalComponent)
                tick_status: RunnerLifecycleStatus = (
                    "terminal_component" if has_terminal else "success"
                )
                await world.event_bus.publish(
                    RunnerTickCompletedEvent(
                        tick=tick,
                        status=tick_status,
                        duration_seconds=tick_duration_ms / 1000,
                        active_entities=self._active_entity_count(world),
                    )
                )

                tick += 1
                runner_state.current_tick = tick

                if has_terminal:
                    logger.info(
                        STANDARD_EVENT_NAMES["RUN_COMPLETE"], reason="terminal_component", world_name=world.name
                    )
                    await self._publish_run_completed(
                        world=world,
                        status="terminal_component",
                        reason="terminal_component",
                        start_time=run_start_time,
                        ticks=tick - start_tick,
                    )
                    completion_published = True
                    return
        except asyncio.CancelledError:
            if not completion_published:
                logger.info(
                    STANDARD_EVENT_NAMES["RUN_COMPLETE"],
                    reason="external_cancellation",
                    world_name=world.name,
                )
                await self._publish_run_completed(
                    world=world,
                    status="cancelled",
                    reason="external_cancellation",
                    start_time=run_start_time,
                    ticks=tick - start_tick,
                )
            raise
        finally:
            reset_run_context(context_token)

    def _has_top_level_component(
        self,
        world: World,
        component_type: type[TerminalComponent] | type[InterruptionComponent],
    ) -> bool:
        return any(True for _ in world.query(component_type))

    def _active_entity_count(self, world: World) -> int:
        return len(world._entity_ids)

    async def _publish_run_completed(
        self,
        *,
        world: World,
        status: RunnerLifecycleStatus,
        reason: str,
        start_time: float,
        ticks: int,
    ) -> None:
        await world.event_bus.publish(
            RunCompletedEvent(
                status=status,
                reason=reason,
                duration_seconds=time.monotonic() - start_time,
                ticks=ticks,
                active_entities=self._active_entity_count(world),
            )
        )

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
