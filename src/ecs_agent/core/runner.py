"""Runner for ECS-based LLM Agent with checkpoint resume support."""

import asyncio
import json
import time
import uuid
from dataclasses import asdict, replace
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ecs_agent.components.definitions import (
    ChildStubComponent,
    AgentTurnComponent,
    ConversationComponent,
    ErrorComponent,
    InterruptionComponent,
    LLMComponent,
    RunnerStateComponent,
    SessionRefComponent,
    TerminalComponent,
)
from ecs_agent.types import (
    EntityId,
    RunCompletedEvent,
    RunStartedEvent,
    RunnerLifecycleStatus,
    RunnerTickCompletedEvent,
    RunnerTickStartedEvent,
    UserInputReceivedEvent,
)
from ecs_agent.core.world import World
from ecs_agent.core.run_result import RunResult, RunStatus, snapshot_run_result
from ecs_agent.core.lifecycle import finish_turn
from ecs_agent.history import record_session_event
from ecs_agent.logging import STANDARD_EVENT_NAMES, get_logger
from ecs_agent.observability.context import current_run_id, reset_run_context, set_run_context
from ecs_agent.serialization import WorldSerializer

logger = get_logger(__name__)


class Runner:
    """Orchestrates the main execution loop."""

    def __init__(self) -> None:
        self.last_result: RunResult | None = None
        self._running = False

    async def run(
        self,
        world: World,
        max_ticks: int | None = 100,
        start_tick: int = 0,
        *,
        trace_id: str | None = None,
        run_id: str | None = None,
        parent_observation_id: str | None = None,
        emit_root_trace: bool = True,
        target_entity: EntityId | None = None,
        cancellation_reason: Callable[[], str] | None = None,
    ) -> RunResult:
        """Run the main execution loop until terminal condition.

        Executes world.process() repeatedly until either:
        1. A TerminalComponent is found on any entity
        2. The legacy absolute max_ticks limit is reached (from start_tick)

        When max_ticks is None the loop runs indefinitely until a
        TerminalComponent appears (useful for interactive / chat agents).

        Budget exhaustion returns MAX_TICKS without adding a terminal entity.
        Existing terminal states are checked before executing any systems.

        Args:
            world: World instance to process
            max_ticks: Absolute tick limit (default 100). With start_tick=5,
                       max_ticks=8 allows three ticks. None means unlimited.
            start_tick: Starting tick count for resume (default 0)
            target_entity: Only this entity's terminal/interruption stops the
                run, when supplied. Used by delegation to ignore other entities.
            cancellation_reason: Optional enclosing deadline classifier, evaluated
                during cancellation before publishing the terminal snapshot.

        Returns:
            Semantic run result, also carried by RunCompletedEvent. External
            cancellation and exceptions still propagate; last_result retains
            their terminal snapshot.
        """
        if start_tick < 0 or (max_ticks is not None and max_ticks < 0):
            logger.error("invalid_run_budget", start_tick=start_tick, max_ticks=max_ticks)
            raise ValueError("start_tick and max_ticks must be non-negative")
        if any(
            turn.status == "legacy" and (target_entity is None or entity == target_entity)
            for entity, (turn,) in world.query(AgentTurnComponent)
        ):
            logger.error("legacy_turn_requires_adoption", world_name=world.name)
            raise ValueError("Legacy checkpoint requires explicit AgentHandle.adopt_turn() before continuation")
        if world._runner_task is not None or world._processing_task is not None or self._running or (
            world._agent_run is not None and world._agent_run.task is not asyncio.current_task()
        ):
            logger.error("runner_already_active", world_name=world.name)
            raise RuntimeError("A Runner is already driving this World or Runner instance")
        world._runner_task = asyncio.current_task()
        self._running = True
        resolved_trace_id = uuid.uuid4().hex if trace_id is None else trace_id
        resolved_run_id = str(uuid.uuid4()) if run_id is None else run_id
        context_token = set_run_context(
            trace_id=resolved_trace_id,
            run_id=resolved_run_id,
        )
        run_start_time = time.monotonic()
        self.last_result = None
        tick = start_tick
        completion_published = False
        try:
            for session_entity, _ in world.query(SessionRefComponent):
                if not world.has_component(session_entity, ChildStubComponent) and (target_entity is None or session_entity == target_entity):
                    record_session_event(world, session_entity, "run_started", {"status": "running", "start_tick": start_tick})
            logger.info(
                STANDARD_EVENT_NAMES["RUN_START"],
                max_ticks=max_ticks,
                start_tick=start_tick,
                world_name=world.name,
            )
            turn = world.get_component(target_entity, AgentTurnComponent) if target_entity is not None else None
            await world.event_bus.publish(
                RunStartedEvent(
                    max_ticks=max_ticks,
                    start_tick=start_tick,
                    active_entities=self._active_entity_count(world),
                    parent_observation_id=parent_observation_id,
                    emit_root_trace=emit_root_trace,
                    entity_id=target_entity,
                    turn_id=turn.turn_id if turn is not None else None,
                    run_id=resolved_run_id,
                )
            )
            active = world._agent_run
            if active is not None and active.input_text is not None:
                await world.event_bus.publish(UserInputReceivedEvent(
                    entity_id=active.entity, prompt="", text=active.input_text,
                    turn_id=turn.turn_id if turn is not None else None,
                    run_id=resolved_run_id, source="agent_handle",
                ))

            # Create or update RunnerStateComponent
            runner_state_entities = list(world.query(RunnerStateComponent))
            if runner_state_entities:
                runner_state_entity, (runner_state,) = runner_state_entities[0]
            else:
                runner_state_entity = world.create_entity()
                runner_state = RunnerStateComponent(current_tick=start_tick)
                world.add_component(runner_state_entity, runner_state)

            cumulative_start = max(runner_state.current_tick, start_tick)
            runner_state.current_tick = cumulative_start
            tick = start_tick
            while True:
                stopped: RunnerLifecycleStatus | None = None
                if self._has_top_level_component(world, TerminalComponent, target_entity):
                    stopped = "terminal_component"
                elif self._is_interrupted(world, target_entity):
                    stopped = "interruption_component"
                if stopped is not None:
                    result = await self._publish_run_completed(
                        world=world, status=stopped, reason=stopped,
                        start_time=run_start_time, ticks=tick - start_tick,
                        target_entity=target_entity,
                    )
                    completion_published = True
                    return result
                if max_ticks is not None and tick >= max_ticks:
                    logger.info(STANDARD_EVENT_NAMES["RUN_COMPLETE"], reason="max_ticks", world_name=world.name)
                    result = await self._publish_run_completed(
                        world=world,
                        status="max_ticks",
                        reason="max_ticks",
                        start_time=run_start_time,
                        ticks=tick - start_tick,
                        target_entity=target_entity,
                    )
                    completion_published = True
                    return result

                cumulative_tick = cumulative_start + tick - start_tick
                logger.debug(STANDARD_EVENT_NAMES["TICK_START"], tick=cumulative_tick, world_name=world.name)
                tick_start_time = time.monotonic()
                await world.event_bus.publish(
                    RunnerTickStartedEvent(
                        tick=cumulative_tick,
                        active_entities=self._active_entity_count(world),
                    )
                )
                try:
                    world.apply_pending_system_operations()
                    await world.process()
                except asyncio.CancelledError:
                    # CancelledError arrived from outside (e.g. asyncio.wait_for timeout)
                    # — re-raise so the caller's timeout handler fires correctly
                    logger.info(
                        STANDARD_EVENT_NAMES["RUN_COMPLETE"],
                        reason="external_cancellation",
                        world_name=world.name,
                    )
                    result = await self._publish_run_completed(
                        world=world,
                        status="cancelled",
                        reason=cancellation_reason() if cancellation_reason is not None else "external_cancellation",
                        start_time=run_start_time,
                        ticks=tick - start_tick,
                        target_entity=target_entity,
                    )
                    completion_published = True
                    raise
                except Exception as exc:
                    tick_duration_ms = (time.monotonic() - tick_start_time) * 1000
                    await world.event_bus.publish(
                        RunnerTickCompletedEvent(
                            tick=cumulative_tick,
                            status="error",
                            duration_seconds=tick_duration_ms / 1000,
                            active_entities=self._active_entity_count(world),
                        )
                    )
                    result = await self._publish_run_completed(
                        world=world,
                        status="error",
                        reason="exception",
                        error=str(exc) or type(exc).__name__,
                        start_time=run_start_time,
                        ticks=tick - start_tick,
                        target_entity=target_entity,
                    )
                    completion_published = True
                    raise

                tick_duration_ms = (time.monotonic() - tick_start_time) * 1000
                logger.debug(
                    STANDARD_EVENT_NAMES["TICK_COMPLETE"],
                    tick=cumulative_tick,
                    duration_ms=tick_duration_ms,
                    world_name=world.name,
                )
                has_terminal = self._has_top_level_component(world, TerminalComponent, target_entity)
                tick_status: RunnerLifecycleStatus = (
                    "terminal_component" if has_terminal else "success"
                )
                await world.event_bus.publish(
                    RunnerTickCompletedEvent(
                        tick=cumulative_tick,
                        status=tick_status,
                        duration_seconds=tick_duration_ms / 1000,
                        active_entities=self._active_entity_count(world),
                    )
                )

                tick += 1
                runner_state.current_tick = cumulative_start + tick - start_tick

                if has_terminal:
                    logger.info(
                        STANDARD_EVENT_NAMES["RUN_COMPLETE"], reason="terminal_component", world_name=world.name
                    )
                    result = await self._publish_run_completed(
                        world=world,
                        status="terminal_component",
                        reason="terminal_component",
                        start_time=run_start_time,
                        ticks=tick - start_tick,
                        target_entity=target_entity,
                    )
                    completion_published = True
                    return result
        except asyncio.CancelledError:
            if not completion_published:
                logger.info(
                    STANDARD_EVENT_NAMES["RUN_COMPLETE"],
                    reason="external_cancellation",
                    world_name=world.name,
                )
                result = await self._publish_run_completed(
                    world=world,
                    status="cancelled",
                    reason=cancellation_reason() if cancellation_reason is not None else "external_cancellation",
                    start_time=run_start_time,
                    ticks=tick - start_tick,
                    target_entity=target_entity,
                )
            raise
        finally:
            reset_run_context(context_token)
            world._runner_task = None
            self._running = False

    def _is_interrupted(self, world: World, target_entity: EntityId | None) -> bool:
        if target_entity is not None:
            return self._has_top_level_component(world, InterruptionComponent, target_entity)
        agents = {
            entity
            for component_type in (LLMComponent, ConversationComponent, InterruptionComponent)
            for entity, _ in world.query(component_type)
            if not world.has_component(entity, ChildStubComponent)
        }
        # Whole-World execution may keep other agents running after one pauses.
        return bool(agents) and all(
            world.has_component(entity, InterruptionComponent) for entity in agents
        )

    def _has_top_level_component(
        self,
        world: World,
        component_type: type[TerminalComponent] | type[InterruptionComponent],
        target_entity: EntityId | None = None,
    ) -> bool:
        return any(
            not world.has_component(entity, ChildStubComponent)
            and (target_entity is None or entity == target_entity)
            for entity, _ in world.query(component_type)
        )

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
        target_entity: EntityId | None = None,
        error: str | None = None,
    ) -> RunResult:
        result = snapshot_run_result(
            world, run_id=current_run_id(), reason=reason, ticks=ticks,
            target_entity=target_entity, error=error,
        )
        self.last_result = result
        for session_entity, _ in world.query(SessionRefComponent):
            if world.has_component(session_entity, ChildStubComponent) or (target_entity is not None and session_entity != target_entity):
                continue
            entity_result = result if result.entity_id == session_entity else snapshot_run_result(
                world, run_id=current_run_id(), reason=reason, ticks=ticks, target_entity=session_entity, error=error,
            )
            payload = asdict(entity_result)
            # Messages already have their own records; lifecycle entries retain
            # outcome flags without duplicating searchable message text.
            payload["has_final_message"] = payload.pop("final_message") is not None
            payload["has_partial_message"] = payload.pop("partial_message") is not None
            try:
                record_session_event(world, session_entity, "run_finished", payload)
            except Exception as exc:
                self.last_result = replace(
                    result, status=RunStatus.ERROR, reason="session_recording_error",
                    error=str(exc) or type(exc).__name__, final_message=None,
                    partial_message=result.final_message or result.partial_message,
                )
                finish_turn(world, self.last_result)
                world.add_component(session_entity, ErrorComponent(error=str(exc), system_name="Runner", timestamp=time.time()))
                world.add_component(session_entity, TerminalComponent(reason="session_recording_error", error=str(exc)))
                logger.error("session_terminal_record_failed", entity_id=int(session_entity), exception=str(exc))
                raise
        finish_turn(world, result)
        await world.event_bus.publish(
            RunCompletedEvent(
                status=status,
                reason=reason,
                duration_seconds=time.monotonic() - start_time,
                ticks=ticks,
                active_entities=self._active_entity_count(world),
                result=result,
            )
        )
        return result

    def save_checkpoint(self, world: World, path: str | Path) -> None:
        """Save world state and runner state to checkpoint file.

        Turn completion and failure states are preserved. Loading a managed
        running turn pauses it or blocks it when tool completion is uncertain.

        Args:
            world: World instance to serialize
            path: Filesystem path for checkpoint JSON file
        """
        checkpoint_path = Path(path)

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
