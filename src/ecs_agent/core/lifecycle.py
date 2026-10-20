"""Shared turn state and execution eligibility for agent-processing systems."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from uuid import uuid4

from ecs_agent.components import AgentTurnComponent, ChildStubComponent
from ecs_agent.core.run_result import RunResult, RunStatus
from ecs_agent.types import EntityId

if TYPE_CHECKING:
    from ecs_agent.core.runner import Runner
    from ecs_agent.core.world import World


@dataclass(slots=True)
class ActiveAgentRun:
    """World-owned driver identity; deliberately excluded from checkpoints."""

    entity: EntityId
    runner: Runner
    run_id: str = field(default_factory=lambda: uuid4().hex)
    input_text: str | None = None
    task: asyncio.Task[RunResult] | None = None


def is_entity_active(world: World, entity: EntityId) -> bool:
    """Check execution scope and managed turn eligibility for a custom system.

    Delegation stubs never execute. Child Worlds retain their own execution
    driver; ownership alone does not activate an unrelated parent-World entity.
    Unmanaged entities preserve legacy behavior outside a scoped agent run.
    """
    active = world._agent_run
    if active is not None and active.entity != entity:
        return False
    if world.has_component(entity, ChildStubComponent):
        return False
    turn = world.get_component(entity, AgentTurnComponent)
    return turn is None or turn.status == "running"


def finish_turn(world: World, result: RunResult) -> None:
    """Commit a managed turn's outcome before completed-event observers run."""
    if result.entity_id is None or result.turn_id is None:
        return
    turn = world.get_component(result.entity_id, AgentTurnComponent)
    if turn is None or turn.turn_id != result.turn_id:
        return
    turn.reason = result.reason
    if turn.input_normalization_state == "running":
        turn.status = "blocked"
        turn.reason = "Input normalization completion is unconfirmed; reconcile the trigger before continuing"
    elif turn.unconfirmed_tool_calls:
        turn.status = "blocked"
        turn.reason = "Tool completion is unconfirmed: " + ", ".join(turn.unconfirmed_tool_calls)
    elif result.success:
        turn.status = "completed"
    elif result.status in {RunStatus.MAX_TICKS, RunStatus.CANCELLED, RunStatus.INTERRUPTED}:
        turn.status = "paused"
    else:
        turn.status = "failed"
    world._agent_results[result.entity_id] = result
