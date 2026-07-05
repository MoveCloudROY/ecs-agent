"""Persist-time snapshot of PhaseComponent into the persisted RuntimeState."""

from __future__ import annotations

from ecs_agent.components import PhaseComponent
from ecs_agent.core import World
from ecs_agent.types import EntityId

from examples.e2e.plan_and_task.phase_graph import PLAN_TASK_PHASE_GRAPH
from examples.e2e.plan_and_task.scratchbook_adapter import (
    PlanTaskScratchbookAdapter as ArtifactAdapter,
)
from examples.e2e.plan_and_task.state_models import ReviewVerdict, RuntimeState


def derive_status(
    phase: str, *, abort_reason: str | None, review_verdicts: list[ReviewVerdict]
) -> str:
    """Pure derivation — status is a view of phase + domain fields, never assigned.

    TASK_READY/TASK_BLOCKED are domain labels for their phases (TASK_BLOCKED is
    only reachable via the graph's on_resume demotion). A phase that declares an
    ApprovalGate but holds no verdict yet is awaiting review.
    """
    spec = PLAN_TASK_PHASE_GRAPH.phases_by_id[phase]
    if spec.terminal:
        return "aborted" if abort_reason else "completed"
    if phase == "TASK_READY":
        return "ready"
    if phase == "TASK_BLOCKED":
        return "blocked"
    if spec.approval is not None and all(v.phase != phase for v in review_verdicts):
        return "needs_review"
    return "active"


def save_state(
    world: World, entity_id: EntityId, state: RuntimeState, adapter: ArtifactAdapter
) -> None:
    """Single state-write path: snapshot PhaseComponent into state, then persist.

    RuntimeState.phase, RuntimeState.graph_hash, and RuntimeState.status are
    stamped here at persist time — no transition site mirrors by hand, so a
    stale phase can never reach disk. In-memory phase guards read
    PhaseComponent (the runtime source of truth), not RuntimeState.
    """
    component = world.get_component(entity_id, PhaseComponent)
    if component is None:
        raise ValueError("phase graph is not bound; build the world first")
    state.phase = component.phase
    state.graph_hash = component.graph_hash
    state.status = derive_status(
        component.phase,
        abort_reason=state.abort_reason,
        review_verdicts=state.review_verdicts,
    )
    adapter.write_state(state)
