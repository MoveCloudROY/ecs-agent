"""Single write path for mirroring PhaseComponent into RuntimeState."""

from __future__ import annotations

from ecs_agent.components import PhaseComponent
from ecs_agent.core import World
from ecs_agent.types import EntityId

from examples.e2e.plan_and_task.phase_graph import PLAN_TASK_PHASE_GRAPH
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


def mirror_phase(world: World, entity_id: EntityId, state: RuntimeState) -> None:
    """Copy the runtime phase and graph hash into the persisted state.

    This is the ONLY place RuntimeState.phase, RuntimeState.graph_hash, and
    RuntimeState.status may be written after a transition. Status is derived
    by derive_status() — handlers never assign it.
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
