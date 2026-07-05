"""Single write path for mirroring PhaseComponent into RuntimeState."""

from __future__ import annotations

from ecs_agent.components import PhaseComponent
from ecs_agent.core import World
from ecs_agent.types import EntityId

from examples.e2e.plan_and_task.phase_graph import PLAN_TASK_PHASE_GRAPH
from examples.e2e.plan_and_task.state_models import RuntimeState


def mirror_phase(world: World, entity_id: EntityId, state: RuntimeState) -> None:
    """Copy the runtime phase into the persisted state and derive status.

    This is the ONLY place RuntimeState.phase may be written after a
    transition. Status defaults to "active" ("completed" for terminal
    phases); handler-specific overrides ("aborted"/"ready"/"blocked"/
    "needs_review") are applied by callers AFTER mirroring.
    """
    component = world.get_component(entity_id, PhaseComponent)
    if component is None:
        raise ValueError("phase graph is not bound; build the world first")
    state.phase = component.phase
    spec = PLAN_TASK_PHASE_GRAPH.phases_by_id[component.phase]
    state.status = "completed" if spec.terminal else "active"
