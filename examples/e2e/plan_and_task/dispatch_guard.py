"""Pre-flight review-order guard for `subagent` dispatches.

The controller already refuses an out-of-order review — but it refuses when the
verdict is *recorded*, which is after the subagent has run. In a live session
that cost five full reviewer runs whose output was thrown away: QA dispatched
before the advisor was approved, an advisor re-run after the phase had moved on,
and plan-QA dispatched two phases early.

This guard applies the controller's own two gates *before* the dispatch, so an
out-of-order review costs one tool call instead of a subagent run. It returns a
corrective string rather than raising: the model gets the reason and the next
action in the tool result and can act on it in the same turn.
"""

from __future__ import annotations

from typing import Any

from ecs_agent.components.definitions import PhaseComponent, ToolRegistryComponent
from ecs_agent.core.world import World
from ecs_agent.logging import get_logger
from ecs_agent.types import EntityId

from examples.e2e.plan_and_task.controller import describe_review_gate, latest_verdict
from examples.e2e.plan_and_task.phase_graph import (
    PLAN_TASK_PHASE_GRAPH,
    REVIEW_PHASE_BY_CATEGORY,
)
from examples.e2e.plan_and_task.state_models import RuntimeState

logger = get_logger(__name__)

_OUT_OF_ORDER_PREFIX = "Error: out-of-order subagent dispatch."


def review_order_refusal(
    current_phase: str, category: str, state: RuntimeState | None
) -> str | None:
    """Refusal text for dispatching `category` from `current_phase`, else None.

    Mirrors the two gates in ``PlanController._record_review``: the target phase
    must be the current one or reachable from it, and leaving a review phase
    requires that phase's own verdict to be 'approved'.

    Args:
        current_phase: the workflow's phase right now.
        category: the subagent category being dispatched.
        state: runtime state holding the verdict ledger; None means no workflow
            is loaded yet, in which case the guard defers.
    """
    target = REVIEW_PHASE_BY_CATEGORY.get(category)
    if target is None or state is None:
        # A free-form or non-review category, or nothing to check against.
        return None
    spec = PLAN_TASK_PHASE_GRAPH.phases_by_id.get(current_phase)
    if spec is None or target == current_phase:
        return None

    reachable = spec.to
    if target not in reachable:
        return (
            f"{_OUT_OF_ORDER_PREFIX} The {category} subagent drives {target}, "
            f"but the workflow is at {current_phase!r} and {target} is not "
            f"reachable from it. {describe_review_gate(state, current_phase, reachable)}"
        )
    if spec.approval is not None and latest_verdict(state, current_phase) != "approved":
        return (
            f"{_OUT_OF_ORDER_PREFIX} The {category} subagent drives {target}, "
            f"which cannot be entered yet. "
            f"{describe_review_gate(state, current_phase, reachable)}"
        )
    return None


def install_review_order_dispatch_guard(
    world: World,
    entity_id: EntityId,
    runtime_state_ref: list[RuntimeState | None],
    *,
    tool_name: str = "subagent",
) -> None:
    """Wrap the installed `subagent` handler with the review-order check.

    Must run after ``install_subagent_tool``; wrapping is idempotent per world
    because the wrapper replaces the entry it captured.
    """
    registry = world.get_component(entity_id, ToolRegistryComponent)
    if registry is None or tool_name not in registry.handlers:
        raise ValueError(f"Entity {entity_id} has no {tool_name!r} tool to guard")

    inner = registry.handlers[tool_name]

    async def guarded(**kwargs: Any) -> Any:
        category = str(kwargs.get("category", ""))
        phase_component = world.get_component(entity_id, PhaseComponent)
        current = (
            phase_component.phase
            if phase_component is not None
            else PLAN_TASK_PHASE_GRAPH.initial
        )
        refusal = review_order_refusal(current, category, runtime_state_ref[0])
        if refusal is not None:
            logger.warning(
                "plan_task_dispatch_out_of_order",
                category=category,
                current_phase=current,
                target_phase=REVIEW_PHASE_BY_CATEGORY.get(category),
            )
            return refusal
        return await inner(**kwargs)

    registry.handlers[tool_name] = guarded
    logger.info(
        "plan_task_dispatch_guard_installed", entity_id=entity_id, tool_name=tool_name
    )
