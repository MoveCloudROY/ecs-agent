"""Workflow state transition system driven by gate expressions."""

from __future__ import annotations

import time

from ecs_agent.components import (
    ErrorComponent,
    RunnerStateComponent,
    TerminalComponent,
    WorkflowDefinitionComponent,
    WorkflowGateSnapshotComponent,
    WorkflowLastTransitionComponent,
    WorkflowRuntimeComponent,
)
from ecs_agent.core import World
from ecs_agent.types import EntityId, WorkflowStateEvaluatedEvent
from ecs_agent.workflows.compiler import CompiledWorkflow
from ecs_agent.workflows.contracts import (
    AbsentPredicate,
    AllOf,
    AnyOf,
    FieldPredicate,
    GateExpr,
    HasPredicate,
    Not,
)


def _evaluate_gate(gate: GateExpr, world: World, entity_id: EntityId) -> bool:
    if isinstance(gate, FieldPredicate):
        component = world.get_component(entity_id, gate.component_type)
        if component is None:
            return False
        return getattr(component, gate.field_name, None) == gate.value
    if isinstance(gate, AbsentPredicate):
        return world.get_component(entity_id, gate.component_type) is None
    if isinstance(gate, HasPredicate):
        return world.get_component(entity_id, gate.component_type) is not None
    if isinstance(gate, AllOf):
        return all(_evaluate_gate(predicate, world, entity_id) for predicate in gate.predicates)
    if isinstance(gate, AnyOf):
        return any(_evaluate_gate(predicate, world, entity_id) for predicate in gate.predicates)
    if isinstance(gate, Not):
        return not _evaluate_gate(gate.predicate, world, entity_id)
    raise ValueError(f"Unknown gate type: {type(gate).__name__}")


def _get_current_tick(world: World) -> int:
    runner_states = world.query(RunnerStateComponent)
    if runner_states:
        _, (runner_state,) = runner_states[0]
        return int(runner_state.current_tick)
    return 0


def _process_entity(
    world: World,
    entity_id: EntityId,
    compiled: CompiledWorkflow,
    runtime: WorkflowRuntimeComponent,
    tick: int,
) -> WorkflowStateEvaluatedEvent:
    transitions = compiled.transitions_by_state.get(runtime.current_state_id, ())
    matched = [transition for transition in transitions if _evaluate_gate(transition.gate, world, entity_id)]
    matched_transition_ids = [transition.transition_id for transition in matched]

    if len(matched) == 0:
        world.add_component(
            entity_id,
            WorkflowGateSnapshotComponent(
                state_id=runtime.current_state_id,
                evaluated_at_tick=tick,
                matched_transition_id=None,
            ),
        )
        return WorkflowStateEvaluatedEvent(
            entity_id=entity_id,
            workflow_id=compiled.workflow_id,
            state_id=runtime.current_state_id,
            current_state_id=runtime.current_state_id,
            tick=tick,
            matched_transition_ids=matched_transition_ids,
            transition_history=list(runtime.transition_history),
            status="no_match",
        )

    if len(matched) == 1:
        transition = matched[0]
        from_state = runtime.current_state_id
        runtime.current_state_id = transition.target_state_id
        runtime.transition_history.append(transition.transition_id)
        world.add_component(
            entity_id,
            WorkflowLastTransitionComponent(
                from_state_id=from_state,
                to_state_id=transition.target_state_id,
                transition_id=transition.transition_id,
                tick=tick,
            ),
        )
        world.add_component(
            entity_id,
            WorkflowGateSnapshotComponent(
                state_id=from_state,
                evaluated_at_tick=tick,
                matched_transition_id=transition.transition_id,
            ),
        )
        return WorkflowStateEvaluatedEvent(
            entity_id=entity_id,
            workflow_id=compiled.workflow_id,
            state_id=from_state,
            current_state_id=runtime.current_state_id,
            tick=tick,
            matched_transition_ids=matched_transition_ids,
            committed_transition_id=transition.transition_id,
            from_state_id=from_state,
            to_state_id=transition.target_state_id,
            transition_history=list(runtime.transition_history),
            status="transition",
        )

    ids = ", ".join(transition.transition_id for transition in matched)
    error = (
        f"WorkflowStateSystem: {len(matched)} transitions matched simultaneously "
        f"from state {runtime.current_state_id!r}: {ids}"
    )
    world.add_component(
        entity_id,
        ErrorComponent(
            error=error,
            system_name="WorkflowStateSystem",
            timestamp=time.time(),
        ),
    )
    world.add_component(entity_id, TerminalComponent(reason="workflow_ambiguous_transition"))
    return WorkflowStateEvaluatedEvent(
        entity_id=entity_id,
        workflow_id=compiled.workflow_id,
        state_id=runtime.current_state_id,
        current_state_id=runtime.current_state_id,
        tick=tick,
        matched_transition_ids=matched_transition_ids,
        transition_history=list(runtime.transition_history),
        status="ambiguous",
        error=error,
    )


class WorkflowStateSystem:
    """Evaluates workflow gates and commits workflow runtime transitions.

    Recommended registration priority is -25 for workflow-enabled agents so
    trigger-mutating prompt normalization can run earlier (for example at -30)
    and system prompt rendering can run later (for example at -20), ensuring
    rendered prompts observe the committed workflow state.
    """

    def __init__(self, priority: int = -25) -> None:
        self.priority = priority

    async def process(self, world: World) -> None:
        tick = _get_current_tick(world)
        for entity_id, (definition, runtime) in world.query(
            WorkflowDefinitionComponent, WorkflowRuntimeComponent
        ):
            event = _process_entity(world, entity_id, definition.compiled, runtime, tick)
            await world.event_bus.publish(event)
