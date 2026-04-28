"""Tests for workflow gate evaluation and runtime state transitions."""

from __future__ import annotations

from dataclasses import dataclass

from ecs_agent.components import (
    ErrorComponent,
    RunnerStateComponent,
    TerminalComponent,
    WorkflowGateSnapshotComponent,
    WorkflowLastTransitionComponent,
    WorkflowRuntimeComponent,
)
from ecs_agent.core import World
from ecs_agent.types import EntityId
from ecs_agent.systems.workflow_state import WorkflowStateSystem
from ecs_agent.workflows.compiler import install_workflow
from ecs_agent.workflows.contracts import all_of, any_of, absent, field, has, not_, workflow


@dataclass(slots=True)
class DoneMarker:
    pass


@dataclass(slots=True)
class ReviewMarker:
    pass


@dataclass(slots=True)
class MarkerA:
    pass


@dataclass(slots=True)
class MarkerB:
    pass


@dataclass(slots=True)
class StatusComponent:
    status: str


def _install_test_workflow(
    world: World,
    entity_id: EntityId,
    *,
    draft_gate: object,
    review_gate: object | None = None,
) -> None:
    states: dict[str, dict[str, object]] = {
        "draft": {
            "bind": {"agent": "p0"},
            "go": {"review": draft_gate},
        },
        "review": {"bind": {"agent": "p0"}},
    }
    if review_gate is not None:
        states["review"] = {
            "bind": {"agent": "p0"},
            "go": {"done": review_gate},
        }
        states["done"] = {"bind": {"agent": "p0"}}

    spec = workflow(
        "test_flow",
        initial="draft",
        profiles={"agent": {"p0": "Draft prompt"}},
        states=states,
    )
    install_workflow(world, entity_id, spec, agent_key="agent")


async def test_zero_gate_matches_is_noop() -> None:
    world = World()
    entity_id = world.create_entity()
    _install_test_workflow(world, entity_id, draft_gate=has(DoneMarker))

    await WorkflowStateSystem().process(world)

    runtime = world.get_component(entity_id, WorkflowRuntimeComponent)
    snapshot = world.get_component(entity_id, WorkflowGateSnapshotComponent)
    assert runtime is not None
    assert runtime.current_state_id == "draft"
    assert runtime.transition_history == []
    assert snapshot is not None
    assert snapshot.state_id == "draft"
    assert snapshot.matched_transition_id is None


async def test_one_match_commits_transition() -> None:
    world = World()
    entity_id = world.create_entity()
    _install_test_workflow(world, entity_id, draft_gate=has(DoneMarker))
    world.add_component(entity_id, DoneMarker())

    await WorkflowStateSystem().process(world)

    runtime = world.get_component(entity_id, WorkflowRuntimeComponent)
    snapshot = world.get_component(entity_id, WorkflowGateSnapshotComponent)
    last_transition = world.get_component(entity_id, WorkflowLastTransitionComponent)
    assert runtime is not None
    assert runtime.current_state_id == "review"
    assert runtime.transition_history == ["draft_to_review"]
    assert snapshot is not None
    assert snapshot.state_id == "draft"
    assert snapshot.matched_transition_id == "draft_to_review"
    assert last_transition is not None
    assert last_transition.from_state_id == "draft"
    assert last_transition.to_state_id == "review"
    assert last_transition.transition_id == "draft_to_review"


async def test_two_matches_produces_error() -> None:
    world = World()
    entity_id = world.create_entity()
    spec = workflow(
        "ambiguous",
        initial="draft",
        profiles={"agent": {"p0": "Draft prompt"}},
        states={
            "draft": {
                "bind": {"agent": "p0"},
                "go": {
                    "review": has(MarkerA),
                    "done": has(MarkerB),
                },
            },
            "review": {"bind": {"agent": "p0"}},
            "done": {"bind": {"agent": "p0"}},
        },
    )
    install_workflow(world, entity_id, spec, agent_key="agent")
    world.add_component(entity_id, MarkerA())
    world.add_component(entity_id, MarkerB())

    await WorkflowStateSystem().process(world)

    error = world.get_component(entity_id, ErrorComponent)
    terminal = world.get_component(entity_id, TerminalComponent)
    assert error is not None
    assert terminal is not None
    assert terminal.reason == "workflow_ambiguous_transition"


async def test_gate_field_predicate_exact_match() -> None:
    world = World()
    entity_id = world.create_entity()
    _install_test_workflow(world, entity_id, draft_gate=field(StatusComponent, "status") == "done")
    world.add_component(entity_id, StatusComponent(status="done"))

    await WorkflowStateSystem().process(world)

    runtime = world.get_component(entity_id, WorkflowRuntimeComponent)
    assert runtime is not None
    assert runtime.current_state_id == "review"

    world = World()
    entity_id = world.create_entity()
    _install_test_workflow(world, entity_id, draft_gate=field(StatusComponent, "status") == "done")
    world.add_component(entity_id, StatusComponent(status="draft"))

    await WorkflowStateSystem().process(world)

    runtime = world.get_component(entity_id, WorkflowRuntimeComponent)
    assert runtime is not None
    assert runtime.current_state_id == "draft"


async def test_gate_absent_predicate() -> None:
    world = World()
    entity_id = world.create_entity()
    _install_test_workflow(world, entity_id, draft_gate=absent(DoneMarker))

    await WorkflowStateSystem().process(world)

    runtime = world.get_component(entity_id, WorkflowRuntimeComponent)
    assert runtime is not None
    assert runtime.current_state_id == "review"

    world = World()
    entity_id = world.create_entity()
    _install_test_workflow(world, entity_id, draft_gate=absent(DoneMarker))
    world.add_component(entity_id, DoneMarker())

    await WorkflowStateSystem().process(world)

    runtime = world.get_component(entity_id, WorkflowRuntimeComponent)
    assert runtime is not None
    assert runtime.current_state_id == "draft"


async def test_gate_has_predicate() -> None:
    world = World()
    entity_id = world.create_entity()
    _install_test_workflow(world, entity_id, draft_gate=has(DoneMarker))
    world.add_component(entity_id, DoneMarker())

    await WorkflowStateSystem().process(world)

    runtime = world.get_component(entity_id, WorkflowRuntimeComponent)
    assert runtime is not None
    assert runtime.current_state_id == "review"

    world = World()
    entity_id = world.create_entity()
    _install_test_workflow(world, entity_id, draft_gate=has(DoneMarker))

    await WorkflowStateSystem().process(world)

    runtime = world.get_component(entity_id, WorkflowRuntimeComponent)
    assert runtime is not None
    assert runtime.current_state_id == "draft"


async def test_gate_all_of_requires_all() -> None:
    world = World()
    entity_id = world.create_entity()
    _install_test_workflow(world, entity_id, draft_gate=all_of([has(MarkerA), has(MarkerB)]))
    world.add_component(entity_id, MarkerA())

    await WorkflowStateSystem().process(world)

    runtime = world.get_component(entity_id, WorkflowRuntimeComponent)
    assert runtime is not None
    assert runtime.current_state_id == "draft"

    world.add_component(entity_id, MarkerB())

    await WorkflowStateSystem().process(world)

    assert runtime.current_state_id == "review"


async def test_gate_any_of_requires_one() -> None:
    world = World()
    entity_id = world.create_entity()
    _install_test_workflow(world, entity_id, draft_gate=any_of([has(MarkerA), has(MarkerB)]))

    await WorkflowStateSystem().process(world)

    runtime = world.get_component(entity_id, WorkflowRuntimeComponent)
    assert runtime is not None
    assert runtime.current_state_id == "draft"

    world.add_component(entity_id, MarkerA())

    await WorkflowStateSystem().process(world)

    assert runtime.current_state_id == "review"


async def test_gate_not_inverts() -> None:
    world = World()
    entity_id = world.create_entity()
    _install_test_workflow(world, entity_id, draft_gate=not_(has(DoneMarker)))

    await WorkflowStateSystem().process(world)

    runtime = world.get_component(entity_id, WorkflowRuntimeComponent)
    assert runtime is not None
    assert runtime.current_state_id == "review"


async def test_transition_history_appended() -> None:
    world = World()
    entity_id = world.create_entity()
    _install_test_workflow(world, entity_id, draft_gate=has(MarkerA), review_gate=has(MarkerB))
    world.add_component(entity_id, MarkerA())

    system = WorkflowStateSystem()
    await system.process(world)

    world.add_component(entity_id, MarkerB())
    await system.process(world)

    runtime = world.get_component(entity_id, WorkflowRuntimeComponent)
    assert runtime is not None
    assert runtime.current_state_id == "done"
    assert runtime.transition_history == ["draft_to_review", "review_to_done"]


async def test_no_workflow_entity_skipped() -> None:
    world = World()
    entity_id = world.create_entity()
    world.add_component(entity_id, DoneMarker())

    await WorkflowStateSystem().process(world)

    assert world.get_component(entity_id, ErrorComponent) is None
    assert world.get_component(entity_id, TerminalComponent) is None


async def test_tick_from_runner_state_component() -> None:
    world = World()
    state_entity = world.create_entity()
    entity_id = world.create_entity()
    world.add_component(state_entity, RunnerStateComponent(current_tick=7))
    _install_test_workflow(world, entity_id, draft_gate=has(DoneMarker))

    await WorkflowStateSystem().process(world)

    snapshot = world.get_component(entity_id, WorkflowGateSnapshotComponent)
    assert snapshot is not None
    assert snapshot.evaluated_at_tick == 7


async def test_ambiguous_transition_reason_text() -> None:
    world = World()
    entity_id = world.create_entity()
    spec = workflow(
        "ambiguous_text",
        initial="draft",
        profiles={"agent": {"p0": "Draft prompt"}},
        states={
            "draft": {
                "bind": {"agent": "p0"},
                "go": {
                    "review": any_of([has(MarkerA), has(MarkerB)]),
                    "done": has(MarkerA),
                },
            },
            "review": {"bind": {"agent": "p0"}},
            "done": {"bind": {"agent": "p0"}},
        },
    )
    install_workflow(world, entity_id, spec, agent_key="agent")
    world.add_component(entity_id, MarkerA())

    await WorkflowStateSystem().process(world)

    error = world.get_component(entity_id, ErrorComponent)
    assert error is not None
    assert "simultaneously" in error.error
