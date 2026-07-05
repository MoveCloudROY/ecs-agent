"""Tests for the explicit phase transition API."""

from __future__ import annotations

import pytest

from ecs_agent.components import (
    PermissionComponent,
    PhaseComponent,
    PhaseDefinitionComponent,
)
from ecs_agent.core import World
from ecs_agent.phases.api import (
    HISTORY_LIMIT,
    InvalidPhaseTransitionError,
    PhaseError,
    PhaseIntegrityError,
    advance,
    allowed_targets,
    bind_phase_graph,
    force,
    is_terminal,
)
from ecs_agent.phases.contracts import PhaseSpec, build_graph
from ecs_agent.types import PhaseChangedEvent


def _graph():
    return build_graph(
        "demo",
        initial="DRAFT",
        phases=[
            PhaseSpec(
                phase_id="DRAFT",
                prompts={"main": "You draft."},
                to=("REVIEW",),
                tools=("submit_draft",),
            ),
            PhaseSpec(phase_id="REVIEW", prompts={"main": "You review."}, to=("DRAFT", "DONE")),
            PhaseSpec(phase_id="DONE", prompts={"main": "Done."}, terminal=True),
        ],
    )


async def _bound_world():
    world = World()
    eid = world.create_entity()
    await bind_phase_graph(world, eid, _graph())
    return world, eid


async def test_bind_fresh_entity_starts_at_initial_with_effects() -> None:
    world, eid = await _bound_world()
    component = world.get_component(eid, PhaseComponent)
    assert component is not None
    assert component.phase == "DRAFT"
    assert component.agent_key == "main"
    assert world.get_component(eid, PhaseDefinitionComponent) is not None
    permissions = world.get_component(eid, PermissionComponent)
    assert permissions is not None
    assert permissions.allowed_tools == ["submit_draft"]


async def test_bind_rejects_agent_key_missing_from_a_phase() -> None:
    world = World()
    eid = world.create_entity()
    with pytest.raises(PhaseError, match="agent_key 'ghost'"):
        await bind_phase_graph(world, eid, _graph(), agent_key="ghost")


async def test_advance_commits_valid_transition_and_publishes_event() -> None:
    world, eid = await _bound_world()
    events: list[PhaseChangedEvent] = []

    async def recorder(event: PhaseChangedEvent) -> None:
        events.append(event)

    world.event_bus.subscribe(PhaseChangedEvent, recorder)
    await advance(world, eid, "REVIEW", reason="draft submitted")

    component = world.get_component(eid, PhaseComponent)
    assert component is not None
    assert component.phase == "REVIEW"
    assert component.history[-1] == {
        "from": "DRAFT",
        "to": "REVIEW",
        "reason": "draft submitted",
        "forced": False,
        "tick": 0,
    }
    assert len(events) == 1
    assert events[0].from_phase == "DRAFT"
    assert events[0].to_phase == "REVIEW"
    assert events[0].forced is False


async def test_advance_rejects_non_adjacent_target() -> None:
    world, eid = await _bound_world()
    with pytest.raises(InvalidPhaseTransitionError, match="allowed: \\['REVIEW'\\]"):
        await advance(world, eid, "DONE", reason="skip ahead")
    component = world.get_component(eid, PhaseComponent)
    assert component is not None
    assert component.phase == "DRAFT"


async def test_advance_rejects_from_terminal_phase() -> None:
    world, eid = await _bound_world()
    await advance(world, eid, "REVIEW", reason="r")
    await advance(world, eid, "DONE", reason="d")
    with pytest.raises(InvalidPhaseTransitionError, match="terminal"):
        await advance(world, eid, "REVIEW", reason="undo")


async def test_force_bypasses_adjacency_and_is_audited() -> None:
    world, eid = await _bound_world()
    await force(world, eid, "DONE", reason="admin recovery")
    component = world.get_component(eid, PhaseComponent)
    assert component is not None
    assert component.phase == "DONE"
    assert component.history[-1]["forced"] is True


async def test_force_rejects_unknown_phase() -> None:
    world, eid = await _bound_world()
    with pytest.raises(PhaseError, match="unknown phase 'GHOST'"):
        await force(world, eid, "GHOST", reason="oops")


async def test_half_bound_entity_raises_integrity_error() -> None:
    world = World()
    eid = world.create_entity()
    world.add_component(eid, PhaseComponent(graph_id="demo", phase="DRAFT", graph_hash="x"))
    with pytest.raises(PhaseIntegrityError, match="bind_phase_graph"):
        await advance(world, eid, "REVIEW", reason="r")


async def test_unbound_entity_raises_integrity_error() -> None:
    world = World()
    eid = world.create_entity()
    with pytest.raises(PhaseIntegrityError, match="no phase graph"):
        await advance(world, eid, "REVIEW", reason="r")


async def test_history_is_bounded() -> None:
    world, eid = await _bound_world()
    for _ in range(HISTORY_LIMIT):
        await advance(world, eid, "REVIEW", reason="loop")
        await advance(world, eid, "DRAFT", reason="loop")
    component = world.get_component(eid, PhaseComponent)
    assert component is not None
    assert len(component.history) == HISTORY_LIMIT


async def test_read_helpers() -> None:
    world, eid = await _bound_world()
    assert allowed_targets(world, eid) == frozenset({"REVIEW"})
    assert is_terminal(world, eid) is False
    await force(world, eid, "DONE", reason="admin")
    assert is_terminal(world, eid) is True


def _tool_graph():
    return build_graph(
        "tooling",
        initial="RESTRICTED",
        phases=[
            PhaseSpec(
                phase_id="RESTRICTED",
                prompts={"main": "restricted"},
                to=("OPEN",),
                tools=("submit_draft",),
            ),
            PhaseSpec(phase_id="OPEN", prompts={"main": "open"}, terminal=True),
        ],
    )


async def test_phase_without_tools_clears_allowlist_when_graph_manages_tools() -> None:
    world = World()
    eid = world.create_entity()
    await bind_phase_graph(world, eid, _tool_graph())
    permissions = world.get_component(eid, PermissionComponent)
    assert permissions is not None
    assert permissions.allowed_tools == ["submit_draft"]

    await advance(world, eid, "OPEN", reason="release restriction")
    permissions = world.get_component(eid, PermissionComponent)
    assert permissions is not None
    # OPEN declares no tools: the managing graph clears the restriction
    # (empty allowed_tools == unrestricted under PermissionSystem semantics).
    assert permissions.allowed_tools == []


async def test_denied_tools_never_touched_by_phase_effects() -> None:
    world = World()
    eid = world.create_entity()
    world.add_component(
        eid, PermissionComponent(allowed_tools=[], denied_tools=["rm_rf"])
    )
    await bind_phase_graph(world, eid, _tool_graph())
    await advance(world, eid, "OPEN", reason="r")
    permissions = world.get_component(eid, PermissionComponent)
    assert permissions is not None
    assert permissions.denied_tools == ["rm_rf"]


async def test_non_managing_graph_never_touches_permissions() -> None:
    world = World()
    eid = world.create_entity()
    world.add_component(
        eid, PermissionComponent(allowed_tools=["user_tool"], denied_tools=[])
    )
    graph = build_graph(
        "no-tools",
        initial="A",
        phases=[
            PhaseSpec(phase_id="A", prompts={"main": "a"}, to=("B",)),
            PhaseSpec(phase_id="B", prompts={"main": "b"}, terminal=True),
        ],
    )
    await bind_phase_graph(world, eid, graph)
    await advance(world, eid, "B", reason="r")
    permissions = world.get_component(eid, PermissionComponent)
    assert permissions is not None
    assert permissions.allowed_tools == ["user_tool"]


def test_manages_tools_flag() -> None:
    assert _tool_graph().manages_tools is True
    plain = build_graph(
        "plain",
        initial="A",
        phases=[PhaseSpec(phase_id="A", prompts={"main": "a"}, terminal=True)],
    )
    assert plain.manages_tools is False
