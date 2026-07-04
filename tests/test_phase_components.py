"""Serialization round-trip for phase components."""

from __future__ import annotations

from ecs_agent.components import (
    PhaseApprovalsComponent,
    PhaseComponent,
    PhaseDefinitionComponent,
)
from ecs_agent.core import World
from ecs_agent.phases.contracts import PhaseSpec, build_graph
from ecs_agent.serialization import WorldSerializer


def _graph():
    return build_graph(
        "demo",
        initial="A",
        phases=[
            PhaseSpec(phase_id="A", prompts={"main": "a"}, to=("B",)),
            PhaseSpec(phase_id="B", prompts={"main": "b"}, terminal=True),
        ],
    )


def test_phase_component_round_trips() -> None:
    world = World()
    eid = world.create_entity()
    world.add_component(
        eid,
        PhaseComponent(
            graph_id="demo",
            phase="B",
            graph_hash="h" * 64,
            agent_key="main",
            entered_at_tick=7,
            history=[{"from": "A", "to": "B", "reason": "test", "forced": False, "tick": 7}],
        ),
    )

    restored = WorldSerializer.from_dict(
        WorldSerializer.to_dict(world), providers={}, tool_handlers={}
    )
    component = restored.get_component(eid, PhaseComponent)
    assert component is not None
    assert component.phase == "B"
    assert component.entered_at_tick == 7
    assert component.history[0]["reason"] == "test"


def test_phase_definition_component_is_skipped() -> None:
    world = World()
    eid = world.create_entity()
    world.add_component(eid, PhaseComponent(graph_id="demo", phase="A", graph_hash="x"))
    world.add_component(eid, PhaseDefinitionComponent(graph=_graph()))

    data = WorldSerializer.to_dict(world)
    entity_data = data["entities"][str(eid)]
    assert PhaseComponent.__name__ in entity_data
    assert PhaseDefinitionComponent.__name__ not in entity_data


def test_phase_approvals_component_round_trips() -> None:
    world = World()
    eid = world.create_entity()
    world.add_component(
        eid,
        PhaseApprovalsComponent(
            records=[{"phase": "A", "verdict": "approved", "notes": None, "decided_at": "t"}]
        ),
    )

    restored = WorldSerializer.from_dict(
        WorldSerializer.to_dict(world), providers={}, tool_handlers={}
    )
    ledger = restored.get_component(eid, PhaseApprovalsComponent)
    assert ledger is not None
    assert ledger.records[0]["verdict"] == "approved"
