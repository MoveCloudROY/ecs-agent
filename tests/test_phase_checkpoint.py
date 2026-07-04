"""Checkpoint/restore contract for phase graphs.

These tests encode the fixes for the audit findings:
- restoring + re-binding must NEVER reset progress (old workflow-DSL rebind bug)
- using a half-restored entity must fail loudly (old silent-freeze bug)
- structural drift is detected; on_resume policy is applied from data
"""

from __future__ import annotations

import pytest

from ecs_agent.components import PhaseComponent
from ecs_agent.core import World
from ecs_agent.phases.api import (
    PhaseGraphMismatchError,
    PhaseIntegrityError,
    advance,
    bind_phase_graph,
)
from ecs_agent.phases.contracts import PhaseSpec, build_graph
from ecs_agent.serialization import WorldSerializer
from ecs_agent.types import PhaseChangedEvent


def _graph(*, running_targets: tuple[str, ...] = ("BLOCKED", "DONE")):
    return build_graph(
        "job",
        initial="READY",
        phases=[
            PhaseSpec(phase_id="READY", prompts={"main": "ready"}, to=("RUNNING",)),
            PhaseSpec(
                phase_id="RUNNING",
                prompts={"main": "running"},
                to=running_targets,
                on_resume="BLOCKED",
            ),
            PhaseSpec(phase_id="BLOCKED", prompts={"main": "blocked"}, to=("RUNNING",)),
            PhaseSpec(phase_id="DONE", prompts={"main": "done"}, terminal=True),
        ],
    )


async def _checkpointed_world_at(phase: str):
    world = World()
    eid = world.create_entity()
    await bind_phase_graph(world, eid, _graph())
    component = world.get_component(eid, PhaseComponent)
    assert component is not None
    if phase != "READY":
        await advance(world, eid, "RUNNING", reason="start")
    data = WorldSerializer.to_dict(world)
    restored = WorldSerializer.from_dict(data, providers={}, tool_handlers={})
    return restored, eid


async def test_restored_entity_without_rebind_fails_loudly() -> None:
    restored, eid = await _checkpointed_world_at("RUNNING")
    with pytest.raises(PhaseIntegrityError, match="half-bound"):
        await advance(restored, eid, "DONE", reason="finish")


async def test_rebind_preserves_progress() -> None:
    restored, eid = await _checkpointed_world_at("READY")
    await bind_phase_graph(restored, eid, _graph())
    component = restored.get_component(eid, PhaseComponent)
    assert component is not None
    assert component.phase == "READY"

    # The critical regression test for the old workflow-DSL rebind bug:
    # re-binding an entity mid-graph must NOT reset it to the initial phase.
    await advance(restored, eid, "RUNNING", reason="start")
    data = WorldSerializer.to_dict(restored)
    restored2 = WorldSerializer.from_dict(data, providers={}, tool_handlers={})
    graph = _graph()
    # RUNNING declares on_resume="BLOCKED": rebind applies the demotion, audited.
    events: list[PhaseChangedEvent] = []

    async def recorder(event: PhaseChangedEvent) -> None:
        events.append(event)

    restored2.event_bus.subscribe(PhaseChangedEvent, recorder)
    await bind_phase_graph(restored2, eid, graph)
    component2 = restored2.get_component(eid, PhaseComponent)
    assert component2 is not None
    assert component2.phase == "BLOCKED"
    assert events[0].reason == "on_resume"
    assert events[0].forced is True


async def test_rebind_rejects_different_graph_id() -> None:
    restored, eid = await _checkpointed_world_at("READY")
    other = build_graph(
        "other-job",
        initial="A",
        phases=[PhaseSpec(phase_id="A", prompts={"main": "a"}, terminal=True)],
    )
    with pytest.raises(PhaseGraphMismatchError, match="refusing to bind"):
        await bind_phase_graph(restored, eid, other)


async def test_rebind_rejects_phase_removed_from_graph() -> None:
    restored, eid = await _checkpointed_world_at("RUNNING")
    shrunk = build_graph(
        "job",
        initial="READY",
        phases=[
            PhaseSpec(phase_id="READY", prompts={"main": "ready"}, to=("DONE",)),
            PhaseSpec(phase_id="DONE", prompts={"main": "done"}, terminal=True),
        ],
    )
    with pytest.raises(PhaseGraphMismatchError, match="no longer exists"):
        await bind_phase_graph(restored, eid, shrunk)


async def test_rebind_tolerates_structural_drift_when_phase_survives() -> None:
    restored, eid = await _checkpointed_world_at("READY")
    drifted = _graph(running_targets=("BLOCKED", "DONE", "READY"))
    await bind_phase_graph(restored, eid, drifted)
    component = restored.get_component(eid, PhaseComponent)
    assert component is not None
    assert component.phase == "READY"
    assert component.graph_hash == drifted.structure_hash
