"""Tests for first-class approval gates."""

from __future__ import annotations

import pytest

from ecs_agent.components import PhaseApprovalsComponent, PhaseComponent
from ecs_agent.core import World
from ecs_agent.phases.api import (
    PhaseError,
    bind_phase_graph,
    latest_verdicts,
    record_approval,
)
from ecs_agent.phases.contracts import ApprovalGate, PhaseSpec, build_graph


def _review_graph():
    return build_graph(
        "review-flow",
        initial="QA_REVIEW",
        phases=[
            PhaseSpec(
                phase_id="QA_REVIEW",
                prompts={"main": "review"},
                to=("WRITE", "INTERVIEW"),
                approval=ApprovalGate(
                    verdicts={"approved": "WRITE", "revise": "INTERVIEW", "blocked": None}
                ),
            ),
            PhaseSpec(phase_id="WRITE", prompts={"main": "write"}, to=("QA_REVIEW",)),
            PhaseSpec(phase_id="INTERVIEW", prompts={"main": "interview"}, to=("QA_REVIEW",)),
        ],
    )


async def _bound():
    world = World()
    eid = world.create_entity()
    await bind_phase_graph(world, eid, _review_graph())
    return world, eid


async def test_approved_advances_and_records() -> None:
    world, eid = await _bound()
    result = await record_approval(world, eid, "approved", notes="lgtm")
    assert result == "WRITE"
    component = world.get_component(eid, PhaseComponent)
    assert component is not None
    assert component.phase == "WRITE"
    assert component.history[-1]["reason"] == "approval:approved"
    ledger = world.get_component(eid, PhaseApprovalsComponent)
    assert ledger is not None
    assert ledger.records[-1]["phase"] == "QA_REVIEW"
    assert ledger.records[-1]["verdict"] == "approved"
    assert ledger.records[-1]["notes"] == "lgtm"


async def test_revise_routes_to_revise_target() -> None:
    world, eid = await _bound()
    result = await record_approval(world, eid, "revise")
    assert result == "INTERVIEW"


async def test_blocked_records_but_stays() -> None:
    world, eid = await _bound()
    result = await record_approval(world, eid, "blocked")
    assert result == "QA_REVIEW"
    component = world.get_component(eid, PhaseComponent)
    assert component is not None
    assert component.phase == "QA_REVIEW"


async def test_invalid_verdict_raises() -> None:
    world, eid = await _bound()
    with pytest.raises(PhaseError, match="invalid verdict 'maybe'"):
        await record_approval(world, eid, "maybe")


async def test_phase_without_gate_raises() -> None:
    world, eid = await _bound()
    await record_approval(world, eid, "approved")  # now in WRITE, which has no gate
    with pytest.raises(PhaseError, match="declares no approval gate"):
        await record_approval(world, eid, "approved")


async def test_latest_verdicts_returns_most_recent_per_phase() -> None:
    world, eid = await _bound()
    await record_approval(world, eid, "blocked")
    await record_approval(world, eid, "approved")
    assert latest_verdicts(world, eid) == {"QA_REVIEW": "approved"}
