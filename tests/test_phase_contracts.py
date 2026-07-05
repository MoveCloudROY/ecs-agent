"""Tests for phase-graph contracts: validation and structural hashing."""

from __future__ import annotations

import pytest

from ecs_agent.phases.contracts import ApprovalGate, PhaseSpec, build_graph


def _spec(phase_id: str, **kwargs) -> PhaseSpec:
    defaults = {"prompts": {"main": f"You are in {phase_id}."}}
    defaults.update(kwargs)
    return PhaseSpec(phase_id=phase_id, **defaults)


def _two_phase_graph():
    return build_graph(
        "demo",
        initial="A",
        phases=[
            _spec("A", to=("B",)),
            _spec("B", terminal=True),
        ],
    )


def test_build_graph_valid_two_phase() -> None:
    graph = _two_phase_graph()
    assert graph.graph_id == "demo"
    assert graph.initial == "A"
    assert set(graph.phases_by_id) == {"A", "B"}
    assert len(graph.structure_hash) == 64


def test_build_graph_rejects_duplicate_phase_ids() -> None:
    with pytest.raises(ValueError, match="Duplicate phase_id"):
        build_graph("demo", initial="A", phases=[_spec("A", to=("A",)), _spec("A", terminal=True)])


def test_build_graph_rejects_unknown_initial() -> None:
    with pytest.raises(ValueError, match="Initial phase"):
        build_graph("demo", initial="MISSING", phases=[_spec("A", terminal=True)])


def test_build_graph_rejects_unknown_target() -> None:
    with pytest.raises(ValueError, match="Target 'NOPE'"):
        build_graph("demo", initial="A", phases=[_spec("A", to=("NOPE",))])


def test_phase_spec_rejects_terminal_with_targets() -> None:
    with pytest.raises(ValueError, match="terminal"):
        PhaseSpec(phase_id="X", prompts={"main": "p"}, to=("Y",), terminal=True)


def test_phase_spec_rejects_nonterminal_dead_end() -> None:
    with pytest.raises(ValueError, match="at least one target"):
        PhaseSpec(phase_id="X", prompts={"main": "p"})


def test_build_graph_rejects_approval_target_outside_adjacency() -> None:
    with pytest.raises(ValueError, match="approval verdict"):
        build_graph(
            "demo",
            initial="A",
            phases=[
                _spec("A", to=("B",), approval=ApprovalGate(verdicts={"approved": "C"})),
                _spec("B", to=("C",)),
                _spec("C", terminal=True),
            ],
        )


def test_build_graph_rejects_unknown_on_resume() -> None:
    with pytest.raises(ValueError, match="on_resume"):
        build_graph("demo", initial="A", phases=[_spec("A", to=("A",), on_resume="GHOST")])


def test_approval_gate_requires_mappings() -> None:
    with pytest.raises(ValueError, match="at least one verdict"):
        ApprovalGate(verdicts={})


def test_structure_hash_is_order_insensitive() -> None:
    a = build_graph("demo", initial="A", phases=[_spec("A", to=("B",)), _spec("B", terminal=True)])
    b = build_graph("demo", initial="A", phases=[_spec("B", terminal=True), _spec("A", to=("B",))])
    assert a.structure_hash == b.structure_hash


def test_structure_hash_ignores_prompt_text() -> None:
    a = _two_phase_graph()
    b = build_graph(
        "demo",
        initial="A",
        phases=[
            PhaseSpec(phase_id="A", prompts={"main": "completely different"}, to=("B",)),
            PhaseSpec(phase_id="B", prompts={"main": "also different"}, terminal=True),
        ],
    )
    assert a.structure_hash == b.structure_hash


def test_structure_hash_changes_on_adjacency_change() -> None:
    a = _two_phase_graph()
    b = build_graph(
        "demo",
        initial="A",
        phases=[_spec("A", to=("B",)), _spec("B", to=("A",))],
    )
    assert a.structure_hash != b.structure_hash


def test_graph_mappings_are_immutable() -> None:
    graph = _two_phase_graph()
    with pytest.raises(TypeError):
        graph.phases_by_id["C"] = _spec("C", terminal=True)  # type: ignore[index]
    spec = graph.phases_by_id["A"]
    with pytest.raises(TypeError):
        spec.prompts["main"] = "mutated"  # type: ignore[index]


def test_approval_gate_verdicts_are_immutable() -> None:
    gate = ApprovalGate(verdicts={"approved": None})
    with pytest.raises(TypeError):
        gate.verdicts["approved"] = "SOMEWHERE"  # type: ignore[index]


def test_build_graph_copies_caller_inputs() -> None:
    prompts = {"main": "original"}
    verdicts: dict[str, str | None] = {"approved": None}
    spec_a = PhaseSpec(
        phase_id="A",
        prompts=prompts,
        to=("B",),
        approval=ApprovalGate(verdicts=verdicts),
    )
    graph = build_graph(
        "copy-check",
        initial="A",
        phases=[spec_a, _spec("B", terminal=True)],
    )
    prompts["main"] = "mutated after build"
    verdicts["approved"] = "B"
    frozen_spec = graph.phases_by_id["A"]
    assert frozen_spec.prompts["main"] == "original"
    assert frozen_spec.approval is not None
    assert frozen_spec.approval.verdicts["approved"] is None
