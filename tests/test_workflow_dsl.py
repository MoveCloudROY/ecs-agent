"""Tests for the public workflow DSL surface."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from ecs_agent import (
    PromptProfileSpec,
    WorkflowSpec,
    all_of,
    any_of,
    absent,
    bind_workflow,
    field,
    has,
    not_,
    prompt_file,
    workflow,
)
from ecs_agent.core import World
from ecs_agent.workflows._components import WorkflowBindingComponent, WorkflowRuntimeComponent
from ecs_agent.workflows.contracts import StateSpec, TransitionSpec


@dataclass(slots=True)
class StatusComponent:
    status: str
    count: int


@dataclass(slots=True)
class ReviewComponent:
    approved: bool


def test_workflow_builds_valid_public_spec_with_shared_profile() -> None:
    spec = workflow(
        "draft-review",
        initial="draft",
        profiles={
            "planner": {
                "shared": "Draft prompt",
                "from_file": prompt_file("prompts/review.txt"),
            }
        },
        states={
            "draft": {
                "bind": {"planner": "shared"},
                "go": {
                    "review": [
                        field(StatusComponent, "status") == "ready",
                        field(StatusComponent, "count") == 2,
                    ]
                },
            },
            "review": {
                "bind": {"planner": "shared"},
                "go": {"done": has(ReviewComponent)},
            },
            "done": {"bind": {"planner": "from_file"}},
        },
    )

    assert isinstance(spec, WorkflowSpec)
    assert spec.workflow_id == "draft-review"
    assert spec.initial_state_id == "draft"
    assert [state.state_id for state in spec.states] == ["draft", "review", "done"]
    assert spec.states[0].transitions[0].target_state_id == "review"
    assert spec.states[0].transitions[0].transition_id == "draft_to_review"
    assert spec.states[1].bind["planner"] == "shared"
    assert spec.profiles["planner"]["shared"].prompt == "Draft prompt"
    assert spec.profiles["planner"]["from_file"].prompt == Path("prompts/review.txt")


def test_all_of_accepts_multiple_field_predicates() -> None:
    gate = all_of(
        [
            field(StatusComponent, "status") == "ready",
            field(StatusComponent, "count") == 3,
        ]
    )

    assert len(gate.predicates) == 2


def test_any_of_accepts_mixed_predicate_types() -> None:
    gate = any_of(
        [
            field(StatusComponent, "status") == "ready",
            absent(ReviewComponent),
            has(StatusComponent),
        ]
    )

    assert len(gate.predicates) == 3


def test_not_wraps_absent_predicate() -> None:
    gate = not_(absent(StatusComponent))

    assert gate.predicate.component_type is StatusComponent


def test_workflow_spec_rejects_duplicate_state_ids() -> None:
    shared_profile = PromptProfileSpec(profile_id="shared", prompt="Prompt")

    with pytest.raises(ValueError, match="Duplicate state_id"):
        WorkflowSpec(
            workflow_id="duplicate-states",
            initial_state_id="draft",
            states=(
                StateSpec(state_id="draft", bind={}, transitions=()),
                StateSpec(state_id="draft", bind={}, transitions=()),
            ),
            profiles={"planner": {"shared": shared_profile}},
        )


def test_workflow_rejects_unknown_initial_state() -> None:
    with pytest.raises(ValueError, match="Initial state"):
        workflow(
            "bad-initial",
            initial="missing",
            profiles={"planner": {"shared": "Prompt"}},
            states={"draft": {"bind": {"planner": "shared"}}},
        )


def test_workflow_rejects_unknown_transition_target() -> None:
    with pytest.raises(ValueError, match="Transition target"):
        workflow(
            "bad-target",
            initial="draft",
            profiles={"planner": {"shared": "Prompt"}},
            states={
                "draft": {
                    "bind": {"planner": "shared"},
                    "go": {"missing": has(StatusComponent)},
                }
            },
        )


def test_all_of_rejects_empty_list() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        all_of([])


def test_gate_builders_reject_callables() -> None:
    with pytest.raises(ValueError, match="callable"):
        any_of([lambda: True])


def test_workflow_rejects_unknown_profile_in_bind() -> None:
    with pytest.raises(ValueError, match="Unknown profile_id"):
        workflow(
            "bad-bind",
            initial="draft",
            profiles={"planner": {"shared": "Prompt"}},
            states={"draft": {"bind": {"planner": "missing"}}},
        )


def test_public_workflows_package_exports_only_dsl_surface() -> None:
    namespace: dict[str, object] = {}

    exec(
        "from ecs_agent.workflows import WorkflowSpec, workflow, field, bind_workflow",
        namespace,
    )

    assert namespace["WorkflowSpec"] is WorkflowSpec
    assert namespace["workflow"] is workflow
    assert namespace["field"] is field
    assert namespace["bind_workflow"] is bind_workflow

    with pytest.raises(ImportError):
        exec("from ecs_agent.workflows import FieldPredicate", {})


def test_bind_workflow_installs_runtime_components() -> None:
    world = World()
    entity_id = world.create_entity()
    spec = WorkflowSpec(
        workflow_id="bindable",
        initial_state_id="draft",
        states=(
            StateSpec(
                state_id="draft",
                bind={"planner": "shared"},
                transitions=(
                    TransitionSpec(
                        transition_id="draft_to_done",
                        target_state_id="done",
                        gate=has(StatusComponent),
                    ),
                ),
            ),
            StateSpec(state_id="done", bind={"planner": "shared"}, transitions=()),
        ),
        profiles={
            "planner": {
                "shared": PromptProfileSpec(profile_id="shared", prompt="Prompt")
            }
        },
    )

    bind_workflow(world, entity_id, spec, agent_key="planner")

    runtime = world.get_component(entity_id, WorkflowRuntimeComponent)
    binding = world.get_component(entity_id, WorkflowBindingComponent)

    assert runtime is not None
    assert runtime.current_state_id == "draft"
    assert binding is not None
    assert binding.agent_key == "planner"
