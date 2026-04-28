"""Tests for workflow compilation and ECS installation helpers."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from collections.abc import Callable
from pathlib import Path
from typing import cast, get_origin

import pytest

from ecs_agent.components import (
    WorkflowBindingComponent,
    WorkflowGateSnapshotComponent,
    WorkflowLastTransitionComponent,
    WorkflowRuntimeComponent,
)
from ecs_agent.core import World
from ecs_agent.serialization import WorldSerializer
from ecs_agent.workflows import bind_workflow, prompt_file, workflow
from ecs_agent.workflows._components import WorkflowDefinitionComponent
from ecs_agent.workflows.compiler import compile_workflow, install_workflow
from ecs_agent.workflows.contracts import (
    PromptProfileSpec,
    StateSpec,
    TransitionSpec,
    WorkflowSpec,
    field,
    has,
)


@dataclass(slots=True)
class StatusComponent:
    status: str
    count: int


@dataclass(slots=True)
class ReviewComponent:
    approved: bool


def test_compile_workflow_groups_transitions_and_preserves_shared_profile_bindings() -> None:
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

    compiled = compile_workflow(spec)

    assert compiled.workflow_id == "draft-review"
    assert compiled.initial_state_id == "draft"
    assert compiled.state_ids == frozenset({"draft", "review", "done"})
    assert tuple(compiled.transitions_by_state) == ("draft", "review", "done")
    assert compiled.transitions_by_state["draft"][0].transition_id == "draft_to_review"
    assert compiled.transitions_by_state["draft"][0].target_state_id == "review"
    assert compiled.transitions_by_state["review"][0].target_state_id == "done"
    assert compiled.transitions_by_state["done"] == ()
    assert compiled.bindings_by_state["draft"]["planner"] == "shared"
    assert compiled.bindings_by_state["review"]["planner"] == "shared"
    assert set(compiled.profile_table["planner"]) == {"shared", "from_file"}
    assert compiled.profile_table["planner"]["shared"].prompt_text == "Draft prompt"
    assert compiled.profile_table["planner"]["from_file"].prompt_path == "prompts/review.txt"


def test_compile_workflow_marks_callable_prompt_profiles_runtime_only() -> None:
    spec = workflow(
        "dynamic-prompt",
        initial="draft",
        profiles={"planner": {"dynamic": PromptProfileSpec("dynamic", lambda: "Prompt")}},
        states={"draft": {"bind": {"planner": "dynamic"}}},
    )

    compiled = compile_workflow(spec)
    profile = compiled.profile_table["planner"]["dynamic"]

    assert profile.prompt_factory is not None
    assert profile.is_serializable is False
    assert compiled.is_serializable is False


def test_compile_workflow_rejects_missing_prompt_file_when_path_validation_enabled(
    tmp_path: Path,
) -> None:
    missing_prompt = tmp_path / "missing.txt"
    spec = workflow(
        "missing-file",
        initial="draft",
        profiles={"planner": {"shared": prompt_file(missing_prompt)}},
        states={"draft": {"bind": {"planner": "shared"}}},
    )

    with pytest.raises(ValueError, match="does not exist"):
        compile_workflow(spec, validate_paths=True)


def test_install_workflow_attaches_definition_runtime_and_binding_components() -> None:
    world = World()
    entity_id = world.create_entity()
    spec = workflow(
        "installable",
        initial="draft",
        profiles={"planner": {"shared": "Prompt"}},
        states={
            "draft": {"bind": {"planner": "shared"}, "go": {"done": has(StatusComponent)}},
            "done": {"bind": {"planner": "shared"}},
        },
    )

    install_workflow(world, entity_id, spec, agent_key="planner")

    definition = world.get_component(entity_id, WorkflowDefinitionComponent)
    runtime = world.get_component(entity_id, WorkflowRuntimeComponent)
    binding = world.get_component(entity_id, WorkflowBindingComponent)

    assert definition is not None
    assert definition.compiled.workflow_id == "installable"
    assert runtime is not None
    assert runtime.current_state_id == spec.initial_state_id
    assert runtime.transition_history == []
    assert binding is not None
    assert binding.agent_key == "planner"


def test_bind_workflow_forwards_to_install_workflow() -> None:
    world = World()
    entity_id = world.create_entity()
    spec = workflow(
        "bindable",
        initial="draft",
        profiles={"planner": {"shared": "Prompt"}},
        states={"draft": {"bind": {"planner": "shared"}}},
    )

    bind_workflow(world, entity_id, spec, agent_key="planner")

    runtime = world.get_component(entity_id, WorkflowRuntimeComponent)
    binding = world.get_component(entity_id, WorkflowBindingComponent)

    assert runtime is not None
    assert runtime.current_state_id == "draft"
    assert binding is not None
    assert binding.agent_key == "planner"


def test_workflow_runtime_components_serialize_and_restore() -> None:
    world = World()
    entity_id = world.create_entity()

    world.add_component(
        entity_id,
        WorkflowRuntimeComponent(
            current_state_id="review",
            transition_history=["draft", "review"],
        ),
    )
    world.add_component(entity_id, WorkflowBindingComponent(agent_key="planner"))
    world.add_component(
        entity_id,
        WorkflowGateSnapshotComponent(
            state_id="review",
            evaluated_at_tick=7,
            matched_transition_id="review_to_done",
        ),
    )
    world.add_component(
        entity_id,
        WorkflowLastTransitionComponent(
            from_state_id="draft",
            to_state_id="review",
            transition_id="draft_to_review",
            tick=7,
        ),
    )

    payload = WorldSerializer.to_dict(world)
    restored = WorldSerializer.from_dict(payload, providers={}, tool_handlers={})

    runtime = restored.get_component(entity_id, WorkflowRuntimeComponent)
    binding = restored.get_component(entity_id, WorkflowBindingComponent)
    gate_snapshot = restored.get_component(entity_id, WorkflowGateSnapshotComponent)
    last_transition = restored.get_component(entity_id, WorkflowLastTransitionComponent)

    assert runtime is not None
    assert runtime.current_state_id == "review"
    assert runtime.transition_history == ["draft", "review"]
    assert binding is not None
    assert binding.agent_key == "planner"
    assert gate_snapshot is not None
    assert gate_snapshot.state_id == "review"
    assert gate_snapshot.evaluated_at_tick == 7
    assert gate_snapshot.matched_transition_id == "review_to_done"
    assert last_transition is not None
    assert last_transition.from_state_id == "draft"
    assert last_transition.to_state_id == "review"
    assert last_transition.transition_id == "draft_to_review"
    assert last_transition.tick == 7


def test_workflow_runtime_components_have_no_callable_fields() -> None:
    for component_class in [
        WorkflowRuntimeComponent,
        WorkflowBindingComponent,
        WorkflowGateSnapshotComponent,
        WorkflowLastTransitionComponent,
    ]:
        for field_info in dataclasses.fields(component_class):
            assert get_origin(field_info.type) is not Callable, component_class.__name__


def test_compile_workflow_rejects_callable_gate_nodes() -> None:
    invalid_transition = _unsafe_transition_spec(gate=lambda: True)
    invalid_spec = _unsafe_workflow_spec(
        states=(
            _unsafe_state_spec(
                state_id="draft",
                bind={"planner": "shared"},
                transitions=(invalid_transition,),
            ),
            _unsafe_state_spec(
                state_id="done",
                bind={"planner": "shared"},
                transitions=(),
            ),
        ),
        profiles={
            "planner": {
                "shared": PromptProfileSpec(profile_id="shared", prompt="Prompt"),
            }
        },
    )

    with pytest.raises(ValueError, match="callable gate expressions"):
        compile_workflow(invalid_spec)


def test_compile_workflow_rejects_non_type_component_references() -> None:
    spec = workflow(
        "bad-component-type",
        initial="draft",
        profiles={"planner": {"shared": "Prompt"}},
        states={
            "draft": {
                "bind": {"planner": "shared"},
                "go": {"done": field("not-a-type", "status") == "ready"},
            },
            "done": {"bind": {"planner": "shared"}},
        },
    )

    with pytest.raises(ValueError, match="component_type"):
        compile_workflow(spec)


def test_compile_workflow_rejects_unknown_profile_reference_in_bindings() -> None:
    invalid_spec = _unsafe_workflow_spec(
        states=(
            _unsafe_state_spec(
                state_id="draft",
                bind={"planner": "missing"},
                transitions=(),
            ),
        ),
        profiles={
            "planner": {
                "shared": PromptProfileSpec(profile_id="shared", prompt="Prompt"),
            }
        },
    )

    with pytest.raises(ValueError, match="Unknown profile_id"):
        compile_workflow(invalid_spec)


def test_compile_workflow_serializes_string_and_path_prompts_without_runtime_callables() -> None:
    spec = workflow(
        "serializable-prompts",
        initial="draft",
        profiles={
            "planner": {
                "inline": "Inline prompt",
                "from_file": prompt_file("prompts/review.txt"),
            }
        },
        states={"draft": {"bind": {"planner": "inline"}}},
    )

    compiled = compile_workflow(spec)

    inline_profile = compiled.profile_table["planner"]["inline"]
    path_profile = compiled.profile_table["planner"]["from_file"]

    assert inline_profile.prompt_text == "Inline prompt"
    assert inline_profile.prompt_factory is None
    assert inline_profile.is_serializable is True
    assert path_profile.prompt_path == "prompts/review.txt"
    assert path_profile.prompt_factory is None
    assert path_profile.is_serializable is True
    assert compiled.is_serializable is True


def _unsafe_state_spec(
    *,
    state_id: str,
    bind: dict[str, str],
    transitions: tuple[TransitionSpec, ...],
) -> StateSpec:
    state = cast(StateSpec, object.__new__(StateSpec))
    object.__setattr__(state, "state_id", state_id)
    object.__setattr__(state, "bind", bind)
    object.__setattr__(state, "transitions", transitions)
    return state


def _unsafe_transition_spec(
    *,
    transition_id: str = "draft_to_done",
    target_state_id: str = "done",
    gate: object,
) -> TransitionSpec:
    transition = cast(TransitionSpec, object.__new__(TransitionSpec))
    object.__setattr__(transition, "transition_id", transition_id)
    object.__setattr__(transition, "target_state_id", target_state_id)
    object.__setattr__(transition, "gate", gate)
    return transition


def _unsafe_workflow_spec(
    *,
    states: tuple[StateSpec, ...],
    profiles: dict[str, dict[str, PromptProfileSpec]],
) -> WorkflowSpec:
    workflow_spec = cast(WorkflowSpec, object.__new__(WorkflowSpec))
    object.__setattr__(workflow_spec, "workflow_id", "unsafe")
    object.__setattr__(workflow_spec, "initial_state_id", states[0].state_id)
    object.__setattr__(workflow_spec, "states", states)
    object.__setattr__(workflow_spec, "profiles", profiles)
    return workflow_spec
