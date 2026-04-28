"""Workflow compiler and ECS installation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from collections.abc import Callable

from ecs_agent.core import World
from ecs_agent.types import EntityId
from ecs_agent.workflows._components import (
    WorkflowBindingComponent,
    WorkflowDefinitionComponent,
    WorkflowRuntimeComponent,
)
from ecs_agent.workflows.contracts import (
    AbsentPredicate,
    AllOf,
    AnyOf,
    FieldPredicate,
    HasPredicate,
    Not,
    PromptProfileSpec,
    StateSpec,
    TransitionSpec,
    WorkflowSpec,
)


@dataclass(frozen=True, slots=True)
class CompiledPromptProfile:
    """Validated prompt profile payload stored by the workflow compiler."""

    profile_id: str
    source_kind: Literal["inline", "path", "callable"]
    prompt_text: str | None = None
    prompt_path: str | None = None
    prompt_factory: Callable[[], str] | None = None
    is_serializable: bool = True

    def __post_init__(self) -> None:
        source_count = sum(
            (
                self.prompt_text is not None,
                self.prompt_path is not None,
                self.prompt_factory is not None,
            )
        )
        if source_count != 1:
            raise ValueError("CompiledPromptProfile requires exactly one prompt source")
        if self.source_kind == "inline" and self.prompt_text is None:
            raise ValueError("Inline prompt profiles require prompt_text")
        if self.source_kind == "path" and self.prompt_path is None:
            raise ValueError("Path prompt profiles require prompt_path")
        if self.source_kind == "callable" and self.prompt_factory is None:
            raise ValueError("Callable prompt profiles require prompt_factory")
        if self.source_kind == "callable" and self.is_serializable:
            raise ValueError("Callable prompt profiles cannot be marked serializable")
        if self.source_kind != "callable" and not self.is_serializable:
            raise ValueError("Only callable prompt profiles can be non-serializable")


@dataclass(frozen=True, slots=True)
class CompiledWorkflow:
    """Frozen, validated workflow model used by ECS workflow runtime components."""

    workflow_id: str
    initial_state_id: str
    state_ids: frozenset[str]
    transitions_by_state: dict[str, tuple[TransitionSpec, ...]]
    bindings_by_state: dict[str, dict[str, str]]
    profile_table: dict[str, dict[str, CompiledPromptProfile]]
    is_serializable: bool = True


def compile_workflow(spec: WorkflowSpec, *, validate_paths: bool = False) -> CompiledWorkflow:
    """Validate and compile a workflow spec into runtime installation data."""

    state_ids = _validate_state_table(spec)
    profile_table = _compile_profiles(spec.profiles, validate_paths=validate_paths)
    transitions_by_state: dict[str, tuple[TransitionSpec, ...]] = {}
    bindings_by_state: dict[str, dict[str, str]] = {}

    for state in spec.states:
        bindings_by_state[state.state_id] = dict(state.bind)
        _validate_bindings(state, profile_table)
        transitions_by_state[state.state_id] = _compile_transitions(state, state_ids)

    return CompiledWorkflow(
        workflow_id=spec.workflow_id,
        initial_state_id=spec.initial_state_id,
        state_ids=frozenset(state_ids),
        transitions_by_state=transitions_by_state,
        bindings_by_state=bindings_by_state,
        profile_table=profile_table,
        is_serializable=_workflow_is_serializable(profile_table),
    )


def install_workflow(
    world: World,
    entity_id: EntityId,
    spec: WorkflowSpec,
    *,
    agent_key: str,
) -> None:
    """Compile a workflow spec and attach runtime components to an entity."""

    compiled = compile_workflow(spec)
    initial_bindings = compiled.bindings_by_state[compiled.initial_state_id]
    if agent_key not in initial_bindings:
        raise ValueError(
            f"Agent key {agent_key!r} is not bound in initial state {compiled.initial_state_id!r}"
        )

    world.add_component(entity_id, WorkflowDefinitionComponent(compiled=compiled))
    world.add_component(
        entity_id,
        WorkflowRuntimeComponent(current_state_id=compiled.initial_state_id),
    )
    world.add_component(entity_id, WorkflowBindingComponent(agent_key=agent_key))


def _validate_state_table(spec: WorkflowSpec) -> set[str]:
    if not spec.workflow_id:
        raise ValueError("WorkflowSpec requires a non-empty workflow_id")
    if not spec.states:
        raise ValueError("WorkflowSpec requires at least one state")

    state_ids = [state.state_id for state in spec.states]
    if len(state_ids) != len(set(state_ids)):
        raise ValueError("Duplicate state_id values are not allowed")
    if spec.initial_state_id not in state_ids:
        raise ValueError(f"Initial state {spec.initial_state_id!r} was not found in states")
    return set(state_ids)


def _validate_bindings(
    state: StateSpec,
    profile_table: dict[str, dict[str, CompiledPromptProfile]],
) -> None:
    for agent_key, profile_id in state.bind.items():
        agent_profiles = profile_table.get(agent_key)
        if agent_profiles is None or profile_id not in agent_profiles:
            raise ValueError(
                "Unknown profile_id "
                f"{profile_id!r} for agent_key {agent_key!r} in state {state.state_id!r}"
            )


def _compile_transitions(
    state: StateSpec,
    state_ids: set[str],
) -> tuple[TransitionSpec, ...]:
    compiled_transitions: list[TransitionSpec] = []
    for transition in state.transitions:
        if transition.target_state_id not in state_ids:
            raise ValueError(
                "Transition target "
                f"{transition.target_state_id!r} from state {state.state_id!r} was not found in states"
            )
        _validate_gate_expr(transition.gate)
        compiled_transitions.append(transition)
    return tuple(compiled_transitions)


def _compile_profiles(
    profiles: dict[str, dict[str, PromptProfileSpec]],
    *,
    validate_paths: bool,
) -> dict[str, dict[str, CompiledPromptProfile]]:
    compiled_profiles: dict[str, dict[str, CompiledPromptProfile]] = {}
    for agent_key, agent_profiles in profiles.items():
        compiled_agent_profiles: dict[str, CompiledPromptProfile] = {}
        for profile_id, profile_spec in agent_profiles.items():
            if profile_spec.profile_id != profile_id:
                raise ValueError(
                    "PromptProfileSpec profile_id must match its profiles mapping key: "
                    f"expected {profile_id!r}, got {profile_spec.profile_id!r}"
                )
            compiled_agent_profiles[profile_id] = _compile_prompt_profile(
                profile_spec,
                validate_paths=validate_paths,
            )
        compiled_profiles[agent_key] = compiled_agent_profiles
    return compiled_profiles


def _compile_prompt_profile(
    profile_spec: PromptProfileSpec,
    *,
    validate_paths: bool,
) -> CompiledPromptProfile:
    prompt = profile_spec.prompt
    if isinstance(prompt, str):
        return CompiledPromptProfile(
            profile_id=profile_spec.profile_id,
            source_kind="inline",
            prompt_text=prompt,
        )
    if isinstance(prompt, Path):
        if validate_paths and not prompt.exists():
            raise ValueError(f"Prompt file {str(prompt)!r} does not exist")
        return CompiledPromptProfile(
            profile_id=profile_spec.profile_id,
            source_kind="path",
            prompt_path=str(prompt),
        )
    if callable(prompt):
        return CompiledPromptProfile(
            profile_id=profile_spec.profile_id,
            source_kind="callable",
            prompt_factory=prompt,
            is_serializable=False,
        )
    raise ValueError(
        "PromptProfileSpec.prompt must be str, Path, or Callable[[], str] at compile time"
    )


def _workflow_is_serializable(
    profile_table: dict[str, dict[str, CompiledPromptProfile]],
) -> bool:
    return all(
        profile.is_serializable
        for agent_profiles in profile_table.values()
        for profile in agent_profiles.values()
    )


def _validate_gate_expr(gate: object) -> None:
    if callable(gate) and not isinstance(gate, type):
        raise ValueError("callable gate expressions are forbidden in v1")
    if isinstance(gate, FieldPredicate):
        _validate_component_type(gate.component_type)
        return
    if isinstance(gate, (AbsentPredicate, HasPredicate)):
        _validate_component_type(gate.component_type)
        return
    if isinstance(gate, (AllOf, AnyOf)):
        for predicate in gate.predicates:
            _validate_gate_expr(predicate)
        return
    if isinstance(gate, Not):
        _validate_gate_expr(gate.predicate)
        return
    raise ValueError(f"Expected GateExpr instance, got {type(gate).__name__}")


def _validate_component_type(component_type: Any) -> None:
    if not isinstance(component_type, type):
        raise ValueError("Workflow gate component_type must be a type instance")
