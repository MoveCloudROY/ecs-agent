"""Typed workflow DSL contracts and public authoring helpers."""

from __future__ import annotations

from dataclasses import dataclass
from inspect import Parameter, signature
from pathlib import Path
from typing import Any, TypeAlias, TypedDict
from collections.abc import Callable

from ecs_agent.core import World
from ecs_agent.types import EntityId


@dataclass(frozen=True, slots=True)
class FieldPredicate:
    """Internal field-comparison gate expression."""

    component_type: type
    field_name: str
    op: str
    value: Any

    def __post_init__(self) -> None:
        if self.op != "==":
            raise ValueError(f"Unsupported field predicate operator {self.op!r}")
        if not self.field_name:
            raise ValueError("FieldPredicate requires a non-empty field_name")


@dataclass(frozen=True, slots=True)
class AbsentPredicate:
    """Gate expression requiring a component to be absent."""

    component_type: type


@dataclass(frozen=True, slots=True)
class HasPredicate:
    """Gate expression requiring a component to be present."""

    component_type: type


@dataclass(frozen=True, slots=True)
class AllOf:
    """Gate expression requiring all child predicates to match."""

    predicates: tuple[GateExpr, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "predicates", _normalize_predicates(self.predicates, "all_of"))


@dataclass(frozen=True, slots=True)
class AnyOf:
    """Gate expression requiring any child predicate to match."""

    predicates: tuple[GateExpr, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "predicates", _normalize_predicates(self.predicates, "any_of"))


@dataclass(frozen=True, slots=True)
class Not:
    """Gate expression negating a child predicate."""

    predicate: GateExpr

    def __post_init__(self) -> None:
        _ensure_gate_expr(self.predicate)


GateExpr: TypeAlias = FieldPredicate | AbsentPredicate | HasPredicate | AllOf | AnyOf | Not

_GATE_EXPR_TYPES = (
    FieldPredicate,
    AbsentPredicate,
    HasPredicate,
    AllOf,
    AnyOf,
    Not,
)


@dataclass(frozen=True, slots=True, eq=False)
class _FieldBuilder:
    """Internal helper for field(Component, name) == value authoring."""

    component_type: type
    field_name: str

    def __eq__(self, other: object) -> FieldPredicate:  # type: ignore[override]
        return FieldPredicate(
            component_type=self.component_type,
            field_name=self.field_name,
            op="==",
            value=other,
        )


@dataclass(frozen=True, slots=True)
class TransitionSpec:
    """Compiled transition specification for a workflow state."""

    transition_id: str
    target_state_id: str
    gate: GateExpr

    def __post_init__(self) -> None:
        if not self.transition_id:
            raise ValueError("TransitionSpec requires a non-empty transition_id")
        if not self.target_state_id:
            raise ValueError("TransitionSpec requires a non-empty target_state_id")
        _ensure_gate_expr(self.gate)


@dataclass(frozen=True, slots=True)
class StateSpec:
    """Compiled state specification with prompt bindings and transitions."""

    state_id: str
    bind: dict[str, str]
    transitions: tuple[TransitionSpec, ...]

    def __post_init__(self) -> None:
        if not self.state_id:
            raise ValueError("StateSpec requires a non-empty state_id")


@dataclass(frozen=True, slots=True)
class PromptProfileSpec:
    """Prompt profile definition for an agent binding."""

    profile_id: str
    prompt: str | Path | Callable[[], str]

    def __post_init__(self) -> None:
        if not self.profile_id:
            raise ValueError("PromptProfileSpec requires a non-empty profile_id")
        if callable(self.prompt):
            _validate_zero_arg_callable(self.prompt)


@dataclass(frozen=True, slots=True)
class WorkflowSpec:
    """Top-level compiled workflow DSL object."""

    workflow_id: str
    initial_state_id: str
    states: tuple[StateSpec, ...]
    profiles: dict[str, dict[str, PromptProfileSpec]]

    def __post_init__(self) -> None:
        if not self.workflow_id:
            raise ValueError("WorkflowSpec requires a non-empty workflow_id")
        if not self.states:
            raise ValueError("WorkflowSpec requires at least one state")

        state_ids = [state.state_id for state in self.states]
        if len(state_ids) != len(set(state_ids)):
            raise ValueError("Duplicate state_id values are not allowed")
        if self.initial_state_id not in state_ids:
            raise ValueError(
                f"Initial state {self.initial_state_id!r} was not found in states"
            )

        valid_state_ids = set(state_ids)
        for state in self.states:
            for agent_key, profile_id in state.bind.items():
                agent_profiles = self.profiles.get(agent_key)
                if agent_profiles is None or profile_id not in agent_profiles:
                    raise ValueError(
                        "Unknown profile_id "
                        f"{profile_id!r} for agent_key {agent_key!r} in state {state.state_id!r}"
                    )
            for transition in state.transitions:
                if transition.target_state_id not in valid_state_ids:
                    raise ValueError(
                        "Transition target "
                        f"{transition.target_state_id!r} from state {state.state_id!r} was not found in states"
                    )


class StateDict(TypedDict, total=False):
    """Authoring shorthand for workflow(state=...) input."""

    bind: dict[str, str]
    go: dict[str, GateExpr | list[GateExpr]]


PromptProfileInput: TypeAlias = PromptProfileSpec | str | Path


def field(component_type: type, field_name: str) -> _FieldBuilder:
    """Return a field gate builder supporting `== value` authoring."""

    return _FieldBuilder(component_type=component_type, field_name=field_name)


def absent(component_type: type) -> AbsentPredicate:
    """Return a gate expression requiring the component to be absent."""

    return AbsentPredicate(component_type=component_type)


def has(component_type: type) -> HasPredicate:
    """Return a gate expression requiring the component to be present."""

    return HasPredicate(component_type=component_type)


def all_of(predicates: list[GateExpr]) -> AllOf:
    """Return a gate expression requiring all predicates to match."""

    return AllOf(predicates=tuple(predicates))


def any_of(predicates: list[GateExpr]) -> AnyOf:
    """Return a gate expression requiring any predicate to match."""

    return AnyOf(predicates=tuple(predicates))


def not_(predicate: GateExpr) -> Not:
    """Return a gate expression negating a predicate."""

    return Not(predicate=predicate)


def to(
    target_state_id: str,
    gate: GateExpr,
    *,
    transition_id: str | None = None,
) -> TransitionSpec:
    """Build an explicit transition spec escape hatch for advanced authoring."""

    resolved_transition_id = transition_id or f"to_{target_state_id}"
    return TransitionSpec(
        transition_id=resolved_transition_id,
        target_state_id=target_state_id,
        gate=gate,
    )


def prompt_file(path: str | Path) -> Path:
    """Mark a prompt profile input as file-backed rather than inline text."""

    return Path(path)


def workflow(
    workflow_id: str,
    *,
    initial: str,
    profiles: dict[str, dict[str, PromptProfileInput]],
    states: dict[str, StateDict],
) -> WorkflowSpec:
    """Build and validate a workflow specification from authoring shorthand."""

    normalized_profiles = _normalize_profiles(profiles)
    normalized_states = tuple(
        _normalize_state_spec(state_id, state_dict) for state_id, state_dict in states.items()
    )
    return WorkflowSpec(
        workflow_id=workflow_id,
        initial_state_id=initial,
        states=normalized_states,
        profiles=normalized_profiles,
    )


def bind_workflow(
    world: World,
    entity_id: EntityId,
    spec: WorkflowSpec,
    *,
    agent_key: str,
) -> None:
    """Install a compiled workflow onto an ECS entity."""

    from ecs_agent.workflows.compiler import install_workflow

    install_workflow(world, entity_id, spec, agent_key=agent_key)


def _ensure_gate_expr(predicate: object) -> None:
    if callable(predicate):
        raise ValueError("callable gate expressions are forbidden in v1")
    if not isinstance(predicate, _GATE_EXPR_TYPES):
        raise ValueError(f"Expected GateExpr instance, got {type(predicate).__name__}")


def _normalize_predicates(
    predicates: tuple[GateExpr, ...],
    builder_name: str,
) -> tuple[GateExpr, ...]:
    if not predicates:
        raise ValueError(f"{builder_name} requires a non-empty list of predicates")
    for predicate in predicates:
        _ensure_gate_expr(predicate)
    return predicates


def _normalize_profiles(
    profiles: dict[str, dict[str, PromptProfileInput]],
) -> dict[str, dict[str, PromptProfileSpec]]:
    normalized_profiles: dict[str, dict[str, PromptProfileSpec]] = {}
    for agent_key, agent_profiles in profiles.items():
        normalized_agent_profiles: dict[str, PromptProfileSpec] = {}
        for profile_id, profile_input in agent_profiles.items():
            if isinstance(profile_input, PromptProfileSpec):
                if profile_input.profile_id != profile_id:
                    raise ValueError(
                        "PromptProfileSpec profile_id must match its profiles mapping key: "
                        f"expected {profile_id!r}, got {profile_input.profile_id!r}"
                    )
                normalized_agent_profiles[profile_id] = profile_input
                continue
            normalized_agent_profiles[profile_id] = PromptProfileSpec(
                profile_id=profile_id,
                prompt=profile_input,
            )
        normalized_profiles[agent_key] = normalized_agent_profiles
    return normalized_profiles


def _normalize_state_spec(state_id: str, state_dict: StateDict) -> StateSpec:
    bind = dict(state_dict.get("bind", {}))
    transitions: list[TransitionSpec] = []
    for target_state_id, gate_input in state_dict.get("go", {}).items():
        gate = all_of(gate_input) if isinstance(gate_input, list) else gate_input
        _ensure_gate_expr(gate)
        transitions.append(
            TransitionSpec(
                transition_id=f"{state_id}_to_{target_state_id}",
                target_state_id=target_state_id,
                gate=gate,
            )
        )
    return StateSpec(state_id=state_id, bind=bind, transitions=tuple(transitions))


def _validate_zero_arg_callable(prompt_factory: Callable[[], str]) -> None:
    try:
        parameters = signature(prompt_factory).parameters.values()
    except (TypeError, ValueError) as exc:
        raise ValueError("Prompt profile callable must be introspectable") from exc

    required_parameters = [
        parameter
        for parameter in parameters
        if parameter.kind
        in (Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD, Parameter.KEYWORD_ONLY)
        and parameter.default is Parameter.empty
    ]
    if required_parameters:
        raise ValueError("Prompt profile callable must accept zero required arguments")
