"""Configurable child-world runtime profiles for subagents (Task 12).

A subagent's child world runs a set of ECS systems. Which systems is chosen by a
*runtime profile*: a name (``SubagentConfig.runtime_profile``, default ``"default"``)
resolved against this process-level registry to a builder that, given a
``ChildProfileContext``, returns the whole system set as ``ChildSystemSpec`` entries.

Whole-set replacement (coarse-grained): a profile fully defines the child system set.
Callables live only here (never on the serializable ``SubagentConfig``), so
checkpoints stay pure data. The ``"default"`` profile reproduces the historical
hardcoded set exactly, including the "add CompactionSystem only when the parent has
compaction config" conditional.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from ecs_agent.core.system import System
from ecs_agent.systems.compaction import CompactionSystem
from ecs_agent.systems.error_handling import ErrorHandlingSystem
from ecs_agent.systems.reasoning import ReasoningSystem
from ecs_agent.systems.system_prompt_render_system import SystemPromptRenderSystem

# Priority of CompactionSystem in a child world (kept identical to the historical
# _SUBAGENT_COMPACTION_PRIORITY constant).
_CHILD_COMPACTION_PRIORITY = -30


@dataclass(slots=True, frozen=True)
class ChildSystemSpec:
    """One entry in a runtime profile: a factory producing an ECS system + its priority.

    ``factory`` is a zero-arg callable returning a *new* system instance (any object
    satisfying the System protocol). It is called once per child-world assembly.
    """

    factory: Callable[[], System]
    priority: int


@dataclass(slots=True, frozen=True)
class ChildProfileContext:
    """Inputs a profile builder may branch on when composing the child system set."""

    parent_has_compaction: bool


ProfileBuilder = Callable[[ChildProfileContext], list[ChildSystemSpec]]

_PROFILES: dict[str, ProfileBuilder] = {}


def register_child_runtime_profile(name: str, builder: ProfileBuilder) -> None:
    """Register (or replace) a named child-world runtime profile."""
    _PROFILES[name] = builder


def resolve_child_runtime_profile(name: str | None) -> ProfileBuilder:
    """Resolve a profile name to its builder. ``None`` resolves to ``"default"``.

    Raises:
        ValueError: if the named runtime profile is not registered.
    """
    resolved_name = "default" if name is None else name
    builder = _PROFILES.get(resolved_name)
    if builder is None:
        raise ValueError(
            f"Unknown subagent runtime profile '{resolved_name}'. "
            f"Registered profiles: {sorted(_PROFILES)}"
        )
    return builder


def default_child_system_specs(ctx: ChildProfileContext) -> list[ChildSystemSpec]:
    """The historical child system set. Reproduces _assemble_child_world exactly."""
    specs: list[ChildSystemSpec] = []
    if ctx.parent_has_compaction:
        specs.append(
            ChildSystemSpec(
                factory=lambda: CompactionSystem(),
                priority=_CHILD_COMPACTION_PRIORITY,
            )
        )
    specs.append(
        ChildSystemSpec(
            factory=lambda: SystemPromptRenderSystem(priority=-20),
            priority=-20,
        )
    )
    specs.append(
        ChildSystemSpec(factory=lambda: ReasoningSystem(priority=0), priority=0)
    )
    specs.append(
        ChildSystemSpec(factory=lambda: ErrorHandlingSystem(priority=99), priority=99)
    )
    return specs


def reset_child_runtime_profiles() -> None:
    """Reset the registry to only the built-in ``"default"`` profile (test teardown)."""
    _PROFILES.clear()
    register_child_runtime_profile("default", default_child_system_specs)


# Register the built-in default profile at import time.
register_child_runtime_profile("default", default_child_system_specs)


__all__ = [
    "ChildSystemSpec",
    "ChildProfileContext",
    "ProfileBuilder",
    "register_child_runtime_profile",
    "resolve_child_runtime_profile",
    "default_child_system_specs",
    "reset_child_runtime_profiles",
]
