"""Declarative authoring inputs and inspectable runtime installation plans."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ecs_agent.core.system import System
from ecs_agent.components import UserPromptConfigComponent
from ecs_agent.prompts.contracts import SystemPromptConfigSpec, TriggerSpec
from ecs_agent.providers.protocol import LLMModel
from ecs_agent.skills.script_skill import ScriptSkill
from ecs_agent.tools.definitions import ToolDefinition
from ecs_agent.types import InheritancePolicy


@dataclass(frozen=True, slots=True)
class SkillRef:
    """A local skill directory, resolved relative to an explicit source directory."""

    path: str | Path
    source_dir: Path | None = None


@dataclass(frozen=True, slots=True)
class ModelRef:
    """A model identifier to resolve after the complete definition validates."""

    name: str


@dataclass(frozen=True, slots=True)
class SystemStep:
    """One World-level role with dependency constraints and a fresh-instance factory."""

    name: str
    factory: Callable[[], System]
    after: tuple[str, ...] = ()
    before: tuple[str, ...] = ()
    priority: int | None = None


@dataclass(slots=True)
class RuntimeConfig:
    """System extensions and additional components for a fresh runtime.

    steps=None selects the built-in complete loop. An explicit steps tuple
    supplies the entire loop; extra_steps adds roles to either choice.
    """

    steps: tuple[SystemStep, ...] | None = None
    extra_steps: tuple[SystemStep, ...] = ()
    components: tuple[object, ...] = ()
    user_prompt: UserPromptConfigComponent | None = None


@dataclass(slots=True)
class AgentDefinition:
    """Author configuration, separate from World-owned conversation and execution."""

    model: LLMModel | ModelRef
    prompt: str | SystemPromptConfigSpec = ""
    tools: list[ToolDefinition | Callable[..., Any]] = field(default_factory=list)
    permissions: dict[str, bool] | None = None
    skills: list[SkillRef | ScriptSkill] = field(default_factory=list)
    triggers: list[TriggerSpec] = field(default_factory=list)
    subagents: dict[str, AgentDefinition] = field(default_factory=dict)
    name: str = "agent"
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    inheritance_policy: InheritancePolicy = field(default_factory=InheritancePolicy)
    inherited_skills: list[str] = field(default_factory=list)
    runtime_profile: str | None = None
    max_ticks: int | None = None
    source: str = "Python definition"


@dataclass(frozen=True, slots=True)
class InstalledStep:
    """A resolved registration and its dependency provenance."""

    name: str
    system_type: str
    priority: int
    after: tuple[str, ...]
    before: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AssemblyReport:
    """A snapshot of installed components, tool origins, and system order."""

    agent_name: str
    components: tuple[str, ...]
    tools: tuple[tuple[str, str], ...]
    systems: tuple[InstalledStep, ...]
    complete: bool
    component_sources: tuple[tuple[str, str], ...] = ()
    diagnostics: tuple[str, ...] = ()

    @property
    def concurrency_groups(self) -> tuple[tuple[int, tuple[str, ...]], ...]:
        """Return roles grouped by the registration priority that runs concurrently."""
        priorities = sorted({step.priority for step in self.systems})
        return tuple((priority, tuple(step.name for step in self.systems if step.priority == priority)) for priority in priorities)
