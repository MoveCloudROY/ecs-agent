"""Shared main/child component installation and complete-loop definitions."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from typing import cast

from ecs_agent.authoring.definitions import SystemStep
from ecs_agent.components import (
    ConversationComponent, LLMComponent, SubagentRegistryComponent,
    SubagentSessionTableComponent, ToolRegistryComponent, UserPromptConfigComponent,
)
from ecs_agent.core.world import World
from ecs_agent.core.system import System
from ecs_agent.logging import get_logger
from ecs_agent.prompts.contracts import SystemPromptConfigSpec, TriggerSpec
from ecs_agent.providers.protocol import LLMModel
from ecs_agent.skills.manager import SkillManager
from ecs_agent.skills.script_skill import ScriptSkill
from ecs_agent.systems.compaction import CompactionSystem
from ecs_agent.systems.error_handling import ErrorHandlingSystem
from ecs_agent.systems.reasoning import ReasoningSystem
from ecs_agent.systems.system_prompt_render_system import SystemPromptRenderSystem
from ecs_agent.systems.tool_execution import ToolExecutionSystem
from ecs_agent.systems.user_prompt_normalization_system import UserPromptNormalizationSystem
from ecs_agent.tools.definitions import ToolDefinition
from ecs_agent.tools.registry import register_tools
from ecs_agent.types import EntityId, SubagentConfig

logger = get_logger(__name__)


def system_role(system: System, *, fallback: str) -> str:
    """Adapt legacy child profiles by implementation role, including subclasses."""
    from ecs_agent.systems.tool_execution import ToolExecutionSystem
    from ecs_agent.systems.subagent import SubagentSystem
    from ecs_agent.systems.subagent_wait import SubagentWaitSystem

    for role, implementation in (
        ("compaction", CompactionSystem), ("normalize", UserPromptNormalizationSystem),
        ("render", SystemPromptRenderSystem), ("reason", ReasoningSystem),
        ("execute", ToolExecutionSystem), ("errors", ErrorHandlingSystem),
        ("delegate", SubagentSystem), ("wait", SubagentWaitSystem),
    ):
        if isinstance(system, implementation):
            return role
    return fallback


def core_steps(*, compaction: bool = False, normalize: bool = True) -> tuple[SystemStep, ...]:
    """Return the shared reasoning loop with dependency-safe conventional priorities."""
    from ecs_agent.systems.tool_execution import ToolExecutionSystem

    steps = []
    if compaction:
        steps.append(SystemStep("compaction", CompactionSystem, priority=-30))
    if normalize:
        steps.append(SystemStep("normalize", UserPromptNormalizationSystem, after=("compaction",) if compaction else (), priority=-25))
    steps.extend([
        SystemStep("render", SystemPromptRenderSystem, after=("normalize",) if normalize else ("compaction",) if compaction else (), priority=-20),
        SystemStep("reason", ReasoningSystem, after=("render",), priority=0),
        SystemStep("execute", ToolExecutionSystem, after=("reason",), priority=5),
        SystemStep("errors", ErrorHandlingSystem, after=("execute",), priority=99),
    ])
    return tuple(steps)


def install_components(
    world: World, entity: EntityId, *, model: LLMModel,
    prompt: SystemPromptConfigSpec, tools: list[ToolDefinition],
    skills: list[ScriptSkill], triggers: list[TriggerSpec], conversation: bool = True,
    user_prompt: UserPromptConfigComponent | None = None,
) -> None:
    """Install base state and local assets once on a fresh agent entity."""
    for component_type in (LLMComponent, SystemPromptConfigSpec, UserPromptConfigComponent, ToolRegistryComponent):
        if world.has_component(entity, component_type):
            raise ValueError(f"Agent assembly would overwrite {component_type.__name__}")
    world.add_component(entity, LLMComponent(model=model, system_prompt=prompt.template_source.inline or ""))
    world.add_component(entity, replace(prompt, placeholders=list(prompt.placeholders)))
    world.add_component(entity, replace(user_prompt, triggers=list(user_prompt.triggers), script_handlers=dict(user_prompt.script_handlers)) if user_prompt is not None else UserPromptConfigComponent(triggers=list(triggers)))
    if conversation:
        world.add_component(entity, ConversationComponent(messages=[]))
    register_tools(world, entity, tools)
    manager = SkillManager()
    installed: list[ScriptSkill] = []
    try:
        for skill in skills:
            # Track before invoking an extension's install hook so a failure
            # inside that hook still receives matching cleanup.
            installed.append(skill)
            manager.install(world, entity, skill)
    except Exception as exc:
        logger.error("agent_assets_install_failed", entity_id=int(entity), exception=str(exc))
        for skill in reversed(installed):
            try:
                manager.uninstall(world, entity, skill.name)
            except Exception as cleanup_error:
                logger.error("agent_assets_cleanup_failed", skill=skill.name, exception=str(cleanup_error))
        raise


def install_delegation(world: World, entity: EntityId, configs: dict[str, SubagentConfig], *, rebind: bool = False) -> None:
    """Bind delegation tools to the installed World-level delegation system."""
    from ecs_agent.systems.subagent import SubagentSystem
    from ecs_agent.systems.subagent_wait import SubagentWaitSystem

    registry = world.get_component(entity, SubagentRegistryComponent)
    if registry is None:
        registry = SubagentRegistryComponent(subagents=configs)
        world.add_component(entity, registry)
    elif registry.subagents != configs:
        raise ValueError("Delegation installation would overwrite the existing child registry")
    if not configs and not registry.free_subagent_config.enabled:
        return
    system = world._authoring_instances.get("delegate")
    if not isinstance(system, SubagentSystem):
        raise ValueError("Subagents require a SubagentSystem in the delegate role")
    if not world.has_component(entity, SubagentSessionTableComponent):
        world.add_component(entity, SubagentSessionTableComponent(sessions={}))
    system.install_subagent_tool(world, entity, override=rebind)
    system.install_subagent_control_tools(world, entity, override=rebind)
    wait = world._authoring_instances.get("wait")
    if isinstance(wait, SubagentWaitSystem):
        wait._resume_callback = system.make_resume_callback()


def delegation_steps(*, complete: bool = True) -> tuple[SystemStep, ...]:
    """Return delegation and optional background-wait support."""
    from ecs_agent.systems.subagent import SubagentSystem
    from ecs_agent.systems.subagent_wait import SubagentWaitSystem

    steps = [SystemStep("delegate", SubagentSystem, after=("render", "normalize"), priority=-2 if complete else -1)]
    if complete:
        steps.append(SystemStep("wait", SubagentWaitSystem, after=("delegate",), before=("reason",), priority=-1))
    return tuple(steps)


def validate_complete_systems(world: World, *, delegation: bool) -> None:
    """Check the installed role implementations before providers are created."""
    from ecs_agent.systems.tool_execution import ToolExecutionSystem

    requirements = {
        "normalize": UserPromptNormalizationSystem, "render": SystemPromptRenderSystem,
        "reason": ReasoningSystem, "execute": ToolExecutionSystem, "errors": ErrorHandlingSystem,
    }
    if delegation:
        from ecs_agent.systems.subagent import SubagentSystem
        from ecs_agent.systems.subagent_wait import SubagentWaitSystem
        requirements.update({"delegate": SubagentSystem, "wait": SubagentWaitSystem})
    for role, expected in requirements.items():
        if not isinstance(world._authoring_instances.get(role), expected):
            raise ValueError(f"Complete runtime requires {expected.__name__} in role '{role}'")
