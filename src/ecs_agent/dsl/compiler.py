"""Compile validated Agent DSL specs into ECS runtime components."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Literal, cast

from ecs_agent.components import (
    LLMComponent,
    PermissionComponent,
    SubagentRegistryComponent,
    SubagentSessionTableComponent,
    ToolRegistryComponent,
)
from ecs_agent.components.definitions import UserPromptConfigComponent
from ecs_agent.core import World
from ecs_agent.dsl.schema import AgentSpec
from ecs_agent.logging import get_logger
from ecs_agent.prompts.contracts import (
    PlaceholderSpec,
    PromptTemplateSource,
    SystemPromptConfigSpec,
    TriggerSpec,
)
from ecs_agent.providers.protocol import LLMModel
from ecs_agent.skills.manager import SkillManager
from ecs_agent.skills.script_skill import ScriptSkill
from ecs_agent.skills.skill import Skill
from ecs_agent.systems.system_prompt_render_system import SystemPromptRenderSystem
from ecs_agent.systems.user_prompt_normalization_system import (
    UserPromptNormalizationSystem,
)
from ecs_agent.systems.subagent import SubagentSystem
from ecs_agent.types import EntityId, SubagentConfig

logger = get_logger(__name__)


def compile_agent_specs(
    specs: dict[str, AgentSpec],
    model_factory: Callable[[str, str], LLMModel],
    *,
    source_dir: Path | None = None,
) -> tuple[EntityId, World]:
    primary_specs = [
        (agent_name, spec)
        for agent_name, spec in specs.items()
        if spec.mode == "primary"
    ]
    primary_count = len(primary_specs)
    if primary_count != 1:
        logger.error(
            "agent_spec_compile_invalid_primary_count",
            primary_count=primary_count,
            total_specs=len(specs),
        )
        raise ValueError(f"Expected exactly one primary agent, found {primary_count}")

    world = World()
    primary_entity = world.create_entity()

    _, primary_spec = primary_specs[0]
    primary_model = model_factory(primary_spec.model, primary_spec.prompt)
    world.add_component(
        primary_entity,
        LLMComponent(
            model=primary_model,
            system_prompt=primary_spec.prompt,
        ),
    )
    placeholder_specs = [
        PlaceholderSpec(name=ph["name"], value=ph["value"])
        for ph in primary_spec.placeholders
    ]
    world.add_component(
        primary_entity,
        SystemPromptConfigSpec(
            template_source=PromptTemplateSource(inline=primary_spec.prompt),
            placeholders=placeholder_specs,
        ),
    )

    if primary_spec.triggers:
        trigger_specs = [
            TriggerSpec(
                pattern=cast(str, t["pattern"]),
                match_mode=cast(
                    Literal["keyword", "prefix", "contains"],
                    t["match_mode"],
                ),
                action=cast(Literal["replace", "inject"], t["action"]),
                content=cast(str, t["content"]),
                priority=int(t.get("priority", 0)),
            )
            for t in primary_spec.triggers
        ]
        world.add_component(
            primary_entity,
            UserPromptConfigComponent(triggers=trigger_specs),
        )
    else:
        world.add_component(primary_entity, UserPromptConfigComponent())

    world.register_system(SystemPromptRenderSystem(), priority=-20)
    world.register_system(UserPromptNormalizationSystem(), priority=-10)

    # Add permission component if tools are specified
    if primary_spec.tools:
        allowed_tools = [
            tool for tool, enabled in primary_spec.tools.items() if enabled
        ]
        world.add_component(
            primary_entity,
            PermissionComponent(allowed_tools=allowed_tools, denied_tools=[]),
        )
        logger.info(
            "agent_tools_permission_attached",
            entity_id=int(primary_entity),
            allowed_count=len(allowed_tools),
        )

    # Install skills declared in DSL
    if primary_spec.skills:
        if source_dir is None:
            logger.warning(
                "agent_spec_skills_skipped_no_source_dir",
                skill_count=len(primary_spec.skills),
            )
        else:
            skill_manager = SkillManager()
            for skill_entry in primary_spec.skills:
                skill_path = (source_dir / skill_entry["path"] / "SKILL.md").resolve()
                skill_obj = Skill(skill_path=skill_path)
                skill_manager.install(world, primary_entity, cast(ScriptSkill, skill_obj))
                logger.info(
                    "agent_skill_installed",
                    entity_id=int(primary_entity),
                    skill_name=skill_obj.name,
                )
    # Always attach ToolRegistryComponent so skills and subagent tools have
    # a registry to write into, regardless of whether subagents are declared.
    world.add_component(primary_entity, ToolRegistryComponent(tools={}, handlers={}))

    subagents: dict[str, SubagentConfig] = {}
    for agent_name, spec in specs.items():
        if spec.mode != "subagent":
            continue
        subagents[agent_name] = SubagentConfig(
            name=agent_name,
            model=model_factory(spec.model, spec.prompt),
            system_prompt=spec.prompt,
        )

    world.add_component(primary_entity, SubagentRegistryComponent(subagents=subagents))

    if subagents:
        world.add_component(
            primary_entity, SubagentSessionTableComponent(sessions={})
        )
        subagent_system = SubagentSystem(priority=-1)
        world.register_system(subagent_system, priority=-1)
        subagent_system.install_subagent_tool(world, primary_entity)
        subagent_system.install_subagent_control_tools(world, primary_entity)
        logger.info(
            "agent_subagent_system_installed",
            entity_id=int(primary_entity),
            subagent_count=len(subagents),
        )

    logger.info(
        "agent_specs_compiled",
        primary_entity_id=int(primary_entity),
        subagent_count=len(subagents),
    )
    return primary_entity, world


__all__ = ["compile_agent_specs"]
