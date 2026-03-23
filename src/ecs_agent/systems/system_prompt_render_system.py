"""Render normalized system prompts from PromptConfigSpec."""

from __future__ import annotations

from typing import cast
from pathlib import Path
from string import Template
import importlib

from ecs_agent.components import (
    LLMComponent,
    RenderedSystemPromptComponent,
    SkillComponent,
    SubagentRegistryComponent,
    ToolRegistryComponent,
)
from ecs_agent.core.world import World
from ecs_agent.logging import get_logger
from ecs_agent.prompts.contracts import PromptConfigSpec, PromptTemplateSource
from ecs_agent.prompts.registry import resolve_placeholder_values
from ecs_agent.types import EntityId


def _load_mcp_client_component() -> type[object] | None:
    try:
        module = importlib.import_module("ecs_agent.mcp.components")
    except ImportError:
        return None

    component = getattr(module, "MCPClientComponent", None)
    if component is None or not isinstance(component, type):
        return None
    return cast(type[object], component)


_MCP_CLIENT_COMPONENT_CLASS = _load_mcp_client_component()

logger = get_logger(__name__)


class SystemPromptRenderSystem:
    def __init__(self, priority: int = 0) -> None:
        self.priority = priority

    async def process(self, world: World) -> None:
        for entity_id, (prompt_config,) in world.query(PromptConfigSpec):
            try:
                rendered, snapshot = _render_system_prompt(
                    world, entity_id, prompt_config
                )
                world.add_component(
                    entity_id,
                    RenderedSystemPromptComponent(
                        text=rendered,
                        placeholder_snapshot=snapshot,
                    ),
                )

                llm_component = world.get_component(entity_id, LLMComponent)
                if llm_component is not None:
                    llm_component.system_prompt = rendered
            except Exception as exc:
                logger.error(
                    "system_prompt_render_failed",
                    entity_id=entity_id,
                    exception=str(exc),
                )
                raise


def _render_system_prompt(
    world: World,
    entity_id: EntityId,
    prompt_config: PromptConfigSpec,
) -> tuple[str, dict[str, str]]:
    template_text = _read_template(prompt_config.template_source)
    template = Template(template_text)

    user_values = _resolve_user_placeholders(prompt_config)
    builtins = _resolve_builtin_placeholders(world, entity_id)
    snapshot = {**user_values, **builtins}

    try:
        rendered = template.substitute(snapshot)
    except KeyError as exc:
        missing = str(exc).strip("'\"")
        raise ValueError(f"unknown placeholders in template: {missing}") from exc
    return rendered, snapshot


def _read_template(template_source: PromptTemplateSource) -> str:
    inline = template_source.inline
    file_path = template_source.file_path
    if inline is not None:
        return inline

    assert file_path is not None
    path = Path(file_path)
    if not path.exists():
        raise ValueError(f"missing template file: {file_path}")
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"unreadable template file: {file_path}") from exc


def _resolve_user_placeholders(prompt_config: PromptConfigSpec) -> dict[str, str]:
    return resolve_placeholder_values(prompt_config.placeholders)


def _resolve_builtin_placeholders(world: World, entity_id: EntityId) -> dict[str, str]:
    return {
        "_installed_tools": _format_bullets(_tool_names(world, entity_id)),
        "_installed_skills": _format_bullets(_skill_names(world, entity_id)),
        "_installed_mcps": _format_bullets(_mcp_tool_names(world, entity_id)),
        "_installed_subagents": _format_bullets(_subagent_names(world, entity_id)),
    }


def _tool_names(world: World, entity_id: EntityId) -> list[str]:
    registry = world.get_component(entity_id, ToolRegistryComponent)
    if registry is None:
        return []
    return sorted(registry.tools)


def _skill_names(world: World, entity_id: EntityId) -> list[str]:
    skill_component = world.get_component(entity_id, SkillComponent)
    if skill_component is None:
        return []
    return sorted(skill_component.skills)


def _subagent_names(world: World, entity_id: EntityId) -> list[str]:
    registry = world.get_component(entity_id, SubagentRegistryComponent)
    if registry is None:
        return []
    return sorted(registry.subagents)


def _mcp_tool_names(world: World, entity_id: EntityId) -> list[str]:
    if _MCP_CLIENT_COMPONENT_CLASS is None:
        return []

    mcp_component = world.get_component(entity_id, _MCP_CLIENT_COMPONENT_CLASS)
    if mcp_component is None:
        return []

    cached_tools = getattr(mcp_component, "cached_tools", [])
    if not isinstance(cached_tools, list):
        return []

    names: list[str] = []
    for tool in cached_tools:
        name = _extract_name(tool)
        if name:
            names.append(name)
    return sorted(set(names))


def _extract_name(tool: object) -> str | None:
    if isinstance(tool, dict):
        name = tool.get("name")
        if isinstance(name, str):
            return name
        return None

    name = getattr(tool, "name", None)
    if isinstance(name, str):
        return name
    return None


def _format_bullets(values: list[str]) -> str:
    if not values:
        return "- none"
    return "\n".join(f"- {value}" for value in values)


__all__ = ["SystemPromptRenderSystem"]
