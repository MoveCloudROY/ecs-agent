"""Render normalized system prompts from SystemPromptConfigSpec."""

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
    SystemPromptComponent,
    ToolRegistryComponent,
)
from ecs_agent.core.world import World
from ecs_agent.logging import get_logger
from ecs_agent.prompts.contracts import SystemPromptConfigSpec, PromptTemplateSource
from ecs_agent.prompts.provider import (
    BuiltinPlaceholderProvider,
    InventoryPlaceholderProvider,
)
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

_BUILTIN_PLACEHOLDER_PROVIDERS: list[BuiltinPlaceholderProvider] = [
    InventoryPlaceholderProvider(),
]

logger = get_logger(__name__)


class SystemPromptRenderSystem:
    def __init__(self, priority: int = 0) -> None:
        self.priority = priority

    async def process(self, world: World) -> None:
        if not _uses_default_builtin_provider_registry():
            raise ValueError("built-in provider aggregation not wired yet")

        for entity_id, (prompt_config,) in world.query(SystemPromptConfigSpec):
            cache_key = _render_cache_key(world, entity_id)
            rendered_component = world.get_component(
                entity_id, RenderedSystemPromptComponent
            )
            if rendered_component is not None:
                previous_cache_key = rendered_component.placeholder_snapshot.get(
                    "_cache_key"
                )
                if previous_cache_key == cache_key:
                    _bridge_rendered_prompt(world, entity_id, rendered_component.text)
                    continue
                world.remove_component(entity_id, RenderedSystemPromptComponent)

                logger.debug(
                    "system_prompt_render_invalidated",
                    entity_id=entity_id,
                    previous_cache_key=previous_cache_key,
                    current_cache_key=cache_key,
                )
            try:
                rendered, snapshot = _render_system_prompt(
                    world, entity_id, prompt_config
                )
                snapshot["_cache_key"] = cache_key
                world.add_component(
                    entity_id,
                    RenderedSystemPromptComponent(
                        text=rendered,
                        placeholder_snapshot=snapshot,
                    ),
                )

                logger.debug(
                    "system_prompt_rendered",
                    entity_id=entity_id,
                    prompt_length=len(rendered),
                    placeholder_count=len(snapshot),
                    prompt_text=rendered,
                )

                _bridge_rendered_prompt(world, entity_id, rendered)
            except Exception as exc:
                logger.error(
                    "system_prompt_render_failed",
                    entity_id=entity_id,
                    exception=str(exc),
                )
                raise


def _uses_default_builtin_provider_registry() -> bool:
    if len(_BUILTIN_PLACEHOLDER_PROVIDERS) != 1:
        return False
    return isinstance(_BUILTIN_PLACEHOLDER_PROVIDERS[0], InventoryPlaceholderProvider)


def _render_system_prompt(
    world: World,
    entity_id: EntityId,
    prompt_config: SystemPromptConfigSpec,
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


def _resolve_user_placeholders(prompt_config: SystemPromptConfigSpec) -> dict[str, str]:
    return resolve_placeholder_values(prompt_config.placeholders)


def _resolve_builtin_placeholders(world: World, entity_id: EntityId) -> dict[str, str]:
    return {
        "_installed_tools": _format_bullets(_tool_entries(world, entity_id)),
        "_installed_skills": _format_bullets(_skill_entries(world, entity_id)),
        "_installed_mcps": _format_bullets(_mcp_tool_entries(world, entity_id)),
        "_installed_subagents": _format_bullets(_subagent_entries(world, entity_id)),
    }


def _render_cache_key(world: World, entity_id: EntityId) -> str:
    tool_names = tuple(name for name, _ in _tool_entries(world, entity_id))
    skill_names = tuple(name for name, _ in _skill_entries(world, entity_id))
    subagent_names = tuple(name for name, _ in _subagent_entries(world, entity_id))
    return (
        f"tools:{','.join(tool_names)}|"
        f"skills:{','.join(skill_names)}|"
        f"subagents:{','.join(subagent_names)}"
    )


def _bridge_rendered_prompt(world: World, entity_id: EntityId, rendered: str) -> None:
    llm_component = world.get_component(entity_id, LLMComponent)
    if llm_component is not None:
        llm_component.system_prompt = rendered

    legacy_system_prompt = world.get_component(entity_id, SystemPromptComponent)
    if legacy_system_prompt is not None:
        legacy_system_prompt.content = rendered


def _tool_entries(world: World, entity_id: EntityId) -> list[tuple[str, str]]:
    registry = world.get_component(entity_id, ToolRegistryComponent)
    if registry is None:
        return []
    return sorted(
        ((name, schema.description) for name, schema in registry.tools.items()),
        key=lambda e: e[0],
    )


def _skill_entries(world: World, entity_id: EntityId) -> list[tuple[str, str]]:
    skill_component = world.get_component(entity_id, SkillComponent)
    if skill_component is None:
        return []
    return sorted(
        ((name, meta.description) for name, meta in skill_component.skills.items()),
        key=lambda e: e[0],
    )


def _subagent_entries(world: World, entity_id: EntityId) -> list[tuple[str, str]]:
    registry = world.get_component(entity_id, SubagentRegistryComponent)
    if registry is None:
        return []
    return sorted(
        ((name, cfg.description) for name, cfg in registry.subagents.items()),
        key=lambda e: e[0],
    )


def _mcp_tool_entries(world: World, entity_id: EntityId) -> list[tuple[str, str]]:
    if _MCP_CLIENT_COMPONENT_CLASS is None:
        return []

    mcp_component = world.get_component(entity_id, _MCP_CLIENT_COMPONENT_CLASS)
    if mcp_component is None:
        return []

    cached_tools = getattr(mcp_component, "cached_tools", [])
    if not isinstance(cached_tools, list):
        return []

    entries: list[tuple[str, str]] = []
    for tool in cached_tools:
        entry = _extract_entry(tool)
        if entry is not None:
            entries.append(entry)
    return sorted(set(entries), key=lambda e: e[0])


def _extract_entry(tool: object) -> tuple[str, str] | None:
    if isinstance(tool, dict):
        name = tool.get("name")
        if isinstance(name, str):
            desc = tool.get("description", "")
            return (name, desc if isinstance(desc, str) else "")
        return None

    name = getattr(tool, "name", None)
    if isinstance(name, str):
        desc = getattr(tool, "description", "")
        return (name, desc if isinstance(desc, str) else "")
    return None


def _format_bullets(entries: list[tuple[str, str]]) -> str:
    if not entries:
        return "- none"
    lines: list[str] = []
    for name, description in entries:
        if description:
            lines.append(f"- {name}: {description}")
        else:
            lines.append(f"- {name}")
    return "\n".join(lines)


__all__ = ["SystemPromptRenderSystem"]
