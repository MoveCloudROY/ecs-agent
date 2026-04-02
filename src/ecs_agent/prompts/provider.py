"""Built-in placeholder provider protocol for system-prompt rendering."""

from __future__ import annotations

import importlib
from typing import Protocol, cast, runtime_checkable

from ecs_agent.components import (
    SkillComponent,
    SubagentRegistryComponent,
    ToolRegistryComponent,
)
from ecs_agent.core.world import World
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


@runtime_checkable
class BuiltinPlaceholderProvider(Protocol):
    """Sync protocol for built-in placeholder providers."""

    provider_id: str

    def resolve_placeholders(self, world: World, entity_id: EntityId) -> dict[str, str]:
        """Return built-in placeholder key→value pairs for this entity.

        Keys MUST start with underscore (reserved for built-ins).
        """

    def provider_fingerprint(self, world: World, entity_id: EntityId) -> str:
        """Return fingerprint that changes when provider output changes."""


class InventoryPlaceholderProvider:
    """Built-in provider for installed tool/skill/mcp/subagent placeholders."""

    provider_id = "inventory"

    def resolve_placeholders(self, world: World, entity_id: EntityId) -> dict[str, str]:
        return {
            "_installed_tools": _format_bullets(self._tool_entries(world, entity_id)),
            "_installed_skills": _format_bullets(self._skill_entries(world, entity_id)),
            "_installed_mcps": _format_bullets(
                self._mcp_tool_entries(world, entity_id)
            ),
            "_installed_subagents": _format_bullets(
                self._subagent_entries(world, entity_id)
            ),
        }

    def provider_fingerprint(self, world: World, entity_id: EntityId) -> str:
        tool_names = tuple(name for name, _ in self._tool_entries(world, entity_id))
        skill_names = tuple(name for name, _ in self._skill_entries(world, entity_id))
        subagent_names = tuple(
            name for name, _ in self._subagent_entries(world, entity_id)
        )
        mcp_names = tuple(name for name, _ in self._mcp_tool_entries(world, entity_id))
        return (
            f"tools:{','.join(tool_names)}|"
            f"skills:{','.join(skill_names)}|"
            f"subagents:{','.join(subagent_names)}|"
            f"mcps:{','.join(mcp_names)}"
        )

    @staticmethod
    def _tool_entries(world: World, entity_id: EntityId) -> list[tuple[str, str]]:
        registry = world.get_component(entity_id, ToolRegistryComponent)
        if registry is None:
            return []
        return sorted(
            ((name, schema.description) for name, schema in registry.tools.items()),
            key=lambda e: e[0],
        )

    @staticmethod
    def _skill_entries(world: World, entity_id: EntityId) -> list[tuple[str, str]]:
        skill_component = world.get_component(entity_id, SkillComponent)
        if skill_component is None:
            return []
        return sorted(
            ((name, meta.description) for name, meta in skill_component.skills.items()),
            key=lambda e: e[0],
        )

    @staticmethod
    def _subagent_entries(world: World, entity_id: EntityId) -> list[tuple[str, str]]:
        registry = world.get_component(entity_id, SubagentRegistryComponent)
        if registry is None:
            return []
        return sorted(
            ((name, cfg.description) for name, cfg in registry.subagents.items()),
            key=lambda e: e[0],
        )

    @staticmethod
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


__all__ = ["BuiltinPlaceholderProvider", "InventoryPlaceholderProvider"]
