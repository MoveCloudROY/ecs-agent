"""Built-in tools skill."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from ecs_agent.core.world import World
from ecs_agent.logging import get_logger
from ecs_agent.skills.script_skill import ScriptSkill
from ecs_agent.tools.discovery import scan_module
from ecs_agent.types import EntityId, ToolSchema

from ecs_agent.tools.builtins import bash_tool, edit_tool, file_tools, glob_tool

logger = get_logger(__name__)


class BuiltinToolsSkill(ScriptSkill):
    """Skill providing read_file, write_file, edit_file, bash."""

    name = "builtin-tools"
    description = "Basic file manipulation and bash execution tools."
    is_tool_bundle = True

    def __init__(self) -> None:
        self._bound_tools: (
            dict[str, tuple[ToolSchema, Callable[..., Awaitable[str]]]] | None
        ) = None

    def tools(self) -> dict[str, tuple[ToolSchema, Callable[..., Awaitable[str]]]]:
        if self._bound_tools is not None:
            return self._bound_tools

        discovered: dict[str, tuple[ToolSchema, Callable[..., Awaitable[str]]]] = {}
        for module in (file_tools, bash_tool, edit_tool, glob_tool):
            discovered.update(scan_module(module))
        return discovered

    def bind_workspace(self, workspace_root: str) -> "BuiltinToolsSkill":
        """Bind workspace_root into all tool handlers and strip it from schemas.

        After calling this, tools no longer expose ``workspace_root`` as a
        parameter visible to the LLM.  The value is injected automatically
        when each handler is invoked.
        """
        original_tools = dict(self.tools())
        bound: dict[str, tuple[ToolSchema, Callable[..., Awaitable[str]]]] = {}
        for tool_name, (schema, handler) in original_tools.items():
            params = schema.parameters
            filtered_props = {
                k: v
                for k, v in params.get("properties", {}).items()
                if k != "workspace_root"
            }
            filtered_required = [
                r for r in params.get("required", []) if r != "workspace_root"
            ]
            new_schema = ToolSchema(
                name=schema.name,
                description=schema.description,
                parameters={
                    "type": params.get("type", "object"),
                    "properties": filtered_props,
                    "required": filtered_required,
                },
                sandbox_compatible=schema.sandbox_compatible,
            )

            async def _bound(
                _h: Callable[..., Awaitable[str]] = handler, **kwargs: object
            ) -> str:
                return await _h(workspace_root=workspace_root, **kwargs)

            bound[tool_name] = (new_schema, _bound)

        self._bound_tools = bound
        return self

    def system_prompt(self) -> str:
        return ""

    def install(self, world: World, entity_id: EntityId) -> None:
        _ = world
        _ = entity_id
        logger.info("builtin_tools_skill_install")

    def uninstall(self, world: World, entity_id: EntityId) -> None:
        _ = world
        _ = entity_id
        logger.info("builtin_tools_skill_uninstall")


__all__ = ["BuiltinToolsSkill"]
