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

    def tools(self) -> dict[str, tuple[ToolSchema, Callable[..., Awaitable[str]]]]:
        discovered: dict[str, tuple[ToolSchema, Callable[..., Awaitable[str]]]] = {}
        for module in (file_tools, bash_tool, edit_tool, glob_tool):
            discovered.update(scan_module(module))
        return discovered

    def bind_workspace(self, workspace_root: str) -> None:
        """Bind workspace_root into all tool handlers and strip it from schemas.

        After calling this, tools no longer expose ``workspace_root`` as a
        parameter visible to the LLM.  The value is injected automatically
        when each handler is invoked.
        """
        original_tools = self.tools()
        bound: dict[str, tuple[ToolSchema, Callable[..., Awaitable[str]]]] = {}
        for tool_name, (schema, handler) in original_tools.items():
            # Strip workspace_root from schema so the LLM does not send it.
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

            # Wrap handler to inject workspace_root automatically.
            async def _bound(
                _h: Callable[..., Awaitable[str]] = handler, **kwargs: str
            ) -> str:
                return await _h(workspace_root=workspace_root, **kwargs)

            bound[tool_name] = (new_schema, _bound)

        # Replace tools() to return the bound versions.
        self.tools = lambda: bound  # type: ignore[method-assign]  # mypy cannot narrow lambda assignment to instance method slot

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
