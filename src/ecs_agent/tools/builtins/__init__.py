"""Built-in tools skill."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from ecs_agent.core.world import World
from ecs_agent.logging import get_logger
from ecs_agent.skills.script_skill import ScriptSkill
from ecs_agent.tools.builtins.file_snapshot import FileSnapshotStore, use_snapshot_store
from ecs_agent.tools.discovery import scan_module
from ecs_agent.types import EntityId, ToolSchema

from ecs_agent.tools.builtins import (
    bash_tool,
    code_execution_tool,
    edit_tool,
    explore_tool,
    file_tools,
    glob_tool,
    grep_tool,
    webfetch_tool,
)

logger = get_logger(__name__)


def _hide_internal_schema_params(schema: ToolSchema) -> ToolSchema:
    params = schema.parameters
    return ToolSchema(
        name=schema.name,
        description=schema.description,
        parameters={
            "type": params.get("type", "object"),
            "properties": dict(params.get("properties", {})),
            "required": list(params.get("required", [])),
        },
        sandbox_compatible=schema.sandbox_compatible,
    )


class BuiltinToolsSkill(ScriptSkill):
    name = "builtin-tools"
    description = (
        "Basic file manipulation, bash execution, and tmux interactive session tools."
    )
    is_tool_bundle = True

    def __init__(self) -> None:
        self._bound_tools: (
            dict[str, tuple[ToolSchema, Callable[..., Awaitable[str]]]] | None
        ) = None
        self._snapshot_store = FileSnapshotStore()

    def tools(self) -> dict[str, tuple[ToolSchema, Callable[..., Awaitable[str]]]]:
        if self._bound_tools is not None:
            return self._bound_tools

        discovered: dict[str, tuple[ToolSchema, Callable[..., Awaitable[str]]]] = {}
        for module in (
            file_tools,
            bash_tool,
            edit_tool,
            glob_tool,
            grep_tool,
            explore_tool,
            webfetch_tool,
            code_execution_tool,
        ):
            discovered.update(scan_module(module))
        return {
            tool_name: (_hide_internal_schema_params(schema), handler)
            for tool_name, (schema, handler) in discovered.items()
        }

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
            has_workspace_root = "workspace_root" in params.get("properties", {})
            uses_snapshot_store = tool_name in {"read_file", "write_file", "edit_file"}
            filtered_props = {
                k: v
                for k, v in params.get("properties", {}).items()
                if k != "workspace_root"
            }
            filtered_required = [
                r
                for r in params.get("required", [])
                if r != "workspace_root"
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

            if has_workspace_root or uses_snapshot_store:
                async def _bound(
                    _h: Callable[..., Awaitable[str]] = handler,
                    _has_workspace_root: bool = has_workspace_root,
                    _uses_snapshot_store: bool = uses_snapshot_store,
                    **kwargs: object,
                ) -> str:
                    injected: dict[str, object] = dict(kwargs)
                    if _has_workspace_root:
                        injected["workspace_root"] = workspace_root
                    if _uses_snapshot_store:
                        with use_snapshot_store(self._snapshot_store):
                            return await _h(**injected)
                    return await _h(**injected)

                bound[tool_name] = (new_schema, _bound)
            else:
                bound[tool_name] = (new_schema, handler)

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
