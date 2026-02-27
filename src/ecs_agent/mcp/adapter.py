from __future__ import annotations

import asyncio
from concurrent.futures import Future, ThreadPoolExecutor
from collections.abc import Awaitable, Callable, Coroutine
from typing import TYPE_CHECKING, Any

from ecs_agent.components import ToolRegistryComponent
from ecs_agent.components.definitions import SkillComponent
from ecs_agent.core.world import World
from ecs_agent.skills.manager import SkillManager
from ecs_agent.skills.protocol import Skill
from ecs_agent.types import EntityId, ToolSchema

if TYPE_CHECKING:
    from ecs_agent.mcp.client import MCPClient

MCPTool = dict[str, Any]
ToolHandler = Callable[..., Awaitable[str]]
MCPToolConverter = Callable[[MCPTool], tuple[ToolSchema, ToolHandler] | None]


def mcp_tool_to_ecs_tool(
    mcp_client: MCPClient, server_name: str, tool: MCPTool
) -> tuple[ToolSchema, ToolHandler]:
    original_name = str(tool.get("name", "")).strip()
    namespaced_name = f"{server_name}/{original_name}"
    description = str(tool.get("description", "")).strip()
    input_schema = tool.get("inputSchema")
    parameters = input_schema if isinstance(input_schema, dict) else {}

    schema = ToolSchema(
        name=namespaced_name,
        description=description,
        parameters=parameters,
    )

    async def handler(**kwargs: Any) -> str:
        return await mcp_client.call_tool(original_name, kwargs)

    return schema, handler


class MCPSkillAdapter(Skill):
    def __init__(
        self,
        mcp_client: MCPClient,
        server_name: str,
        converter: MCPToolConverter | None = None,
    ) -> None:
        self.name = f"mcp-{server_name}"
        self.description = f"MCP tool adapter for server '{server_name}'."
        self._mcp_client = mcp_client
        self._server_name = server_name
        self._converter = converter
        self._manager = SkillManager()
        self._tool_bundle: dict[str, tuple[ToolSchema, ToolHandler]] = {}
        self._is_installing = False
        self._is_uninstalling = False
        self._loaded = False

    def tools(self) -> dict[str, tuple[ToolSchema, ToolHandler]]:
        self._ensure_loaded()
        return dict(self._tool_bundle)

    def system_prompt(self) -> str:
        self._ensure_loaded()
        tool_names = sorted(self._tool_bundle.keys())
        if not tool_names:
            return ""

        names_text = ", ".join(tool_names)
        return (
            "Tier 1 MCP skill summary: this skill provides external tools from a single "
            f"MCP server. Available tools are {names_text}. Use these tools only when needed "
            "for concrete actions. If you need complete parameter schemas, call the "
            "load_skill_details tool with this skill name."
        )

    def install(self, world: World, entity_id: EntityId) -> None:
        if self._is_installing:
            return

        skill_component = world.get_component(entity_id, SkillComponent)
        if skill_component is not None and self.name in skill_component.skills:
            return

        self._is_installing = True
        try:
            self._ensure_loaded()
            self._manager.install(world, entity_id, self)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to install MCP skill '{self.name}': {exc}"
            ) from exc
        finally:
            self._is_installing = False

    def uninstall(self, world: World, entity_id: EntityId) -> None:
        if self._is_uninstalling:
            return

        self._is_uninstalling = True
        try:
            skill_component = world.get_component(entity_id, SkillComponent)
            metadata = (
                skill_component.skills.get(self.name)
                if skill_component is not None
                else None
            )
            registry = world.get_component(entity_id, ToolRegistryComponent)
            if (
                metadata is not None
                and registry is not None
                and skill_component is not None
            ):
                for tool_name in metadata.tool_names:
                    registry.tools.pop(tool_name, None)
                    registry.handlers.pop(tool_name, None)
                skill_component.skills.pop(self.name, None)

            self._run_sync(self._mcp_client.disconnect())
            self._tool_bundle = {}
            self._loaded = False
        finally:
            self._is_uninstalling = False

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return

        self._run_sync(self._load_tools())
        self._loaded = True

    async def _load_tools(self) -> None:
        if not self._mcp_client.is_connected:
            await self._mcp_client.connect()

        raw_tools = await self._mcp_client.list_tools()
        converter = self._converter or (
            lambda tool: mcp_tool_to_ecs_tool(self._mcp_client, self._server_name, tool)
        )

        converted: dict[str, tuple[ToolSchema, ToolHandler]] = {}
        for tool in raw_tools:
            item = converter(tool)
            if item is None:
                continue

            schema, handler = item
            converted[schema.name] = (schema, handler)

        self._tool_bundle = converted

    def _run_sync(self, operation: Coroutine[Any, Any, Any]) -> Any:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(operation)

        def _run_in_thread() -> Any:
            return asyncio.run(operation)

        with ThreadPoolExecutor(max_workers=1) as executor:
            future: Future[Any] = executor.submit(_run_in_thread)
            return future.result()
