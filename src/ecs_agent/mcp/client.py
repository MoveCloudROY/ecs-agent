from __future__ import annotations

from typing import Any
from collections.abc import Awaitable
import inspect
from typing import cast

from mcp import ClientSession  # type: ignore[import-not-found]
from mcp.client.stdio import stdio_client  # type: ignore[import-not-found]
from mcp.client.sse import sse_client  # type: ignore[import-not-found]
from mcp.client.streamable_http import streamablehttp_client  # type: ignore[import-not-found]

from ecs_agent.logging import get_logger
from ecs_agent.mcp.components import MCPConfigComponent

logger = get_logger(__name__)


class MCPClient:
    def __init__(self, config: MCPConfigComponent) -> None:
        self.server_name = config.server_name
        self.transport_type = config.transport_type
        self.config = config.config
        self._session: ClientSession | None = None
        self._tools_cache: list[dict[str, Any]] = []

    @property
    def is_connected(self) -> bool:
        return self._session is not None

    async def connect(self) -> None:
        if self.is_connected:
            return

        try:
            read, write = await self._create_transport_streams()
            session = ClientSession(read, write)
            await session.initialize()
            self._session = session
            logger.info(
                "mcp_connect",
                server=self.server_name,
                transport=self.transport_type,
            )
        except Exception as exc:
            logger.error(
                "mcp_connect_failed",
                server=self.server_name,
                transport=self.transport_type,
                exception=str(exc),
            )
            raise RuntimeError(
                f"Failed to connect to MCP server '{self.server_name}': {exc}"
            ) from exc

    async def disconnect(self) -> None:
        if not self._session:
            return

        close = getattr(self._session, "close", None)
        if callable(close):
            close_result = close()
            if isinstance(close_result, Awaitable) or inspect.isawaitable(close_result):
                await close_result

        self._session = None
        self._tools_cache = []
        logger.info("mcp_disconnect", server=self.server_name)

    async def list_tools(self) -> list[dict[str, Any]]:
        session = self._require_connected_session()
        result = await session.list_tools()
        tools = getattr(result, "tools", [])
        self._tools_cache = [self._serialize_tool(tool) for tool in tools]
        return list(self._tools_cache)

    async def call_tool(self, name: str, args: dict[str, Any]) -> str:
        session = self._require_connected_session()
        if self._tools_cache and name not in {
            tool["name"] for tool in self._tools_cache
        }:
            raise ValueError(
                f"Unknown tool '{name}' for MCP server '{self.server_name}'"
            )

        result = await session.call_tool(name, arguments=args)
        content = getattr(result, "content", None)
        if not content:
            return ""

        first = content[0]
        text_value = getattr(first, "text", None)
        if isinstance(text_value, str):
            return text_value

        if isinstance(first, dict):
            dict_text = first.get("text")
            if isinstance(dict_text, str):
                return dict_text

        return str(first)

    async def _create_transport_streams(self) -> tuple[Any, Any]:
        if self.transport_type == "stdio":
            streams = await stdio_client(self.config)
            return cast(tuple[Any, Any], streams)

        if self.transport_type == "sse":
            url = self._require_config_key("url")
            streams = await sse_client(url)
            return cast(tuple[Any, Any], streams)

        if self.transport_type == "http":
            url = self._require_config_key("url")
            streams = await streamablehttp_client(url)
            return cast(tuple[Any, Any], streams)

        raise ValueError(f"Unknown transport type: {self.transport_type}")

    def _require_connected_session(self) -> ClientSession:
        if self._session is None:
            raise RuntimeError("Not connected to MCP server")
        return self._session

    def _require_config_key(self, key: str) -> str:
        value = self.config.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"Missing required MCP config key '{key}'")
        return value

    def _serialize_tool(self, tool: Any) -> dict[str, Any]:
        if isinstance(tool, dict):
            return {
                "name": tool.get("name", ""),
                "description": tool.get("description", ""),
                "inputSchema": tool.get("inputSchema", {}),
            }

        return {
            "name": getattr(tool, "name", ""),
            "description": getattr(tool, "description", ""),
            "inputSchema": getattr(tool, "inputSchema", {}),
        }
