from __future__ import annotations

from collections.abc import Awaitable
from contextlib import AsyncExitStack
import inspect
from typing import Any

from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.streamable_http import streamablehttp_client

from ecs_agent.logging import get_logger
from ecs_agent.mcp.components import MCPConfigComponent

logger = get_logger(__name__)


class MCPClient:
    def __init__(self, config: MCPConfigComponent) -> None:
        self.server_name = config.server_name
        self.transport_type = config.transport_type
        self.config = config.config
        self._session: ClientSession | None = None
        self._transport_exit_stack: AsyncExitStack | None = None
        self._tools_cache: list[dict[str, Any]] = []

    @property
    def is_connected(self) -> bool:
        return self._session is not None

    async def connect(self) -> None:
        if self.is_connected:
            return

        transport_exit_stack = AsyncExitStack()
        try:
            read, write = await self._create_transport_streams(transport_exit_stack)
            session = ClientSession(read, write)
            await session.initialize()
            self._session = session
            self._transport_exit_stack = transport_exit_stack
            logger.info(
                "mcp_connect",
                server=self.server_name,
                transport=self.transport_type,
            )
        except Exception as exc:
            await transport_exit_stack.aclose()
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

        if self._transport_exit_stack is not None:
            await self._transport_exit_stack.aclose()
            self._transport_exit_stack = None

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

    async def _create_transport_streams(
        self, transport_exit_stack: AsyncExitStack
    ) -> tuple[Any, Any]:
        if self.transport_type == "stdio":
            params = self._create_stdio_server_parameters()
            streams = await transport_exit_stack.enter_async_context(
                stdio_client(params)
            )
            return self._extract_stream_pair(streams)

        if self.transport_type == "sse":
            url = self._require_config_key("url")
            streams = await transport_exit_stack.enter_async_context(sse_client(url))
            return self._extract_stream_pair(streams)

        if self.transport_type == "http":
            url = self._require_config_key("url")
            streams = await transport_exit_stack.enter_async_context(
                streamablehttp_client(url)
            )
            return self._extract_stream_pair(streams)

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

    def _create_stdio_server_parameters(self) -> StdioServerParameters:
        command = self._require_config_key("command")

        raw_args = self.config.get("args", [])
        if not isinstance(raw_args, list) or not all(
            isinstance(item, str) for item in raw_args
        ):
            raise ValueError("MCP stdio config key 'args' must be a list[str]")
        args = list(raw_args)

        raw_env = self.config.get("env")
        env: dict[str, str] | None
        if raw_env is None:
            env = None
        elif isinstance(raw_env, dict) and all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in raw_env.items()
        ):
            env = {key: value for key, value in raw_env.items()}
        else:
            raise ValueError("MCP stdio config key 'env' must be a dict[str, str]")

        raw_cwd = self.config.get("cwd")
        if raw_cwd is not None and not isinstance(raw_cwd, str):
            raise ValueError("MCP stdio config key 'cwd' must be a string")
        cwd = raw_cwd

        return StdioServerParameters(command=command, args=args, env=env, cwd=cwd)

    def _extract_stream_pair(self, streams: tuple[Any, ...]) -> tuple[Any, Any]:
        if len(streams) < 2:
            raise RuntimeError("MCP transport did not provide read/write streams")
        return streams[0], streams[1]

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
