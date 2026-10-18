"""MCP client for external tool servers."""

from __future__ import annotations

import asyncio
import contextlib
from contextlib import AsyncExitStack
from typing import Any

from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.streamable_http import streamablehttp_client

from ecs_agent.logging import get_logger
from ecs_agent.mcp.components import MCPConfigComponent

logger = get_logger(__name__)


class MCPClient:
    """Client for one MCP server, holding the session in an owner task.

    The transport and ``ClientSession`` contexts are entered AND exited inside
    a single background task (`_run_connection`): entering the session context
    is what starts the SDK's receive loop (without it ``initialize()`` awaits
    a response nobody reads — a permanent hang), and anyio cancel scopes are
    task-bound, so exiting from a different task raises. The owner task keeps
    the session alive until ``disconnect()`` signals it to unwind.
    """

    def __init__(self, config: MCPConfigComponent) -> None:
        self.server_name = config.server_name
        self.transport_type = config.transport_type
        self.config = config.config
        self._session: ClientSession | None = None
        self._tools_cache: list[dict[str, Any]] = []
        self._owner_task: asyncio.Task[None] | None = None
        self._ready: asyncio.Event | None = None
        self._stop: asyncio.Event | None = None
        self._connect_error: BaseException | None = None
        self._was_connected = False

    @property
    def is_connected(self) -> bool:
        return self._session is not None

    async def connect(self, timeout: float = 30.0) -> None:
        """Establish the session, bounded by ``timeout`` seconds.

        Raises:
            RuntimeError: When the server cannot be reached, rejects the
                handshake, or does not answer within the timeout.
        """
        if self.is_connected:
            return

        self._ready = asyncio.Event()
        self._stop = asyncio.Event()
        self._connect_error = None
        self._owner_task = asyncio.create_task(self._run_connection())
        try:
            await asyncio.wait_for(self._ready.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            self._owner_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._owner_task
            self._owner_task = None
            logger.error(
                "mcp_connect_failed",
                server=self.server_name,
                transport=self.transport_type,
                exception=f"timed out after {timeout}s",
            )
            raise RuntimeError(
                f"Failed to connect to MCP server '{self.server_name}': "
                f"timed out after {timeout}s"
            )

        if self._connect_error is not None:
            error = self._connect_error
            self._owner_task = None
            logger.error(
                "mcp_connect_failed",
                server=self.server_name,
                transport=self.transport_type,
                exception=str(error),
            )
            raise RuntimeError(
                f"Failed to connect to MCP server '{self.server_name}': {error}"
            ) from error

        self._was_connected = True
        logger.info(
            "mcp_connect",
            server=self.server_name,
            transport=self.transport_type,
        )

    async def _run_connection(self) -> None:
        """Owner task: the session's full lifecycle lives in this one task."""
        try:
            async with AsyncExitStack() as stack:
                read, write = await self._create_transport_streams(stack)
                session = await stack.enter_async_context(
                    ClientSession(read, write)
                )
                await session.initialize()
                self._session = session
                assert self._ready is not None and self._stop is not None
                self._ready.set()
                await self._stop.wait()
        except Exception as exc:
            self._connect_error = exc
        finally:
            self._session = None
            if self._ready is not None:
                self._ready.set()

    async def disconnect(self) -> None:
        task = self._owner_task
        if task is None:
            return

        assert self._stop is not None
        self._stop.set()
        try:
            await asyncio.wait_for(task, timeout=10.0)
        except asyncio.TimeoutError:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        finally:
            self._owner_task = None
            self._session = None
            self._tools_cache = []

        logger.info("mcp_disconnect", server=self.server_name)

    async def list_tools(self) -> list[dict[str, Any]]:
        session = await self._connected_session()
        result = await session.list_tools()
        tools = getattr(result, "tools", [])
        self._tools_cache = [self._serialize_tool(tool) for tool in tools]
        return list(self._tools_cache)

    async def call_tool(self, name: str, args: dict[str, Any]) -> str:
        session = await self._connected_session()
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

        raise TypeError(
            f"MCP tool returned non-text content of type {type(first).__name__}"
        )

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

    async def _connected_session(self) -> ClientSession:
        """Return the live session, reconnecting on the current loop if needed.

        Self-healing matters for the sync install path: tools loaded through a
        temporary event loop lose their session when that loop is torn down;
        the first real tool call then reconnects on the loop that will
        actually use the session.
        """
        if self._session is None and self._was_connected:
            # A previous session existed but its loop is gone (e.g. tools
            # loaded through the temporary-loop sync bridge): re-establish on
            # the loop that is actually calling the tools.
            await self.connect()
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
