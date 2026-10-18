from __future__ import annotations

import asyncio
import builtins
from contextlib import asynccontextmanager
import importlib
import sys
import types
from dataclasses import dataclass, field
from unittest.mock import AsyncMock

import pytest


def _reset_mcp_modules() -> None:
    for module_name in [
        "mcp",
        "mcp.client",
        "mcp.client.stdio",
        "mcp.client.sse",
        "mcp.client.streamable_http",
        "ecs_agent.mcp",
        "ecs_agent.mcp.client",
        "ecs_agent.mcp.components",
    ]:
        sys.modules.pop(module_name, None)


def _install_fake_mcp(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    _reset_mcp_modules()

    state: dict[str, object] = {
        "initialize": AsyncMock(),
        "close": AsyncMock(),
        "list_tools": AsyncMock(),
        "call_tool": AsyncMock(),
        "stdio_client": AsyncMock(),
        "sse_client": AsyncMock(),
        "streamablehttp_client": AsyncMock(),
    }

    class FakeSession:
        def __init__(self, read: object, write: object) -> None:
            self.read = read
            self.write = write
            self.initialize = state["initialize"]
            self.close = state["close"]
            self.list_tools = state["list_tools"]
            self.call_tool = state["call_tool"]

        async def __aenter__(self) -> "FakeSession":
            # Entering the session context is what starts the SDK receive
            # loop; MCPClient now enters it before initialize().
            return self

        async def __aexit__(
            self,
            exc_type: object,
            exc: object,
            tb: object,
        ) -> None:
            # The real ClientSession cleans up here; the fake routes it to the
            # close mock so disconnect assertions stay meaningful.
            await self.close()

    mcp_module = types.ModuleType("mcp")
    mcp_module.ClientSession = FakeSession  # type: ignore[attr-defined]

    client_module = types.ModuleType("mcp.client")
    stdio_module = types.ModuleType("mcp.client.stdio")
    sse_module = types.ModuleType("mcp.client.sse")
    streamable_http_module = types.ModuleType("mcp.client.streamable_http")

    @dataclass(slots=True)
    class FakeStdioServerParameters:
        command: str
        args: list[str] = field(default_factory=list)
        env: dict[str, str] | None = None
        cwd: str | None = None
        encoding: str = "utf-8"
        encoding_error_handler: str = "strict"

    @asynccontextmanager
    async def stdio_client(
        config: FakeStdioServerParameters,
    ) -> types.AsyncGeneratorType:
        mock = state["stdio_client"]
        assert isinstance(mock, AsyncMock)
        await mock(config)
        yield object(), object()

    @asynccontextmanager
    async def sse_client(url: str) -> types.AsyncGeneratorType:
        mock = state["sse_client"]
        assert isinstance(mock, AsyncMock)
        await mock(url)
        yield object(), object()

    @asynccontextmanager
    async def streamablehttp_client(url: str) -> types.AsyncGeneratorType:
        mock = state["streamablehttp_client"]
        assert isinstance(mock, AsyncMock)
        await mock(url)
        yield object(), object(), (lambda: None)

    stdio_module.StdioServerParameters = FakeStdioServerParameters  # type: ignore[attr-defined]
    stdio_module.stdio_client = stdio_client  # type: ignore[attr-defined]
    sse_module.sse_client = sse_client  # type: ignore[attr-defined]
    streamable_http_module.streamablehttp_client = streamablehttp_client  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "mcp", mcp_module)
    monkeypatch.setitem(sys.modules, "mcp.client", client_module)
    monkeypatch.setitem(sys.modules, "mcp.client.stdio", stdio_module)
    monkeypatch.setitem(sys.modules, "mcp.client.sse", sse_module)
    monkeypatch.setitem(
        sys.modules, "mcp.client.streamable_http", streamable_http_module
    )
    return state


def _import_mcp_client_module() -> types.ModuleType:
    return importlib.import_module("ecs_agent.mcp.client")


@pytest.mark.asyncio
async def test_constructor_accepts_config_component(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_mcp(monkeypatch)
    components_module = importlib.import_module("ecs_agent.mcp.components")
    client_module = _import_mcp_client_module()

    config = components_module.MCPConfigComponent(
        server_name="filesystem",
        transport_type="stdio",
        config={"command": "uvx", "args": ["mcp-server"]},
    )

    client = client_module.MCPClient(config)

    assert client.server_name == "filesystem"
    assert client.transport_type == "stdio"
    assert client.config == {"command": "uvx", "args": ["mcp-server"]}
    assert client.is_connected is False


@pytest.mark.asyncio
async def test_connect_stdio(monkeypatch: pytest.MonkeyPatch) -> None:
    state = _install_fake_mcp(monkeypatch)
    components_module = importlib.import_module("ecs_agent.mcp.components")
    client_module = _import_mcp_client_module()

    config = components_module.MCPConfigComponent(
        server_name="filesystem",
        transport_type="stdio",
        config={"command": "uvx", "args": ["mcp-server"]},
    )
    client = client_module.MCPClient(config)

    await client.connect()

    stdio_mock = state["stdio_client"]
    initialize_mock = state["initialize"]
    assert isinstance(stdio_mock, AsyncMock)
    assert isinstance(initialize_mock, AsyncMock)
    stdio_mock.assert_awaited_once()
    params = stdio_mock.await_args.args[0]
    assert getattr(params, "command") == "uvx"
    assert getattr(params, "args") == ["mcp-server"]
    initialize_mock.assert_awaited_once()
    assert client.is_connected is True


@pytest.mark.asyncio
async def test_connect_sse(monkeypatch: pytest.MonkeyPatch) -> None:
    state = _install_fake_mcp(monkeypatch)
    components_module = importlib.import_module("ecs_agent.mcp.components")
    client_module = _import_mcp_client_module()

    config = components_module.MCPConfigComponent(
        server_name="remote",
        transport_type="sse",
        config={"url": "https://example.com/sse"},
    )
    client = client_module.MCPClient(config)

    await client.connect()

    sse_mock = state["sse_client"]
    initialize_mock = state["initialize"]
    assert isinstance(sse_mock, AsyncMock)
    assert isinstance(initialize_mock, AsyncMock)
    sse_mock.assert_awaited_once_with("https://example.com/sse")
    initialize_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_connect_http(monkeypatch: pytest.MonkeyPatch) -> None:
    state = _install_fake_mcp(monkeypatch)
    components_module = importlib.import_module("ecs_agent.mcp.components")
    client_module = _import_mcp_client_module()

    config = components_module.MCPConfigComponent(
        server_name="http-server",
        transport_type="http",
        config={"url": "https://example.com/mcp"},
    )
    client = client_module.MCPClient(config)

    await client.connect()

    http_mock = state["streamablehttp_client"]
    initialize_mock = state["initialize"]
    assert isinstance(http_mock, AsyncMock)
    assert isinstance(initialize_mock, AsyncMock)
    http_mock.assert_awaited_once_with("https://example.com/mcp")
    initialize_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_disconnect_closes_session(monkeypatch: pytest.MonkeyPatch) -> None:
    state = _install_fake_mcp(monkeypatch)
    components_module = importlib.import_module("ecs_agent.mcp.components")
    client_module = _import_mcp_client_module()

    client = client_module.MCPClient(
        components_module.MCPConfigComponent(
            server_name="filesystem",
            transport_type="stdio",
            config={"command": "uvx", "args": ["mcp-server"]},
        )
    )
    await client.connect()
    await client.disconnect()

    close_mock = state["close"]
    assert isinstance(close_mock, AsyncMock)
    close_mock.assert_awaited_once()
    assert client.is_connected is False


@pytest.mark.asyncio
async def test_list_tools_returns_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    state = _install_fake_mcp(monkeypatch)
    components_module = importlib.import_module("ecs_agent.mcp.components")
    client_module = _import_mcp_client_module()

    tool = types.SimpleNamespace(
        name="read_file",
        description="Read file contents",
        inputSchema={"type": "object", "properties": {"path": {"type": "string"}}},
    )
    list_tools_mock = state["list_tools"]
    assert isinstance(list_tools_mock, AsyncMock)
    list_tools_mock.return_value = types.SimpleNamespace(tools=[tool])

    client = client_module.MCPClient(
        components_module.MCPConfigComponent(
            server_name="filesystem",
            transport_type="stdio",
            config={"command": "uvx", "args": ["mcp-server"]},
        )
    )
    await client.connect()

    tools = await client.list_tools()

    assert tools == [
        {
            "name": "read_file",
            "description": "Read file contents",
            "inputSchema": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
            },
        }
    ]


@pytest.mark.asyncio
async def test_call_tool_dispatches(monkeypatch: pytest.MonkeyPatch) -> None:
    state = _install_fake_mcp(monkeypatch)
    components_module = importlib.import_module("ecs_agent.mcp.components")
    client_module = _import_mcp_client_module()

    list_tools_mock = state["list_tools"]
    assert isinstance(list_tools_mock, AsyncMock)
    list_tools_mock.return_value = types.SimpleNamespace(
        tools=[types.SimpleNamespace(name="read_file", description="", inputSchema={})]
    )

    call_tool_mock = state["call_tool"]
    assert isinstance(call_tool_mock, AsyncMock)
    call_tool_mock.return_value = types.SimpleNamespace(
        content=[types.SimpleNamespace(text="file contents")]
    )

    client = client_module.MCPClient(
        components_module.MCPConfigComponent(
            server_name="filesystem",
            transport_type="stdio",
            config={"command": "uvx", "args": ["mcp-server"]},
        )
    )
    await client.connect()
    await client.list_tools()

    result = await client.call_tool("read_file", {"path": "README.md"})

    call_tool_mock.assert_awaited_once_with(
        "read_file", arguments={"path": "README.md"}
    )
    assert result == "file contents"


@pytest.mark.asyncio
async def test_call_tool_disconnected_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_mcp(monkeypatch)
    components_module = importlib.import_module("ecs_agent.mcp.components")
    client_module = _import_mcp_client_module()

    client = client_module.MCPClient(
        components_module.MCPConfigComponent(
            server_name="filesystem",
            transport_type="stdio",
            config={"command": "uvx", "args": ["mcp-server"]},
        )
    )

    with pytest.raises(RuntimeError, match="Not connected"):
        await client.call_tool("read_file", {"path": "README.md"})


@pytest.mark.asyncio
async def test_unknown_tool_name_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    state = _install_fake_mcp(monkeypatch)
    components_module = importlib.import_module("ecs_agent.mcp.components")
    client_module = _import_mcp_client_module()

    list_tools_mock = state["list_tools"]
    assert isinstance(list_tools_mock, AsyncMock)
    list_tools_mock.return_value = types.SimpleNamespace(
        tools=[types.SimpleNamespace(name="read_file", description="", inputSchema={})]
    )

    client = client_module.MCPClient(
        components_module.MCPConfigComponent(
            server_name="filesystem",
            transport_type="stdio",
            config={"command": "uvx", "args": ["mcp-server"]},
        )
    )
    await client.connect()
    await client.list_tools()

    with pytest.raises(ValueError, match="Unknown tool"):
        await client.call_tool("write_file", {"path": "README.md"})


@pytest.mark.asyncio
async def test_connect_failure_raises_clear_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _install_fake_mcp(monkeypatch)
    components_module = importlib.import_module("ecs_agent.mcp.components")
    client_module = _import_mcp_client_module()

    initialize_mock = state["initialize"]
    assert isinstance(initialize_mock, AsyncMock)
    initialize_mock.side_effect = RuntimeError("boom")

    client = client_module.MCPClient(
        components_module.MCPConfigComponent(
            server_name="filesystem",
            transport_type="stdio",
            config={"command": "uvx", "args": ["mcp-server"]},
        )
    )

    with pytest.raises(RuntimeError, match="Failed to connect"):
        await client.connect()


def test_import_without_mcp(monkeypatch: pytest.MonkeyPatch) -> None:
    _reset_mcp_modules()
    original_import = builtins.__import__

    def fake_import(
        name: str,
        globals_dict: dict[str, object] | None = None,
        locals_dict: dict[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        _ = globals_dict, locals_dict, fromlist, level
        if name == "mcp" or name.startswith("mcp."):
            raise ImportError("No module named 'mcp'")
        return original_import(name, globals_dict, locals_dict, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match="MCP support requires the 'mcp' package"):
        importlib.import_module("ecs_agent.mcp")


def test_mcp_components_dataclasses(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_mcp(monkeypatch)
    components_module = importlib.import_module("ecs_agent.mcp.components")

    config = components_module.MCPConfigComponent(
        server_name="filesystem",
        transport_type="stdio",
        config={"command": "uvx"},
    )
    client = components_module.MCPClientComponent(
        session=object(),
        connected=True,
        cached_tools=[{"name": "read_file"}],
    )

    assert config.server_name == "filesystem"
    assert config.transport_type == "stdio"
    assert config.config == {"command": "uvx"}
    assert client.connected is True
    assert client.cached_tools == [{"name": "read_file"}]


# ---------------------------------------------------------------------------
# Real in-memory MCP server integration (verifies the session lifecycle)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_connect_list_call_roundtrip_against_real_inmemory_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """connect/list_tools/call_tool must work against a real MCP session.

    Regression: MCPClient never entered the ClientSession context, so the
    SDK's receive loop was never started and initialize() awaited a response
    nobody would ever read — connect() hung forever against any real server.
    """
    import contextlib

    # Drop any fake mcp modules a prior test left in sys.modules so this test
    # binds against the real SDK.
    _reset_mcp_modules()

    from mcp.server.fastmcp import FastMCP
    from mcp.shared.memory import create_client_server_memory_streams

    MCPClient = _import_mcp_client_module().MCPClient
    MCPConfigComponent = importlib.import_module(
        "ecs_agent.mcp.components"
    ).MCPConfigComponent

    server = FastMCP("itest")

    @server.tool()
    def echo(text: str) -> str:
        return f"echo:{text}"

    async with create_client_server_memory_streams() as (
        client_streams,
        server_streams,
    ):
        server_read, server_write = server_streams
        server_task = asyncio.create_task(
            server._mcp_server.run(
                server_read,
                server_write,
                server._mcp_server.create_initialization_options(),
                raise_exceptions=True,
            )
        )

        config = MCPConfigComponent(
            server_name="itest",
            transport_type="stdio",
            config={"command": "unused"},
        )
        client = MCPClient(config)

        async def fake_transport_streams(transport_exit_stack: object) -> object:
            del transport_exit_stack
            return client_streams

        monkeypatch.setattr(
            client, "_create_transport_streams", fake_transport_streams
        )

        try:
            await asyncio.wait_for(client.connect(), timeout=5.0)
            assert client.is_connected

            tools = await asyncio.wait_for(client.list_tools(), timeout=5.0)
            assert any(tool["name"] == "echo" for tool in tools)

            result = await asyncio.wait_for(
                client.call_tool("echo", {"text": "hi"}), timeout=5.0
            )
            assert result == "echo:hi"

            await asyncio.wait_for(client.disconnect(), timeout=5.0)
            assert not client.is_connected
        finally:
            server_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await server_task
