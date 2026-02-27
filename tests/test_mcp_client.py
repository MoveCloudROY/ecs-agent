from __future__ import annotations

import builtins
import importlib
import sys
import types
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

    mcp_module = types.ModuleType("mcp")
    mcp_module.ClientSession = FakeSession  # type: ignore[attr-defined]

    client_module = types.ModuleType("mcp.client")
    stdio_module = types.ModuleType("mcp.client.stdio")
    sse_module = types.ModuleType("mcp.client.sse")
    streamable_http_module = types.ModuleType("mcp.client.streamable_http")

    async def stdio_client(config: dict[str, object]) -> tuple[object, object]:
        mock = state["stdio_client"]
        assert isinstance(mock, AsyncMock)
        await mock(config)
        return object(), object()

    async def sse_client(url: str) -> tuple[object, object]:
        mock = state["sse_client"]
        assert isinstance(mock, AsyncMock)
        await mock(url)
        return object(), object()

    async def streamablehttp_client(url: str) -> tuple[object, object]:
        mock = state["streamablehttp_client"]
        assert isinstance(mock, AsyncMock)
        await mock(url)
        return object(), object()

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
    stdio_mock.assert_awaited_once_with(config.config)
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
