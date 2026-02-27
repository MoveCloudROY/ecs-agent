from __future__ import annotations

from dataclasses import dataclass
import importlib
import sys
import types
from typing import Any

import pytest

from ecs_agent.components import ToolRegistryComponent
from ecs_agent.core import World
from ecs_agent.skills.manager import SkillManager
from ecs_agent.types import ToolSchema


def _reset_mcp_modules() -> None:
    for module_name in [
        "mcp",
        "mcp.client",
        "mcp.client.stdio",
        "mcp.client.sse",
        "mcp.client.streamable_http",
        "ecs_agent.mcp",
        "ecs_agent.mcp.client",
        "ecs_agent.mcp.adapter",
    ]:
        sys.modules.pop(module_name, None)


@pytest.fixture(autouse=True)
def _install_fake_mcp(monkeypatch: pytest.MonkeyPatch) -> None:
    _reset_mcp_modules()
    mcp_module = types.ModuleType("mcp")
    mcp_module.ClientSession = object  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mcp", mcp_module)

    client_module = types.ModuleType("mcp.client")
    stdio_module = types.ModuleType("mcp.client.stdio")
    sse_module = types.ModuleType("mcp.client.sse")
    streamable_http_module = types.ModuleType("mcp.client.streamable_http")

    async def stdio_client(_: dict[str, object]) -> tuple[object, object]:
        return object(), object()

    async def sse_client(_: str) -> tuple[object, object]:
        return object(), object()

    async def streamablehttp_client(_: str) -> tuple[object, object]:
        return object(), object()

    stdio_module.stdio_client = stdio_client  # type: ignore[attr-defined]
    sse_module.sse_client = sse_client  # type: ignore[attr-defined]
    streamable_http_module.streamablehttp_client = streamablehttp_client  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mcp.client", client_module)
    monkeypatch.setitem(sys.modules, "mcp.client.stdio", stdio_module)
    monkeypatch.setitem(sys.modules, "mcp.client.sse", sse_module)
    monkeypatch.setitem(
        sys.modules, "mcp.client.streamable_http", streamable_http_module
    )


def _import_adapter_module() -> types.ModuleType:
    return importlib.import_module("ecs_agent.mcp.adapter")


@dataclass(slots=True)
class FakeMCPClient:
    server_name: str
    tools_payload: list[dict[str, Any]]
    call_result: str = "ok"
    fail_connect: bool = False
    connected: bool = False
    connect_calls: int = 0
    disconnect_calls: int = 0
    list_tools_calls: int = 0
    call_tool_calls: list[tuple[str, dict[str, Any]]] | None = None

    def __post_init__(self) -> None:
        self.call_tool_calls = []

    @property
    def is_connected(self) -> bool:
        return self.connected

    async def connect(self) -> None:
        self.connect_calls += 1
        if self.fail_connect:
            raise RuntimeError("boom")
        self.connected = True

    async def disconnect(self) -> None:
        self.disconnect_calls += 1
        self.connected = False

    async def list_tools(self) -> list[dict[str, Any]]:
        self.list_tools_calls += 1
        return list(self.tools_payload)

    async def call_tool(self, name: str, args: dict[str, Any]) -> str:
        assert self.call_tool_calls is not None
        self.call_tool_calls.append((name, args))
        return self.call_result


def _tool_payload(name: str, description: str = "desc") -> dict[str, Any]:
    return {
        "name": name,
        "description": description,
        "inputSchema": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Path to file"}},
            "required": ["path"],
        },
    }


def test_mcp_tool_to_ecs_tool_maps_schema() -> None:
    adapter_module = _import_adapter_module()
    client = FakeMCPClient(server_name="fs", tools_payload=[])
    schema, _ = adapter_module.mcp_tool_to_ecs_tool(client, "fs", _tool_payload("read"))
    assert schema == ToolSchema(
        name="fs/read",
        description="desc",
        parameters={
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Path to file"}},
            "required": ["path"],
        },
    )


@pytest.mark.asyncio
async def test_default_converter_handler_calls_mcp_client_call_tool() -> None:
    adapter_module = _import_adapter_module()
    client = FakeMCPClient(server_name="fs", tools_payload=[], call_result="payload")
    _, handler = adapter_module.mcp_tool_to_ecs_tool(
        client, "fs", _tool_payload("read")
    )
    result = await handler(path="README.md")
    assert result == "payload"
    assert client.call_tool_calls == [("read", {"path": "README.md"})]


def test_adapter_tool_namespacing() -> None:
    adapter_module = _import_adapter_module()
    client = FakeMCPClient(
        server_name="fs",
        tools_payload=[
            _tool_payload("read"),
            _tool_payload("write"),
            _tool_payload("edit"),
        ],
    )
    adapter = adapter_module.MCPSkillAdapter(client, "fs")
    bundle = adapter.tools()
    assert set(bundle) == {"fs/read", "fs/write", "fs/edit"}


def test_adapter_tools_multiple_tools_from_same_server() -> None:
    adapter_module = _import_adapter_module()
    client = FakeMCPClient(
        server_name="shell",
        tools_payload=[
            _tool_payload("bash", "Run shell"),
            _tool_payload("pwd", "Print cwd"),
        ],
    )
    adapter = adapter_module.MCPSkillAdapter(client, "shell")
    bundle = adapter.tools()
    assert bundle["shell/bash"][0].description == "Run shell"
    assert bundle["shell/pwd"][0].description == "Print cwd"


def test_adapter_connects_once_when_discovering_tools() -> None:
    adapter_module = _import_adapter_module()
    client = FakeMCPClient(server_name="fs", tools_payload=[_tool_payload("read")])
    adapter = adapter_module.MCPSkillAdapter(client, "fs")
    adapter.tools()
    adapter.tools()
    assert client.connect_calls == 1
    assert client.list_tools_calls == 1


def test_custom_converter_filters_tools() -> None:
    adapter_module = _import_adapter_module()

    def converter(tool: dict[str, Any]) -> tuple[ToolSchema, Any] | None:
        if tool["name"] == "write":
            return None
        schema = ToolSchema(
            name=f"fs/{tool['name']}", description="ok", parameters={"type": "object"}
        )

        async def handler(**_: Any) -> str:
            return "ok"

        return schema, handler

    client = FakeMCPClient(
        server_name="fs", tools_payload=[_tool_payload("read"), _tool_payload("write")]
    )
    adapter = adapter_module.MCPSkillAdapter(client, "fs", converter=converter)
    assert set(adapter.tools()) == {"fs/read"}


def test_custom_converter_transforms_tool_schema_and_name() -> None:
    adapter_module = _import_adapter_module()

    def converter(tool: dict[str, Any]) -> tuple[ToolSchema, Any] | None:
        schema = ToolSchema(
            name=f"custom/{tool['name']}_v2",
            description="custom",
            parameters={"type": "object", "properties": {"value": {"type": "integer"}}},
        )

        async def handler(**_: Any) -> str:
            return "transformed"

        return schema, handler

    client = FakeMCPClient(server_name="fs", tools_payload=[_tool_payload("read")])
    adapter = adapter_module.MCPSkillAdapter(client, "fs", converter=converter)
    bundle = adapter.tools()
    assert set(bundle) == {"custom/read_v2"}
    assert bundle["custom/read_v2"][0].description == "custom"


def test_system_prompt_tier1_is_compact_and_lists_tools() -> None:
    adapter_module = _import_adapter_module()
    client = FakeMCPClient(
        server_name="fs",
        tools_payload=[
            _tool_payload("read"),
            _tool_payload("write"),
            _tool_payload("edit"),
        ],
    )
    adapter = adapter_module.MCPSkillAdapter(client, "fs")
    prompt = adapter.system_prompt()
    assert "Tier 1" in prompt
    assert "fs/read" in prompt
    assert "fs/write" in prompt
    assert "fs/edit" in prompt
    assert len(prompt.split()) <= 200


def test_install_merges_tools_via_skill_manager() -> None:
    adapter_module = _import_adapter_module()
    world = World()
    entity_id = world.create_entity()
    client = FakeMCPClient(
        server_name="fs", tools_payload=[_tool_payload("read"), _tool_payload("write")]
    )
    adapter = adapter_module.MCPSkillAdapter(client, "fs")

    adapter.install(world, entity_id)

    registry = world.get_component(entity_id, ToolRegistryComponent)
    assert registry is not None
    assert "fs/read" in registry.tools
    assert "fs/write" in registry.tools
    assert client.connect_calls == 1
    assert client.list_tools_calls == 1


def test_install_raises_clear_error_when_connect_fails() -> None:
    adapter_module = _import_adapter_module()
    world = World()
    entity_id = world.create_entity()
    client = FakeMCPClient(server_name="fs", tools_payload=[], fail_connect=True)
    adapter = adapter_module.MCPSkillAdapter(client, "fs")

    with pytest.raises(RuntimeError, match="Failed to install MCP skill"):
        adapter.install(world, entity_id)


def test_uninstall_removes_tools_and_disconnects() -> None:
    adapter_module = _import_adapter_module()
    world = World()
    entity_id = world.create_entity()
    client = FakeMCPClient(
        server_name="fs", tools_payload=[_tool_payload("read"), _tool_payload("write")]
    )
    adapter = adapter_module.MCPSkillAdapter(client, "fs")
    adapter.install(world, entity_id)

    adapter.uninstall(world, entity_id)
    registry = world.get_component(entity_id, ToolRegistryComponent)
    assert registry is not None
    assert "fs/read" not in registry.tools
    assert "fs/write" not in registry.tools
    assert client.disconnect_calls == 1


@pytest.mark.asyncio
async def test_load_skill_details_meta_tool_registration() -> None:
    adapter_module = _import_adapter_module()
    world = World()
    entity_id = world.create_entity()
    client = FakeMCPClient(server_name="fs", tools_payload=[_tool_payload("read")])
    adapter = adapter_module.MCPSkillAdapter(client, "fs")
    adapter.install(world, entity_id)

    registry = world.get_component(entity_id, ToolRegistryComponent)
    assert registry is not None
    assert "load_skill_details" in registry.tools
    details = await registry.handlers["load_skill_details"](skill_name="mcp-fs")
    assert "fs/read" in details
    assert "Path to file" in details


@pytest.mark.asyncio
async def test_progressive_disclosure() -> None:
    adapter_module = _import_adapter_module()
    world = World()
    entity_id = world.create_entity()
    manager = SkillManager()
    client = FakeMCPClient(server_name="fs", tools_payload=[_tool_payload("read")])
    adapter = adapter_module.MCPSkillAdapter(client, "fs")

    manager.install(world, entity_id, adapter)
    tier1 = adapter.system_prompt()
    assert len(tier1.split()) <= 200

    registry = world.get_component(entity_id, ToolRegistryComponent)
    assert registry is not None
    tier2 = await registry.handlers["load_skill_details"](skill_name="mcp-fs")
    assert "fs/read" in tier2
    assert "required" in tier2
    assert len(tier2) > len(tier1)


@pytest.mark.asyncio
async def test_meta_tool_handles_unknown_skill_name() -> None:
    adapter_module = _import_adapter_module()
    world = World()
    entity_id = world.create_entity()
    client = FakeMCPClient(server_name="fs", tools_payload=[_tool_payload("read")])
    adapter = adapter_module.MCPSkillAdapter(client, "fs")
    adapter.install(world, entity_id)

    registry = world.get_component(entity_id, ToolRegistryComponent)
    assert registry is not None
    result = await registry.handlers["load_skill_details"](skill_name="missing")
    assert result == "Skill 'missing' is not installed."


def test_tier2_details_contain_full_schema_json() -> None:
    adapter_module = _import_adapter_module()
    world = World()
    entity_id = world.create_entity()
    client = FakeMCPClient(server_name="fs", tools_payload=[_tool_payload("read")])
    adapter = adapter_module.MCPSkillAdapter(client, "fs")
    adapter.install(world, entity_id)

    details = SkillManager().format_skill_details(world, entity_id, "mcp-fs")
    assert details is not None
    assert '"path"' in details
    assert '"type": "string"' in details
