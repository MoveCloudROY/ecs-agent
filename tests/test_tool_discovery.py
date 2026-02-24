from __future__ import annotations

import asyncio
import types
from typing import Any

import pytest

from ecs_agent.tools import scan_module, tool
from ecs_agent.tools.discovery import _TOOL_REGISTRY
from ecs_agent.types import ToolSchema


@pytest.mark.asyncio
async def test_tool_decorator_on_async_function_builds_schema_from_type_hints() -> None:
    @tool(name="greet", description="Say hello")
    async def greet(name: str, times: int = 1) -> str:
        return f"Hello {name}" * times

    schema = getattr(greet, "_tool_schema")
    handler = getattr(greet, "_tool_handler")

    assert isinstance(schema, ToolSchema)
    assert schema.name == "greet"
    assert schema.description == "Say hello"
    assert schema.parameters == {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "times": {"type": "integer"},
        },
        "required": ["name"],
    }
    assert await handler(name="Ada", times=2) == "Hello AdaHello Ada"


@pytest.mark.asyncio
async def test_tool_decorator_on_sync_function_wraps_handler_with_executor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeLoop:
        def __init__(self) -> None:
            self.called = False
            self.executor: Any = "sentinel"

        def run_in_executor(self, executor: Any, fn: Any) -> asyncio.Future[str]:
            self.called = True
            self.executor = executor
            future: asyncio.Future[str] = asyncio.get_running_loop().create_future()
            future.set_result(fn())
            return future

    fake_loop = FakeLoop()
    monkeypatch.setattr(asyncio, "get_event_loop", lambda: fake_loop)

    @tool(name="upcase", description="Uppercase text")
    def upcase(text: str) -> str:
        return text.upper()

    handler = getattr(upcase, "_tool_handler")
    result = await handler(text="hello")

    assert result == "HELLO"
    assert fake_loop.called is True
    assert fake_loop.executor is None


def test_scan_module_returns_name_to_schema_and_handler_pairs() -> None:
    module = types.ModuleType("mod")

    @tool(name="fn_a", description="A")
    async def fn_a() -> str:
        return "a"

    @tool(name="fn_b", description="B")
    async def fn_b(x: str) -> str:
        return x

    module.fn_a = fn_a
    module.fn_b = fn_b

    result = scan_module(module)

    assert set(result.keys()) == {"fn_a", "fn_b"}
    assert isinstance(result["fn_a"][0], ToolSchema)
    assert callable(result["fn_a"][1])
    assert isinstance(result["fn_b"][0], ToolSchema)
    assert callable(result["fn_b"][1])


def test_scan_module_raises_value_error_on_duplicate_tool_names() -> None:
    module = types.ModuleType("dup_mod")

    @tool(name="dup", description="First")
    async def first() -> str:
        return "first"

    @tool(name="dup", description="Second")
    async def second() -> str:
        return "second"

    module.first = first
    module.second = second

    with pytest.raises(ValueError, match="Duplicate tool name: dup"):
        scan_module(module)


def test_tool_without_arguments_uses_function_name_and_docstring() -> None:
    @tool()
    async def summarize(topic: str) -> str:
        """Summarize a topic."""
        return topic

    schema = getattr(summarize, "_tool_schema")
    assert schema.name == "summarize"
    assert schema.description == "Summarize a topic."


def test_parameter_type_mapping_str_int_float_bool_and_unknown() -> None:
    class CustomType:
        pass

    @tool(name="types", description="Map types")
    async def typed(
        text: str,
        count: int,
        ratio: float,
        enabled: bool,
        custom: CustomType,
    ) -> str:
        return text

    properties = getattr(typed, "_tool_schema").parameters["properties"]
    assert properties["text"]["type"] == "string"
    assert properties["count"]["type"] == "integer"
    assert properties["ratio"]["type"] == "number"
    assert properties["enabled"]["type"] == "boolean"
    assert properties["custom"]["type"] == "string"


def test_required_fields_only_include_parameters_without_defaults() -> None:
    @tool(name="requireds", description="Required fields")
    async def mixed(
        required_text: str, optional_count: int = 1, optional_flag: bool = False
    ) -> str:
        return required_text

    required = getattr(mixed, "_tool_schema").parameters["required"]
    assert required == ["required_text"]


@pytest.mark.asyncio
async def test_decorator_returns_original_function_object() -> None:
    async def fn(value: str) -> str:
        return value

    decorated = tool(name="orig", description="Original function")(fn)

    assert decorated is fn
    handler = getattr(fn, "_tool_handler")
    assert await handler(value="ok") == "ok"


def test_decorated_functions_are_registered_in_module_registry() -> None:
    _TOOL_REGISTRY.clear()

    @tool(name="registry_test", description="Registry")
    async def registry_tool() -> str:
        return "ok"

    schema = getattr(registry_tool, "_tool_schema")
    assert "registry_test" in _TOOL_REGISTRY
    assert _TOOL_REGISTRY["registry_test"] == schema
