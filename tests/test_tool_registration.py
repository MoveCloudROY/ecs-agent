"""Tool authoring contracts from annotated Python input to real execution."""

from __future__ import annotations

import inspect
import types
from typing import Annotated, Any, Dict, List, Literal

import pytest
from jsonschema import Draft202012Validator

from ecs_agent.components import (
    ConversationComponent, LLMComponent, PermissionComponent, ToolApprovalComponent,
    ToolRegistryComponent,
)
from ecs_agent.core import Runner, World
from ecs_agent.providers.fake_model import FakeModel
from ecs_agent.skills import SkillManager
from ecs_agent.systems.reasoning import ReasoningSystem
from ecs_agent.systems.tool_execution import ToolExecutionSystem
from ecs_agent.tools import ToolDefinition, get_tool_definition, register_tools, scan_module, tool
from ecs_agent.tools.builtins import BuiltinToolsSkill
from ecs_agent.types import ApprovalPolicy, CompletionResult, Message, ToolCall, ToolSchema


@tool()
def nested(
    values: list[dict[str, int | None]],
    mode: Annotated[Literal["fast", "slow"], "Processing mode"] = "fast",
) -> str:
    return f"{mode}: {values}"


def test_nested_schema_validates_values_and_preserves_literal_constraints() -> None:
    schema = get_tool_definition(nested).schema.parameters
    validator = Draft202012Validator(schema)
    validator.validate({"values": [{"a": 1, "b": None}]})
    assert schema["properties"]["mode"] == {
        "type": "string", "enum": ["fast", "slow"], "description": "Processing mode",
    }
    for invalid in (
        {"values": [{"a": "1"}]}, {"values": "[]"},
        {"values": [], "mode": "unknown"}, {"values": [], "extra": 1}, {},
    ):
        assert not validator.is_valid(invalid)


def test_union_keeps_each_branchs_array_and_enum_constraints() -> None:
    @tool()
    def choice(value: list[int] | Literal["all"] | None) -> str:
        return str(value)

    validator = Draft202012Validator(get_tool_definition(choice).schema.parameters)
    for value in ([1, 2], "all", None):
        validator.validate({"value": value})
    for value in (["1"], "other", 1):
        assert not validator.is_valid({"value": value})
    assert not validator.is_valid({})


async def test_nullable_required_and_defaults_reach_the_original_function() -> None:
    @tool()
    def nullable(required: str | None, optional: str | None = "default") -> str:
        return f"{required}/{optional}"

    definition = get_tool_definition(nullable)
    validator = Draft202012Validator(definition.schema.parameters)
    validator.validate({"required": None})
    assert not validator.is_valid({})
    assert await definition.handler(required=None) == "None/default"
    assert await definition.handler(required="value", optional=None) == "value/None"
    assert nullable("direct") == "direct/default"
    assert inspect.signature(nullable) == inspect.signature(definition.handler)


@pytest.mark.parametrize("annotation", [Any, object, list, List, Dict, dict[int, str], tuple[int, str], "UnresolvedInput"])
def test_unsupported_annotations_report_the_parameter(annotation: Any) -> None:
    def unsupported(value: Any) -> str:
        return str(value)

    unsupported.__annotations__["value"] = annotation
    with pytest.raises(ValueError, match="Tool 'unsupported' parameter 'value'"):
        tool()(unsupported)


def test_unannotated_and_non_keyword_signatures_fail_early() -> None:
    def unannotated(value):
        return str(value)

    def positional(value: int, /) -> str:
        return str(value)

    def variadic(*values: int) -> str:
        return str(values)

    def keywords(**values: int) -> str:
        return str(values)

    for function in (unannotated, positional, variadic, keywords):
        with pytest.raises(ValueError, match="parameter"):
            tool()(function)


def test_return_annotation_does_not_change_input_contract() -> None:
    def function(value: int) -> str:
        return str(value)

    function.__annotations__["return"] = "UnresolvedReturn"
    decorated = tool()(function)
    assert decorated is function
    assert get_tool_definition(decorated).schema.parameters["properties"] == {"value": {"type": "integer"}}


async def test_registration_merges_skill_tools_and_accepts_module_discovery(tmp_path) -> None:
    world = World()
    entity = world.create_entity()
    SkillManager().install(world, entity, BuiltinToolsSkill().bind_workspace(str(tmp_path)))
    before = world.get_component(entity, ToolRegistryComponent)
    assert before is not None
    saved = dict(before.handlers)
    module = types.ModuleType("test_tools")
    module.nested = nested
    registry = register_tools(world, entity, scan_module(module))
    assert registry is before
    assert set(registry.tools) == set(registry.handlers) == set(saved) | {"nested"}
    assert all(registry.handlers[name] is handler for name, handler in saved.items())
    assert await registry.handlers["nested"](values=[]) == "fast: []"


def test_duplicate_batch_is_atomic_for_new_and_existing_registries() -> None:
    world = World()
    entity = world.create_entity()
    with pytest.raises(ValueError, match="Duplicate tool name"):
        register_tools(world, entity, [nested, nested])
    assert world.get_component(entity, ToolRegistryComponent) is None
    registry = register_tools(world, entity, [nested])
    saved_schema = registry.tools["nested"]

    @tool()
    async def added() -> str:
        return "added"

    with pytest.raises(ValueError, match="Duplicate tool name"):
        register_tools(world, entity, [added, nested])
    assert set(registry.tools) == set(registry.handlers) == {"nested"}
    assert registry.tools["nested"] is saved_schema


def test_invalid_schema_or_binding_leaves_registry_untouched() -> None:
    world = World()
    entity = world.create_entity()

    async def adapted(value: int) -> str:
        return str(value)

    invalid = [
        {"type": "invalid"},
        {"type": "object", "properties": {"wrong": {"type": "integer"}}, "required": ["wrong"]},
        {"type": "object", "properties": {"value": {"type": "integer"}}},
    ]
    for parameters in invalid:
        definition = ToolDefinition(ToolSchema("adapted", "", parameters), adapted)
        with pytest.raises(ValueError, match="adapted"):
            register_tools(world, entity, [nested, definition])
        assert world.get_component(entity, ToolRegistryComponent) is None


async def test_explicit_adapter_receives_json_and_converts_it() -> None:
    class Quantity:
        def __init__(self, value: int) -> None:
            self.value = value

    async def adapter(value: int) -> str:
        return str(Quantity(value).value * 2)

    world = World()
    entity = world.create_entity()
    schema = ToolSchema("double", "Double a quantity", {
        "type": "object", "properties": {"value": {"type": "integer"}}, "required": ["value"],
    })
    registry = register_tools(world, entity, [ToolDefinition(schema, adapter)])
    schema.parameters["properties"].clear()
    assert "value" in registry.tools["double"].parameters["properties"]
    assert await registry.handlers["double"](value=3) == "6"


@pytest.mark.parametrize("policy", ["allow", "deny_permission", "deny_approval", "invalid_arguments"])
async def test_model_generated_calls_use_schema_registration_and_authorization(policy: str) -> None:
    executed: list[list[int]] = []

    @tool()
    async def total(values: list[int], mode: Literal["sum"] = "sum") -> str:
        executed.append(values)
        return str(sum(values))

    world = World()
    entity = world.create_entity()
    arguments = {"values": [1, 2]} if policy != "invalid_arguments" else {"values": ["wrong"]}
    model = FakeModel([
        CompletionResult(message=Message(role="assistant", content="", tool_calls=[ToolCall("call-1", "total", arguments)])),
        CompletionResult(message=Message(role="assistant", content="finished")),
    ])
    world.add_component(entity, LLMComponent(model=model))
    world.add_component(entity, ConversationComponent([Message(role="user", content="Calculate the total")]))
    register_tools(world, entity, [total])
    if policy == "deny_permission":
        world.add_component(entity, PermissionComponent(allowed_tools=[]))
    if policy == "deny_approval":
        world.add_component(entity, ToolApprovalComponent(policy=ApprovalPolicy.ALWAYS_DENY))
    world.register_system(ReasoningSystem(), priority=0)
    world.register_system(ToolExecutionSystem(), priority=5)
    result = await Runner().run(world, max_ticks=3, target_entity=entity)
    assert executed == ([[1, 2]] if policy == "allow" else [])
    if policy == "allow":
        assert result.success
        conversation = world.get_component(entity, ConversationComponent)
        assert conversation is not None
        outputs = [message.content for message in conversation.messages if message.role == "tool"]
        assert outputs == ["3"]
