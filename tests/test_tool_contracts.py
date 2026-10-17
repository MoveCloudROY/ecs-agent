"""Tool identities and complete spans are checked at execution and replay."""

import json

import pytest

from ecs_agent.components import ConversationComponent, PendingToolCallsComponent, TerminalComponent, ToolRegistryComponent
from ecs_agent.core import World
from ecs_agent.providers._message_contract import validate_tool_spans
from ecs_agent.systems.tool_execution import ToolExecutionSystem
from ecs_agent.tools.validation import tool_call_error
from ecs_agent.types import Message, ToolCall, ToolSchema
from tests.test_message_flow_contracts import reply, sdk_world


def call_message(*ids: str) -> Message:
    """Build one complete assistant tool batch."""
    return Message(role="assistant", content="", tool_calls=[
        ToolCall(id=call_id, name="lookup", arguments={}) for call_id in ids
    ])


def result_message(call_id: str) -> Message:
    """Build one tool result."""
    return Message(role="tool", content="result", tool_call_id=call_id)


@pytest.mark.parametrize("messages", [
    [result_message("orphan")],
    [call_message("a"), result_message("unknown")],
    [call_message("a"), result_message("a"), result_message("a")],
    [call_message("a", "b"), result_message("b"), result_message("a")],
    [call_message("a"), Message(role="user", content="next"), result_message("a")],
    [call_message("a", "a"), result_message("a")],
    [call_message("a"), result_message("a"), call_message("a"), result_message("a")],
    [call_message("a")],
])
async def test_corrupt_tool_history_is_rejected_before_http(messages: list[Message]) -> None:
    _, _, model, transport = await sdk_world()
    try:
        with pytest.raises(ValueError):
            await model.complete(messages)
        assert not transport.requests
    finally:
        await model.aclose()


async def test_explicit_response_reference_allows_remote_tool_results_only_in_prefix() -> None:
    _, _, model, transport = await sdk_world()
    transport.responses = [reply("resp_next")]
    try:
        messages = [result_message("remote_a"), result_message("remote_b"),
                    Message(role="user", content="continue")]
        await model.complete(messages, thread_response_id="resp_stored")
        body = json.loads(transport.requests[0].content)
        assert body["previous_response_id"] == "resp_stored"
        assert body["store"] is False
        assert [item.get("call_id") for item in body["input"][:2]] == ["remote_a", "remote_b"]
        with pytest.raises(ValueError, match="Orphan"):
            validate_tool_spans([Message(role="user", content="new"), result_message("orphan")],
                                allow_external_results=True)
        with pytest.raises(ValueError, match="Duplicate"):
            validate_tool_spans([result_message("a"), result_message("a")], allow_external_results=True)
    finally:
        await model.aclose()


@pytest.mark.parametrize("ids", [("a", "a"), ("", "b")])
async def test_invalid_manual_queue_executes_no_side_effects(ids: tuple[str, str]) -> None:
    world = World()
    entity = world.create_entity()
    executed: list[bool] = []

    async def lookup() -> str:
        executed.append(True)
        return "result"

    world.add_component(entity, ConversationComponent([]))
    world.add_component(entity, ToolRegistryComponent(
        tools={"lookup": ToolSchema(name="lookup", description="lookup", parameters={})},
        handlers={"lookup": lookup},
    ))
    world.add_component(entity, PendingToolCallsComponent(call_message(*ids).tool_calls))
    await ToolExecutionSystem().process(world)
    assert not executed
    assert world.get_component(entity, TerminalComponent).reason == "tool_execution_error"
    assert not world.has_component(entity, PendingToolCallsComponent)


def test_unknown_tool_and_unresolvable_schema_are_not_valid_calls() -> None:
    call = ToolCall(id="a", name="unknown", arguments={})
    assert tool_call_error([call], []) == "Unknown tool 'unknown'"
    call.name = "remote"
    schema = ToolSchema(name="remote", description="remote schema", parameters={
        "$ref": "https://schema.invalid/never-retrieve.json",
    })
    assert tool_call_error([call], [schema]).startswith("Invalid schema")
    schema.parameters = {"type": "not-a-schema-type"}
    assert tool_call_error([call], [schema]).startswith("Invalid schema")
