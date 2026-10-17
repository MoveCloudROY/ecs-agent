"""Responses output is a replayable protocol unit through real SDK transport."""

import json

import httpx
import pytest

from ecs_agent.components import (
    CheckpointComponent, CompactionConfigComponent, ConversationComponent,
    PendingToolCallsComponent, StreamingComponent, TerminalComponent, ToolRegistryComponent,
)
from ecs_agent.prompts.message_assembly import _drop_oldest_tool_span
from ecs_agent.serialization import WorldSerializer
from ecs_agent.systems.checkpoint import CheckpointSystem
from ecs_agent.systems.compaction import CompactionSystem
from ecs_agent.systems.reasoning import ReasoningSystem
from ecs_agent.systems.tool_execution import ToolExecutionSystem
from ecs_agent.types import Message, ToolCall, ToolSchema
from tests.test_message_flow_contracts import reply, sdk_world


OUTPUT = [
    {"type": "reasoning", "id": "rs_original", "summary": [], "encrypted_content": "opaque-state"},
    {"type": "message", "id": "msg_comment", "role": "assistant", "status": "completed",
     "phase": "commentary", "content": [{"type": "output_text", "text": "checking"}]},
    {"type": "function_call", "id": "fc_original", "call_id": "call_original",
     "name": "lookup", "arguments": '{ "query" : "value" }', "status": "completed"},
]


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("terminal_only", [False, True])
async def test_raw_output_survives_tool_loop_and_serialization(stream: bool, terminal_only: bool) -> None:
    world, entity, model, transport = await sdk_world()
    body = reply("resp_original", output=OUTPUT).json()
    if stream:
        events = []
        if not terminal_only:
            for index, item in enumerate(OUTPUT):
                events.extend([
                    {"type": "response.output_item.added", "output_index": index, "item": item},
                    {"type": "response.output_item.done", "output_index": index, "item": item},
                ])
        events.append({"type": "response.completed", "response": body})
        first = httpx.Response(200, headers={"content-type": "text/event-stream"}, content="".join(
            f"data: {json.dumps(event)}\n\n" for event in events
        ))
        world.add_component(entity, StreamingComponent(enabled=True))
    else:
        first = httpx.Response(200, json=body)
    transport.responses = [first, reply("resp_final")]
    calls: list[str] = []

    async def lookup(query: str) -> str:
        calls.append(query)
        return "found"

    world.add_component(entity, ToolRegistryComponent(
        tools={"lookup": ToolSchema(name="lookup", description="lookup", parameters={"type": "object"})},
        handlers={"lookup": lookup},
    ))
    try:
        await ReasoningSystem().process(world)
        conv = world.get_component(entity, ConversationComponent)
        assert conv is not None
        message = conv.messages[-1]
        assert message.provider_output is not None
        assert message.provider_output.items == OUTPUT
        assert message.provider_output.complete
        # Use the public world snapshot path, not merely copying the dataclass.
        restored = WorldSerializer.from_dict(WorldSerializer.to_dict(world), providers={"default": model, model.model_id: model}, tool_handlers={"lookup": lookup})
        restored_conv = restored.get_component(entity, ConversationComponent)
        assert restored_conv is not None
        assert restored_conv.messages[-1].provider_output == message.provider_output
        world = restored
        world.remove_component(entity, StreamingComponent)
        await ToolExecutionSystem().process(world)
        await ReasoningSystem().process(world)
        sent = json.loads(transport.requests[-1].content)["input"]
        assert sent[1:4] == OUTPUT
        assert sent[4] == {"type": "function_call_output", "call_id": "call_original", "output": "found"}
        assert calls == ["value"]
    finally:
        await model.aclose()


async def test_pending_raw_tool_turn_survives_compaction_and_checkpoint_undo() -> None:
    world, entity, model, transport = await sdk_world()
    transport.responses = [reply("resp_original", output=OUTPUT), reply("resp_final")]
    called: list[str] = []

    async def lookup(query: str) -> str:
        called.append(query)
        return "found"

    world.add_component(entity, ToolRegistryComponent(
        tools={"lookup": ToolSchema(name="lookup", description="lookup", parameters={"type": "object"})},
        handlers={"lookup": lookup},
    ))
    try:
        await ReasoningSystem().process(world)
        world.add_component(entity, CompactionConfigComponent(threshold_tokens=1))
        conv = world.get_component(entity, ConversationComponent)
        before = WorldSerializer._message_to_dict(conv.messages[-1])
        await CompactionSystem().process(world)
        assert len(transport.requests) == 1
        assert WorldSerializer._message_to_dict(conv.messages[-1]) == before
        assert world.has_component(entity, PendingToolCallsComponent)
        world.add_component(entity, CheckpointComponent())
        await CheckpointSystem().process(world)
        conv.messages.clear()
        world.remove_component(entity, PendingToolCallsComponent)
        await CheckpointSystem.undo(world, providers={model.model_id: model}, tool_handlers={"lookup": lookup})
        await ToolExecutionSystem().process(world)
        await ReasoningSystem().process(world)
        sent = json.loads(transport.requests[-1].content)["input"]
        assert sent[1:4] == OUTPUT
        assert sent[4] == {"type": "function_call_output", "call_id": "call_original", "output": "found"}
        assert called == ["value"]
    finally:
        await model.aclose()


def test_trimming_preserves_partial_multi_tool_span_and_drops_closed_unit() -> None:
    assistant = Message(role="assistant", content="working", tool_calls=[
        ToolCall(id="a", name="lookup", arguments={}),
        ToolCall(id="b", name="lookup", arguments={}),
    ])
    prefix = [Message(role="user", content="task")]
    partial = [*prefix, assistant, Message(role="tool", tool_call_id="a", content="first")]
    assert _drop_oldest_tool_span(partial) == partial
    closed = [*partial, Message(role="tool", tool_call_id="b", content="second")]
    assert _drop_oldest_tool_span(closed) == prefix


async def test_origin_change_projects_visible_history_without_private_state() -> None:
    world, entity, model, transport = await sdk_world()
    transport.responses = [reply("resp_old", output=OUTPUT[:2]), reply("resp_new")]
    try:
        await ReasoningSystem().process(world)
        model._base_url = "https://another.invalid/v1"
        world.remove_component(entity, TerminalComponent)
        conv = world.get_component(entity, ConversationComponent)
        assert conv is not None
        conv.messages.append(Message(role="user", content="new provider"))
        await ReasoningSystem().process(world)
        sent = json.loads(transport.requests[-1].content)
        assert "opaque-state" not in json.dumps(sent)
        assert "checking" in json.dumps(sent)
    finally:
        await model.aclose()


async def test_snapshot_never_deepcopies_bound_handlers_owning_live_sdk_clients() -> None:
    world, entity, model, transport = await sdk_world()
    transport.responses = [reply("resp_warm")]

    class BoundTools:
        """A realistic bound handler owns a reference to a warm model client."""

        def __init__(self) -> None:
            self.client = model

        async def lookup(self) -> str:
            return self.client.model_id

    handler = BoundTools().lookup
    world.add_component(entity, ToolRegistryComponent(
        tools={"lookup": ToolSchema(name="lookup", description="lookup", parameters={"type": "object"})},
        handlers={"lookup": handler},
    ))
    try:
        await ReasoningSystem().process(world)
        data = WorldSerializer.to_dict(world)
        restored = WorldSerializer.from_dict(data, providers={model.model_id: model}, tool_handlers={"lookup": handler})
        registry = restored.get_component(entity, ToolRegistryComponent)
        assert registry is not None and await registry.handlers["lookup"]() == model.model_id
        assert not model._client.is_closed
    finally:
        await model.aclose()
