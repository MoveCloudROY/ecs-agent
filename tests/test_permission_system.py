import pytest

from ecs_agent.components import (
    ConversationComponent,
    PendingToolCallsComponent,
    PermissionComponent,
)
from ecs_agent.core import World
from ecs_agent.systems.permission import PermissionSystem
from ecs_agent.types import Message, ToolCall, ToolDeniedEvent


def _call(call_id: str, name: str) -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments={})


@pytest.mark.asyncio
async def test_denied_tool_is_removed_adds_conversation_message_and_publishes_event() -> (
    None
):
    world = World()
    entity_id = world.create_entity()
    world.add_component(entity_id, ConversationComponent(messages=[]))
    world.add_component(
        entity_id,
        PendingToolCallsComponent(
            tool_calls=[_call("c1", "bash"), _call("c2", "read")]
        ),
    )
    world.add_component(entity_id, PermissionComponent(denied_tools=["bash"]))

    seen: list[ToolDeniedEvent] = []

    async def on_denied(event: ToolDeniedEvent) -> None:
        seen.append(event)

    world.event_bus.subscribe(ToolDeniedEvent, on_denied)

    await PermissionSystem().process(world)

    pending = world.get_component(entity_id, PendingToolCallsComponent)
    conversation = world.get_component(entity_id, ConversationComponent)
    assert pending is not None
    assert [call.id for call in pending.tool_calls] == ["c2"]
    assert conversation is not None
    assert conversation.messages[-1].role == "tool"
    assert conversation.messages[-1].tool_call_id == "c1"
    assert "denied" in conversation.messages[-1].content.lower()
    assert len(seen) == 1
    assert seen[0].entity_id == entity_id
    assert seen[0].tool_call_id == "c1"


@pytest.mark.asyncio
async def test_allowed_tools_list_allows_only_explicit_tools() -> None:
    world = World()
    entity_id = world.create_entity()
    world.add_component(entity_id, ConversationComponent(messages=[]))
    world.add_component(
        entity_id,
        PendingToolCallsComponent(
            tool_calls=[_call("c1", "bash"), _call("c2", "read")]
        ),
    )
    world.add_component(entity_id, PermissionComponent(allowed_tools=["bash"]))

    await PermissionSystem().process(world)

    pending = world.get_component(entity_id, PendingToolCallsComponent)
    assert pending is not None
    assert [call.id for call in pending.tool_calls] == ["c1"]


@pytest.mark.asyncio
async def test_allowed_tools_empty_allows_all_except_denied() -> None:
    world = World()
    entity_id = world.create_entity()
    world.add_component(entity_id, ConversationComponent(messages=[]))
    world.add_component(
        entity_id,
        PendingToolCallsComponent(
            tool_calls=[_call("c1", "bash"), _call("c2", "read")]
        ),
    )
    world.add_component(entity_id, PermissionComponent())

    await PermissionSystem().process(world)

    pending = world.get_component(entity_id, PendingToolCallsComponent)
    assert pending is not None
    assert [call.id for call in pending.tool_calls] == ["c1", "c2"]


@pytest.mark.asyncio
async def test_denied_list_has_priority_over_allowed_list() -> None:
    world = World()
    entity_id = world.create_entity()
    world.add_component(entity_id, ConversationComponent(messages=[]))
    world.add_component(
        entity_id, PendingToolCallsComponent(tool_calls=[_call("c1", "bash")])
    )
    world.add_component(
        entity_id,
        PermissionComponent(allowed_tools=["bash"], denied_tools=["bash"]),
    )

    await PermissionSystem().process(world)

    assert world.get_component(entity_id, PendingToolCallsComponent) is None


@pytest.mark.asyncio
async def test_all_denied_calls_remove_pending_component() -> None:
    world = World()
    entity_id = world.create_entity()
    world.add_component(entity_id, ConversationComponent(messages=[]))
    world.add_component(
        entity_id, PendingToolCallsComponent(tool_calls=[_call("c1", "bash")])
    )
    world.add_component(entity_id, PermissionComponent(denied_tools=["bash"]))

    await PermissionSystem().process(world)

    assert world.get_component(entity_id, PendingToolCallsComponent) is None


@pytest.mark.asyncio
async def test_entity_without_permission_component_is_skipped() -> None:
    world = World()
    entity_id = world.create_entity()
    world.add_component(
        entity_id, PendingToolCallsComponent(tool_calls=[_call("c1", "bash")])
    )

    await PermissionSystem().process(world)

    pending = world.get_component(entity_id, PendingToolCallsComponent)
    assert pending is not None
    assert [call.id for call in pending.tool_calls] == ["c1"]


@pytest.mark.asyncio
async def test_denied_message_is_added_when_conversation_exists() -> None:
    world = World()
    entity_id = world.create_entity()
    world.add_component(
        entity_id,
        ConversationComponent(messages=[Message(role="user", content="hi")]),
    )
    world.add_component(
        entity_id, PendingToolCallsComponent(tool_calls=[_call("c1", "bash")])
    )
    world.add_component(entity_id, PermissionComponent(denied_tools=["bash"]))

    await PermissionSystem().process(world)

    conversation = world.get_component(entity_id, ConversationComponent)
    assert conversation is not None
    assert len(conversation.messages) == 2
    assert conversation.messages[-1].tool_call_id == "c1"
    assert "denied" in conversation.messages[-1].content.lower()


@pytest.mark.asyncio
async def test_multiple_calls_mixed_allow_and_deny() -> None:
    world = World()
    entity_id = world.create_entity()
    world.add_component(entity_id, ConversationComponent(messages=[]))
    world.add_component(
        entity_id,
        PendingToolCallsComponent(
            tool_calls=[
                _call("c1", "bash"),
                _call("c2", "read"),
                _call("c3", "write"),
            ]
        ),
    )
    world.add_component(
        entity_id,
        PermissionComponent(allowed_tools=["read", "write"], denied_tools=["write"]),
    )

    await PermissionSystem().process(world)

    pending = world.get_component(entity_id, PendingToolCallsComponent)
    conversation = world.get_component(entity_id, ConversationComponent)
    assert pending is not None
    assert [call.id for call in pending.tool_calls] == ["c2"]
    assert conversation is not None
    denied_ids = [
        msg.tool_call_id for msg in conversation.messages if msg.role == "tool"
    ]
    assert denied_ids == ["c1", "c3"]
