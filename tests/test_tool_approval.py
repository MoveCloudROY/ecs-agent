import asyncio

import pytest

from ecs_agent.components import (
    ConversationComponent,
    PendingToolCallsComponent,
    ToolApprovalComponent,
)
from ecs_agent.core import World
from ecs_agent.systems.tool_approval import ToolApprovalSystem
from ecs_agent.types import (
    ApprovalPolicy,
    Message,
    ToolApprovedEvent,
    ToolApprovalRequestedEvent,
    ToolCall,
    ToolDeniedEvent,
)


@pytest.mark.asyncio
async def test_always_approve_keeps_tool_call_and_publishes_approved_event() -> None:
    world = World()
    entity_id = world.create_entity()
    world.add_component(entity_id, ConversationComponent(messages=[]))
    world.add_component(
        entity_id,
        PendingToolCallsComponent(
            tool_calls=[ToolCall(id="call-1", name="ping", arguments={})]
        ),
    )
    world.add_component(
        entity_id,
        ToolApprovalComponent(policy=ApprovalPolicy.ALWAYS_APPROVE),
    )

    seen: list[ToolApprovedEvent] = []

    async def on_approved(event: ToolApprovedEvent) -> None:
        seen.append(event)

    world.event_bus.subscribe(ToolApprovedEvent, on_approved)

    await ToolApprovalSystem().process(world)

    pending = world.get_component(entity_id, PendingToolCallsComponent)
    approval = world.get_component(entity_id, ToolApprovalComponent)
    assert pending is not None
    assert approval is not None
    assert [call.id for call in pending.tool_calls] == ["call-1"]
    assert approval.approved_calls == ["call-1"]
    assert approval.denied_calls == []
    assert len(seen) == 1
    assert seen[0].entity_id == entity_id
    assert seen[0].tool_call_id == "call-1"


@pytest.mark.asyncio
async def test_always_deny_removes_tool_call_adds_message_and_publishes_denied_event() -> (
    None
):
    world = World()
    entity_id = world.create_entity()
    world.add_component(entity_id, ConversationComponent(messages=[]))
    world.add_component(
        entity_id,
        PendingToolCallsComponent(
            tool_calls=[ToolCall(id="call-1", name="ping", arguments={})]
        ),
    )
    world.add_component(
        entity_id,
        ToolApprovalComponent(policy=ApprovalPolicy.ALWAYS_DENY),
    )

    seen: list[ToolDeniedEvent] = []

    async def on_denied(event: ToolDeniedEvent) -> None:
        seen.append(event)

    world.event_bus.subscribe(ToolDeniedEvent, on_denied)

    await ToolApprovalSystem().process(world)

    pending = world.get_component(entity_id, PendingToolCallsComponent)
    approval = world.get_component(entity_id, ToolApprovalComponent)
    conversation = world.get_component(entity_id, ConversationComponent)
    assert pending is None
    assert approval is not None
    assert approval.approved_calls == []
    assert approval.denied_calls == ["call-1"]
    assert conversation is not None
    assert conversation.messages[-1] == Message(
        role="system", content="Tool call call-1 denied by policy"
    )
    assert len(seen) == 1
    assert seen[0].entity_id == entity_id
    assert seen[0].tool_call_id == "call-1"
    assert seen[0].reason == "Denied by ALWAYS_DENY policy"


@pytest.mark.asyncio
async def test_require_approval_publishes_request_and_keeps_call_when_handler_approves() -> (
    None
):
    world = World()
    entity_id = world.create_entity()
    world.add_component(entity_id, ConversationComponent(messages=[]))
    world.add_component(
        entity_id,
        PendingToolCallsComponent(
            tool_calls=[ToolCall(id="call-1", name="ping", arguments={})]
        ),
    )
    world.add_component(
        entity_id,
        ToolApprovalComponent(policy=ApprovalPolicy.REQUIRE_APPROVAL, timeout=0.2),
    )

    requests: list[ToolApprovalRequestedEvent] = []

    async def on_request(event: ToolApprovalRequestedEvent) -> None:
        requests.append(event)
        event.approval_future.set_result(True)

    world.event_bus.subscribe(ToolApprovalRequestedEvent, on_request)

    await ToolApprovalSystem().process(world)

    pending = world.get_component(entity_id, PendingToolCallsComponent)
    approval = world.get_component(entity_id, ToolApprovalComponent)
    assert pending is not None
    assert approval is not None
    assert [call.id for call in pending.tool_calls] == ["call-1"]
    assert approval.approved_calls == ["call-1"]
    assert approval.denied_calls == []
    assert len(requests) == 1
    assert requests[0].entity_id == entity_id
    assert requests[0].tool_call.id == "call-1"


@pytest.mark.asyncio
async def test_require_approval_timeout_denies_call_when_future_is_never_resolved() -> (
    None
):
    world = World()
    entity_id = world.create_entity()
    world.add_component(entity_id, ConversationComponent(messages=[]))
    world.add_component(
        entity_id,
        PendingToolCallsComponent(
            tool_calls=[ToolCall(id="call-1", name="ping", arguments={})]
        ),
    )
    world.add_component(
        entity_id,
        ToolApprovalComponent(policy=ApprovalPolicy.REQUIRE_APPROVAL, timeout=0.01),
    )

    async def on_request(event: ToolApprovalRequestedEvent) -> None:
        _ = event
        await asyncio.sleep(0.05)

    world.event_bus.subscribe(ToolApprovalRequestedEvent, on_request)

    await ToolApprovalSystem().process(world)

    pending = world.get_component(entity_id, PendingToolCallsComponent)
    approval = world.get_component(entity_id, ToolApprovalComponent)
    conversation = world.get_component(entity_id, ConversationComponent)
    assert pending is None
    assert approval is not None
    assert approval.approved_calls == []
    assert approval.denied_calls == ["call-1"]
    assert conversation is not None
    assert conversation.messages[-1] == Message(
        role="system", content="Tool call call-1 denied by policy"
    )


@pytest.mark.asyncio
async def test_entity_without_tool_approval_component_is_skipped() -> None:
    world = World()
    entity_id = world.create_entity()
    world.add_component(entity_id, ConversationComponent(messages=[]))
    world.add_component(
        entity_id,
        PendingToolCallsComponent(
            tool_calls=[ToolCall(id="call-1", name="ping", arguments={})]
        ),
    )

    await ToolApprovalSystem().process(world)

    pending = world.get_component(entity_id, PendingToolCallsComponent)
    assert pending is not None
    assert [call.id for call in pending.tool_calls] == ["call-1"]


@pytest.mark.asyncio
async def test_require_approval_processes_each_call_individually() -> None:
    world = World()
    entity_id = world.create_entity()
    world.add_component(entity_id, ConversationComponent(messages=[]))
    world.add_component(
        entity_id,
        PendingToolCallsComponent(
            tool_calls=[
                ToolCall(id="call-1", name="ping", arguments={}),
                ToolCall(id="call-2", name="pong", arguments={}),
            ]
        ),
    )
    world.add_component(
        entity_id,
        ToolApprovalComponent(policy=ApprovalPolicy.REQUIRE_APPROVAL, timeout=0.2),
    )

    requested_ids: list[str] = []

    async def on_request(event: ToolApprovalRequestedEvent) -> None:
        requested_ids.append(event.tool_call.id)
        event.approval_future.set_result(event.tool_call.id == "call-1")

    world.event_bus.subscribe(ToolApprovalRequestedEvent, on_request)

    await ToolApprovalSystem().process(world)

    pending = world.get_component(entity_id, PendingToolCallsComponent)
    approval = world.get_component(entity_id, ToolApprovalComponent)
    conversation = world.get_component(entity_id, ConversationComponent)
    assert pending is not None
    assert approval is not None
    assert [call.id for call in pending.tool_calls] == ["call-1"]
    assert approval.approved_calls == ["call-1"]
    assert approval.denied_calls == ["call-2"]
    assert requested_ids == ["call-1", "call-2"]
    assert conversation is not None
    assert conversation.messages[-1] == Message(
        role="system", content="Tool call call-2 denied by policy"
    )
