"""Tests for ToolApprovalComponent.timeout=None (infinite wait)."""

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
    ToolApprovalRequestedEvent,
    ToolCall,
)


@pytest.mark.asyncio
async def test_approval_timeout_none_waits_indefinitely() -> None:
    """timeout=None should wait for the future without timing out."""
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
        ToolApprovalComponent(
            policy=ApprovalPolicy.REQUIRE_APPROVAL,
            timeout=None,  # Infinite wait
        ),
    )

    async def delayed_approve(event: ToolApprovalRequestedEvent) -> None:
        # Simulate a delay, then approve
        await asyncio.sleep(0.05)
        event.approval_future.set_result(True)

    world.event_bus.subscribe(ToolApprovalRequestedEvent, delayed_approve)

    await ToolApprovalSystem().process(world)

    pending = world.get_component(entity_id, PendingToolCallsComponent)
    approval = world.get_component(entity_id, ToolApprovalComponent)
    assert pending is not None
    assert approval is not None
    assert [c.id for c in pending.tool_calls] == ["call-1"]
    assert approval.approved_calls == ["call-1"]


@pytest.mark.asyncio
async def test_approval_timeout_none_deny_works() -> None:
    """timeout=None should still respect a deny decision."""
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
        ToolApprovalComponent(
            policy=ApprovalPolicy.REQUIRE_APPROVAL,
            timeout=None,
        ),
    )

    async def deny_handler(event: ToolApprovalRequestedEvent) -> None:
        event.approval_future.set_result(False)

    world.event_bus.subscribe(ToolApprovalRequestedEvent, deny_handler)

    await ToolApprovalSystem().process(world)

    pending = world.get_component(entity_id, PendingToolCallsComponent)
    approval = world.get_component(entity_id, ToolApprovalComponent)
    assert pending is None  # Removed because denied
    assert approval is not None
    assert approval.denied_calls == ["call-1"]


@pytest.mark.asyncio
async def test_approval_default_timeout_still_30() -> None:
    """Default timeout should remain 30.0 for backward compatibility."""
    comp = ToolApprovalComponent(policy=ApprovalPolicy.REQUIRE_APPROVAL)
    assert comp.timeout == 30.0
