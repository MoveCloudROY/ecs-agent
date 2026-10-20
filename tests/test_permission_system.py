"""Permission preflight and execution contracts for allowlists and denylists."""

import pytest

from ecs_agent.components import ConversationComponent, PendingToolCallsComponent, PermissionComponent
from ecs_agent.core import Runner
from ecs_agent.providers._message_contract import tool_span_error
from ecs_agent.systems.permission import PermissionSystem
from ecs_agent.systems.reasoning import ReasoningSystem
from ecs_agent.systems.tool_execution import ToolExecutionSystem
from ecs_agent.types import ToolDeniedEvent
from tests.authorization_support import scenario


@pytest.mark.parametrize("policy,expected", [
    (None, ["read", "write"]),
    (PermissionComponent(), ["read", "write"]),
    (PermissionComponent(allowed_tools=[]), []),
    (PermissionComponent(allowed_tools=["read"]), ["read"]),
    (PermissionComponent(denied_tools=["write"]), ["read"]),
    (PermissionComponent(allowed_tools=["read"], denied_tools=["read"]), []),
])
async def test_permission_policy_is_enforced_without_optional_preflight(policy, expected) -> None:
    case = scenario()
    if policy is not None:
        case.world.add_component(case.entity, policy)
    case.world.register_system(ToolExecutionSystem(), priority=5)
    assert (await Runner().run(case.world, max_ticks=4)).success
    assert [name for name, _ in case.called] == expected
    conversation = case.world.get_component(case.entity, ConversationComponent)
    assert tool_span_error(conversation.messages) is None
    denied = [event for event in case.events if isinstance(event, ToolDeniedEvent)]
    assert len(denied) == 2 - len(expected)


async def test_preflight_keeps_original_queue_and_defers_messages_and_events() -> None:
    case = scenario()
    case.world.add_component(case.entity, PermissionComponent(allowed_tools=["read"]))
    await ReasoningSystem().process(case.world)
    pending = case.world.get_component(case.entity, PendingToolCallsComponent)
    conversation = case.world.get_component(case.entity, ConversationComponent)
    before = list(conversation.messages)
    await PermissionSystem().process(case.world)
    assert case.world.get_component(case.entity, PendingToolCallsComponent) is pending
    assert [call.id for call in pending.tool_calls] == ["read-1", "write-1"]
    assert conversation.messages == before
    assert not any(isinstance(event, ToolDeniedEvent) for event in case.events)
    await ToolExecutionSystem().process(case.world)
    assert case.called == [("read", 1)]
    assert case.world.get_component(case.entity, PendingToolCallsComponent) is None
    assert tool_span_error(conversation.messages) is None
    assert [message.tool_call_id for message in conversation.messages if message.role == "tool"] == ["read-1", "write-1"]
