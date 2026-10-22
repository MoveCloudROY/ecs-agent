"""Live subagent delegation tests.

Select a profile with --live; see docs/testing.md.
"""

import asyncio
import json

import pytest

from ecs_agent.components import (
    ConversationComponent,
    LLMComponent,
    SubagentRegistryComponent,
    ToolRegistryComponent,
)
from ecs_agent.components.definitions import (
    SubagentNotificationQueueComponent,
    SubagentSessionTableComponent,
    SubagentWaitComponent,
)
from ecs_agent.core import World
from ecs_agent.systems.subagent import SubagentSystem
from ecs_agent.systems.subagent_wait import SubagentWaitSystem
from ecs_agent.types import (
    SubagentConfig,
    SubagentStreamStartEvent,
)

from tests.support.models import LiveModelFactory


BACKGROUND_PROMPT = (
    "Say 'hello' in one word. "
    "End your response with: "
    "<subagent_background_result>"
    "<summary>hello</summary>"
    "<full_result>hello</full_result>"
    "</subagent_background_result>"
)


async def _execute_live_subagent(
    *,
    stream: bool = False,
    live_model_factory: LiveModelFactory,
) -> None:
    world = World(name="subagent-live")
    entity_id = world.create_entity()
    model = live_model_factory.openai()

    world.add_component(
        entity_id,
        LLMComponent(
            model=model,
            system_prompt="You are a helpful parent agent.",
        ),
    )
    world.add_component(entity_id, ConversationComponent(messages=[]))
    world.add_component(entity_id, ToolRegistryComponent(tools={}, handlers={}))
    world.add_component(
        entity_id,
        SubagentRegistryComponent(
            subagents={
                "worker": SubagentConfig(
                    name="worker",
                    model=model,
                    system_prompt="Answer the user's request directly and briefly.",
                    max_ticks=3,
                )
            }
        ),
    )
    world.add_component(entity_id, SubagentSessionTableComponent(sessions={}))

    # Use max_background_concurrency=1 to test queuing
    subagent_system = SubagentSystem(max_background_concurrency=1)
    await subagent_system.process(world)
    subagent_system.install_subagent_control_tools(world, entity_id)

    # Track stream events if requested
    stream_events = []
    if stream:

        async def on_stream_start(event: SubagentStreamStartEvent) -> None:
            stream_events.append(event)

        world.event_bus.subscribe(SubagentStreamStartEvent, on_stream_start)

    tools = world.get_component(entity_id, ToolRegistryComponent)
    assert tools is not None

    # Launch two background sessions
    subagent_tool = tools.handlers["subagent"]

    resp1_json = await subagent_tool(
        category="worker",
        prompt="Say 'one'.",
        background=True,
        stream=stream,
    )
    resp2_json = await subagent_tool(
        category="worker",
        prompt="Say 'two'.",
        background=True,
    )

    resp1 = json.loads(resp1_json)
    resp2 = json.loads(resp2_json)

    session_id1 = resp1["session_id"]
    session_id2 = resp2["session_id"]

    # Verify one is running/queued (timing might be tight, but session 2 should be queued if session 1 is running)
    status_tool = tools.handlers["subagent_status"]

    # Wait a bit for the scheduler to pick up the first one
    await asyncio.sleep(0.5)

    status2_json = await status_tool(session_id=session_id2)
    status2 = json.loads(status2_json)

    # Session 2 should be queued because concurrency is 1
    assert status2["lifecycle_status"] == "queued"
    assert status2["queue_position"] == 0

    # Wait for both to succeed
    result_tool = tools.handlers["subagent_result"]

    res1_json = await result_tool(session_id=session_id1, timeout=30)
    res2_json = await result_tool(session_id=session_id2, timeout=30)

    res1 = json.loads(res1_json)
    res2 = json.loads(res2_json)

    assert res1["status"] == "success"
    assert res2["status"] == "success"
    assert res1["inline_content"] is not None
    assert res2["inline_content"] is not None
    assert "one" in res1["inline_content"].lower()
    assert "two" in res2["inline_content"].lower()

    if stream:
        # Verify stream start event was received
        assert len(stream_events) > 0
        assert any(e.session_id == session_id1 for e in stream_events)


async def _execute_live_notification_flow(
    *, live_model_factory: LiveModelFactory
) -> None:
    world = World(name="subagent-live-notification")
    entity_id = world.create_entity()
    model = live_model_factory.openai()

    world.add_component(
        entity_id,
        LLMComponent(
            model=model,
            system_prompt="You are a helpful parent agent.",
        ),
    )
    world.add_component(entity_id, ConversationComponent(messages=[]))
    world.add_component(entity_id, ToolRegistryComponent(tools={}, handlers={}))
    world.add_component(
        entity_id,
        SubagentRegistryComponent(
            subagents={
                "worker": SubagentConfig(
                    name="worker",
                    model=model,
                    system_prompt="Answer the user's request directly and briefly.",
                    max_ticks=3,
                )
            }
        ),
    )
    world.add_component(entity_id, SubagentSessionTableComponent(sessions={}))

    subagent_system = SubagentSystem(max_background_concurrency=1)
    wait_system = SubagentWaitSystem(priority=-5)
    await subagent_system.process(world)
    subagent_system.install_subagent_control_tools(world, entity_id)
    world.register_system(wait_system, priority=-5)

    tools = world.get_component(entity_id, ToolRegistryComponent)
    assert tools is not None

    launch_result = json.loads(
        await tools.handlers["subagent"](
            category="worker",
            prompt=BACKGROUND_PROMPT,
            background=True,
        )
    )
    session_id = launch_result["session_id"]

    wait_ack = await tools.handlers["subagent_wait"](session_ids=[session_id])
    assert (
        wait_ack == "Waiting for background subagents. "
        "Will be notified when all sessions complete."
    )
    assert world.has_component(entity_id, SubagentWaitComponent)

    await world.process()

    assert not world.has_component(entity_id, SubagentWaitComponent)

    queue = world.get_component(entity_id, SubagentNotificationQueueComponent)
    assert queue is not None
    assert len(queue.notifications) == 1
    assert queue.notifications[0].session_id == session_id
    assert queue.notifications[0].terminal_status == "succeeded"
    assert queue.notifications[0].delivered_at is not None

    conversation = world.get_component(entity_id, ConversationComponent)
    assert conversation is not None
    # The wait system injects completion notifications as a role="user" message
    # in the conversation tail (cache-stable), not into the system prompt.
    notification_messages = [
        message for message in conversation.messages if message.role == "user"
    ]
    assert len(notification_messages) == 1
    assert "Background subagent updates:" in notification_messages[0].content
    assert session_id in notification_messages[0].content

    summary_result = json.loads(
        await tools.handlers["subagent_result"](
            session_id=session_id,
            read_method="summary",
        )
    )
    if summary_result.get("status") == "success":
        assert summary_result["read_method"] == "summary"
        assert summary_result["inline_content"] is not None
        assert "hello" in summary_result["inline_content"].lower()
    else:
        assert "Summary not available" in summary_result.get("error", "")

    full_result = json.loads(
        await tools.handlers["subagent_result"](
            session_id=session_id,
            read_method="full",
        )
    )
    assert full_result["status"] == "success"
    assert full_result["lifecycle_status"] == "succeeded"
    assert full_result["inline_content"] is not None
    assert "hello" in full_result["inline_content"].lower()


@pytest.mark.live(formats=["openai_chat_completions"], capabilities=["text", "tools"])
@pytest.mark.subagent
@pytest.mark.asyncio
async def test_subagent_compatible_mode(
    *, live_model_factory: LiveModelFactory
) -> None:
    await _execute_live_subagent(
        live_model_factory=live_model_factory,
    )


@pytest.mark.live(formats=["openai_responses"], capabilities=["text", "tools"])
@pytest.mark.subagent
@pytest.mark.asyncio
async def test_subagent_responses_mode(*, live_model_factory: LiveModelFactory) -> None:
    await _execute_live_subagent(
        live_model_factory=live_model_factory,
    )


@pytest.mark.live(
    formats=["openai_chat_completions"], capabilities=["text", "tools", "streaming"]
)
@pytest.mark.subagent
@pytest.mark.asyncio
async def test_subagent_streaming_smoke(
    *, live_model_factory: LiveModelFactory
) -> None:
    await _execute_live_subagent(
        stream=True,
        live_model_factory=live_model_factory,
    )


@pytest.mark.live(formats=["openai_chat_completions"], capabilities=["text", "tools"])
@pytest.mark.subagent
@pytest.mark.asyncio
async def test_aliyun_completions_background_completion_notification_flow(
    *, live_model_factory: LiveModelFactory
) -> None:
    await _execute_live_notification_flow(
        live_model_factory=live_model_factory,
    )


@pytest.mark.live(formats=["openai_responses"], capabilities=["text", "tools"])
@pytest.mark.subagent
@pytest.mark.asyncio
async def test_aliyun_responses_background_completion_notification_flow(
    *, live_model_factory: LiveModelFactory
) -> None:
    await _execute_live_notification_flow(
        live_model_factory=live_model_factory,
    )
