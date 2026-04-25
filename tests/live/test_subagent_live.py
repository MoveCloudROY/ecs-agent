"""Live subagent delegation tests."""

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
from ecs_agent.providers import ApiFormat, OpenAIModel, ProviderConfig
from ecs_agent.systems.subagent import SubagentSystem
from ecs_agent.systems.subagent_wait import SubagentWaitSystem
from ecs_agent.types import (
    SubagentConfig,
    SubagentStreamStartEvent,
)

BACKGROUND_PROMPT = (
    "Say 'hello' in one word. "
    "End your response with: "
    "<subagent_background_result>"
    "<summary>hello</summary>"
    "<full_result>hello</full_result>"
    "</subagent_background_result>"
)


def _build_provider(
    api_key: str, *, base_url: str, api_format: ApiFormat
) -> OpenAIModel:
    model = "qwen3.5-flash"
    return OpenAIModel(
        config=ProviderConfig(
            provider_id="aliyun",
            base_url=base_url,
            api_key=api_key,
            api_format=api_format,
        ),
        model=model,
    )


async def _execute_live_subagent(
    live_api_key: str,
    *,
    base_url: str,
    api_format: ApiFormat,
    stream: bool = False,
) -> None:
    world = World(name="subagent-live")
    entity_id = world.create_entity()
    model = _build_provider(
        live_api_key,
        base_url=base_url,
        api_format=api_format,
    )

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
    live_api_key: str,
    *,
    base_url: str,
    api_format: ApiFormat,
) -> None:
    world = World(name="subagent-live-notification")
    entity_id = world.create_entity()
    model = _build_provider(
        live_api_key,
        base_url=base_url,
        api_format=api_format,
    )

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
        wait_ack
        == "Waiting for background subagents. Will be notified when they complete."
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
    system_messages = [
        message for message in conversation.messages if message.role == "system"
    ]
    assert len(system_messages) == 1
    assert "Background subagent updates:" in system_messages[0].content
    assert session_id in system_messages[0].content

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


@pytest.mark.asyncio
async def test_subagent_compatible_mode(live_api_key: str) -> None:
    await _execute_live_subagent(
        live_api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_format=ApiFormat.OPENAI_CHAT_COMPLETIONS,
    )


@pytest.mark.asyncio
async def test_subagent_responses_mode(live_api_key: str) -> None:
    await _execute_live_subagent(
        live_api_key,
        base_url="https://dashscope.aliyuncs.com/api/v2/apps/protocols/compatible-mode/v1",
        api_format=ApiFormat.OPENAI_RESPONSES,
    )


@pytest.mark.asyncio
async def test_subagent_streaming_smoke(live_api_key: str) -> None:
    await _execute_live_subagent(
        live_api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_format=ApiFormat.OPENAI_CHAT_COMPLETIONS,
        stream=True,
    )


@pytest.mark.asyncio
async def test_aliyun_completions_background_completion_notification_flow(
    live_api_key: str,
) -> None:
    await _execute_live_notification_flow(
        live_api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_format=ApiFormat.OPENAI_CHAT_COMPLETIONS,
    )


@pytest.mark.asyncio
async def test_aliyun_responses_background_completion_notification_flow(
    live_api_key: str,
) -> None:
    await _execute_live_notification_flow(
        live_api_key,
        base_url=(
            "https://dashscope.aliyuncs.com/api/v2/apps/protocols/compatible-mode/v1"
        ),
        api_format=ApiFormat.OPENAI_RESPONSES,
    )
