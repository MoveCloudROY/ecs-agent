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
from ecs_agent.components.definitions import SubagentSessionTableComponent
from ecs_agent.core import World
from ecs_agent.providers import ApiFormat, OpenAIProvider, ProviderConfig
from ecs_agent.systems.subagent import SubagentSystem
from ecs_agent.types import (
    SubagentConfig,
    SubagentStreamStartEvent,
)


def _build_provider(
    api_key: str, *, base_url: str, api_format: ApiFormat
) -> OpenAIProvider:
    model = "qwen3.5-flash"
    return OpenAIProvider(
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
    provider = _build_provider(
        live_api_key,
        base_url=base_url,
        api_format=api_format,
    )

    world.add_component(
        entity_id,
        LLMComponent(
            provider=provider,
            model=provider.model,
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
                    provider=provider,
                    model=provider.model,
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

    # Launch two background sessions
    subagent_tool = world.get_component(entity_id, ToolRegistryComponent).handlers[
        "subagent"
    ]

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
    status_tool = world.get_component(entity_id, ToolRegistryComponent).handlers[
        "subagent_status"
    ]

    # Wait a bit for the scheduler to pick up the first one
    await asyncio.sleep(0.5)

    status2_json = await status_tool(session_id=session_id2)
    status2 = json.loads(status2_json)

    # Session 2 should be queued because concurrency is 1
    assert status2["lifecycle_status"] == "queued"
    assert status2["queue_position"] == 0

    # Wait for both to succeed
    result_tool = world.get_component(entity_id, ToolRegistryComponent).handlers[
        "subagent_result"
    ]

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


@pytest.mark.live
@pytest.mark.asyncio
async def test_subagent_compatible_mode(live_api_key: str) -> None:
    await _execute_live_subagent(
        live_api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_format=ApiFormat.OPENAI_CHAT_COMPLETIONS,
    )


@pytest.mark.live
@pytest.mark.asyncio
async def test_subagent_responses_mode(live_api_key: str) -> None:
    await _execute_live_subagent(
        live_api_key,
        base_url="https://dashscope.aliyuncs.com/api/v2/apps/protocols/compatible-mode/v1",
        api_format=ApiFormat.OPENAI_RESPONSES,
    )


@pytest.mark.live
@pytest.mark.asyncio
async def test_subagent_streaming_smoke(live_api_key: str) -> None:
    await _execute_live_subagent(
        live_api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_format=ApiFormat.OPENAI_CHAT_COMPLETIONS,
        stream=True,
    )
