"""Live tests for subagent_wait wait-all semantics and subagent_resume with real LLM."""

from __future__ import annotations

import asyncio
import json
import os

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
from ecs_agent.logging import configure_logging
from ecs_agent.providers import ApiFormat, OpenAIModel, ProviderConfig
from ecs_agent.providers.claude_model import ClaudeModel
from ecs_agent.systems.subagent import SubagentSystem
from ecs_agent.systems.subagent_wait import SubagentWaitSystem
from ecs_agent.types import SubagentConfig


def _build_openai_provider(api_key: str, base_url: str, model: str) -> OpenAIModel:
    return OpenAIModel(
        config=ProviderConfig(
            provider_id="mimo",
            base_url=base_url,
            api_key=api_key,
            api_format=ApiFormat.OPENAI_CHAT_COMPLETIONS,
        ),
        model=model,
    )


def _build_anthropic_provider(
    api_key: str, base_url: str, model: str
) -> ClaudeModel:
    return ClaudeModel(
        config=ProviderConfig(
            provider_id="mimo-anthropic",
            base_url=base_url,
            api_key=api_key,
            api_format=ApiFormat.ANTHROPIC_MESSAGES,
        ),
        model=model,
    )


def _live_env() -> tuple[str, str, str] | None:
    api_key = os.getenv("LLM_API_KEY")
    base_url = os.getenv("LLM_BASE_URL")
    model = os.getenv("LLM_MODEL")
    if not api_key or not base_url or not model:
        return None
    return api_key, base_url, model


@pytest.mark.asyncio
async def test_live_wait_all_resolves_when_all_sessions_complete() -> None:
    env = _live_env()
    if env is None:
        pytest.skip("LLM_API_KEY / LLM_BASE_URL / LLM_MODEL not set")
    api_key, base_url, model = env
    configure_logging(json_output=False, level="WARNING")

    world = World(name="live-wait-all")
    parent = world.create_entity()
    provider = _build_openai_provider(api_key, base_url, model)

    world.add_component(
        parent,
        LLMComponent(model=provider, system_prompt="You are a parent agent."),
    )
    world.add_component(parent, ConversationComponent(messages=[]))
    world.add_component(parent, ToolRegistryComponent(tools={}, handlers={}))
    world.add_component(parent, SubagentSessionTableComponent(sessions={}))
    world.add_component(parent, SubagentNotificationQueueComponent())
    world.add_component(
        parent,
        SubagentRegistryComponent(
            subagents={
                "worker": SubagentConfig(
                    name="worker",
                    model=provider,
                    system_prompt="Reply with one word.",
                    max_ticks=3,
                ),
            }
        ),
    )

    subagent_system = SubagentSystem(max_background_concurrency=2)
    wait_system = SubagentWaitSystem(priority=-5)
    subagent_system.install_subagent_tool(world, parent)
    subagent_system.install_subagent_control_tools(world, parent)
    world.register_system(wait_system, priority=-5)

    tools = world.get_component(parent, ToolRegistryComponent)
    assert tools is not None

    launch_a = json.loads(
        await tools.handlers["subagent"](
            category="worker",
            prompt="Say 'alpha'.",
            background=True,
        )
    )
    launch_b = json.loads(
        await tools.handlers["subagent"](
            category="worker",
            prompt="Say 'beta'.",
            background=True,
        )
    )
    sid_a = launch_a["session_id"]
    sid_b = launch_b["session_id"]

    wait_ack = await tools.handlers["subagent_wait"](
        session_ids=[sid_a, sid_b],
        timeout=60.0,
    )
    assert "all sessions complete" in wait_ack

    await asyncio.wait_for(world.process(), timeout=120.0)

    assert not world.has_component(parent, SubagentWaitComponent)

    conversation = world.get_component(parent, ConversationComponent)
    assert conversation is not None
    user_notifications = [
        m for m in conversation.messages
        if m.role == "user" and "Background subagent updates:" in m.content
    ]
    assert len(user_notifications) == 1
    assert sid_a in user_notifications[0].content
    assert sid_b in user_notifications[0].content

    result_a = json.loads(
        await tools.handlers["subagent_result"](session_id=sid_a)
    )
    result_b = json.loads(
        await tools.handlers["subagent_result"](session_id=sid_b)
    )
    assert result_a["status"] == "success"
    assert result_b["status"] == "success"


@pytest.mark.asyncio
async def test_live_wait_all_does_not_wake_on_partial_completion() -> None:
    env = _live_env()
    if env is None:
        pytest.skip("LLM_API_KEY / LLM_BASE_URL / LLM_MODEL not set")
    api_key, base_url, model = env
    configure_logging(json_output=False, level="WARNING")

    world = World(name="live-wait-all-partial")
    parent = world.create_entity()
    provider = _build_openai_provider(api_key, base_url, model)

    world.add_component(
        parent,
        LLMComponent(model=provider, system_prompt="You are a parent agent."),
    )
    world.add_component(parent, ConversationComponent(messages=[]))
    world.add_component(parent, ToolRegistryComponent(tools={}, handlers={}))
    world.add_component(parent, SubagentSessionTableComponent(sessions={}))
    world.add_component(parent, SubagentNotificationQueueComponent())
    world.add_component(
        parent,
        SubagentRegistryComponent(
            subagents={
                "fast": SubagentConfig(
                    name="fast",
                    model=provider,
                    system_prompt="Reply immediately with one word.",
                    max_ticks=3,
                ),
                "slow": SubagentConfig(
                    name="slow",
                    model=provider,
                    system_prompt=(
                        "Think briefly, then reply with one word. "
                        "Take a moment before answering."
                    ),
                    max_ticks=3,
                ),
            }
        ),
    )

    subagent_system = SubagentSystem(max_background_concurrency=2)
    wait_system = SubagentWaitSystem(priority=-5)
    subagent_system.install_subagent_tool(world, parent)
    subagent_system.install_subagent_control_tools(world, parent)
    world.register_system(wait_system, priority=-5)

    tools = world.get_component(parent, ToolRegistryComponent)
    assert tools is not None

    launch_fast = json.loads(
        await tools.handlers["subagent"](
            category="fast",
            prompt="Say 'quick'.",
            background=True,
        )
    )
    launch_slow = json.loads(
        await tools.handlers["subagent"](
            category="slow",
            prompt="Say 'deliberate'.",
            background=True,
        )
    )
    sid_fast = launch_fast["session_id"]
    sid_slow = launch_slow["session_id"]

    await tools.handlers["subagent_wait"](
        session_ids=[sid_fast, sid_slow],
        timeout=60.0,
    )

    await asyncio.wait_for(world.process(), timeout=120.0)

    assert not world.has_component(parent, SubagentWaitComponent)

    conversation = world.get_component(parent, ConversationComponent)
    assert conversation is not None
    user_notifications = [
        m for m in conversation.messages
        if m.role == "user" and "Background subagent updates:" in m.content
    ]
    assert len(user_notifications) == 1
    assert sid_fast in user_notifications[0].content
    assert sid_slow in user_notifications[0].content


@pytest.mark.asyncio
async def test_live_subagent_resume_restarts_failed_session() -> None:
    env = _live_env()
    if env is None:
        pytest.skip("LLM_API_KEY / LLM_BASE_URL / LLM_MODEL not set")
    api_key, base_url, model = env
    configure_logging(json_output=False, level="WARNING")

    world = World(name="live-resume")
    parent = world.create_entity()
    provider = _build_openai_provider(api_key, base_url, model)

    world.add_component(
        parent,
        LLMComponent(model=provider, system_prompt="You are a parent agent."),
    )
    world.add_component(parent, ConversationComponent(messages=[]))
    world.add_component(parent, ToolRegistryComponent(tools={}, handlers={}))
    world.add_component(parent, SubagentSessionTableComponent(sessions={}))
    world.add_component(parent, SubagentNotificationQueueComponent())
    world.add_component(
        parent,
        SubagentRegistryComponent(
            subagents={
                "worker": SubagentConfig(
                    name="worker",
                    model=provider,
                    system_prompt="Reply with one word.",
                    max_ticks=3,
                ),
            }
        ),
    )

    subagent_system = SubagentSystem(max_background_concurrency=2)
    subagent_system.install_subagent_tool(world, parent)
    subagent_system.install_subagent_control_tools(world, parent)

    tools = world.get_component(parent, ToolRegistryComponent)
    assert tools is not None

    from ecs_agent.types import SubagentSessionRecord

    failed_metadata = SubagentSessionRecord(
        session_id="ses-live-failed",
        category="worker",
        prompt="Say 'hello'.",
        parent_entity_id=parent,
        created_at="2026-06-22T00:00:00Z",
        updated_at="2026-06-22T00:00:00Z",
        status="failed",
        error="simulated failure for resume test",
        background=True,
    )
    await subagent_system._runtime_manager.restore_session_metadata(
        failed_metadata
    )
    await subagent_system._runtime_manager.sync_to_component(world, parent)

    resume_payload = json.loads(
        await tools.handlers["subagent_resume"](
            session_id="ses-live-failed",
        )
    )
    assert resume_payload["status"] == "resumed"
    assert resume_payload["original_session_id"] == "ses-live-failed"
    new_sid = resume_payload["new_session_id"]
    assert new_sid != "ses-live-failed"

    new_task = await subagent_system._runtime_manager.get_task(new_sid)
    assert new_task is not None
    await asyncio.wait_for(new_task, timeout=60.0)

    new_session = await subagent_system._runtime_manager.get_session(new_sid)
    assert new_session is not None
    assert new_session.status == "succeeded"


@pytest.mark.asyncio
async def test_live_notification_role_is_user_not_system() -> None:
    env = _live_env()
    if env is None:
        pytest.skip("LLM_API_KEY / LLM_BASE_URL / LLM_MODEL not set")
    api_key, base_url, model = env
    configure_logging(json_output=False, level="WARNING")

    world = World(name="live-role-check")
    parent = world.create_entity()
    provider = _build_openai_provider(api_key, base_url, model)

    world.add_component(
        parent,
        LLMComponent(model=provider, system_prompt="You are a parent agent."),
    )
    world.add_component(parent, ConversationComponent(messages=[]))
    world.add_component(parent, ToolRegistryComponent(tools={}, handlers={}))
    world.add_component(parent, SubagentSessionTableComponent(sessions={}))
    world.add_component(parent, SubagentNotificationQueueComponent())
    world.add_component(
        parent,
        SubagentRegistryComponent(
            subagents={
                "worker": SubagentConfig(
                    name="worker",
                    model=provider,
                    system_prompt="Reply with one word.",
                    max_ticks=3,
                ),
            }
        ),
    )

    subagent_system = SubagentSystem(max_background_concurrency=1)
    wait_system = SubagentWaitSystem(priority=-5)
    subagent_system.install_subagent_tool(world, parent)
    subagent_system.install_subagent_control_tools(world, parent)
    world.register_system(wait_system, priority=-5)

    tools = world.get_component(parent, ToolRegistryComponent)
    assert tools is not None

    launch = json.loads(
        await tools.handlers["subagent"](
            category="worker",
            prompt="Say 'hi'.",
            background=True,
        )
    )
    sid = launch["session_id"]

    await tools.handlers["subagent_wait"](session_ids=[sid], timeout=60.0)
    await asyncio.wait_for(world.process(), timeout=120.0)

    conversation = world.get_component(parent, ConversationComponent)
    assert conversation is not None
    system_role_msgs = [m for m in conversation.messages if m.role == "system"]
    user_notification_msgs = [
        m for m in conversation.messages
        if m.role == "user" and "Background subagent updates:" in m.content
    ]
    assert len(user_notification_msgs) == 1
    assert all(
        "Background subagent updates:" not in m.content
        for m in system_role_msgs
    )


def _anthropic_env() -> tuple[str, str, str] | None:
    api_key = os.getenv("LLM_API_KEY")
    base_url = os.getenv("ANTHROPIC_BASE_URL")
    model = os.getenv("LLM_MODEL")
    if not api_key or not base_url or not model:
        return None
    return api_key, base_url, model


@pytest.mark.asyncio
async def test_live_anthropic_wait_all_and_notification_role() -> None:
    env = _anthropic_env()
    if env is None:
        pytest.skip("LLM_API_KEY / ANTHROPIC_BASE_URL / LLM_MODEL not set")
    api_key, base_url, model = env
    configure_logging(json_output=False, level="WARNING")

    world = World(name="live-anthropic-wait-all")
    parent = world.create_entity()
    provider = _build_anthropic_provider(api_key, base_url, model)

    world.add_component(
        parent,
        LLMComponent(model=provider, system_prompt="You are a parent agent."),
    )
    world.add_component(parent, ConversationComponent(messages=[]))
    world.add_component(parent, ToolRegistryComponent(tools={}, handlers={}))
    world.add_component(parent, SubagentSessionTableComponent(sessions={}))
    world.add_component(parent, SubagentNotificationQueueComponent())
    world.add_component(
        parent,
        SubagentRegistryComponent(
            subagents={
                "worker": SubagentConfig(
                    name="worker",
                    model=provider,
                    system_prompt="Reply with one word.",
                    max_ticks=3,
                ),
            }
        ),
    )

    subagent_system = SubagentSystem(max_background_concurrency=2)
    wait_system = SubagentWaitSystem(priority=-5)
    subagent_system.install_subagent_tool(world, parent)
    subagent_system.install_subagent_control_tools(world, parent)
    world.register_system(wait_system, priority=-5)

    tools = world.get_component(parent, ToolRegistryComponent)
    assert tools is not None

    launch_a = json.loads(
        await tools.handlers["subagent"](
            category="worker",
            prompt="Say 'alpha'.",
            background=True,
        )
    )
    launch_b = json.loads(
        await tools.handlers["subagent"](
            category="worker",
            prompt="Say 'beta'.",
            background=True,
        )
    )
    sid_a = launch_a["session_id"]
    sid_b = launch_b["session_id"]

    await tools.handlers["subagent_wait"](
        session_ids=[sid_a, sid_b],
        timeout=60.0,
    )
    await asyncio.wait_for(world.process(), timeout=120.0)

    assert not world.has_component(parent, SubagentWaitComponent)

    conversation = world.get_component(parent, ConversationComponent)
    assert conversation is not None
    system_role_msgs = [m for m in conversation.messages if m.role == "system"]
    user_notification_msgs = [
        m for m in conversation.messages
        if m.role == "user" and "Background subagent updates:" in m.content
    ]
    assert len(user_notification_msgs) == 1
    assert sid_a in user_notification_msgs[0].content
    assert sid_b in user_notification_msgs[0].content
    assert all(
        "Background subagent updates:" not in m.content
        for m in system_role_msgs
    )


@pytest.mark.asyncio
async def test_live_anthropic_subagent_resume() -> None:
    env = _anthropic_env()
    if env is None:
        pytest.skip("LLM_API_KEY / ANTHROPIC_BASE_URL / LLM_MODEL not set")
    api_key, base_url, model = env
    configure_logging(json_output=False, level="WARNING")

    world = World(name="live-anthropic-resume")
    parent = world.create_entity()
    provider = _build_anthropic_provider(api_key, base_url, model)

    world.add_component(
        parent,
        LLMComponent(model=provider, system_prompt="You are a parent agent."),
    )
    world.add_component(parent, ConversationComponent(messages=[]))
    world.add_component(parent, ToolRegistryComponent(tools={}, handlers={}))
    world.add_component(parent, SubagentSessionTableComponent(sessions={}))
    world.add_component(
        parent,
        SubagentRegistryComponent(
            subagents={
                "worker": SubagentConfig(
                    name="worker",
                    model=provider,
                    system_prompt="Reply with one word.",
                    max_ticks=3,
                ),
            }
        ),
    )

    subagent_system = SubagentSystem(max_background_concurrency=1)
    subagent_system.install_subagent_tool(world, parent)
    subagent_system.install_subagent_control_tools(world, parent)

    from ecs_agent.types import SubagentSessionRecord

    failed_metadata = SubagentSessionRecord(
        session_id="ses-anthropic-failed",
        category="worker",
        prompt="Say 'hello'.",
        parent_entity_id=parent,
        created_at="2026-06-22T00:00:00Z",
        updated_at="2026-06-22T00:00:00Z",
        status="failed",
        error="simulated failure for anthropic resume test",
        background=True,
    )
    await subagent_system._runtime_manager.restore_session_metadata(
        failed_metadata
    )
    await subagent_system._runtime_manager.sync_to_component(world, parent)

    tools = world.get_component(parent, ToolRegistryComponent)
    assert tools is not None

    resume_payload = json.loads(
        await tools.handlers["subagent_resume"](
            session_id="ses-anthropic-failed",
        )
    )
    assert resume_payload["status"] == "resumed"
    new_sid = resume_payload["new_session_id"]

    new_task = await subagent_system._runtime_manager.get_task(new_sid)
    assert new_task is not None
    await asyncio.wait_for(new_task, timeout=60.0)

    new_session = await subagent_system._runtime_manager.get_session(new_sid)
    assert new_session is not None
    assert new_session.status == "succeeded"
