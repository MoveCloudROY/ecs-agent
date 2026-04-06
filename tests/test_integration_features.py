"""Integration tests for five-features plan spanning all 5 new features.

Tests the interplay between:
1. Enhanced logging
2. Responses API
3. Tree conversation
4. Markdown skills
5. Subagent delegation
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from ecs_agent.components import (
    ConversationComponent,
    ConversationTreeComponent,
    LLMComponent,
    MessageBusConfigComponent,
    SubagentNotificationQueueComponent,
    SubagentRegistryComponent,
    SubagentSessionTableComponent,
    SubagentWaitComponent,
    ToolRegistryComponent,
)
from ecs_agent.conversation_tree import add_message, linearize
from ecs_agent.core import Runner, World
from ecs_agent.logging import configure_logging, get_logger
from ecs_agent.providers import FakeProvider
from ecs_agent.providers.config import ApiFormat, ProviderConfig
from ecs_agent.providers.openai_provider import OpenAIProvider
from ecs_agent.serialization import WorldSerializer
from ecs_agent.skills.skill import Skill
from ecs_agent.systems.error_handling import ErrorHandlingSystem
from ecs_agent.systems.memory import MemorySystem
from ecs_agent.systems.reasoning import ReasoningSystem
from ecs_agent.systems.message_bus import MessageBusSystem
from ecs_agent.systems.subagent import SubagentSystem
from ecs_agent.systems.subagent_wait import SubagentWaitSystem
from ecs_agent.types import (
    CompletionResult,
    ConversationBranch,
    ConversationMessage,
    Message,
    SubagentConfig,
    SubagentSessionRecord,
    SubagentStreamDeltaEvent,
    SubagentStreamEndEvent,
    SubagentStreamStartEvent,
)


@pytest.mark.skipif(not os.environ.get("LLM_API_KEY"), reason="LLM_API_KEY not set")
async def test_responses_api_with_real_llm() -> None:
    """Test Responses API with real LLM provider.

    Note: DashScope may not support Responses API, so this test handles both
    Responses API and fallback to Chat Completions.
    """
    api_key = os.environ.get("LLM_API_KEY", "")
    base_url = os.environ.get(
        "LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    model = os.environ.get("LLM_MODEL", "qwen3.5-plus")

    world = World()

    # Create provider with Responses API enabled (will fallback to Chat Completions if not supported)
    provider = OpenAIProvider(
        config=ProviderConfig(
            provider_id="openai",
            base_url=base_url,
            api_key=api_key,
            api_format=ApiFormat.OPENAI_RESPONSES,
        ),
        model=model,
    )

    entity = world.create_entity()
    world.add_component(entity, LLMComponent(provider=provider, model=model))
    world.add_component(
        entity,
        ConversationComponent(
            messages=[Message(role="user", content="Say 'hello' and nothing else")]
        ),
    )

    world.register_system(ReasoningSystem(priority=0), priority=0)
    world.register_system(MemorySystem(), priority=10)
    world.register_system(ErrorHandlingSystem(priority=99), priority=99)

    runner = Runner()
    await runner.run(world, max_ticks=3)

    # Verify response was generated
    conv = world.get_component(entity, ConversationComponent)
    assert conv is not None
    assert len(conv.messages) >= 2  # user + assistant
    assert conv.messages[-1].role == "assistant"
    assert len(conv.messages[-1].content) > 0


async def test_tree_conversation_with_reasoning_system() -> None:
    """Test ConversationTreeComponent with ReasoningSystem integration."""
    world = World()

    # Create provider with deterministic response
    provider = FakeProvider(
        responses=[
            CompletionResult(
                message=Message(role="assistant", content="Test response 1")
            ),
            CompletionResult(
                message=Message(role="assistant", content="Test response 2")
            ),
        ]
    )

    entity = world.create_entity()

    # Initialize tree conversation with root message
    root_msg = ConversationMessage(
        id="msg_root",
        parent_message_id=None,
        role="user",
        content="Hello",
    )
    branch = ConversationBranch(
        branch_id="branch_main",
        leaf_message_id="msg_root",
    )

    world.add_component(
        entity,
        ConversationTreeComponent(
            messages={"msg_root": root_msg},
            current_branch_id="branch_main",
            branches={"branch_main": branch},
        ),
    )

    world.add_component(entity, LLMComponent(provider=provider, model="fake"))

    # Add flat conversation for reasoning system
    world.add_component(
        entity,
        ConversationComponent(messages=[Message(role="user", content="Hello")]),
    )

    world.register_system(ReasoningSystem(priority=0), priority=0)
    world.register_system(MemorySystem(), priority=10)
    world.register_system(ErrorHandlingSystem(priority=99), priority=99)

    runner = Runner()
    await runner.run(world, max_ticks=3)

    # Verify response was added to flat conversation
    conv = world.get_component(entity, ConversationComponent)
    assert conv is not None
    assert len(conv.messages) >= 2
    assert conv.messages[-1].role == "assistant"

    # Verify tree component still exists (not modified by default)
    tree = world.get_component(entity, ConversationTreeComponent)
    assert tree is not None
    assert tree.current_branch_id == "branch_main"


async def test_markdown_skill_install_and_use() -> None:
    """Test loading markdown Skill from SKILL.md fixture and using it."""
    # Create temporary SKILL.md fixture
    with tempfile.TemporaryDirectory() as tmpdir:
        skill_path = Path(tmpdir) / "SKILL.md"
        skill_path.write_text("""---
name: test-skill
tier: 1
description: Test skill for integration testing
tags: [test]
---

# Test Skill

This is a test skill.

## System Prompt

You are a test assistant.

## Tools

None
""")

        # Load skill
        skill = Skill(skill_path=skill_path)

        # Verify metadata
        assert skill.description == "Test skill for integration testing"

        # Verify content — system_prompt() is a method that returns full body
        assert "You are a test assistant" in skill.system_prompt()

        # Create world and entity with skill installed
        world = World()
        entity = world.create_entity()

        provider = FakeProvider(
            responses=[
                CompletionResult(
                    message=Message(role="assistant", content="I am a test assistant")
                )
            ]
        )

        # Apply skill to entity (system prompt)
        world.add_component(
            entity,
            LLMComponent(
                provider=provider, model="fake", system_prompt=skill.system_prompt()
            ),
        )
        world.add_component(
            entity,
            ConversationComponent(
                messages=[Message(role="user", content="Who are you?")]
            ),
        )

        world.register_system(ReasoningSystem(priority=0), priority=0)
        world.register_system(MemorySystem(), priority=10)
        world.register_system(ErrorHandlingSystem(priority=99), priority=99)

        runner = Runner()
        await runner.run(world, max_ticks=3)

        # Verify skill was used
        llm = world.get_component(entity, LLMComponent)
        assert llm is not None
        assert "You are a test assistant" in llm.system_prompt


async def test_subagent_delegation_end_to_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test full subagent delegation flow from parent to child."""
    import ecs_agent.systems.subagent_runtime as runtime_module

    monkeypatch.setattr(runtime_module, "_GLOBAL_SCHEDULER", None)

    world = World()

    # Create parent entity
    parent_entity = world.create_entity()

    # Create FakeProvider-based subagent config
    subagent_config = SubagentConfig(
        name="test-subagent",
        provider=FakeProvider(
            responses=[
                CompletionResult(
                    message=Message(role="assistant", content="Subagent result")
                )
            ]
        ),
        model="fake",
        system_prompt="You are a test subagent",
        max_ticks=5,
        skills=[],
    )

    # Register subagent
    registry = SubagentRegistryComponent(subagents={"test-subagent": subagent_config})
    world.add_component(parent_entity, registry)

    # Add LLM and conversation to parent
    world.add_component(
        parent_entity,
        LLMComponent(
            provider=FakeProvider(responses=[]),
            model="fake",
        ),
    )
    world.add_component(
        parent_entity,
        ConversationComponent(messages=[]),
    )

    # Add tool registry (SubagentSystem will populate subagent tool)
    world.add_component(parent_entity, ToolRegistryComponent(tools={}, handlers={}))
    world.add_component(parent_entity, MessageBusConfigComponent(request_timeout=1.0))

    # Register SubagentSystem
    world.register_system(SubagentSystem(priority=0), priority=0)
    world.register_system(MessageBusSystem(priority=5), priority=5)
    world.register_system(ErrorHandlingSystem(priority=99), priority=99)

    # Process to register subagent tool
    await world.process()

    # Get subagent handler
    tool_registry = world.get_component(parent_entity, ToolRegistryComponent)
    assert tool_registry is not None
    assert "subagent" in tool_registry.handlers

    subagent_handler = tool_registry.handlers["subagent"]

    # Call subagent
    result = await subagent_handler(category="test-subagent", prompt="Do something")

    # Verify result
    assert isinstance(result, str)
    assert result == "Subagent result"


async def test_subagent_queue_saturation_respects_global_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ecs_agent.systems.subagent_runtime as runtime_module

    monkeypatch.setattr(runtime_module, "_GLOBAL_SCHEDULER", None)

    world = World()
    parent = world.create_entity()
    release_running = asyncio.Event()

    async def blocking_completion(*args: object, **kwargs: object) -> CompletionResult:
        del args, kwargs
        return await _wait_then_complete(release_running)

    blocking_provider = FakeProvider(responses=[])
    blocking_provider.complete = AsyncMock(  # type: ignore[method-assign]
        side_effect=blocking_completion
    )

    world.add_component(parent, ToolRegistryComponent(tools={}, handlers={}))
    world.add_component(parent, SubagentSessionTableComponent(sessions={}))
    world.add_component(parent, MessageBusConfigComponent(request_timeout=1.0))
    world.add_component(
        parent,
        SubagentRegistryComponent(
            subagents={
                "worker": SubagentConfig(
                    name="worker",
                    provider=blocking_provider,
                    model="fake",
                    system_prompt="Block until released.",
                )
            }
        ),
    )

    system = SubagentSystem(priority=0, max_background_concurrency=2)
    system.install_subagent_control_tools(world, parent)
    world.register_system(system, priority=0)
    world.register_system(MessageBusSystem(priority=5), priority=5)
    world.register_system(ErrorHandlingSystem(priority=99), priority=99)

    await world.process()

    tools = world.get_component(parent, ToolRegistryComponent)
    assert tools is not None

    handler = tools.handlers["subagent"]
    payloads = [
        json.loads(
            await handler(
                category="worker",
                prompt=f"job-{index}",
                background=True,
            )
        )
        for index in range(3)
    ]

    await asyncio.sleep(0)

    status_handler = tools.handlers["subagent_status"]
    statuses = [
        json.loads(await status_handler(session_id=payload["session_id"]))
        for payload in payloads
    ]

    running_count = sum(
        1 for status in statuses if status["lifecycle_status"] == "running"
    )
    queued_statuses = [
        status for status in statuses if status["lifecycle_status"] == "queued"
    ]

    assert running_count == 2
    assert len(queued_statuses) == 1
    assert queued_statuses[0]["queue_position"] == 0

    release_running.set()
    for payload in payloads:
        result = json.loads(
            await tools.handlers["subagent_result"](
                session_id=payload["session_id"], timeout=2.0
            )
        )
        assert result["status"] == "success"
        assert result["lifecycle_status"] == "succeeded"


async def test_subagent_background_stream_events_are_visible_to_parent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ecs_agent.systems.subagent_runtime as runtime_module

    monkeypatch.setattr(runtime_module, "_GLOBAL_SCHEDULER", None)

    world = World()
    parent = world.create_entity()
    received: list[
        SubagentStreamStartEvent | SubagentStreamDeltaEvent | SubagentStreamEndEvent
    ] = []

    async def on_start(event: SubagentStreamStartEvent) -> None:
        received.append(event)

    async def on_delta(event: SubagentStreamDeltaEvent) -> None:
        received.append(event)

    async def on_end(event: SubagentStreamEndEvent) -> None:
        received.append(event)

    world.event_bus.subscribe(SubagentStreamStartEvent, on_start)
    world.event_bus.subscribe(SubagentStreamDeltaEvent, on_delta)
    world.event_bus.subscribe(SubagentStreamEndEvent, on_end)

    provider = FakeProvider(
        responses=[
            CompletionResult(
                message=Message(role="assistant", content="streamed integration output")
            )
        ]
    )

    world.add_component(parent, ToolRegistryComponent(tools={}, handlers={}))
    world.add_component(parent, SubagentSessionTableComponent(sessions={}))
    world.add_component(parent, MessageBusConfigComponent(request_timeout=1.0))
    world.add_component(
        parent,
        SubagentRegistryComponent(
            subagents={
                "stream-worker": SubagentConfig(
                    name="stream-worker",
                    provider=provider,
                    model="fake",
                    system_prompt="Return one short streamed answer.",
                )
            }
        ),
    )

    system = SubagentSystem(priority=0, max_background_concurrency=1)
    system.install_subagent_control_tools(world, parent)
    world.register_system(system, priority=0)
    world.register_system(MessageBusSystem(priority=5), priority=5)
    world.register_system(ErrorHandlingSystem(priority=99), priority=99)

    await world.process()

    tools = world.get_component(parent, ToolRegistryComponent)
    assert tools is not None

    launch_result = json.loads(
        await tools.handlers["subagent"](
            category="stream-worker",
            prompt="Stream the answer back to the parent.",
            background=True,
            stream=True,
        )
    )
    session_id = launch_result["session_id"]

    result = json.loads(
        await tools.handlers["subagent_result"](session_id=session_id, timeout=2.0)
    )
    assert result["status"] == "success"
    assert result["lifecycle_status"] == "succeeded"

    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert received
    assert isinstance(received[0], SubagentStreamStartEvent)
    assert received[0].session_id == session_id
    assert any(isinstance(event, SubagentStreamDeltaEvent) for event in received)
    assert any(isinstance(event, SubagentStreamEndEvent) for event in received)


async def test_subagent_wait_injects_notification_and_enables_explicit_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ecs_agent.systems.subagent_runtime as runtime_module

    monkeypatch.setattr(runtime_module, "_GLOBAL_SCHEDULER", None)

    world = World()
    parent = world.create_entity()

    provider = FakeProvider(
        responses=[
            CompletionResult(
                message=Message(
                    role="assistant",
                    content=(
                        "<subagent_background_result>\n"
                        "<summary>Cached parent summary.</summary>\n"
                        "<full_result>Full background result for the parent.</full_result>\n"
                        "</subagent_background_result>"
                    ),
                )
            )
        ]
    )

    world.add_component(
        parent,
        ConversationComponent(messages=[Message(role="user", content="launch worker")]),
    )
    world.add_component(parent, ToolRegistryComponent(tools={}, handlers={}))
    world.add_component(parent, SubagentSessionTableComponent(sessions={}))
    world.add_component(parent, MessageBusConfigComponent(request_timeout=1.0))
    world.add_component(
        parent,
        SubagentRegistryComponent(
            subagents={
                "wait-worker": SubagentConfig(
                    name="wait-worker",
                    provider=provider,
                    model="fake",
                    system_prompt="Return a wrapped background result.",
                )
            }
        ),
    )

    subagent_system = SubagentSystem(priority=-1, max_background_concurrency=1)
    wait_system = SubagentWaitSystem(priority=-5)
    subagent_system.install_subagent_control_tools(world, parent)
    world.register_system(wait_system, priority=-5)
    world.register_system(subagent_system, priority=-1)
    world.register_system(ErrorHandlingSystem(priority=99), priority=99)

    await world.process()

    tools = world.get_component(parent, ToolRegistryComponent)
    assert tools is not None

    launch_result = json.loads(
        await tools.handlers["subagent"](
            category="wait-worker",
            prompt="Do the background task.",
            background=True,
        )
    )
    session_id = launch_result["session_id"]

    wait_ack = await tools.handlers["subagent_wait"](session_ids=[session_id])
    assert (
        wait_ack
        == "Waiting for background subagents. Will be notified when they complete."
    )
    assert world.has_component(parent, SubagentWaitComponent)

    wait_task = asyncio.create_task(wait_system.process(world))
    await asyncio.wait_for(wait_task, timeout=2.0)

    assert not world.has_component(parent, SubagentWaitComponent)

    queue = world.get_component(parent, SubagentNotificationQueueComponent)
    assert queue is not None
    assert len(queue.notifications) == 1
    assert queue.notifications[0].session_id == session_id
    assert queue.notifications[0].terminal_status == "succeeded"
    assert queue.notifications[0].delivered_at is not None

    conversation = world.get_component(parent, ConversationComponent)
    assert conversation is not None
    system_messages = [
        message for message in conversation.messages if message.role == "system"
    ]
    assert len(system_messages) == 1
    assert "Background subagent updates:" in system_messages[0].content
    assert session_id in system_messages[0].content
    assert 'read_method="summary"' in system_messages[0].content

    summary_result = json.loads(
        await tools.handlers["subagent_result"](
            session_id=session_id,
            read_method="summary",
        )
    )
    assert summary_result["status"] == "success"
    assert summary_result["read_method"] == "summary"
    assert summary_result["inline_content"] == "Cached parent summary."

    full_result = json.loads(
        await tools.handlers["subagent_result"](
            session_id=session_id,
            read_method="full",
        )
    )
    assert full_result["status"] == "success"
    assert full_result["lifecycle_status"] == "succeeded"
    assert full_result["inline_content"] == "Full background result for the parent."


async def test_subagent_queued_session_survives_restore_and_reenqueue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ecs_agent.systems.subagent_runtime as runtime_module

    monkeypatch.setattr(runtime_module, "_GLOBAL_SCHEDULER", None)

    world = World()
    parent = world.create_entity()
    queued_provider = FakeProvider(
        responses=[
            CompletionResult(message=Message(role="assistant", content="queued done"))
        ]
    )

    world.add_component(parent, ToolRegistryComponent(tools={}, handlers={}))
    world.add_component(
        parent,
        SubagentSessionTableComponent(
            sessions={
                "queued-session": SubagentSessionRecord(
                    session_id="queued-session",
                    category="queued-worker",
                    prompt="Wait for restore.",
                    parent_entity_id=parent,
                    created_at="2026-04-05T10:00:00Z",
                    updated_at="2026-04-05T10:00:00Z",
                    background=True,
                    status="queued",
                ),
                "running-session": SubagentSessionRecord(
                    session_id="running-session",
                    category="blocking-worker",
                    prompt="Was running when checkpointed.",
                    parent_entity_id=parent,
                    created_at="2026-04-05T09:59:00Z",
                    updated_at="2026-04-05T10:01:00Z",
                    background=True,
                    status="running",
                    started_at="2026-04-05T10:00:10Z",
                ),
            }
        ),
    )
    world.add_component(parent, MessageBusConfigComponent(request_timeout=1.0))
    world.add_component(
        parent,
        SubagentRegistryComponent(
            subagents={
                "blocking-worker": SubagentConfig(
                    name="blocking-worker",
                    provider=queued_provider,
                    model="fake",
                    system_prompt="Block until released.",
                ),
                "queued-worker": SubagentConfig(
                    name="queued-worker",
                    provider=queued_provider,
                    model="fake",
                    system_prompt="Finish quickly once admitted.",
                ),
            }
        ),
    )

    serialized = WorldSerializer.to_dict(world)
    subagents = serialized["entities"][str(int(parent))]["SubagentRegistryComponent"][
        "subagents"
    ]
    subagents["queued-worker"]["provider"] = "queued-worker"
    subagents["blocking-worker"]["provider"] = "queued-worker"

    monkeypatch.setattr(runtime_module, "_GLOBAL_SCHEDULER", None)
    restored = WorldSerializer.from_dict(
        serialized,
        providers={
            "queued-worker": queued_provider,
        },
        tool_handlers={},
    )

    restored_system = SubagentSystem(priority=0, max_background_concurrency=1)
    restored_system.install_subagent_control_tools(restored, parent)

    async def fake_execute_core(
        world_arg: World,
        parent_entity_id: object,
        subagent_name: str,
        task: str,
        correlation_id: str,
        traceparent: str,
        config_snapshot: SubagentConfig,
    ) -> tuple[str, bool, str | None]:
        del (
            world_arg,
            parent_entity_id,
            subagent_name,
            task,
            correlation_id,
            traceparent,
        )
        del config_snapshot
        return ("queued done", True, None)

    restored_system._execute_subagent_core = fake_execute_core  # type: ignore[method-assign]
    await restored_system.process(restored)

    restored_tools = restored.get_component(parent, ToolRegistryComponent)
    assert restored_tools is not None

    restored_status = json.loads(
        await restored_tools.handlers["subagent_status"](session_id="queued-session")
    )
    assert restored_status["session_id"] == "queued-session"
    assert restored_status["lifecycle_status"] in {"queued", "running", "succeeded"}

    queued_result = json.loads(
        await restored_tools.handlers["subagent_result"](
            session_id="queued-session", timeout=2.0
        )
    )

    assert queued_result["status"] == "success"
    assert queued_result["lifecycle_status"] == "succeeded"

    restored_table = restored.get_component(parent, SubagentSessionTableComponent)
    assert restored_table is not None
    assert restored_table.sessions["queued-session"].result_excerpt == "queued done"

    blocking_result = json.loads(
        await restored_tools.handlers["subagent_result"](
            session_id="running-session", timeout=None
        )
    )
    assert blocking_result["status"] == "terminal"
    assert blocking_result["lifecycle_status"] == "failed"
    assert blocking_result["error"] == "restored_without_live_task_handle"


async def _wait_then_complete(
    release_event: asyncio.Event,
    *,
    content: str = "done",
) -> CompletionResult:
    await release_event.wait()
    return CompletionResult(message=Message(role="assistant", content=content))


async def test_enhanced_logging_in_system_execution() -> None:
    """Test enhanced logging with caller info in system execution."""
    # Configure logging with enhanced features
    configure_logging(level="INFO", colors=False)

    world = World()
    entity = world.create_entity()

    provider = FakeProvider(
        responses=[
            CompletionResult(
                message=Message(role="assistant", content="Logged response")
            )
        ]
    )

    world.add_component(entity, LLMComponent(provider=provider, model="fake"))
    world.add_component(
        entity,
        ConversationComponent(messages=[Message(role="user", content="Test")]),
    )

    world.register_system(ReasoningSystem(priority=0), priority=0)
    world.register_system(MemorySystem(), priority=10)
    world.register_system(ErrorHandlingSystem(priority=99), priority=99)

    # Get logger and verify it works
    logger = get_logger("test_integration")
    logger.info("test_event", entity_id=entity, message="Testing enhanced logging")

    runner = Runner()
    await runner.run(world, max_ticks=3)

    # Verify execution completed (logging is tested via output, not assertions)
    conv = world.get_component(entity, ConversationComponent)
    assert conv is not None
    assert len(conv.messages) >= 2


async def test_all_new_components_serializable() -> None:
    """Test that all new components can be serialized and deserialized."""
    world = World()
    entity = world.create_entity()

    # Add all new components
    from ecs_agent.components import ResponsesAPIStateComponent

    # ConversationTreeComponent
    root_msg = ConversationMessage(
        id="msg_1", parent_message_id=None, role="user", content="Test"
    )
    branch = ConversationBranch(branch_id="branch_1", leaf_message_id="msg_1")
    world.add_component(
        entity,
        ConversationTreeComponent(
            messages={"msg_1": root_msg},
            current_branch_id="branch_1",
            branches={"branch_1": branch},
        ),
    )

    # ResponsesAPIStateComponent
    world.add_component(
        entity,
        ResponsesAPIStateComponent(
            previous_response_id="resp_122",
        ),
    )

    # SubagentRegistryComponent
    subagent_config = SubagentConfig(
        name="test",
        provider=FakeProvider(responses=[]),
        model="fake",
        system_prompt="",
        max_ticks=10,
        skills=[],
    )
    world.add_component(
        entity,
        SubagentRegistryComponent(subagents={"test": subagent_config}),
    )

    # Serialize
    serialized = WorldSerializer.to_dict(world)

    # Verify components are in serialized data
    entity_data = serialized["entities"][str(entity)]
    assert "ConversationTreeComponent" in entity_data
    assert "ResponsesAPIStateComponent" in entity_data
    assert "SubagentRegistryComponent" in entity_data

    # Deserialize
    restored = WorldSerializer.from_dict(serialized, providers={}, tool_handlers={})

    # Verify all components restored
    tree = restored.get_component(entity, ConversationTreeComponent)
    assert tree is not None
    assert tree.current_branch_id == "branch_1"

    responses_api = restored.get_component(entity, ResponsesAPIStateComponent)
    assert responses_api is not None
    assert responses_api.previous_response_id == "resp_122"

    registry = restored.get_component(entity, SubagentRegistryComponent)
    assert registry is not None
    assert "test" in registry.subagents

    # Verify SubagentConfig inheritance_policy is serializable
    restored_config = registry.subagents["test"]
    assert restored_config.inheritance_policy is not None
    assert restored_config.inheritance_policy.enabled is True
    assert restored_config.inheritance_policy.inherit_system_prompt is True
    assert restored_config.inheritance_policy.inherit_tools == []
    assert restored_config.inheritance_policy.inherit_permissions is False
    assert restored_config.inheritance_policy.tool_conflict_policy == "skip"
    assert restored_config.inheritance_policy.missing_skill_policy == "warn"


def test_subagent_doc_consistency_stale_symbols() -> None:
    """Fail if docs reference non-existent symbols.

    This test detects stale documentation that refers to symbols that were
    removed or renamed in the codebase. It scans docs/features/subagent.md
    for references to DELEGATE_TOOL_SCHEMA and delegate_tool_handler, which
    do not exist in the public API (SubagentSystem auto-registers subagent tool).
    """
    doc_path = Path("docs/features/subagent.md")
    assert doc_path.exists(), f"Doc file not found: {doc_path}"

    doc_content = doc_path.read_text()

    # Check for stale symbol: DELEGATE_TOOL_SCHEMA
    assert "DELEGATE_TOOL_SCHEMA" not in doc_content, (
        "Docs reference DELEGATE_TOOL_SCHEMA which does not exist. "
        "SubagentSystem auto-registers subagent tool inline. "
        "See docs/features/subagent.md line ~78."
    )

    # Check for stale symbol: delegate_tool_handler
    assert "delegate_tool_handler" not in doc_content, (
        "Docs reference delegate_tool_handler which does not exist. "
        "The subagent tool is auto-registered by SubagentSystem without manual handler setup. "
        "See docs/features/subagent.md lines ~102, ~107."
    )


def test_subagent_doc_consistency_event_subscriptions() -> None:
    """Fail if docs use outdated string-topic event subscriptions.

    This test detects stale event subscription patterns in subagent docs.
    Modern API uses typed event classes (e.g., world.event_bus.subscribe(DelegationStartedEvent, handler)),
    not string topic names (e.g., world.event_bus.subscribe('delegation_started', handler)).
    """
    doc_path = Path("docs/features/subagent.md")
    assert doc_path.exists(), f"Doc file not found: {doc_path}"

    doc_content = doc_path.read_text()

    # Check for outdated string-topic subscriptions
    stale_patterns = [
        (
            'world.event_bus.subscribe("delegation_started"',
            "subscription with string 'delegation_started'",
        ),
        (
            'world.event_bus.subscribe("delegation_completed"',
            "subscription with string 'delegation_completed'",
        ),
    ]

    for stale_pattern, description in stale_patterns:
        assert stale_pattern not in doc_content, (
            f"Docs use outdated {description}. "
            f"Should use typed event classes instead: "
            f"world.event_bus.subscribe(DelegationStartedEvent, handler) or "
            f"world.event_bus.subscribe(DelegationCompletedEvent, handler). "
            f"See EventBus.subscribe() signature in src/ecs_agent/core/event_bus.py."
        )


def test_subagent_doc_consistency_installer_api() -> None:
    """Fail if docs do not mention subagent tool auto-registration.

    This test verifies that documentation explains the automatic subagent tool
    registration by SubagentSystem.
    """
    doc_path = Path("docs/features/subagent.md")
    if not doc_path.exists():
        pytest.skip(f"Subagent docs not found at {doc_path}")

    doc_content = doc_path.read_text()
    doc_content_lower = doc_content.lower()

    # Check for subagent tool auto-registration mention (flexible substring check)
    has_subagent_mention = "subagent" in doc_content
    has_auto_mention = "auto" in doc_content_lower or "automatic" in doc_content_lower
    has_register_mention = "register" in doc_content_lower

    assert has_subagent_mention and (has_auto_mention or has_register_mention), (
        "Docs should clearly mention the subagent tool and its automatic registration. "
        "Expected to find 'subagent' (tool name) and either 'auto' or 'register' (mechanism). "
        "SubagentSystem auto-registers the subagent tool for entities with both "
        "SubagentRegistryComponent and ToolRegistryComponent."
    )

    # Check for backward compatibility mention (auto-registration without manual setup)
    has_backward_compat = (
        ("auto" in doc_content_lower or "automatic" in doc_content_lower)
        and "register" in doc_content_lower
    ) or "without manual" in doc_content_lower

    assert has_backward_compat, (
        "Docs should mention automatic subagent tool registration. "
        "Users should not need to manually register the subagent tool via ToolRegistryComponent. "
        "SubagentSystem handles this automatically."
    )


def test_subagent_doc_consistency_inheritance_policy() -> None:
    """Fail if docs do not explain InheritancePolicy configuration.

    This test verifies that documentation explains the InheritancePolicy fields
    and behaviors, including enabled flag, inheritance toggles, tool conflict handling,
    and skill-level policies.
    """
    doc_path = Path("docs/features/subagent.md")
    if not doc_path.exists():
        pytest.skip(f"Subagent docs not found at {doc_path}")

    doc_content = doc_path.read_text()
    doc_content_lower = doc_content.lower()

    # Check for inheritance policy mention
    assert "inherit" in doc_content_lower or "policy" in doc_content_lower, (
        "Docs should mention InheritancePolicy or inheritance behavior. "
        "SubagentConfig includes an inheritance_policy field that controls what parent "
        "capabilities are inherited by child agents."
    )

    # Check for inherit_tools or tool inheritance mention
    has_tool_inheritance = "inherit_tool" in doc_content_lower or (
        "tool" in doc_content_lower and "inherit" in doc_content_lower
    )

    assert has_tool_inheritance, (
        "Docs should explain tool inheritance. "
        "InheritancePolicy.inherit_tools field controls which parent tools are inherited by child."
    )

    # Check for conflict resolution mention
    has_conflict_mention = (
        "conflict" in doc_content_lower
        or "override" in doc_content_lower
        or "skip" in doc_content_lower
    )

    assert has_conflict_mention, (
        "Docs should explain conflict resolution for tool inheritance. "
        "InheritancePolicy.tool_conflict_policy controls resolution (skip/error/override)."
    )
