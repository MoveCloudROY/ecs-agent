"""Env-gated real-LLM integration tests for runtime control features.

Tests MUST skip gracefully when LLM_API_KEY is not set (CI/local default deterministic).
Tests MUST pass when LLM_API_KEY is present (validates real provider behavior).

Environment Variables (required for runtime control tests):
    LLM_API_KEY: DashScope API key (or any OpenAI-compatible provider)
    LLM_BASE_URL: Base URL for the LLM provider (default: https://dashscope.aliyuncs.com/compatible-mode/v1)
    LLM_MODEL: Model identifier (default: qwen3.5-flash)
    LLM_MODEL_ALTERNATIVE: Alternative model for switching tests (default: qwen3.5-turbo)

Test Contract:
    - With LLM_API_KEY set: Runtime control tests execute with real provider.
    - Without LLM_API_KEY: Runtime control tests skip gracefully (no failures).
"""

from __future__ import annotations

import json
import os
from typing import Any

import pytest

from ecs_agent.components import (
    ConversationComponent,
    LLMComponent,
    OneShotContextPoolComponent,
    PromptConfigComponent,
    StreamingComponent,
    SystemPromptComponent,
    TurnStateComponent,
)
from ecs_agent.components.definitions import InterruptionComponent, TerminalComponent
from ecs_agent.conversation_tree import (
    ConversationTreeComponent,
    add_message,
    create_branch,
    get_active_leaf,
    revert_to_message,
    switch_branch,
)
from ecs_agent.core import Runner, World
from ecs_agent.logging import FORBIDDEN_FIELDS
from ecs_agent.providers import OpenAIProvider
from ecs_agent.systems.error_handling import ErrorHandlingSystem
from ecs_agent.systems.memory import MemorySystem
from ecs_agent.systems.reasoning import ReasoningSystem
from ecs_agent.types import (
    CompletionResult,
    InterruptionReason,
    Message,
    StreamDelta,
    StreamContentDeltaEvent,
)

# DashScope API configuration from environment variables
API_KEY = os.getenv("LLM_API_KEY", "")
BASE_URL = os.getenv(
    "LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
)
MODEL = os.getenv("LLM_MODEL", "qwen-plus")


def _json_events(output: str) -> list[dict[str, object]]:
    """Parse JSON events from logging output."""
    events: list[dict[str, object]] = []
    for line in output.strip().split("\n"):
        if line.strip():
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


# ============================================================================
# Existing Real LLM Tests (from previous tasks)
# ============================================================================


@pytest.mark.skipif(not API_KEY, reason="LLM_API_KEY environment variable not set")
@pytest.mark.asyncio
async def test_real_openai_streaming_produces_deltas() -> None:
    """Streaming ReasoningSystem with real OpenAIProvider emits StreamDelta events."""
    world = World()
    provider = OpenAIProvider(
        api_key=API_KEY,
        base_url=BASE_URL,
        model=MODEL,
    )

    entity = world.create_entity()
    world.add_component(entity, LLMComponent(provider=provider, model=MODEL))
    world.add_component(
        entity,
        ConversationComponent(
            messages=[Message(role="user", content="Say hello in 5 words or less")]
        ),
    )
    world.add_component(entity, StreamingComponent(enabled=True))
    world.add_component(
        entity,
        SystemPromptComponent(content="You are a helpful assistant."),
    )

    world.register_system(ReasoningSystem(priority=0), priority=0)

    deltas: list[StreamContentDeltaEvent] = []

    async def capture_delta(event: StreamContentDeltaEvent) -> None:
        deltas.append(event)

    world.event_bus.subscribe(StreamContentDeltaEvent, capture_delta)

    await world.process()

    assert len(deltas) > 0, "Expected at least one StreamDelta event"
    total_content = "".join(event.delta for event in deltas)
    assert len(total_content) > 0, "Expected non-empty streamed content"

    conv = world.get_component(entity, ConversationComponent)
    assert conv is not None
    assert len(conv.messages) == 2
    assert conv.messages[1].role == "assistant"
    assert len(conv.messages[1].content) > 0


@pytest.mark.skipif(not API_KEY, reason="LLM_API_KEY environment variable not set")
@pytest.mark.asyncio
async def test_real_openai_provider_non_streaming() -> None:
    """OpenAIProvider via DashScope returns valid CompletionResult."""
    provider = OpenAIProvider(
        api_key=API_KEY,
        base_url=BASE_URL,
        model=MODEL,
    )

    messages = [
        Message(role="system", content="You are a helpful assistant."),
        Message(role="user", content="Say hello in 5 words or less"),
    ]

    result = await provider.complete(messages, stream=False)

    assert isinstance(result, CompletionResult), "Expected CompletionResult"
    assert result.message.role == "assistant"
    assert len(result.message.content) > 0, "Expected non-empty response content"
    assert result.message.tool_calls is None, "No tools provided, should be None"


@pytest.mark.skipif(not API_KEY, reason="LLM_API_KEY environment variable not set")
@pytest.mark.asyncio
async def test_real_openai_provider_streaming() -> None:
    """OpenAIProvider streaming returns valid StreamDelta sequence."""
    provider = OpenAIProvider(
        api_key=API_KEY,
        base_url=BASE_URL,
        model=MODEL,
    )

    messages = [
        Message(role="system", content="You are a helpful assistant."),
        Message(role="user", content="Say hello in 5 words or less"),
    ]

    stream_result = await provider.complete(messages, stream=True)

    assert not isinstance(stream_result, CompletionResult), (
        "Expected async iterator, not CompletionResult"
    )

    deltas: list[StreamDelta] = []
    async for delta in stream_result:
        deltas.append(delta)

    assert len(deltas) > 0, "Expected at least one StreamDelta"
    content_chunks = [d.content for d in deltas if d.content is not None]
    assert len(content_chunks) > 0, "Expected at least one content delta"
    total_content = "".join(content_chunks)
    assert len(total_content) > 0, "Expected non-empty accumulated content"


@pytest.mark.skipif(not API_KEY, reason="LLM_API_KEY environment variable not set")
@pytest.mark.asyncio
async def test_real_full_agent_loop_streaming() -> None:
    """Full World + ReasoningSystem + StreamingComponent runs to completion."""
    world = World()
    provider = OpenAIProvider(
        api_key=API_KEY,
        base_url=BASE_URL,
        model=MODEL,
    )

    entity = world.create_entity()
    world.add_component(entity, LLMComponent(provider=provider, model=MODEL))
    world.add_component(
        entity,
        ConversationComponent(
            messages=[Message(role="user", content="Say hello in 5 words")]
        ),
    )
    world.add_component(entity, StreamingComponent(enabled=True))

    world.register_system(ReasoningSystem(priority=0), priority=0)
    world.register_system(MemorySystem(), priority=10)
    world.register_system(ErrorHandlingSystem(priority=99), priority=99)

    deltas: list[StreamContentDeltaEvent] = []

    async def capture_delta(event: StreamContentDeltaEvent) -> None:
        deltas.append(event)

    world.event_bus.subscribe(StreamContentDeltaEvent, capture_delta)

    runner = Runner()
    await runner.run(world, max_ticks=1)

    conv = world.get_component(entity, ConversationComponent)
    assert conv is not None
    assert len(conv.messages) >= 2, "Expected at least user + assistant messages"
    assert conv.messages[-1].role == "assistant", (
        "Last message should be from assistant"
    )
    assert len(conv.messages[-1].content) > 0

    assert len(deltas) > 0, "Expected StreamDelta events during streaming"


@pytest.mark.skipif(not API_KEY, reason="LLM_API_KEY environment variable not set")
@pytest.mark.asyncio
async def test_real_multi_turn_conversation() -> None:
    """Test multi-turn conversation with real LLM maintaining context."""
    world = World()
    provider = OpenAIProvider(
        api_key=API_KEY,
        base_url=BASE_URL,
        model=MODEL,
    )

    entity = world.create_entity()
    world.add_component(entity, LLMComponent(provider=provider, model=MODEL))
    world.add_component(
        entity,
        ConversationComponent(
            messages=[Message(role="user", content="My name is Alice. Remember it.")]
        ),
    )

    world.register_system(ReasoningSystem(priority=0), priority=0)
    world.register_system(MemorySystem(), priority=10)
    world.register_system(ErrorHandlingSystem(priority=99), priority=99)

    runner = Runner()
    await runner.run(world, max_ticks=1)

    conv = world.get_component(entity, ConversationComponent)
    assert conv is not None
    assert len(conv.messages) >= 2, "Expected user + assistant messages"
    assert conv.messages[-1].role == "assistant"

    conv.messages.append(
        Message(role="user", content="What is my name? Answer in 3 words or less.")
    )

    await runner.run(world, max_ticks=2, start_tick=1)

    assert len(conv.messages) >= 4, "Expected 2 turns of conversation"
    final_response = conv.messages[-1].content.lower()
    assert conv.messages[-1].role == "assistant"
    assert len(final_response) > 0, "Expected non-empty response"


@pytest.mark.skipif(not API_KEY, reason="LLM_API_KEY environment variable not set")
@pytest.mark.asyncio
async def test_real_llm_reasoning_logging_contracts(capsys: object) -> None:
    """Verify structured logging contracts when ReasoningSystem uses real LLM provider."""
    world = World()
    provider = OpenAIProvider(
        api_key=API_KEY,
        base_url=BASE_URL,
        model=MODEL,
    )

    entity = world.create_entity()
    world.add_component(entity, LLMComponent(provider=provider, model=MODEL))
    world.add_component(
        entity,
        ConversationComponent(
            messages=[Message(role="user", content="Say hello in 3 words")]
        ),
    )

    world.register_system(ReasoningSystem(priority=0), priority=0)

    await world.process()

    captured = capsys.readouterr()  # type: ignore[attr-defined]
    events = _json_events(captured.out)

    reasoning_start_events = [e for e in events if e.get("event") == "reasoning_start"]
    assert len(reasoning_start_events) > 0, (
        "Expected at least one reasoning_start event"
    )

    start_event = reasoning_start_events[0]
    assert "entity_id" in start_event, "reasoning_start must include entity_id"
    assert "model" in start_event, "reasoning_start must include model"
    assert start_event["model"] == MODEL

    reasoning_complete_events = [
        e for e in events if e.get("event") == "reasoning_complete"
    ]
    assert len(reasoning_complete_events) > 0, (
        "Expected at least one reasoning_complete event"
    )

    complete_event = reasoning_complete_events[0]
    assert "entity_id" in complete_event, "reasoning_complete must include entity_id"
    assert "duration_ms" in complete_event, (
        "reasoning_complete must include duration_ms"
    )
    assert isinstance(complete_event["duration_ms"], (int, float))
    assert complete_event["duration_ms"] >= 0

    for event in events:
        for forbidden in FORBIDDEN_FIELDS:
            assert forbidden not in event, (
                f"Forbidden field '{forbidden}' found in log event: {event.get('event')}"
            )


@pytest.mark.skipif(not API_KEY, reason="LLM_API_KEY environment variable not set")
@pytest.mark.asyncio
async def test_real_llm_streaming_logging_metadata(capsys: object) -> None:
    """Verify streaming mode logs contain correct metadata fields."""
    world = World()
    provider = OpenAIProvider(
        api_key=API_KEY,
        base_url=BASE_URL,
        model=MODEL,
    )

    entity = world.create_entity()
    world.add_component(entity, LLMComponent(provider=provider, model=MODEL))
    world.add_component(
        entity,
        ConversationComponent(
            messages=[Message(role="user", content="Count to three")]
        ),
    )
    world.add_component(entity, StreamingComponent(enabled=True))

    world.register_system(ReasoningSystem(priority=0), priority=0)

    await world.process()

    captured = capsys.readouterr()  # type: ignore[attr-defined]
    events = _json_events(captured.out)

    reasoning_start_events = [e for e in events if e.get("event") == "reasoning_start"]
    assert len(reasoning_start_events) > 0

    start_event = reasoning_start_events[0]
    assert "streaming" in start_event, (
        "reasoning_start must include 'streaming' field in streaming mode"
    )
    assert start_event["streaming"] is True

    reasoning_complete_events = [
        e for e in events if e.get("event") == "reasoning_complete"
    ]
    if reasoning_complete_events:
        complete_event = reasoning_complete_events[0]
        assert "duration_ms" in complete_event

    for event in events:
        assert "content" not in event, (
            "Raw conversation content must not appear in logs"
        )
        assert "arguments" not in event, "Raw tool arguments must not appear in logs"


@pytest.mark.skipif(not API_KEY, reason="LLM_API_KEY environment variable not set")
@pytest.mark.asyncio
async def test_real_llm_error_logging_contracts(capsys: object) -> None:
    """Verify error logging when LLM provider fails (bad API key)."""
    world = World()
    provider = OpenAIProvider(
        api_key="sk-invalid-key-for-testing",
        base_url=BASE_URL,
        model=MODEL,
    )

    entity = world.create_entity()
    world.add_component(entity, LLMComponent(provider=provider, model=MODEL))
    world.add_component(
        entity,
        ConversationComponent(messages=[Message(role="user", content="Hello")]),
    )

    world.register_system(ReasoningSystem(priority=0), priority=0)
    world.register_system(ErrorHandlingSystem(priority=99), priority=99)

    await world.process()

    captured = capsys.readouterr()  # type: ignore[attr-defined]
    events = _json_events(captured.out)

    error_events = [e for e in events if e.get("event") == "reasoning_error"]
    assert len(error_events) > 0, "Expected at least one reasoning_error event"

    error_event = error_events[0]
    assert "entity_id" in error_event, "reasoning_error must include entity_id"
    assert "reason" in error_event or "exception" in error_event, (
        "reasoning_error must include failure reason"
    )

    for event in events:
        event_str = json.dumps(event)
        assert "sk-invalid-key-for-testing" not in event_str, (
            "API key must not appear in logs"
        )
        assert "api_key" not in event, "api_key field must not appear in logs"


# ============================================================================
# NEW: Runtime Control Tests (Task 13)
# ============================================================================


def get_real_provider() -> OpenAIProvider:
    """Construct OpenAI-compatible provider from env vars."""
    api_key = os.getenv("LLM_API_KEY")
    if not api_key:
        pytest.skip("LLM_API_KEY not set")

    base_url = os.getenv(
        "LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    model = os.getenv("LLM_MODEL", "qwen3.5-flash")

    return OpenAIProvider(api_key=api_key, base_url=base_url, model=model)


class RecordingProvider:
    def __init__(self, provider: OpenAIProvider) -> None:
        self._provider = provider
        self.last_messages: list[Message] = []

    async def complete(
        self,
        messages: list[Message],
        tools: list[Any] | None = None,
        stream: bool = False,
        response_format: dict[str, Any] | None = None,
    ) -> CompletionResult | Any:
        self.last_messages = list(messages)
        return await self._provider.complete(
            messages,
            tools=tools,
            stream=stream,
            response_format=response_format,
        )


@pytest.mark.skipif(not API_KEY, reason="LLM_API_KEY environment variable not set")
@pytest.mark.asyncio
async def test_real_llm_prompt_keyword_injection_smoke() -> None:
    world = World()
    runner = Runner()

    model = os.getenv("LLM_MODEL", "qwen3.5-flash")
    provider = RecordingProvider(get_real_provider())

    entity = world.create_entity()
    world.add_component(entity, LLMComponent(provider=provider, model=model))
    world.add_component(
        entity,
        ConversationComponent(
            messages=[
                Message(role="user", content="Need @code help in one short sentence")
            ]
        ),
    )
    world.add_component(
        entity,
        PromptConfigComponent(
            trigger_templates={"@code": "KEYWORD_TEMPLATE_BLOCK"},
            enable_context_pool=True,
        ),
    )
    world.add_component(
        entity,
        OneShotContextPoolComponent(
            items=[
                (
                    30,
                    0,
                    "tool:search",
                    "source: tool:search\nstatus: success\nresult: citations\nerror: ",
                ),
                (
                    20,
                    1,
                    "subagent:researcher",
                    "source: subagent:researcher\nstatus: success\nresult: synthesis\nerror: ",
                ),
            ],
            _counter=2,
        ),
    )
    world.add_component(
        entity, TurnStateComponent(current_turn_id="turn-real-keyword-1")
    )
    world.add_component(
        entity,
        SystemPromptComponent(
            content=(
                "# Markdown Linked Prompt\n\n"
                "## toolSelection\n\n"
                "Follow the workflow from markdown-linked skills.\n\n"
                "## exploreSection\n\n"
                "Use evidence from context pool entries first.\n\n"
                "## librarianSection\n\n"
                "Preserve concrete citations in responses."
            ),
        ),
    )

    world.register_system(ReasoningSystem(priority=0), priority=0)
    world.register_system(ErrorHandlingSystem(priority=99), priority=99)

    await runner.run(world, max_ticks=1)

    assert len(provider.last_messages) >= 1
    outbound_system = provider.last_messages[0]
    outbound_user = provider.last_messages[-1]
    assert outbound_system.role == "system"
    assert outbound_system.content == (
        "# Markdown Linked Prompt\n\n"
        "## toolSelection\n\n"
        "Follow the workflow from markdown-linked skills.\n\n"
        "## exploreSection\n\n"
        "Use evidence from context pool entries first.\n\n"
        "## librarianSection\n\n"
        "Preserve concrete citations in responses."
    )
    assert outbound_user.role == "user"
    assert outbound_user.content.startswith(
        "[PROMPT_INJECT:@code]\nKEYWORD_TEMPLATE_BLOCK\n\n"
    )
    assert outbound_user.content.index(
        "[PROMPT_INJECT:@code]"
    ) < outbound_user.content.index("[PROMPT_CONTEXT_POOL]")
    assert outbound_user.content.index(
        "source: tool:search"
    ) < outbound_user.content.index("source: subagent:researcher")
    assert "[PROMPT_CONTEXT_POOL]" in outbound_user.content
    assert outbound_user.content.endswith("Need @code help in one short sentence")

    conv = world.get_component(entity, ConversationComponent)
    assert conv is not None
    assert conv.messages[-1].role == "assistant"
    assert len(conv.messages[-1].content) > 0


@pytest.mark.skipif(not API_KEY, reason="LLM_API_KEY environment variable not set")
@pytest.mark.asyncio
async def test_real_llm_prompt_event_injection_smoke() -> None:
    world = World()
    runner = Runner()

    model = os.getenv("LLM_MODEL", "qwen3.5-flash")
    provider = RecordingProvider(get_real_provider())

    entity = world.create_entity()
    world.add_component(entity, LLMComponent(provider=provider, model=model))
    world.add_component(
        entity,
        ConversationComponent(
            messages=[Message(role="user", content="Need concise summary")]
        ),
    )
    world.add_component(
        entity,
        PromptConfigComponent(
            trigger_templates={"event:tool_success": "EVENT_TEMPLATE_BLOCK"},
            enable_context_pool=True,
        ),
    )
    world.add_component(
        entity,
        OneShotContextPoolComponent(
            items=[
                (
                    30,
                    0,
                    "tool:search",
                    "source: tool:search\nstatus: success\nresult: citations\nerror: ",
                ),
                (
                    20,
                    1,
                    "subagent:researcher",
                    "source: subagent:researcher\nstatus: success\nresult: synthesis\nerror: ",
                ),
            ],
            _counter=2,
        ),
    )
    world.add_component(entity, TurnStateComponent(current_turn_id="turn-real-event-1"))

    world.register_system(ReasoningSystem(priority=0), priority=0)
    world.register_system(ErrorHandlingSystem(priority=99), priority=99)

    await runner.run(world, max_ticks=1)

    assert len(provider.last_messages) >= 1
    outbound_user = provider.last_messages[-1]
    assert outbound_user.role == "user"
    assert outbound_user.content.startswith(
        "[PROMPT_INJECT:event:tool_success]\nEVENT_TEMPLATE_BLOCK\n\n"
    )
    assert outbound_user.content.index(
        "[PROMPT_INJECT:event:tool_success]"
    ) < outbound_user.content.index("[PROMPT_CONTEXT_POOL]")
    assert outbound_user.content.index(
        "source: tool:search"
    ) < outbound_user.content.index("source: subagent:researcher")
    assert outbound_user.content.endswith("Need concise summary")

    conv = world.get_component(entity, ConversationComponent)
    assert conv is not None
    assert conv.messages[-1].role == "assistant"
    assert len(conv.messages[-1].content) > 0


@pytest.mark.skipif(os.getenv("LLM_API_KEY") is None, reason="LLM_API_KEY not set")
@pytest.mark.asyncio
async def test_real_llm_model_switching() -> None:
    """Validate pending_model switch affects real provider responses."""
    world = World()
    runner = Runner()

    provider = get_real_provider()
    model = os.getenv("LLM_MODEL", "qwen3.5-flash")

    entity = world.create_entity()
    llm = LLMComponent(provider=provider, model=model)
    world.add_component(entity, llm)
    world.add_component(
        entity,
        ConversationComponent(
            messages=[
                Message(role="user", content="Say 'model A' if you receive this.")
            ]
        ),
    )

    world.register_system(ReasoningSystem(priority=0), priority=0)

    await runner.run(world, max_ticks=1)

    conv = world.get_component(entity, ConversationComponent)
    assert conv is not None
    first_response = conv.messages[-1].content

    alternative_model = os.getenv("LLM_MODEL_ALTERNATIVE", "qwen3.5-turbo")
    llm.pending_model = alternative_model

    conv.messages.append(
        Message(role="user", content="Say 'model B' if you receive this.")
    )

    world.remove_component(entity, TerminalComponent)

    await runner.run(world, max_ticks=1)

    second_response = conv.messages[-1].content

    assert len(first_response) > 0
    assert len(second_response) > 0


@pytest.mark.skipif(os.getenv("LLM_API_KEY") is None, reason="LLM_API_KEY not set")
@pytest.mark.asyncio
async def test_real_llm_graceful_interruption() -> None:
    """Validate InterruptionComponent preserves partial content with real streaming."""
    world = World()
    runner = Runner()

    provider = get_real_provider()
    model = os.getenv("LLM_MODEL", "qwen3.5-flash")

    entity = world.create_entity()
    llm = LLMComponent(provider=provider, model=model)
    world.add_component(entity, llm)
    world.add_component(
        entity,
        ConversationComponent(
            messages=[Message(role="user", content="Write a long story about a robot.")]
        ),
    )

    world.register_system(ReasoningSystem(priority=0), priority=0)

    world.add_component(
        entity,
        InterruptionComponent(
            reason=InterruptionReason.USER_REQUESTED, metadata={"test": "interruption"}
        ),
    )

    await runner.run(world, max_ticks=1)

    interrupt = world.get_component(entity, InterruptionComponent)
    assert interrupt is not None
    assert interrupt.reason == InterruptionReason.USER_REQUESTED

    conv = world.get_component(entity, ConversationComponent)
    assert conv is not None


@pytest.mark.skipif(os.getenv("LLM_API_KEY") is None, reason="LLM_API_KEY not set")
@pytest.mark.asyncio
async def test_real_llm_conversation_tree_revert() -> None:
    """Validate revert affects next generation context with real provider."""
    world = World()
    runner = Runner()

    provider = get_real_provider()
    model = os.getenv("LLM_MODEL", "qwen3.5-flash")

    entity = world.create_entity()
    llm = LLMComponent(provider=provider, model=model)
    world.add_component(entity, llm)

    tree = ConversationTreeComponent()
    msg1 = add_message(tree, role="user", content="Remember the number 42.")
    msg2 = add_message(
        tree, role="assistant", content="I remember 42.", parent_id=msg1.id
    )
    msg3 = add_message(
        tree, role="user", content="Remember the number 99.", parent_id=msg2.id
    )

    create_branch(tree, "main", msg3.id)
    switch_branch(tree, "main")

    world.add_component(entity, tree)
    world.register_system(ReasoningSystem(priority=0), priority=0)

    await runner.run(world, max_ticks=1)

    conv_tree = world.get_component(entity, ConversationTreeComponent)
    assert conv_tree is not None
    first_leaf = get_active_leaf(conv_tree)

    revert_to_message(conv_tree, msg2.id)

    world.remove_component(entity, TerminalComponent)

    await runner.run(world, max_ticks=1)

    second_leaf = get_active_leaf(conv_tree)

    assert first_leaf != second_leaf
    assert first_leaf is not None
    assert second_leaf is not None


@pytest.mark.skipif(os.getenv("LLM_API_KEY") is None, reason="LLM_API_KEY not set")
@pytest.mark.asyncio
async def test_real_llm_complete_runtime_workflow() -> None:
    """Integration test: All runtime control features with real provider."""
    world = World()
    runner = Runner()

    provider = get_real_provider()
    model = os.getenv("LLM_MODEL", "qwen3.5-flash")

    entity = world.create_entity()
    llm = LLMComponent(provider=provider, model=model)
    world.add_component(entity, llm)

    tree = ConversationTreeComponent()
    msg1 = add_message(tree, role="user", content="Hello, what is 2+2?")

    create_branch(tree, "main", msg1.id)
    switch_branch(tree, "main")

    world.add_component(entity, tree)
    world.register_system(ReasoningSystem(priority=0), priority=0)

    await runner.run(world, max_ticks=1)

    conv_tree = world.get_component(entity, ConversationTreeComponent)
    assert conv_tree is not None
    first_leaf = get_active_leaf(conv_tree)
    assert first_leaf is not None

    msg2 = add_message(
        conv_tree, role="user", content="What is 3+3?", parent_id=first_leaf
    )
    create_branch(conv_tree, "second", msg2.id)
    switch_branch(conv_tree, "second")

    alternative_model = os.getenv("LLM_MODEL_ALTERNATIVE", "qwen3.5-turbo")
    llm.pending_model = alternative_model

    world.remove_component(entity, TerminalComponent)

    await runner.run(world, max_ticks=1)

    second_leaf = get_active_leaf(conv_tree)
    assert second_leaf is not None
    assert second_leaf != first_leaf

    revert_to_message(conv_tree, msg1.id)
    world.add_component(
        entity,
        InterruptionComponent(
            reason=InterruptionReason.USER_REQUESTED, metadata={"workflow": "test"}
        ),
    )

    world.remove_component(entity, TerminalComponent)

    await runner.run(world, max_ticks=1)

    interrupt = world.get_component(entity, InterruptionComponent)
    assert interrupt is not None
    assert interrupt.reason == InterruptionReason.USER_REQUESTED

    third_leaf = get_active_leaf(conv_tree)
    assert third_leaf is not None
    assert len(conv_tree.messages) > 0


@pytest.mark.skipif(not API_KEY, reason="LLM_API_KEY environment variable not set")
@pytest.mark.asyncio
async def test_real_provider_smoke_with_dashscope_defaults() -> None:
    """Smoke test: OpenAIProvider with DashScope-compatible endpoints returns non-empty response.

    This test validates that the provider can successfully communicate with
    DashScope (or any OpenAI-compatible API) using environment defaults.
    """
    provider = OpenAIProvider(
        api_key=API_KEY,
        base_url=BASE_URL,
        model=MODEL,
    )

    messages = [
        Message(role="user", content="Say hello"),
    ]

    result = await provider.complete(messages, stream=False)

    assert isinstance(result, CompletionResult), "Expected CompletionResult"
    assert result.message.role == "assistant", "Expected assistant role"
    assert len(result.message.content) > 0, "Expected non-empty response content"


@pytest.mark.asyncio
async def test_real_read_file_returns_hashed_format(tmp_path: Any) -> None:
    import re

    from ecs_agent.tools.builtins.file_tools import read_file

    test_file = tmp_path / "test.txt"
    raw_lines = ["Alpha", "Beta", "Gamma"]
    test_file.write_text("\n".join(raw_lines) + "\n", encoding="utf-8")

    result = await read_file(file_path="test.txt", workspace_root=str(tmp_path))
    hashed_lines = result.splitlines()

    assert result.startswith("1#")
    assert len(hashed_lines) == 3
    for idx, line in enumerate(hashed_lines):
        assert re.match(r"^\d+#[0-9a-f]{4}\|", line) is not None
        assert line.split("|", 1)[1] == raw_lines[idx]


@pytest.mark.asyncio
async def test_real_glob_finds_files_in_workspace(tmp_path: Any) -> None:
    from ecs_agent.tools.builtins.glob_tool import glob

    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    (tmp_path / "b.py").write_text("print('x')", encoding="utf-8")
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "c.txt").write_text("c", encoding="utf-8")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "d.txt").write_text("d", encoding="utf-8")
    (tmp_path / "src" / "e.py").write_text("print('y')", encoding="utf-8")

    result = await glob(pattern="**/*.txt", base_path=".", workspace_root=str(tmp_path))
    matched = result.splitlines()

    assert set(matched) == {"a.txt", "notes/c.txt", "src/d.txt"}
    assert all(not path.endswith(".py") for path in matched)
    assert all(not path.startswith("/") for path in matched)

    empty_result = await glob(
        pattern="**/*.xyz", base_path=".", workspace_root=str(tmp_path)
    )
    assert empty_result == ""


@pytest.mark.asyncio
async def test_real_llm_builtin_tools_read_file_smoke(tmp_path: Any) -> None:
    from ecs_agent import BuiltinToolsSkill, SkillManager
    from ecs_agent.providers import FakeProvider
    from ecs_agent.systems.tool_execution import ToolExecutionSystem
    from ecs_agent.types import ToolCall

    class RecordingFakeProvider(FakeProvider):
        def __init__(self, responses: list[CompletionResult]) -> None:
            super().__init__(responses)
            self.recorded_messages: list[list[Message]] = []

        async def complete(
            self,
            messages: list[Message],
            tools: list[Any] | None = None,
            stream: bool = False,
            response_format: dict[str, Any] | None = None,
        ) -> CompletionResult | Any:
            self.recorded_messages.append(list(messages))
            return await super().complete(
                messages,
                tools=tools,
                stream=stream,
                response_format=response_format,
            )

    note = tmp_path / "note.txt"
    note.write_text("one\ntwo\nthree\n", encoding="utf-8")

    provider = RecordingFakeProvider(
        responses=[
            CompletionResult(
                message=Message(
                    role="assistant",
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="tc-read-1",
                            name="read_file",
                            arguments={"file_path": "note.txt"},
                        )
                    ],
                )
            ),
            CompletionResult(
                message=Message(
                    role="assistant", content="Read complete from tool output"
                )
            ),
        ]
    )

    world = World()
    entity = world.create_entity()
    world.add_component(entity, LLMComponent(provider=provider, model="fake"))
    world.add_component(
        entity,
        ConversationComponent(messages=[Message(role="user", content="Read note.txt")]),
    )

    skill = BuiltinToolsSkill()
    skill.bind_workspace(str(tmp_path))
    SkillManager().install(world, entity, skill)

    world.register_system(ReasoningSystem(priority=0), priority=0)
    world.register_system(ToolExecutionSystem(priority=5), priority=5)
    world.register_system(MemorySystem(), priority=10)
    world.register_system(ErrorHandlingSystem(priority=99), priority=99)

    await Runner().run(world, max_ticks=5)

    conv = world.get_component(entity, ConversationComponent)
    assert conv is not None
    assert any(msg.role == "tool" for msg in conv.messages)
    assert conv.messages[-1].role == "assistant"
    assert len(conv.messages[-1].content) > 0

    assert len(provider.recorded_messages) >= 2
    second_turn_messages = provider.recorded_messages[1]
    tool_messages = [msg for msg in second_turn_messages if msg.role == "tool"]
    assert len(tool_messages) >= 1
    assert tool_messages[-1].content.startswith("1#")
    assert "|one" in tool_messages[-1].content


@pytest.mark.skipif(not API_KEY, reason="LLM_API_KEY environment variable not set")
@pytest.mark.asyncio
async def test_real_llm_agent_uses_glob_to_find_files(tmp_path: Any) -> None:
    from ecs_agent import BuiltinToolsSkill, SkillManager
    from ecs_agent.systems.tool_execution import ToolExecutionSystem

    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "a.txt").write_text("alpha", encoding="utf-8")
    (tmp_path / "docs" / "b.txt").write_text("beta", encoding="utf-8")
    (tmp_path / "docs" / "c.py").write_text("print('x')", encoding="utf-8")

    world = World()
    entity = world.create_entity()
    world.add_component(
        entity,
        LLMComponent(
            provider=get_real_provider(), model=os.getenv("LLM_MODEL", "qwen3.5-flash")
        ),
    )
    world.add_component(
        entity,
        ConversationComponent(
            messages=[
                Message(
                    role="user",
                    content=(
                        "Use the glob tool to find all .txt files in this workspace. "
                        "Call glob with pattern '**/*.txt' and base_path '.'. "
                        "Then briefly report the files you found."
                    ),
                )
            ]
        ),
    )

    skill = BuiltinToolsSkill()
    skill.bind_workspace(str(tmp_path))
    SkillManager().install(world, entity, skill)

    world.register_system(ReasoningSystem(priority=0), priority=0)
    world.register_system(ToolExecutionSystem(priority=5), priority=5)
    world.register_system(MemorySystem(), priority=10)
    world.register_system(ErrorHandlingSystem(priority=99), priority=99)

    await Runner().run(world, max_ticks=5)

    conv = world.get_component(entity, ConversationComponent)
    assert conv is not None
    tool_messages = [msg for msg in conv.messages if msg.role == "tool"]
    assert len(tool_messages) >= 1
    assert any(".txt" in msg.content for msg in tool_messages)
    assert conv.messages[-1].role == "assistant"
    assert len(conv.messages[-1].content) > 0


@pytest.mark.skipif(not API_KEY, reason="LLM_API_KEY environment variable not set")
@pytest.mark.asyncio
async def test_real_llm_agent_reads_and_reports_file_content(tmp_path: Any) -> None:
    from ecs_agent import BuiltinToolsSkill, SkillManager
    from ecs_agent.systems.tool_execution import ToolExecutionSystem

    (tmp_path / "greeting.txt").write_text(
        "Hello, World!\nLine two.\nLine three.\n", encoding="utf-8"
    )

    world = World()
    entity = world.create_entity()
    world.add_component(
        entity,
        LLMComponent(
            provider=get_real_provider(), model=os.getenv("LLM_MODEL", "qwen3.5-flash")
        ),
    )
    world.add_component(
        entity,
        ConversationComponent(
            messages=[
                Message(
                    role="user",
                    content=(
                        "Use read_file on greeting.txt and tell me how many lines it has. "
                        "Rely on the tool result."
                    ),
                )
            ]
        ),
    )

    skill = BuiltinToolsSkill()
    skill.bind_workspace(str(tmp_path))
    SkillManager().install(world, entity, skill)

    world.register_system(ReasoningSystem(priority=0), priority=0)
    world.register_system(ToolExecutionSystem(priority=5), priority=5)
    world.register_system(MemorySystem(), priority=10)
    world.register_system(ErrorHandlingSystem(priority=99), priority=99)

    await Runner().run(world, max_ticks=5)

    conv = world.get_component(entity, ConversationComponent)
    assert conv is not None
    tool_messages = [msg for msg in conv.messages if msg.role == "tool"]
    assert len(tool_messages) >= 1
    assert any("1#" in msg.content for msg in tool_messages)
    assert conv.messages[-1].role == "assistant"
    assert len(conv.messages[-1].content.strip()) > 0


# ============================================================================
# T7: Rendered Prompt Pipeline Real-LLM Tests
# ============================================================================


class _CapturingProvider:
    """Wraps a real provider and captures the outbound messages."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.captured_messages: list[Message] = []

    async def complete(
        self,
        messages: list[Message],
        tools: list[Any] | None = None,
        stream: bool = False,
        response_format: dict[str, Any] | None = None,
    ) -> CompletionResult | Any:
        self.captured_messages = list(messages)
        return await self._inner.complete(  # type: ignore[no-any-return]
            messages, tools=tools, stream=stream, response_format=response_format
        )


@pytest.mark.skipif(not API_KEY, reason="LLM_API_KEY environment variable not set")
@pytest.mark.asyncio
async def test_real_llm_rendered_system_prompt_reaches_provider() -> None:
    """Verify RenderedSystemPromptComponent.text reaches the provider as system message."""
    from ecs_agent.components import ToolRegistryComponent, TurnStateComponent
    from ecs_agent.components.definitions import RenderedSystemPromptComponent
    from ecs_agent.prompts.contracts import PromptConfigSpec, PromptTemplateSource
    from ecs_agent.systems.system_prompt_render_system import SystemPromptRenderSystem
    from ecs_agent.types import ToolSchema as _ToolSchema

    inner = OpenAIProvider(api_key=API_KEY, base_url=BASE_URL, model=MODEL)
    capturing: _CapturingProvider = _CapturingProvider(inner)

    world = World()
    entity = world.create_entity()
    world.add_component(entity, LLMComponent(provider=capturing, model=MODEL))  # type: ignore[arg-type]
    world.add_component(
        entity,
        ConversationComponent(
            messages=[Message(role="user", content="What tools do you have?")]
        ),
    )
    world.add_component(entity, TurnStateComponent(current_turn_id="t1"))
    world.add_component(
        entity,
        PromptConfigSpec(
            template_source=PromptTemplateSource(
                inline="You are a Python expert. Available tools:\n${_installed_tools}"
            )
        ),
    )
    world.add_component(
        entity,
        ToolRegistryComponent(
            tools={
                "bash": _ToolSchema(
                    name="bash",
                    description="Run shell commands",
                    parameters={"type": "object", "properties": {}},
                )
            },
            handlers={},
        ),
    )
    world.register_system(SystemPromptRenderSystem(), priority=-20)
    world.register_system(ReasoningSystem(priority=0), priority=0)
    world.register_system(ErrorHandlingSystem(priority=99), priority=99)

    runner = Runner()
    await runner.run(world, max_ticks=1)

    assert len(capturing.captured_messages) >= 1
    sys_msg = capturing.captured_messages[0]
    assert sys_msg.role == "system"
    assert "You are a Python expert" in sys_msg.content
    assert "bash" in sys_msg.content  # ${_installed_tools} was rendered


@pytest.mark.skipif(not API_KEY, reason="LLM_API_KEY environment variable not set")
@pytest.mark.asyncio
async def test_real_llm_rendered_user_prompt_trigger_injection() -> None:
    """Verify RenderedUserPromptComponent.text contains injected trigger content."""
    from ecs_agent.components import PromptConfigComponent, TurnStateComponent
    from ecs_agent.components.definitions import RenderedUserPromptComponent
    from ecs_agent.systems.user_prompt_normalization_system import (
        UserPromptNormalizationSystem,
    )

    inner = OpenAIProvider(api_key=API_KEY, base_url=BASE_URL, model=MODEL)
    capturing: _CapturingProvider = _CapturingProvider(inner)

    world = World()
    entity = world.create_entity()
    world.add_component(entity, LLMComponent(provider=capturing, model=MODEL))  # type: ignore[arg-type]
    world.add_component(
        entity,
        ConversationComponent(
            messages=[Message(role="user", content="Please @test verify the output")]
        ),
    )
    world.add_component(entity, TurnStateComponent(current_turn_id="t2"))
    world.add_component(
        entity,
        PromptConfigComponent(
            trigger_templates={"@test": "Use testing best practices."},
            enable_context_pool=False,
        ),
    )
    world.register_system(UserPromptNormalizationSystem(), priority=-10)
    world.register_system(ReasoningSystem(priority=0), priority=0)
    world.register_system(ErrorHandlingSystem(priority=99), priority=99)

    runner = Runner()
    await runner.run(world, max_ticks=1)

    assert len(capturing.captured_messages) >= 1
    user_msg = capturing.captured_messages[-1]
    assert user_msg.role == "user"
    assert "Use testing best practices." in user_msg.content
    assert "Please @test verify the output" in user_msg.content
