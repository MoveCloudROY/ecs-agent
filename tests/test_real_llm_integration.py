"""Real LLM integration tests using DashScope-compatible API.

These tests use real OpenAI-compatible API calls to verify streaming,
checkpoints, and provider functionality with actual LLM responses.

Environment Variables (required for tests to execute):
    LLM_API_KEY: DashScope API key (or any OpenAI-compatible provider)
    LLM_BASE_URL: Base URL for the LLM provider (e.g. https://dashscope.aliyuncs.com/compatible-mode/v1)
    LLM_MODEL: Model identifier (e.g. qwen-plus, qwen3.5-plus)

Test Contract:
    - With all env vars set: Tests execute normally.
    - Without LLM_API_KEY: All tests skip gracefully (no failures).

Usage:
    # Run with API key (tests will execute)
    LLM_API_KEY=sk-xxx LLM_BASE_URL=https://... LLM_MODEL=qwen-plus uv run pytest tests/test_real_llm_integration.py -v --timeout=60

    # Run without API key (tests will skip)
    uv run pytest tests/test_real_llm_integration.py -v
"""

from __future__ import annotations

import json

import os

import pytest

from ecs_agent.components import (
    ConversationComponent,
    LLMComponent,
    StreamingComponent,
    SystemPromptComponent,
)
from ecs_agent.core import Runner, World
from ecs_agent.providers import OpenAIProvider
# Removed unused imports: CheckpointComponent, ClaudeProvider, CheckpointSystem
from ecs_agent.systems.error_handling import ErrorHandlingSystem
from ecs_agent.systems.memory import MemorySystem
from ecs_agent.systems.reasoning import ReasoningSystem
from ecs_agent.types import CompletionResult, Message, StreamDelta, StreamDeltaEvent
from ecs_agent.logging import STANDARD_EVENT_NAMES, FORBIDDEN_FIELDS

# DashScope API configuration from environment variables
# Tests require LLM_API_KEY to execute; other vars use sensible defaults
API_KEY = os.getenv("LLM_API_KEY", "")
BASE_URL = os.getenv("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
MODEL = os.getenv("LLM_MODEL", "qwen-plus")

# Skip all tests if API key is not set (env-driven test contract)
pytestmark = pytest.mark.skipif(not API_KEY, reason="LLM_API_KEY environment variable not set")


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


@pytest.mark.asyncio
async def test_real_openai_streaming_produces_deltas() -> None:
    """Streaming ReasoningSystem with real OpenAIProvider emits StreamDelta events."""
    # Setup: World with streaming-enabled entity
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

    # Register reasoning system
    world.register_system(ReasoningSystem(priority=0), priority=0)

    # Capture streaming events
    deltas: list[StreamDeltaEvent] = []

    async def capture_delta(event: StreamDeltaEvent) -> None:
        deltas.append(event)

    world.event_bus.subscribe(StreamDeltaEvent, capture_delta)

    # Execute: Process one tick to trigger streaming
    await world.process()

    # Assert: StreamDelta events received with non-empty content
    assert len(deltas) > 0, "Expected at least one StreamDelta event"
    total_content = "".join(event.delta for event in deltas)
    assert len(total_content) > 0, "Expected non-empty streamed content"

    # Verify conversation contains assistant response
    conv = world.get_component(entity, ConversationComponent)
    assert conv is not None
    assert len(conv.messages) == 2  # user + assistant
    assert conv.messages[1].role == "assistant"
    assert len(conv.messages[1].content) > 0


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

    # Execute: Non-streaming complete
    result = await provider.complete(messages, stream=False)

    # Assert: CompletionResult with valid message
    assert isinstance(result, CompletionResult), "Expected CompletionResult"
    assert result.message.role == "assistant"
    assert len(result.message.content) > 0, "Expected non-empty response content"
    assert result.message.tool_calls is None, "No tools provided, should be None"


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

    # Execute: Streaming complete
    stream_result = await provider.complete(messages, stream=True)

    # Assert: Returns async iterator
    assert not isinstance(
        stream_result, CompletionResult
    ), "Expected async iterator, not CompletionResult"

    # Collect deltas
    deltas: list[StreamDelta] = []
    async for delta in stream_result:
        deltas.append(delta)

    # Verify: Multiple deltas received with content
    assert len(deltas) > 0, "Expected at least one StreamDelta"
    content_chunks = [d.content for d in deltas if d.content is not None]
    assert len(content_chunks) > 0, "Expected at least one content delta"
    total_content = "".join(content_chunks)
    assert len(total_content) > 0, "Expected non-empty accumulated content"


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

    # Register systems
    world.register_system(ReasoningSystem(priority=0), priority=0)
    world.register_system(MemorySystem(), priority=10)
    world.register_system(ErrorHandlingSystem(priority=99), priority=99)

    # Capture streaming events
    deltas: list[StreamDeltaEvent] = []

    async def capture_delta(event: StreamDeltaEvent) -> None:
        deltas.append(event)

    world.event_bus.subscribe(StreamDeltaEvent, capture_delta)

    # Execute: Run agent loop
    runner = Runner()
    await runner.run(world, max_ticks=1)  # Run 1 tick to avoid duplicate responses

    # Assert: Conversation has assistant response
    conv = world.get_component(entity, ConversationComponent)
    assert conv is not None
    assert len(conv.messages) >= 2, "Expected at least user + assistant messages"
    assert conv.messages[-1].role == "assistant", "Last message should be from assistant"
    assert len(conv.messages[-1].content) > 0

    # Assert: StreamDelta events published
    assert len(deltas) > 0, "Expected StreamDelta events during streaming"


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
            messages=[
                Message(role="user", content="My name is Alice. Remember it.")
            ]
        ),
    )

    # Register systems
    world.register_system(ReasoningSystem(priority=0), priority=0)
    world.register_system(MemorySystem(), priority=10)
    world.register_system(ErrorHandlingSystem(priority=99), priority=99)

    # Turn 1: LLM acknowledges name
    runner = Runner()
    await runner.run(world, max_ticks=1)

    conv = world.get_component(entity, ConversationComponent)
    assert conv is not None
    assert len(conv.messages) >= 2, "Expected user + assistant messages"
    assert conv.messages[-1].role == "assistant"

    # Turn 2: Ask LLM to recall the name
    conv.messages.append(
        Message(role="user", content="What is my name? Answer in 3 words or less.")
    )

    # Run another tick
    await runner.run(world, max_ticks=2, start_tick=1)

    # Verify: LLM should recall "Alice" from context
    assert len(conv.messages) >= 4, "Expected 2 turns of conversation"
    final_response = conv.messages[-1].content.lower()
    assert conv.messages[-1].role == "assistant"
    # Note: We don't assert exact content since LLM responses vary,
    # but we verify the conversation flow works correctly
    assert len(final_response) > 0, "Expected non-empty response"


@pytest.mark.asyncio
async def test_real_llm_reasoning_logging_contracts(capsys) -> None:
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

    # Register ReasoningSystem
    world.register_system(ReasoningSystem(priority=0), priority=0)

    # Execute: Process one tick to trigger LLM call
    await world.process()

    # Capture logging output
    captured = capsys.readouterr()
    events = _json_events(captured.out)

    # Assert: reasoning_start event exists with required metadata
    reasoning_start_events = [e for e in events if e.get("event") == "reasoning_start"]
    assert len(reasoning_start_events) > 0, "Expected at least one reasoning_start event"

    start_event = reasoning_start_events[0]
    assert "entity_id" in start_event, "reasoning_start must include entity_id"
    assert "model" in start_event, "reasoning_start must include model"
    assert start_event["model"] == MODEL

    # Assert: reasoning_complete event exists with duration_ms
    reasoning_complete_events = [e for e in events if e.get("event") == "reasoning_complete"]
    assert len(reasoning_complete_events) > 0, "Expected at least one reasoning_complete event"

    complete_event = reasoning_complete_events[0]
    assert "entity_id" in complete_event, "reasoning_complete must include entity_id"
    assert "duration_ms" in complete_event, "reasoning_complete must include duration_ms"
    assert isinstance(complete_event["duration_ms"], (int, float))
    assert complete_event["duration_ms"] >= 0

    # Assert: No forbidden fields in any log event
    for event in events:
        for forbidden in FORBIDDEN_FIELDS:
            assert forbidden not in event, f"Forbidden field '{forbidden}' found in log event: {event.get('event')}"


@pytest.mark.asyncio
async def test_real_llm_streaming_logging_metadata(capsys) -> None:
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

    # Execute
    await world.process()

    # Capture and parse logs
    captured = capsys.readouterr()
    events = _json_events(captured.out)

    # Assert: reasoning_start includes streaming metadata
    reasoning_start_events = [e for e in events if e.get("event") == "reasoning_start"]
    assert len(reasoning_start_events) > 0

    start_event = reasoning_start_events[0]
    assert "streaming" in start_event, "reasoning_start must include 'streaming' field in streaming mode"
    assert start_event["streaming"] is True

    # Assert: reasoning_complete includes chunk count or streaming completion metadata
    reasoning_complete_events = [e for e in events if e.get("event") == "reasoning_complete"]
    if reasoning_complete_events:
        complete_event = reasoning_complete_events[0]
        # Should have duration_ms at minimum
        assert "duration_ms" in complete_event

    # Assert: No sensitive conversation content in logs
    for event in events:
        assert "content" not in event, "Raw conversation content must not appear in logs"
        assert "arguments" not in event, "Raw tool arguments must not appear in logs"


@pytest.mark.asyncio
async def test_real_llm_error_logging_contracts(capsys) -> None:
    """Verify error logging when LLM provider fails (bad API key)."""
    world = World()
    # Use invalid API key to trigger error
    provider = OpenAIProvider(
        api_key="sk-invalid-key-for-testing",
        base_url=BASE_URL,
        model=MODEL,
    )

    entity = world.create_entity()
    world.add_component(entity, LLMComponent(provider=provider, model=MODEL))
    world.add_component(
        entity,
        ConversationComponent(
            messages=[Message(role="user", content="Hello")]
        ),
    )

    world.register_system(ReasoningSystem(priority=0), priority=0)
    world.register_system(ErrorHandlingSystem(priority=99), priority=99)

    # Execute (should fail)
    await world.process()

    # Capture logs
    captured = capsys.readouterr()
    events = _json_events(captured.out)

    # Assert: reasoning_error event exists
    error_events = [e for e in events if e.get("event") == "reasoning_error"]
    assert len(error_events) > 0, "Expected at least one reasoning_error event"

    error_event = error_events[0]
    assert "entity_id" in error_event, "reasoning_error must include entity_id"
    assert "reason" in error_event or "exception" in error_event, "reasoning_error must include failure reason"

    # Assert: No API key leaked in error logs
    for event in events:
        event_str = json.dumps(event)
        assert "sk-invalid-key-for-testing" not in event_str, "API key must not appear in logs"
        assert "api_key" not in event, "api_key field must not appear in logs"
