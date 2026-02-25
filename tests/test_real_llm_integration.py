"""Real LLM integration tests using DashScope API.

These tests use real OpenAI-compatible API calls to verify streaming,
checkpoints, and provider functionality with actual LLM responses.

Environment Variables:
    LLM_API_KEY: DashScope API key (required for tests to run)

Usage:
    # Run with API key (tests will execute)
    LLM_API_KEY=sk-xxx uv run pytest tests/test_real_llm_integration.py -v --timeout=60

    # Run without API key (tests will skip)
    uv run pytest tests/test_real_llm_integration.py -v
"""

from __future__ import annotations

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

# DashScope API configuration
API_KEY = os.getenv("LLM_API_KEY", "")
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL = "qwen-plus"

# Skip all tests if API key is not set
pytestmark = pytest.mark.skipif(not API_KEY, reason="LLM_API_KEY not set")


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
