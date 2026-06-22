"""Streaming system agent example demonstrating real-time response streaming via EventBus.

This example shows system-level streaming (different from model-level streaming):
- Creates an agent entity with StreamingComponent(enabled=True)
- ReasoningSystem detects the streaming flag and publishes EventBus events
- Subscribes to StreamStartEvent, StreamReasoningDeltaEvent, StreamReasoningEndEvent,
  StreamContentStartEvent, StreamContentDeltaEvent, StreamEndEvent
- Prints reasoning chunks and final content chunks in distinct phases
- Uses OpenAIModel with DashScope (env: LLM_API_KEY, LLM_BASE_URL, LLM_MODEL)
- Falls back to FakeModel for demo without API key

Key difference from streaming_agent.py:
- streaming_agent.py: Provider-level streaming (await model.complete(stream=True))
- This example: System-level streaming (EventBus events published by ReasoningSystem)
"""

from __future__ import annotations

import asyncio
import os
import sys

from ecs_agent.components import ConversationComponent, LLMComponent, StreamingComponent
from ecs_agent.core import Runner, World
from ecs_agent.logging import configure_logging
from ecs_agent.providers import FakeModel, Model
from ecs_agent.providers.config import ApiFormat
from ecs_agent.providers.protocol import LLMModel
from ecs_agent.systems.error_handling import ErrorHandlingSystem
from ecs_agent.systems.reasoning import ReasoningSystem
from ecs_agent.types import (
    CompletionResult,
    Message,
    StreamContentStartEvent,
    StreamContentDeltaEvent,
    StreamEndEvent,
    StreamReasoningDeltaEvent,
    StreamReasoningEndEvent,
    StreamStartEvent,
    Usage,
)


async def main() -> None:
    """Run a streaming agent demonstrating system-level real-time response output."""
    # --- Configure logging ---
    configure_logging(json_output=False)

    # --- Load config from environment ---
    api_key = os.environ.get("LLM_API_KEY", "")
    base_url = os.environ.get(
        "LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    model = os.environ.get("LLM_MODEL", "qwen3.5-flash")

    # --- Create LLM model ---
    model: LLMModel
    if api_key:
        print(f"Using model: {model}")
        print(f"Base URL: {base_url}")
        model = Model(model, base_url=base_url, api_key=api_key, api_format=ApiFormat.OPENAI_CHAT_COMPLETIONS)
    else:
        print("No LLM_API_KEY provided. Using FakeModel for demonstration.")
        print("To use a real API, set LLM_API_KEY, LLM_BASE_URL, and LLM_MODEL.")
        print()

        # Create a fake model that simulates streaming character-by-character
        model = FakeModel(
            responses=[
                CompletionResult(
                    message=Message(
                        role="assistant",
                        content="This is a simulated streaming response. Each character is sent as a separate delta, demonstrating real-time token streaming via the EventBus.",
                    ),
                    usage=Usage(
                        prompt_tokens=12, completion_tokens=28, total_tokens=40
                    ),
                )
            ]
        )

    # --- Create World with agent entity ---
    world = World()
    agent_id = world.create_entity()

    # Add LLMComponent
    world.add_component(
        agent_id,
        LLMComponent(
            model=model,
            
            system_prompt="You are a helpful assistant that provides concise answers.",
        ),
    )

    # Add ConversationComponent with user message
    world.add_component(
        agent_id,
        ConversationComponent(
            messages=[
                Message(
                    role="user",
                    content="Explain streaming in LLMs in one sentence.",
                )
            ]
        ),
    )

    # Add StreamingComponent to enable system-level streaming
    world.add_component(
        agent_id,
        StreamingComponent(enabled=True, non_blocking_delta_publish=True),
    )

    # --- Subscribe to streaming events ---
    streamed_content = []
    reasoning_seen = False

    async def on_stream_start(event: StreamStartEvent) -> None:
        """Called when streaming starts."""
        print("\n[Streaming started]")

    async def on_stream_delta(event: StreamContentDeltaEvent) -> None:
        """Called for each streamed delta (token/chunk)."""
        if event.delta:  # delta is a string, not an object
            # Print delta without newline for real-time effect
            sys.stdout.write(event.delta)
            sys.stdout.flush()
            streamed_content.append(event.delta)

    async def on_stream_reasoning_delta(event: StreamReasoningDeltaEvent) -> None:
        nonlocal reasoning_seen
        if event.reasoning_delta:
            reasoning_seen = True
            sys.stdout.write(event.reasoning_delta)
            sys.stdout.flush()

    async def on_stream_reasoning_end(event: StreamReasoningEndEvent) -> None:
        _ = event
        if reasoning_seen:
            print("\n[Reasoning ended]")

    async def on_stream_content_start(event: StreamContentStartEvent) -> None:
        _ = event
        print("[Content started]")

    async def on_stream_end(event: StreamEndEvent) -> None:
        """Called when streaming ends."""
        print("\n[Streaming ended]")

    # Register event handlers
    world.event_bus.subscribe(StreamStartEvent, on_stream_start)
    world.event_bus.subscribe(StreamContentDeltaEvent, on_stream_delta)
    world.event_bus.subscribe(StreamReasoningDeltaEvent, on_stream_reasoning_delta)
    world.event_bus.subscribe(StreamReasoningEndEvent, on_stream_reasoning_end)
    world.event_bus.subscribe(StreamContentStartEvent, on_stream_content_start)
    world.event_bus.subscribe(StreamEndEvent, on_stream_end)

    # --- Register systems ---
    # ReasoningSystem (priority 0) executes the LLM inference
    # ErrorHandlingSystem (priority 99) catches any errors
    world.register_system(ReasoningSystem(priority=0), priority=0)
    world.register_system(ErrorHandlingSystem(priority=99), priority=99)

    # --- Run the agent ---
    print("Running streaming agent...")
    print("-" * 60)

    runner = Runner()
    await runner.run(world, max_ticks=3)

    # --- Print final results ---
    print()
    print("-" * 60)
    print(
        f"Total streamed content length: {sum(len(c) for c in streamed_content)} chars"
    )

    # Print final conversation
    conv = world.get_component(agent_id, ConversationComponent)
    if conv is not None:
        print()
        print("Final conversation:")
        for msg in conv.messages:
            if msg.role == "user":
                print(f"  User: {msg.content}")
            elif msg.role == "assistant":
                print(f"  Assistant: {msg.content[:100]}...")


if __name__ == "__main__":
    asyncio.run(main())
