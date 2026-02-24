"""Streaming agent example demonstrating real-time response streaming.

This example shows how to use the streaming feature:
- Creates an OpenAIProvider (with FakeProvider fallback for testing)
- Calls provider.complete(messages, stream=True) to get streaming deltas
- Iterates AsyncIterator[StreamDelta] and prints content chunks in real-time
- Shows final delta with usage stats (prompt_tokens, completion_tokens, total_tokens)

Usage:
  1. Copy .env.example to .env and fill in your API credentials
  2. Run: uv run python examples/streaming_agent.py

Environment variables:
  LLM_API_KEY   — API key for the LLM provider (required, can be fake for testing)
  LLM_BASE_URL  — Base URL for the API (default: https://dashscope.aliyuncs.com/compatible-mode/v1)
  LLM_MODEL     — Model name (default: qwen3.5-plus)

Output:
  The agent streams the response word-by-word in real-time, showing each chunk
  as it arrives from the LLM. At the end, usage statistics are displayed.
"""

from __future__ import annotations

import asyncio
import os
import sys

from ecs_agent.logging import configure_logging
from ecs_agent.providers import FakeProvider, OpenAIProvider
from ecs_agent.types import CompletionResult, Message, Usage


async def main() -> None:
    """Run a streaming agent that demonstrates real-time response output."""
    # --- Configure logging ---
    configure_logging(json_output=False)

    # --- Load config from environment ---
    api_key = os.environ.get("LLM_API_KEY", "")
    base_url = os.environ.get(
        "LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    model = os.environ.get("LLM_MODEL", "qwen3.5-plus")

    # --- Create LLM provider ---
    if api_key:
        print(f"Using OpenAIProvider with model: {model}")
        print(f"Base URL: {base_url}")
        provider = OpenAIProvider(api_key=api_key, base_url=base_url, model=model)
    else:
        print("No LLM_API_KEY provided. Using FakeProvider for demonstration.")
        print("To use a real API, set LLM_API_KEY, LLM_BASE_URL, and LLM_MODEL.")
        print()
        # Create a fake provider that streams character-by-character
        provider = FakeProvider(
            responses=[
                CompletionResult(
                    message=Message(
                        role="assistant",
                        content="This is a streamed response from the FakeProvider. Each character arrives as a separate chunk, simulating real-time streaming from an LLM.",
                    ),
                    usage=Usage(
                        prompt_tokens=15, completion_tokens=35, total_tokens=50
                    ),
                )
            ]
        )

    # --- Prepare messages ---
    messages = [
        Message(
            role="user",
            content="Explain the concept of streaming in LLMs in a single paragraph.",
        )
    ]

    # --- Stream the response ---
    print("Streaming response:")
    print("-" * 60)

    # Call provider with streaming enabled
    delta_iterator = await provider.complete(messages, stream=True)

    # Track final delta info
    final_delta = None
    total_tokens = 0

    # Iterate through deltas and print content in real-time
    async for delta in delta_iterator:
        if delta.content:
            # Use sys.stdout.write to avoid newlines between chunks
            sys.stdout.write(delta.content)
            sys.stdout.flush()

        # Keep track of final delta for usage stats
        if delta.finish_reason:
            final_delta = delta

    # --- Print final newline and stats ---
    print()
    print("-" * 60)

    if final_delta and final_delta.usage:
        usage = final_delta.usage
        print(f"Tokens used:")
        print(f"  Prompt tokens:     {usage.prompt_tokens}")
        print(f"  Completion tokens: {usage.completion_tokens}")
        print(f"  Total tokens:      {usage.total_tokens}")
    else:
        print("(No usage statistics available)")


if __name__ == "__main__":
    asyncio.run(main())
