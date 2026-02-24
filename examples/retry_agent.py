"""Retry provider example with custom retry configuration.

This example demonstrates how to wrap an LLM provider with RetryProvider to add
automatic retry logic with exponential backoff for transient failures.

The RetryProvider:
- Wraps any LLMProvider (e.g., OpenAIProvider, FakeProvider)
- Retries non-streaming completion calls on network/HTTP errors
- Uses exponential backoff with configurable parameters (max_attempts, min_wait, max_wait)
- Skips retry logic for streaming calls (passes through directly)
- Logs retry attempts with structured fields (attempt number, error, wait time)

Custom RetryConfig allows fine-tuning:
- max_attempts: Maximum number of attempts (default: 3)
- multiplier: Exponential backoff multiplier (default: 1.0)
- min_wait: Minimum wait time between retries in seconds (default: 4.0)
- max_wait: Maximum wait time between retries in seconds (default: 60.0)
- retry_status_codes: HTTP status codes to retry on (default: 429, 500, 502, 503, 504)

Usage:
  1. Copy .env.example to .env and fill in your API credentials
  2. Run: uv run python examples/retry_agent.py

Environment variables:
  LLM_API_KEY   — API key for the LLM provider (required)
  LLM_BASE_URL  — Base URL for the API (default: https://dashscope.aliyuncs.com/compatible-mode/v1)
  LLM_MODEL     — Model name (default: qwen3.5-plus)
"""

from __future__ import annotations

import asyncio
import os
import sys

from ecs_agent.logging import configure_logging, get_logger
from ecs_agent.providers import FakeProvider, OpenAIProvider
from ecs_agent.providers.retry_provider import RetryProvider
from ecs_agent.types import CompletionResult, Message, RetryConfig, Usage

logger = get_logger(__name__)


async def main() -> None:
    """Run an agent with retry-wrapped provider."""
    # --- Configure logging ---
    configure_logging(json_output=False)

    # --- Load config from environment ---
    api_key = os.environ.get("LLM_API_KEY", "")
    base_url = os.environ.get(
        "LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    model = os.environ.get("LLM_MODEL", "qwen3.5-plus")

    # --- Create base provider ---
    if api_key:
        print(f"Using OpenAIProvider with model: {model}")
        print(f"Base URL: {base_url}")
        base_provider = OpenAIProvider(api_key=api_key, base_url=base_url, model=model)
    else:
        print("No LLM_API_KEY provided. Using FakeProvider for demonstration.")
        # Create a fake provider with a realistic response
        fake_result = CompletionResult(
            message=Message(
                role="assistant",
                content="This is a demonstration response from the retry-wrapped provider. "
                "In production, the RetryProvider would transparently retry "
                "on transient errors like rate limits (429) and server errors (500, 502, 503, 504).",
            ),
            usage=Usage(
                prompt_tokens=15,
                completion_tokens=45,
                total_tokens=60,
            ),
        )
        base_provider = FakeProvider(responses=[fake_result])

    # --- Create custom retry configuration ---
    # This demonstrates custom retry parameters beyond the defaults
    retry_config = RetryConfig(
        max_attempts=5,  # Allow up to 5 attempts instead of default 3
        multiplier=2.0,  # Use 2x exponential backoff instead of 1x
        min_wait=2.0,  # Start with 2 seconds instead of 4
        max_wait=30.0,  # Cap at 30 seconds instead of 60
        retry_status_codes=(429, 500, 502, 503, 504),  # Retry on these HTTP errors
    )

    print()
    print("Retry Configuration:")
    print(f"  max_attempts: {retry_config.max_attempts}")
    print(f"  multiplier: {retry_config.multiplier}")
    print(f"  min_wait: {retry_config.min_wait}s")
    print(f"  max_wait: {retry_config.max_wait}s")
    print(f"  retry_status_codes: {retry_config.retry_status_codes}")
    print()

    # --- Wrap provider with retry logic ---
    provider = RetryProvider(base_provider, retry_config=retry_config)

    # --- Make a completion request ---
    messages = [
        Message(
            role="user",
            content="Explain how the RetryProvider works with exponential backoff.",
        )
    ]

    print("Making completion request through retry-wrapped provider...")
    print()

    try:
        result = await provider.complete(messages=messages)
        print("Completion Result:")
        print(f"  Role: {result.message.role}")
        print(f"  Content: {result.message.content}")
        if result.usage:
            print(
                f"  Tokens - Prompt: {result.usage.prompt_tokens}, "
                f"Completion: {result.usage.completion_tokens}, "
                f"Total: {result.usage.total_tokens}"
            )
        print()
        print("✓ Completion succeeded (no retries needed for this request)")

    except Exception as e:
        logger.error(
            "completion_failed",
            error=str(e),
            error_type=type(e).__name__,
        )
        print(f"✗ Completion failed with error: {e}")
        sys.exit(1)

    # --- Demonstrate retry behavior in logging ---
    print()
    print("Notes:")
    print("- If this request had encountered transient errors (429, 500, etc.),")
    print(
        "  the RetryProvider would have automatically retried with exponential backoff"
    )
    print("- Retry attempts are logged at WARNING level with structured fields:")
    print("  {attempt, error, wait_seconds}")
    print(
        "- Streaming calls bypass retry logic and are passed directly to the base provider"
    )


if __name__ == "__main__":
    asyncio.run(main())
