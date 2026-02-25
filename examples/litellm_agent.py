"""LiteLLMProvider agent example supporting 100+ LLM providers.

This example demonstrates using the LiteLLMProvider to support a wide range of LLM
providers with a unified interface. LiteLLM normalizes API calls to OpenAI format.

Usage:
  1. Install litellm: pip install litellm
  2. Set environment variables:
     - LLM_API_KEY: Your API key
     - LLM_MODEL: Model in format "provider/model" (e.g., "openai/gpt-4" or "anthropic/claude-3-opus-20240229")
  3. Run: uv run python examples/litellm_agent.py

If litellm is not installed or API key is missing, the example gracefully falls back to FakeProvider.
"""

from __future__ import annotations

import asyncio
import os
import sys
from typing import Any

from ecs_agent.components import (
    ConversationComponent,
    LLMComponent,
    ToolRegistryComponent,
)
from ecs_agent.core import Runner, World
from ecs_agent.providers import FakeProvider
from ecs_agent.providers.protocol import LLMProvider
from ecs_agent.types import CompletionResult, Message, ToolSchema
from ecs_agent.systems.error_handling import ErrorHandlingSystem
from ecs_agent.systems.memory import MemorySystem
from ecs_agent.systems.reasoning import ReasoningSystem
from ecs_agent.systems.tool_execution import ToolExecutionSystem


# Try to import LiteLLMProvider; gracefully handle if litellm is not installed
try:
    from ecs_agent.providers import LiteLLMProvider

    HAS_LITELLM = True
except (ImportError, AttributeError):
    HAS_LITELLM = False


async def search_database(query: str) -> str:
    """Simulate searching a database."""
    results = {
        "users": "Found 42 users matching the search",
        "products": "Found 156 products in inventory",
        "orders": "Found 89 recent orders",
    }
    return results.get(query.lower(), f"No results for '{query}'")


async def main() -> None:
    """Run LiteLLMProvider agent example."""
    # Check if litellm is installed
    if not HAS_LITELLM:
        print("litellm is not installed.")
        print("Install with: pip install litellm")
        sys.exit(0)

    # Load config from environment
    api_key = os.environ.get("LLM_API_KEY", "")
    model = os.environ.get("LLM_MODEL", "")

    # Decide which provider to use
    provider: LLMProvider
    if api_key and model:
        print(f"Using LiteLLMProvider: {model}")
        provider = LiteLLMProvider(
            model=model,
            api_key=api_key,
        )
    else:
        print("No API key or model specified. Using FakeProvider instead.")
        provider = FakeProvider(
            responses=[
                CompletionResult(
                    message=Message(
                        role="assistant",
                        content="I searched the database and found relevant results for you.",
                    )
                )
            ]
        )
        model = "fake"

    # Create World
    world = World()
    agent_id = world.create_entity()

    # Add components
    world.add_component(
        agent_id,
        LLMComponent(provider=provider, model=model),
    )
    world.add_component(
        agent_id,
        ConversationComponent(
            messages=[
                Message(
                    role="user",
                    content="Search for users in the database and tell me what you find.",
                )
            ]
        ),
    )

    # Register tools
    world.add_component(
        agent_id,
        ToolRegistryComponent(
            tools={
                "search_database": ToolSchema(
                    name="search_database",
                    description="Search the database for entities",
                    parameters={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "What to search for (users, products, or orders)",
                            }
                        },
                        "required": ["query"],
                    },
                ),
            },
            handlers={"search_database": search_database},
        ),
    )

    # Register systems
    world.register_system(ReasoningSystem(priority=0), priority=0)
    world.register_system(ToolExecutionSystem(priority=5), priority=5)
    world.register_system(MemorySystem(), priority=10)
    world.register_system(ErrorHandlingSystem(priority=99), priority=99)

    # Run the agent
    print("Running agent...\n")
    runner = Runner()
    await runner.run(world, max_ticks=5)

    # Print conversation
    conv = world.get_component(agent_id, ConversationComponent)
    if conv:
        print("\n" + "=" * 60)
        print("CONVERSATION")
        print("=" * 60)
        for msg in conv.messages:
            if msg.role == "user":
                print(f"\n[User] {msg.content}")
            elif msg.role == "assistant":
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        print(f"\n[Tool Call] {tc.name}({tc.arguments})")
                else:
                    print(f"\n[Assistant] {msg.content}")
            elif msg.role == "tool":
                print(f"[Tool Result] {msg.content}")


if __name__ == "__main__":
    asyncio.run(main())
