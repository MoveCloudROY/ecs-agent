"""ClaudeProvider agent example with tool calling and streaming.

This example demonstrates using the ClaudeProvider with both non-streaming and
streaming modes. ClaudeProvider uses Anthropic's API (or DashScope compatible endpoint)
and supports tool calling.

Usage:
  1. Set environment variables:
     - LLM_API_KEY: Your API key
     - LLM_BASE_URL: API endpoint (default: https://api.anthropic.com)
     - LLM_MODEL: Model name (default: claude-3-5-haiku-latest)
  2. Run: uv run python examples/claude_agent.py

Without API credentials, this example falls back to FakeProvider for demonstration.
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


# Try to import ClaudeProvider; fall back gracefully if not available
try:
    from ecs_agent.providers import ClaudeProvider

    HAS_CLAUDE = True
except (ImportError, AttributeError):
    HAS_CLAUDE = False


async def get_weather(city: str) -> str:
    """Simulate getting weather for a city."""
    weather_db = {
        "beijing": "Beijing: Sunny, 22°C, humidity 40%",
        "shanghai": "Shanghai: Cloudy, 20°C, humidity 55%",
        "shenzhen": "Shenzhen: Rainy, 24°C, humidity 75%",
    }
    return weather_db.get(city.lower(), f"Weather for {city} not available")


async def get_time(city: str) -> str:
    """Simulate getting current time in a city."""
    time_db = {
        "beijing": "14:30 (UTC+8)",
        "shanghai": "14:30 (UTC+8)",
        "newyork": "02:30 (UTC-5)",
    }
    return time_db.get(city.lower(), f"Time in {city} not available")


async def main() -> None:
    """Run ClaudeProvider agent example."""
    # Load config from environment
    api_key = os.environ.get("LLM_API_KEY", "")
    base_url = os.environ.get("LLM_BASE_URL", "https://api.anthropic.com")
    model = os.environ.get("LLM_MODEL", "claude-3-5-haiku-latest")

    # Decide which provider to use
    provider: LLMProvider
    if api_key and HAS_CLAUDE:
        print(f"Using ClaudeProvider: {model}")
        provider = ClaudeProvider(
            api_key=api_key,
            base_url=base_url,
            model=model,
        )
    else:
        print("Using FakeProvider (no API key or ClaudeProvider unavailable)")
        provider = FakeProvider(
            responses=[
                CompletionResult(
                    message=Message(
                        role="assistant",
                        content="The weather in Beijing is sunny with a temperature of 22°C. The current time is 14:30.",
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
                    content="What's the weather like in Beijing? And what time is it there?",
                )
            ]
        ),
    )

    # Register tools
    world.add_component(
        agent_id,
        ToolRegistryComponent(
            tools={
                "get_weather": ToolSchema(
                    name="get_weather",
                    description="Get the current weather for a city",
                    parameters={
                        "type": "object",
                        "properties": {
                            "city": {
                                "type": "string",
                                "description": "City name",
                            }
                        },
                        "required": ["city"],
                    },
                ),
                "get_time": ToolSchema(
                    name="get_time",
                    description="Get the current time in a city",
                    parameters={
                        "type": "object",
                        "properties": {
                            "city": {
                                "type": "string",
                                "description": "City name",
                            }
                        },
                        "required": ["city"],
                    },
                ),
            },
            handlers={"get_weather": get_weather, "get_time": get_time},
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
