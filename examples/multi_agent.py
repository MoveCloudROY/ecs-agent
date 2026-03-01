"""Multi-agent collaboration example with dual-mode LLM provider selection.

This example demonstrates:
- Creating a World with ReasoningSystem, CollaborationSystem, MemorySystem, and ErrorHandlingSystem
- Creating two Agent Entities (researcher and summarizer)
- Setting up CollaborationComponent with peers and inbox
- Agent A sends a message to Agent B via inbox
- Running the agents to process collaboration messages
- Printing both agents' conversations

Dual-mode provider selection: uses FakeProvider by default (no API key needed),
or switches to OpenAIProvider when LLM_API_KEY environment variable is set.
Environment variables:
  LLM_API_KEY: Trigger for OpenAIProvider mode (if set, uses real LLM)
  LLM_BASE_URL: Base URL for LLM API (defaults to https://dashscope.aliyuncs.com/compatible-mode/v1)
  LLM_MODEL: Model name (defaults to qwen3.5-flash)
"""

from __future__ import annotations

import asyncio
import os

from ecs_agent.components import (
    CollaborationComponent,
    ConversationComponent,
    LLMComponent,
)
from ecs_agent.core import Runner, World
from ecs_agent.providers import FakeProvider, OpenAIProvider
from ecs_agent.providers.protocol import LLMProvider
from ecs_agent.systems.collaboration import CollaborationSystem
from ecs_agent.systems.error_handling import ErrorHandlingSystem
from ecs_agent.systems.memory import MemorySystem
from ecs_agent.systems.reasoning import ReasoningSystem
from ecs_agent.types import CompletionResult, Message


async def main() -> None:
    """Run a multi-agent collaboration example."""
    # --- Environment variable configuration ---
    api_key = os.environ.get("LLM_API_KEY", "")
    base_url = os.environ.get(
        "LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    model = os.environ.get("LLM_MODEL", "qwen3.5-flash")

    # --- Create LLM providers (two separate instances) ---
    provider_a: LLMProvider
    provider_b: LLMProvider
    if api_key:
        print(f"Using OpenAIProvider with model: {model}")
        print(f"Base URL: {base_url}")
        provider_a = OpenAIProvider(api_key=api_key, base_url=base_url, model=model)
        provider_b = OpenAIProvider(api_key=api_key, base_url=base_url, model=model)
    else:
        print("No LLM_API_KEY provided. Using FakeProvider for demonstration.")
        print("To use a real API, set LLM_API_KEY, LLM_BASE_URL, and LLM_MODEL.")
        print()

        # Create FakeProvider for Agent A (researcher)
        provider_a = FakeProvider(
            responses=[
                CompletionResult(
                    message=Message(
                        role="assistant",
                        content="I've analyzed the data and found interesting patterns.",
                    )
                )
            ]
        )

        # Create FakeProvider for Agent B (summarizer)
        provider_b = FakeProvider(
            responses=[
                CompletionResult(
                    message=Message(
                        role="assistant",
                        content="Thank you! I'll summarize the key findings for you.",
                    )
                )
            ]
        )

    # Create World
    world = World()

    # Create Agent A (researcher)
    agent_a_id = world.create_entity()
    world.add_component(
        agent_a_id,
        LLMComponent(
            provider=provider_a,
            model=model if api_key else "fake",
            system_prompt="You are a researcher agent.",
        ),
    )
    world.add_component(
        agent_a_id,
        ConversationComponent(
            messages=[Message(role="user", content="Start researching the topic.")]
        ),
    )

    # Create Agent B (summarizer)
    agent_b_id = world.create_entity()
    world.add_component(
        agent_b_id,
        LLMComponent(
            provider=provider_b,
            model=model if api_key else "fake",
            system_prompt="You are a summarizer agent.",
        ),
    )
    world.add_component(
        agent_b_id,
        ConversationComponent(messages=[]),
    )

    # Set up collaboration: Agent A sends message to Agent B
    world.add_component(
        agent_a_id,
        CollaborationComponent(peers=[agent_b_id], inbox=[]),
    )
    world.add_component(
        agent_b_id,
        CollaborationComponent(
            peers=[agent_a_id],
            inbox=[
                (
                    agent_a_id,
                    Message(role="assistant", content="I found interesting data."),
                )
            ],
        ),
    )

    # Register Systems
    world.register_system(ReasoningSystem(priority=0), priority=0)
    world.register_system(CollaborationSystem(priority=5), priority=5)
    world.register_system(MemorySystem(), priority=10)
    world.register_system(ErrorHandlingSystem(priority=99), priority=99)

    # Run
    runner = Runner()
    await runner.run(world, max_ticks=5)

    # Print results
    print("Agent A (researcher) conversation:")
    conv_a = world.get_component(agent_a_id, ConversationComponent)
    if conv_a is not None:
        for msg in conv_a.messages:
            print(f"  {msg.role}: {msg.content}")
    else:
        print("  No conversation found")

    print("\nAgent B (summarizer) conversation:")
    conv_b = world.get_component(agent_b_id, ConversationComponent)
    if conv_b is not None:
        for msg in conv_b.messages:
            print(f"  {msg.role}: {msg.content}")
    else:
        print("  No conversation found")


if __name__ == "__main__":
    asyncio.run(main())
