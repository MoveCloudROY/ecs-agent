"""Multi-agent collaboration example with dual-mode LLM model selection.

This example demonstrates:
- Creating a World with ReasoningSystem, MessageBusSystem, and ErrorHandlingSystem
- Creating two Agent Entities (researcher and summarizer)
- Setting up agents to communicate via MessageBusSystem pub/sub
- Agent A publishes a message to Agent B via MessageBusSystem
- Agent B receives the message in its conversation
- Running the agents to process collaboration messages
- Printing both agents' conversations

Dual-mode model selection: uses FakeModel by default (no API key needed),
or switches to OpenAIModel when LLM_API_KEY environment variable is set.
Environment variables:
  LLM_API_KEY: Trigger for OpenAIModel mode (if set, uses real LLM)
  LLM_BASE_URL: Base URL for LLM API (defaults to https://dashscope.aliyuncs.com/compatible-mode/v1)
  LLM_MODEL: Model name (defaults to qwen3.5-flash)
"""

from __future__ import annotations

import asyncio
import os

from ecs_agent.components import (
    ConversationComponent,
    LLMComponent,
    MessageBusConfigComponent,
    MessageBusSubscriptionComponent,
)
from ecs_agent.core import Runner, World
from ecs_agent.providers import FakeModel, Model
from ecs_agent.providers.config import ApiFormat
from ecs_agent.providers.protocol import LLMModel
from ecs_agent.systems.message_bus import MessageBusSystem
from ecs_agent.systems.error_handling import ErrorHandlingSystem
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
    model_a: LLMModel
    model_b: LLMModel
    if api_key:
        print(f"Using model: {model}")
        print(f"Base URL: {base_url}")
        model_a = Model(model, base_url=base_url, api_key=api_key, api_format=ApiFormat.OPENAI_CHAT_COMPLETIONS)
        model_b = Model(model, base_url=base_url, api_key=api_key, api_format=ApiFormat.OPENAI_CHAT_COMPLETIONS)
    else:
        print("No LLM_API_KEY provided. Using FakeModel for demonstration.")
        print("To use a real API, set LLM_API_KEY, LLM_BASE_URL, and LLM_MODEL.")
        print()

        # Create FakeModel for Agent A (researcher)
        model_a = FakeModel(
            responses=[
                CompletionResult(
                    message=Message(
                        role="assistant",
                        content="I've analyzed the data and found interesting patterns.",
                    )
                )
            ]
        )

        # Create FakeModel for Agent B (summarizer)
        model_b = FakeModel(
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
            model=model_a,
            
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
            model=model_b,
            
            system_prompt="You are a summarizer agent.",
        ),
    )
    world.add_component(
        agent_b_id,
        ConversationComponent(
            messages=[Message(role="user", content="Waiting for research results...")]
        ),
    )

    # Set up message bus: Agent A will publish, Agent B will subscribe
    # Register MessageBusConfigComponent on a dedicated entity
    bus_entity = world.create_entity()
    world.add_component(bus_entity, MessageBusConfigComponent())
    world.add_component(bus_entity, MessageBusSubscriptionComponent())

    # Register both agents to use message bus
    world.add_component(
        agent_a_id,
        MessageBusSubscriptionComponent(),
    )
    world.add_component(
        agent_b_id,
        MessageBusSubscriptionComponent(),
    )

    # Register Systems
    world.register_system(ReasoningSystem(priority=0), priority=0)
    message_bus_system = MessageBusSystem(priority=5)
    world.register_system(message_bus_system, priority=5)
    world.register_system(ErrorHandlingSystem(priority=99), priority=99)

    # Run initial tick to let Agent A reason
    runner = Runner()
    await runner.run(world, max_ticks=1)

    # Now have Agent A publish a message to Agent B via the message bus
    message_bus_system.subscribe(
        topic="research-results", subscriber_id=str(agent_b_id)
    )
    message_to_b = Message(
        role="assistant",
        content="I found interesting data while researching this topic.",
    )
    await message_bus_system.publish(
        topic="research-results",
        message={"content": message_to_b.content, "role": message_to_b.role},
    )

    # Run more ticks to let Agent B receive and process the message
    await runner.run(world, max_ticks=4)

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
