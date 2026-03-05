"""Simple chat agent example using the ECS-based LLM Agent framework.

This example demonstrates dual-mode LLM provider selection:
- Without LLM_API_KEY: Uses FakeProvider for deterministic testing
- With LLM_API_KEY: Uses OpenAIProvider with DashScope/Qwen

Also demonstrates:
- Creating a World with ReasoningSystem, MemorySystem, and ErrorHandlingSystem
- Creating an Agent Entity with LLMComponent and ConversationComponent
- Running the agent with a user message
- Printing the conversation history
"""

import asyncio
import os

from ecs_agent.components import ConversationComponent, LLMComponent
from ecs_agent.core import Runner, World
from ecs_agent.logging import configure_logging
from ecs_agent.providers import FakeProvider, OpenAIProvider
from ecs_agent.providers.protocol import LLMProvider
from ecs_agent.systems.error_handling import ErrorHandlingSystem
from ecs_agent.systems.memory import MemorySystem
from ecs_agent.systems.reasoning import ReasoningSystem
from ecs_agent.types import CompletionResult, Message


async def main() -> None:
    """Run a simple chat agent example."""
    configure_logging(json_output=False)

    # Create World
    world = World()

    # --- Create LLM provider ---
    api_key: str = os.environ.get("LLM_API_KEY", "")
    base_url: str = os.environ.get(
        "LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    model: str = os.environ.get("LLM_MODEL", "qwen3.5-flash")

    provider: LLMProvider
    if api_key:
        print(f"Using OpenAIProvider with model: {model}")
        provider = OpenAIProvider(api_key=api_key, base_url=base_url, model=model)
    else:
        print("No LLM_API_KEY set. Using FakeProvider for demonstration.")
        provider = FakeProvider(
            responses=[
                CompletionResult(
                    message=Message(
                        role="assistant",
                        content="Hello! I'm doing great, thank you for asking! How can I help you today?",
                    )
                )
            ]
        )

    # Create Agent Entity
    agent_id = world.create_entity()
    world.add_component(
        agent_id,
        LLMComponent(
            provider=provider,
            model=model if api_key else "fake",
            system_prompt="You are a helpful assistant.",
        ),
    )
    world.add_component(
        agent_id,
        ConversationComponent(
            messages=[Message(role="user", content="Hello, how are you?")]
        ),
    )

    # Register Systems
    world.register_system(ReasoningSystem(priority=0), priority=0)
    world.register_system(MemorySystem(), priority=10)
    world.register_system(ErrorHandlingSystem(priority=99), priority=99)

    # Run
    runner = Runner()
    await runner.run(world, max_ticks=3)

    # Print results
    conv = world.get_component(agent_id, ConversationComponent)
    if conv is not None:
        print("Conversation:")
        for msg in conv.messages:
            print(f"  {msg.role}: {msg.content}")
    else:
        print("No conversation found")


if __name__ == "__main__":
    asyncio.run(main())
