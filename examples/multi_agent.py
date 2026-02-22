"""Multi-agent collaboration example using the ECS-based LLM Agent framework.

This example demonstrates:
- Creating a World with ReasoningSystem, CollaborationSystem, MemorySystem, and ErrorHandlingSystem
- Creating two Agent Entities (researcher and summarizer)
- Setting up CollaborationComponent with peers and inbox
- Agent A sends a message to Agent B via inbox
- Running the agents to process collaboration messages
- Printing both agents' conversations
"""

import asyncio

from ecs_agent.components import (
    CollaborationComponent,
    ConversationComponent,
    LLMComponent,
)
from ecs_agent.core import Runner, World
from ecs_agent.providers import FakeProvider
from ecs_agent.systems.collaboration import CollaborationSystem
from ecs_agent.systems.error_handling import ErrorHandlingSystem
from ecs_agent.systems.memory import MemorySystem
from ecs_agent.systems.reasoning import ReasoningSystem
from ecs_agent.types import CompletionResult, Message


async def main() -> None:
    """Run a multi-agent collaboration example."""
    # Create World
    world = World()

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

    # Create Agent A (researcher)
    agent_a_id = world.create_entity()
    world.add_component(
        agent_a_id,
        LLMComponent(
            provider=provider_a,
            model="fake",
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
            model="fake",
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
