"""Tool-use agent example using the ECS-based LLM Agent framework.

This example demonstrates:
- Creating a World with ReasoningSystem, ToolExecutionSystem, MemorySystem, and ErrorHandlingSystem
- Defining an async tool handler (add function)
- Creating an Agent Entity with LLMComponent (FakeProvider), ConversationComponent, and ToolRegistryComponent
- FakeProvider returns tool_call for 'add', then final answer
- Running the agent to execute tools and print the result
"""

import asyncio

from ecs_agent.components import (
    ConversationComponent,
    LLMComponent,
    ToolRegistryComponent,
)
from ecs_agent.core import Runner, World
from ecs_agent.providers import FakeProvider
from ecs_agent.systems.error_handling import ErrorHandlingSystem
from ecs_agent.systems.memory import MemorySystem
from ecs_agent.systems.reasoning import ReasoningSystem
from ecs_agent.systems.tool_execution import ToolExecutionSystem
from ecs_agent.types import CompletionResult, Message, ToolCall, ToolSchema


async def add(a: str, b: str) -> str:
    """Add two numbers (string arguments)."""
    return str(int(a) + int(b))


async def main() -> None:
    """Run a tool-use agent example."""
    # Create World
    world = World()

    # Create FakeProvider with tool call response + final answer
    provider = FakeProvider(
        responses=[
            CompletionResult(
                message=Message(
                    role="assistant",
                    content="",
                    tool_calls=[
                        ToolCall(id="call-1", name="add", arguments={"a": "2", "b": "3"})
                    ],
                )
            ),
            CompletionResult(
                message=Message(role="assistant", content="The answer is 5")
            ),
        ]
    )

    # Create Agent Entity
    agent_id = world.create_entity()
    world.add_component(
        agent_id,
        LLMComponent(
            provider=provider,
            model="fake",
            system_prompt="You are a helpful calculator assistant.",
        ),
    )
    world.add_component(
        agent_id,
        ConversationComponent(
            messages=[Message(role="user", content="What is 2 + 3?")]
        ),
    )

    # Register tools
    world.add_component(
        agent_id,
        ToolRegistryComponent(
            tools={
                "add": ToolSchema(
                    name="add",
                    description="Add two numbers",
                    parameters={
                        "type": "object",
                        "properties": {
                            "a": {"type": "string"},
                            "b": {"type": "string"},
                        },
                        "required": ["a", "b"],
                    },
                )
            },
            handlers={"add": add},
        ),
    )

    # Register Systems
    world.register_system(ReasoningSystem(priority=0), priority=0)
    world.register_system(ToolExecutionSystem(priority=5), priority=5)
    world.register_system(MemorySystem(), priority=10)
    world.register_system(ErrorHandlingSystem(priority=99), priority=99)

    # Run
    runner = Runner()
    await runner.run(world, max_ticks=5)

    # Print results
    conv = world.get_component(agent_id, ConversationComponent)
    if conv is not None:
        print("Conversation:")
        for msg in conv.messages:
            if msg.tool_calls:
                print(
                    f"  {msg.role}: [tool_calls: {', '.join(tc.name for tc in msg.tool_calls)}]"
                )
            elif msg.tool_call_id:
                print(f"  {msg.role} (tool result): {msg.content}")
            else:
                print(f"  {msg.role}: {msg.content}")
    else:
        print("No conversation found")


if __name__ == "__main__":
    asyncio.run(main())
