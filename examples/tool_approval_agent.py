"""Tool Approval Agent Example

Demonstrates the @tool decorator, scan_module tool discovery, approval workflow,
and sandboxed tool execution. Uses FakeProvider with pre-configured responses
including tool calls that must pass through the approval system.

The example shows:
  - Defining @tool-decorated functions (get_weather, send_email)
  - Using scan_module() to auto-discover and register tools
  - Setting up ToolApprovalComponent with ALWAYS_APPROVE policy
  - Running ToolApprovalSystem (priority=-5) before ToolExecutionSystem (priority=5)
  - Automatic approval and execution of tools within the sandbox
"""

import asyncio
import sys

from ecs_agent.components import (
    ConversationComponent,
    LLMComponent,
    ToolApprovalComponent,
    ToolRegistryComponent,
)
from ecs_agent.core import Runner, World
from ecs_agent.providers import FakeProvider
from ecs_agent.systems.error_handling import ErrorHandlingSystem
from ecs_agent.systems.memory import MemorySystem
from ecs_agent.systems.reasoning import ReasoningSystem
from ecs_agent.systems.tool_approval import ToolApprovalSystem
from ecs_agent.systems.tool_execution import ToolExecutionSystem
from ecs_agent.tools.discovery import scan_module, tool
from ecs_agent.types import ApprovalPolicy, CompletionResult, Message, ToolCall


@tool()
async def get_weather(location: str) -> str:
    """Get the weather for a location (simulated)."""
    return f"The weather in {location} is sunny and 72°F."


@tool()
async def send_email(recipient: str, subject: str, body: str) -> str:
    """Send an email to a recipient (simulated)."""
    return f"Email sent to {recipient} with subject '{subject}'."


async def main() -> None:
    """Run a tool approval agent example."""
    # Create World
    world = World()

    # Create FakeProvider with tool calls that will be approved
    provider = FakeProvider(
        responses=[
            CompletionResult(
                message=Message(
                    role="assistant",
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="call-1",
                            name="get_weather",
                            arguments={"location": "San Francisco"},
                        ),
                        ToolCall(
                            id="call-2",
                            name="send_email",
                            arguments={
                                "recipient": "user@example.com",
                                "subject": "Weather Report",
                                "body": "Check the weather for San Francisco",
                            },
                        ),
                    ],
                )
            ),
            CompletionResult(
                message=Message(
                    role="assistant",
                    content="I've checked the weather and sent you an email report.",
                )
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
            system_prompt="You are a helpful assistant that can check weather and send emails.",
        ),
    )
    world.add_component(
        agent_id,
        ConversationComponent(
            messages=[
                Message(
                    role="user",
                    content="What's the weather in San Francisco? Send me an email with the report.",
                )
            ]
        ),
    )

    # Scan current module for @tool-decorated functions
    tool_registry = scan_module(sys.modules[__name__])

    # Convert scan_module output to ToolRegistryComponent format
    tools = {name: schema for name, (schema, _) in tool_registry.items()}
    handlers = {name: handler for name, (_, handler) in tool_registry.items()}

    # Register tools
    world.add_component(
        agent_id,
        ToolRegistryComponent(
            tools=tools,
            handlers=handlers,
        ),
    )

    # Add tool approval component with ALWAYS_APPROVE policy
    world.add_component(
        agent_id,
        ToolApprovalComponent(policy=ApprovalPolicy.ALWAYS_APPROVE),
    )

    # Register Systems
    # ToolApprovalSystem runs at priority -5 (before tool execution)
    world.register_system(ToolApprovalSystem(priority=-5), priority=-5)
    world.register_system(ReasoningSystem(priority=0), priority=0)
    # ToolExecutionSystem runs at priority 5 (after approval)
    world.register_system(ToolExecutionSystem(priority=5), priority=5)
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
