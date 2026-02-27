"""
Example demonstrating PermissionComponent and PermissionSystem.

This script shows how to restrict an agent's access to specific tools using
a whitelist/blacklist policy.
"""

import asyncio

from ecs_agent.core import World, Runner
from ecs_agent.components import (
    LLMComponent,
    ConversationComponent,
    PermissionComponent,
    PendingToolCallsComponent,
    ToolRegistryComponent,
)
from ecs_agent.providers import FakeProvider
from ecs_agent.systems.permission import PermissionSystem
from ecs_agent.systems.tool_execution import ToolExecutionSystem
from ecs_agent.types import CompletionResult, Message, ToolCall, ToolSchema


async def main() -> None:
    world = World()

    # 1. Register tools in the registry
    async def safe_tool(**kwargs) -> str:
        return "Safe operation successful"

    async def dangerous_tool(**kwargs) -> str:
        return "Dangerous operation successful"

    tools = {
        "safe_tool": ToolSchema(
            name="safe_tool", description="A safe tool", parameters={}
        ),
        "dangerous_tool": ToolSchema(
            name="dangerous_tool", description="A dangerous tool", parameters={}
        ),
    }
    handlers = {"safe_tool": safe_tool, "dangerous_tool": dangerous_tool}

    # 2. Create agent with permissions
    agent = world.create_entity()
    world.add_component(agent, ToolRegistryComponent(tools=tools, handlers=handlers))

    # Deny 'dangerous_tool' explicitly
    world.add_component(agent, PermissionComponent(denied_tools=["dangerous_tool"]))

    # Setup conversation and LLM
    world.add_component(agent, LLMComponent(provider=FakeProvider([]), model="fake"))
    world.add_component(agent, ConversationComponent(messages=[]))

    # 3. Register PermissionSystem (priority -10) and ToolExecutionSystem (priority 5)
    world.register_system(PermissionSystem(priority=-10), priority=-10)
    world.register_system(ToolExecutionSystem(priority=5), priority=5)

    # 4. Attempt to call both tools
    print("Attempting to call 'safe_tool' and 'dangerous_tool'...")
    world.add_component(
        agent,
        PendingToolCallsComponent(
            tool_calls=[
                ToolCall(id="c1", name="safe_tool", arguments={}),
                ToolCall(id="c2", name="dangerous_tool", arguments={}),
            ]
        ),
    )

    # Run one tick
    runner = Runner()
    await runner.run(world, max_ticks=1)

    # 5. Verify results
    conv = world.get_component(agent, ConversationComponent)
    if conv:
        print("\nConversation History:")
        for msg in conv.messages:
            status = "ALLOWED" if "denied" not in msg.content else "DENIED"
            print(f"[{msg.role}] {status}: {msg.content}")


if __name__ == "__main__":
    asyncio.run(main())
