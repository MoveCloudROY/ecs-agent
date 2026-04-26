"""Example demonstrating PermissionComponent and PermissionSystem with dual-mode LLM model.

This script shows how to restrict an agent's access to specific tools using
a whitelist/blacklist policy. Supports both FakeModel (no API key) and
OpenAIModel (with LLM_API_KEY) for flexible testing.
"""

import asyncio
import os

from ecs_agent.core import World, Runner
from ecs_agent.components import (
    LLMComponent,
    ConversationComponent,
    PermissionComponent,
    PendingToolCallsComponent,
    ToolRegistryComponent,
)
from ecs_agent.providers import FakeModel, Model
from ecs_agent.providers.config import ApiFormat
from ecs_agent.systems.permission import PermissionSystem
from ecs_agent.systems.tool_execution import ToolExecutionSystem
from ecs_agent.types import CompletionResult, Message, ToolCall, ToolSchema


async def main() -> None:
    # --- Read environment variables for LLM model ---
    api_key = os.environ.get("LLM_API_KEY", "")
    base_url = os.environ.get(
        "LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    model = os.environ.get("LLM_MODEL", "qwen3.5-flash")
    
    # --- Create LLM model ---
    if api_key:
        print(f"Using model: {model}")
        print(f"Base URL: {base_url}")
        model = Model(model, base_url=base_url, api_key=api_key, api_format=ApiFormat.OPENAI_CHAT_COMPLETIONS)
    else:
        print("No LLM_API_KEY provided. Using FakeModel for demonstration.")
        print("To use a real API, set LLM_API_KEY, LLM_BASE_URL, and LLM_MODEL.")
        print()
        model = FakeModel([])
    
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
    world.add_component(agent, LLMComponent(model=model,
))
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
