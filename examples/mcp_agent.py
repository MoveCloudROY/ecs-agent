"""
Demonstrates MCP server integration via MCPSkillAdapter.

This example shows how to:
1. Initialize an MCP client with a mock transport (to avoid external dependencies).
2. Wrap the MCP client in an MCPSkillAdapter.
3. Install the adapter as a Skill into the agent's world.
4. Verify that MCP tools are correctly namespaced and accessible.
"""

import asyncio
import json
from typing import Any

from ecs_agent import SkillManager
from ecs_agent.components import ConversationComponent, LLMComponent
from ecs_agent.core import Runner, World
from ecs_agent.providers import FakeProvider
from ecs_agent.systems.reasoning import ReasoningSystem
from ecs_agent.systems.tool_execution import ToolExecutionSystem
from ecs_agent.types import CompletionResult, Message, ToolCall

# Check if MCP is available
try:
    from ecs_agent.mcp.adapter import MCPSkillAdapter
    from ecs_agent.mcp.client import MCPClient
    from ecs_agent.mcp.components import MCPConfigComponent

    HAS_MCP = True
except ImportError:
    HAS_MCP = False


class MockMCPClient:
    """A mock MCP client that simulates server behavior for demonstration."""

    def __init__(self, config: Any) -> None:
        self.server_name = config.server_name
        self.is_connected = False

    async def connect(self) -> None:
        self.is_connected = True

    async def disconnect(self) -> None:
        self.is_connected = False

    async def list_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "get_weather",
                "description": "Get current weather for a city.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            }
        ]

    async def call_tool(self, name: str, args: dict[str, Any]) -> str:
        if name == "get_weather":
            city = args.get("city", "unknown")
            return f"The weather in {city} is sunny, 22°C."
        return "Unknown tool"


async def main() -> None:
    if not HAS_MCP:
        print("MCP dependencies not found. Install with: uv pip install -e '.[mcp]'")
        return

    world = World()
    agent = world.create_entity()

    # 1. Setup Mock MCP Infrastructure
    config = MCPConfigComponent(
        server_name="weather-service",
        transport_type="stdio",
        config={"command": "mock"},
    )

    mock_client = MockMCPClient(config)

    # 2. Wrap in Adapter
    skill = MCPSkillAdapter(mock_client, server_name="weather-service")  # type: ignore[arg-type]

    # 3. Setup Agent and Systems
    provider = FakeProvider(
        responses=[
            CompletionResult(
                message=Message(
                    role="assistant",
                    content="Checking weather...",
                    tool_calls=[
                        ToolCall(
                            id="call_weather",
                            name="weather-service/get_weather",
                            arguments={"city": "San Francisco"},
                        )
                    ],
                )
            ),
            CompletionResult(
                message=Message(
                    role="assistant", content="It's sunny in San Francisco!"
                )
            ),
        ]
    )

    world.add_component(agent, LLMComponent(provider=provider, model="fake"))
    world.add_component(
        agent,
        ConversationComponent(
            messages=[Message(role="user", content="How is the weather in SF?")]
        ),
    )

    # 4. Install MCP Skill
    manager = SkillManager()
    manager.install(world, agent, skill)

    world.register_system(ReasoningSystem(), priority=0)
    world.register_system(ToolExecutionSystem(), priority=5)

    # 5. Run loop
    print("Starting MCP agent demo...")
    runner = Runner()
    await runner.run(world, max_ticks=5)

    # Verify that the tool was called and results captured
    conv = world.get_component(agent, ConversationComponent)
    if conv:
        for msg in conv.messages:
            print(f"{msg.role}: {msg.content}")


if __name__ == "__main__":
    asyncio.run(main())
