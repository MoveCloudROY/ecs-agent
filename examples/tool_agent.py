"""Tool-use agent example using the ECS-based LLM Agent framework.

This example demonstrates:
- Creating a World with ReasoningSystem, ToolExecutionSystem, and ErrorHandlingSystem
- Defining async tool handlers (add, multiply)
- Creating an Agent Entity with a real LLM model (OpenAIModel + RetryModel)
- The LLM decides which tools to call and synthesizes the final answer
- Running the agent to execute tools and print the result

Usage:
  1. Copy .env.example to .env and fill in your API credentials
  2. Run: uv run python examples/tool_agent.py

Environment variables:
  LLM_API_KEY   — API key for the LLM model (required)
  LLM_BASE_URL  — Base URL for the API (default: https://dashscope.aliyuncs.com/compatible-mode/v1)
  LLM_MODEL     — Model name (default: qwen3.5-plus)
"""

from __future__ import annotations

import asyncio
import os
import sys

from ecs_agent.components import (
    ConversationComponent,
    LLMComponent,
)
from ecs_agent.core import AgentHandle, World
from ecs_agent.providers import Model
from ecs_agent.providers.config import ApiFormat
from ecs_agent.providers.retry_model import RetryModel
from ecs_agent.systems.error_handling import ErrorHandlingSystem
from ecs_agent.systems.reasoning import ReasoningSystem
from ecs_agent.systems.tool_execution import ToolExecutionSystem
from ecs_agent.tools import register_tools, tool
from ecs_agent.types import RetryConfig


@tool()
async def add(a: str, b: str) -> str:
    """Add two numbers (string arguments)."""
    return str(int(a) + int(b))


@tool()
async def multiply(a: str, b: str) -> str:
    """Multiply two numbers (string arguments)."""
    return str(int(a) * int(b))


async def main() -> None:
    """Run a tool-use agent example with a real LLM."""
    # --- Load config from environment ---
    api_key = os.environ.get("LLM_API_KEY", "")
    if not api_key:
        print("Error: LLM_API_KEY environment variable is required.")
        print("Copy .env.example to .env and fill in your API key.")
        sys.exit(1)

    base_url = os.environ.get(
        "LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    model = os.environ.get("LLM_MODEL", "qwen3.5-plus")
    connect_timeout = float(os.environ.get("LLM_CONNECT_TIMEOUT", "10"))
    read_timeout = float(os.environ.get("LLM_READ_TIMEOUT", "120"))
    write_timeout = float(os.environ.get("LLM_WRITE_TIMEOUT", "10"))
    pool_timeout = float(os.environ.get("LLM_POOL_TIMEOUT", "10"))
    max_retries = int(os.environ.get("LLM_MAX_RETRIES", "3"))

    print(f"Using model: {model}")
    print(f"Base URL: {base_url}")
    print()

    # --- Create LLM model ---
    base_model = Model(model, base_url=base_url, api_key=api_key, api_format=ApiFormat.OPENAI_CHAT_COMPLETIONS, connect_timeout=connect_timeout, read_timeout=read_timeout, write_timeout=write_timeout, pool_timeout=pool_timeout)
    model = RetryModel(
        base_model,
        retry_config=RetryConfig(
            max_attempts=max_retries,
            multiplier=1.0,
            min_wait=1.0,
            max_wait=8.0,
        ),
    )

    # --- Create World ---
    world = World()

    # Create Agent Entity
    agent_id = world.create_entity()
    world.add_component(
        agent_id,
        LLMComponent(
            model=model,
            
            system_prompt="You are a helpful calculator assistant. Use the provided tools to compute results.",
        ),
    )
    world.add_component(agent_id, ConversationComponent(messages=[]))
    register_tools(world, agent_id, [add, multiply])

    # Register Systems
    world.register_system(ReasoningSystem(priority=0), priority=0)
    world.register_system(ToolExecutionSystem(priority=5), priority=5)
    world.register_system(ErrorHandlingSystem(priority=99), priority=99)

    agent = AgentHandle(world, agent_id)
    try:
        result = await agent.run_turn("What is 2 + 3? And what is 7 * 8?", max_steps=5)
        if result.success:
            assert result.final_message is not None
            print(f"Assistant: {result.final_message.content}")
        else:
            print(f"Run stopped: {result.status} — {result.error or result.reason}")
    finally:
        await model.aclose()


if __name__ == "__main__":
    asyncio.run(main())
