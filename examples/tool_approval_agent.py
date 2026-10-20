"""Tool Approval Agent Example with real LLM.

Demonstrates the @tool decorator, scan_module tool discovery, approval workflow,
and sandboxed tool execution using a real LLM model.

The example shows:
  - Defining @tool-decorated functions (get_weather, send_email)
  - Using scan_module() to auto-discover and register tools
  - Setting up ToolApprovalComponent with ALWAYS_APPROVE policy
  - Running ToolApprovalSystem (priority=2) before ToolExecutionSystem (priority=5)
  - The LLM autonomously decides which tools to call; approval system gates execution

Usage:
  1. Copy .env.example to .env and fill in your API credentials
  2. Run: uv run python examples/tool_approval_agent.py

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
    ToolApprovalComponent,
)
from ecs_agent.core import AgentHandle, World
from ecs_agent.providers import Model
from ecs_agent.providers.config import ApiFormat
from ecs_agent.providers.retry_model import RetryModel
from ecs_agent.systems.error_handling import ErrorHandlingSystem
from ecs_agent.systems.reasoning import ReasoningSystem
from ecs_agent.systems.tool_approval import ToolApprovalSystem
from ecs_agent.systems.tool_execution import ToolExecutionSystem
from ecs_agent.tools import register_tools, scan_module, tool
from ecs_agent.types import ApprovalPolicy, RetryConfig


@tool()
async def get_weather(location: str) -> str:
    """Get the weather for a location (simulated)."""
    return f"The weather in {location} is sunny and 72°F."


@tool()
async def send_email(recipient: str, subject: str, body: str) -> str:
    """Send an email to a recipient (simulated)."""
    return f"Email sent to {recipient} with subject '{subject}'."


async def main() -> None:
    """Run a tool approval agent example with a real LLM."""
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
            
            system_prompt=(
                "You are a helpful assistant that can check weather and send emails. "
                "Use the provided tools to fulfill the user's request."
            ),
        ),
    )
    world.add_component(agent_id, ConversationComponent(messages=[]))

    register_tools(world, agent_id, scan_module(sys.modules[__name__]))

    # Add tool approval component with ALWAYS_APPROVE policy
    world.add_component(
        agent_id,
        ToolApprovalComponent(policy=ApprovalPolicy.ALWAYS_APPROVE),
    )

    # Register Systems
    # ToolApprovalSystem runs at priority 2 (before tool execution)
    world.register_system(ToolApprovalSystem(priority=2), priority=2)
    world.register_system(ReasoningSystem(priority=0), priority=0)
    # ToolExecutionSystem runs at priority 5 (after approval)
    world.register_system(ToolExecutionSystem(priority=5), priority=5)
    world.register_system(ErrorHandlingSystem(priority=99), priority=99)

    # Run
    agent = AgentHandle(world, agent_id)
    try:
        result = await agent.run_turn(
            "What's the weather in San Francisco? Send me an email with the report to user@example.com.",
            max_steps=5,
        )
        print(f"Run status: {result.status}")
        if result.final_message is not None:
            print(f"Assistant: {result.final_message.content}")
        elif result.error:
            print(f"Error: {result.error}")
    finally:
        await model.aclose()

    # Print results
    conv = world.get_component(agent_id, ConversationComponent)
    if conv is not None:
        print("=" * 60)
        print("CONVERSATION HISTORY")
        print("=" * 60)
        for msg in conv.messages:
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    print(f"\n[Action] {tc.name}({tc.arguments})")
            elif msg.tool_call_id:
                print(f"[Result] {msg.content}")
            elif msg.role == "user":
                print(f"\n[User] {msg.content}")
            elif msg.role == "assistant":
                print(f"\n[Assistant] {msg.content}")
    else:
        print("No conversation found")


if __name__ == "__main__":
    asyncio.run(main())
