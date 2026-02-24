"""ReAct (Reasoning + Acting) agent example using the ECS-based LLM Agent framework.

This example demonstrates the full ReAct pattern:
- PlanningSystem drives a multi-step plan, calling the LLM at each step
- The LLM can choose to call tools (Action) or reason directly (Thought)
- ToolExecutionSystem executes tool calls and appends results (Observation)
- The next plan step sees tool results in conversation, enabling the loop:
  Thought → Action → Observation → Thought → ...

Usage:
  1. Copy .env.example to .env and fill in your API credentials
  2. Run: uv run python examples/react_agent.py

Environment variables:
  LLM_API_KEY   — API key for the LLM provider (required)
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
    PlanComponent,
    SystemPromptComponent,
    ToolRegistryComponent,
)
from ecs_agent.core import Runner, World
from ecs_agent.providers import OpenAIProvider
from ecs_agent.systems.error_handling import ErrorHandlingSystem
from ecs_agent.systems.memory import MemorySystem
from ecs_agent.systems.planning import PlanningSystem
from ecs_agent.systems.tool_execution import ToolExecutionSystem
from ecs_agent.types import Message, PlanStepCompletedEvent, ToolSchema


# ---------------------------------------------------------------------------
# Tool definitions — these are the "Actions" the agent can take
# ---------------------------------------------------------------------------


async def get_weather(city: str) -> str:
    """Simulate fetching weather data for a city."""
    weather_db = {
        "beijing": "Sunny, 28°C, humidity 35%",
        "shanghai": "Cloudy, 25°C, humidity 65%",
        "tokyo": "Rainy, 20°C, humidity 80%",
        "new york": "Partly cloudy, 22°C, humidity 50%",
    }
    result = weather_db.get(city.lower())
    if result:
        return f"Weather in {city}: {result}"
    return f"Weather data not available for {city}"


async def get_population(city: str) -> str:
    """Simulate fetching population data for a city."""
    population_db = {
        "beijing": "21.54 million",
        "shanghai": "24.87 million",
        "tokyo": "13.96 million",
        "new york": "8.34 million",
    }
    result = population_db.get(city.lower())
    if result:
        return f"Population of {city}: {result}"
    return f"Population data not available for {city}"


# ---------------------------------------------------------------------------
# Event handler — observe the ReAct loop in real-time
# ---------------------------------------------------------------------------


async def on_step_completed(event: PlanStepCompletedEvent) -> None:
    """Print progress as each plan step completes."""
    print(f"  ✓ Step {event.step_index + 1} completed: {event.step_description}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    """Run a ReAct agent that researches and compares two cities."""
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

    print(f"Using model: {model}")
    print(f"Base URL: {base_url}")
    print()

    # --- Create LLM provider ---
    provider = OpenAIProvider(api_key=api_key, base_url=base_url, model=model)

    # --- Define the plan (ReAct steps) ---
    plan_steps = [
        "Look up the weather in Beijing using the get_weather tool",
        "Look up the weather in Shanghai using the get_weather tool",
        "Look up the population of Beijing using the get_population tool",
        "Look up the population of Shanghai using the get_population tool",
        "Compare Beijing and Shanghai based on all the data collected, and give a recommendation for which city to visit",
    ]

    # --- Define tool schemas ---
    tools = {
        "get_weather": ToolSchema(
            name="get_weather",
            description="Get the current weather for a city",
            parameters={
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city name, e.g. 'Beijing'",
                    }
                },
                "required": ["city"],
            },
        ),
        "get_population": ToolSchema(
            name="get_population",
            description="Get the population of a city",
            parameters={
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city name, e.g. 'Beijing'",
                    }
                },
                "required": ["city"],
            },
        ),
    }

    # --- Build the ECS World ---
    world = World()

    # Create the agent entity
    main_agent = world.create_entity()
    # sub_agent = world.create_entity()

    # Attach components
    world.add_component(
        main_agent, LLMComponent(provider=provider, model=model)
    )
    world.add_component(
        main_agent,
        ConversationComponent(
            messages=[
                Message(
                    role="user",
                    content="Help me compare Beijing and Shanghai. I want to know about their weather and population.",
                )
            ],
            max_messages=50,
        ),
    )
    world.add_component(main_agent, PlanComponent(steps=plan_steps))
    world.add_component(
        main_agent,
        SystemPromptComponent(
            content=(
                "You are a helpful research assistant using the ReAct pattern. "
                "For each step, think about what information you need, "
                "then use the available tools to gather data. "
                "In the final step, synthesize all observations into a clear answer. "
                "Always use tools when a step asks you to — do not make up data."
            ),
        ),
    )
    world.add_component(
        main_agent,
        ToolRegistryComponent(
            tools=tools,
            handlers={
                "get_weather": get_weather,
                "get_population": get_population,
            },
        ),
    )

    # Register systems (order matters: planning → tool execution → memory → error)
    world.register_system(PlanningSystem(priority=0), priority=0)
    world.register_system(ToolExecutionSystem(priority=5), priority=5)
    world.register_system(MemorySystem(), priority=10)
    world.register_system(ErrorHandlingSystem(priority=99), priority=99)

    # Subscribe to plan step events for real-time progress
    world.event_bus.subscribe(PlanStepCompletedEvent, on_step_completed)

    # --- Run the ReAct loop ---
    print(f"Running ReAct agent with {len(plan_steps)}-step plan...")
    print()

    runner = Runner()
    await runner.run(world, max_ticks=20)

    # --- Print results ---
    print()
    print("=" * 60)
    print("CONVERSATION HISTORY")
    print("=" * 60)

    conv = world.get_component(main_agent, ConversationComponent)
    if conv is not None:
        for i, msg in enumerate(conv.messages):
            if msg.role == "user":
                print(f"\n[User] {msg.content}")
            elif msg.role == "assistant":
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        args = tc.arguments
                        print(f"\n[Action] {tc.name}({args})")
                else:
                    print(f"\n[Thought] {msg.content}")
            elif msg.role == "tool":
                print(f"[Observation] {msg.content}")
            elif msg.role == "system":
                pass  # Skip system messages in output

    plan = world.get_component(main_agent, PlanComponent)
    if plan is not None:
        print()
        print("=" * 60)
        status = "COMPLETED" if plan.completed else f"IN PROGRESS (step {plan.current_step}/{len(plan.steps)})"
        print(f"Plan status: {status}")
        print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
