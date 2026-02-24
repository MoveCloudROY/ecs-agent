"""Plan-and-Execute agent example with dynamic replanning.

This example demonstrates the Plan-and-Execute pattern with dynamic replanning:
- PlanningSystem drives a multi-step plan, calling the LLM at each step
- ToolExecutionSystem executes any tool calls the LLM makes
- ReplanningSystem reviews execution results after each step and may revise
  remaining plan steps based on what was learned
- This creates an adaptive planning loop:
    Plan → Execute Step → Observe Results → Revise Plan → Execute Next Step → ...

Usage:
  1. Copy .env.example to .env and fill in your API credentials
  2. Run: uv run python examples/plan_and_execute_agent.py

Environment variables:
  LLM_API_KEY   — API key for the LLM provider (required)
  LLM_BASE_URL  — Base URL for the API (default: https://dashscope.aliyuncs.com/compatible-mode/v1)
  LLM_MODEL     — Model name (default: qwen3.5-plus)
  LLM_CONNECT_TIMEOUT — Connection timeout in seconds (default: 10)
  LLM_READ_TIMEOUT    — Read timeout in seconds (default: 120)
  LLM_WRITE_TIMEOUT   — Write timeout in seconds (default: 10)
  LLM_POOL_TIMEOUT    — Connection pool timeout in seconds (default: 10)
  LLM_MAX_RETRIES     — Retries for transient network/HTTP errors (default: 3)
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
from ecs_agent.providers.retry_provider import RetryProvider
from ecs_agent.systems.error_handling import ErrorHandlingSystem
from ecs_agent.systems.memory import MemorySystem
from ecs_agent.systems.planning import PlanningSystem
from ecs_agent.systems.replanning import ReplanningSystem
from ecs_agent.systems.tool_execution import ToolExecutionSystem
from ecs_agent.types import (
    Message,
    PlanRevisedEvent,
    PlanStepCompletedEvent,
    RetryConfig,
    ToolSchema,
)


# ---------------------------------------------------------------------------
# Tool definitions — simulated tools for travel planning
# ---------------------------------------------------------------------------


async def get_weather(city: str) -> str:
    """Simulate fetching weather forecast for a city."""
    weather_db = {
        "beijing": (
            "Beijing 3-day forecast:\n"
            "  Day 1: Rainy, 15°C, humidity 75%\n"
            "  Day 2: Sunny, 22°C, humidity 40%\n"
            "  Day 3: Sunny, 24°C, humidity 35%"
        ),
        "shanghai": "Shanghai: Cloudy, 20-25°C, moderate humidity",
    }
    result = weather_db.get(city.lower())
    if result:
        return result
    return f"Weather data not available for {city}"


async def search_attractions(city: str) -> str:
    """Simulate searching for tourist attractions."""
    attractions_db = {
        "beijing": (
            "Top attractions in Beijing:\n"
            "  1. 故宫 (Forbidden City) — Imperial palace, indoor, 3-4 hours\n"
            "  2. 长城 (Great Wall) — Outdoor, full day trip\n"
            "  3. 天坛 (Temple of Heaven) — Park & temple, 2-3 hours\n"
            "  4. 颐和园 (Summer Palace) — Gardens & lake, 3-4 hours\n"
            "  5. 798艺术区 (798 Art District) — Indoor galleries, 2-3 hours\n"
            "  6. 国家博物馆 (National Museum) — Indoor, 3-4 hours"
        ),
    }
    result = attractions_db.get(city.lower())
    if result:
        return result
    return f"No attraction data for {city}"


async def search_restaurants(city: str, cuisine_type: str = "") -> str:
    """Simulate searching for restaurants."""
    restaurants_db = {
        "beijing": (
            "Recommended restaurants in Beijing:\n"
            "  1. 全聚德 (Quanjude) — Peking Duck, ¥200-300/person\n"
            "  2. 便宜坊 (Bianyifang) — Peking Duck, ¥150-250/person\n"
            "  3. 东来顺 (Donglaishun) — Hot Pot, ¥150-200/person\n"
            "  4. 护国寺小吃 (Huguosi Snacks) — Local snacks, ¥30-50/person\n"
            "  5. 南锣鼓巷小吃街 (Nanluoguxiang Food Street) — Street food, ¥20-60/person"
        ),
    }
    result = restaurants_db.get(city.lower())
    if result:
        return result
    return f"No restaurant data for {city}"


async def check_transport(from_city: str, to_city: str) -> str:
    """Simulate checking transport between cities."""
    return (
        f"Transport from {from_city} to {to_city}:\n"
        f"  High-speed rail: 4.5 hours, ¥550\n"
        f"  Flight: 2 hours, ¥800-1200\n"
        f"  Bus: 12 hours, ¥200"
    )


# ---------------------------------------------------------------------------
# Event handlers — observe the Plan-and-Execute loop in real-time
# ---------------------------------------------------------------------------


async def on_step_completed(event: PlanStepCompletedEvent) -> None:
    """Print progress as each plan step completes."""
    print(f"  ✓ Step {event.step_index + 1} completed: {event.step_description}")


async def on_plan_revised(event: PlanRevisedEvent) -> None:
    """Print when the plan is dynamically revised."""
    print()
    print("  📋 Plan revised!")
    print(f"    Old plan ({len(event.old_steps)} steps): {event.old_steps}")
    print(f"    New plan ({len(event.new_steps)} steps): {event.new_steps}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    """Run a Plan-and-Execute agent that plans a Beijing 3-day trip."""
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
    print(
        "Timeouts: "
        f"connect={connect_timeout}s read={read_timeout}s "
        f"write={write_timeout}s pool={pool_timeout}s"
    )
    print(f"LLM retries: {max_retries}")
    print()

    # --- Create LLM provider ---
    base_provider = OpenAIProvider(
        api_key=api_key,
        base_url=base_url,
        model=model,
        connect_timeout=connect_timeout,
        read_timeout=read_timeout,
        write_timeout=write_timeout,
        pool_timeout=pool_timeout,
    )
    provider = RetryProvider(
        base_provider,
        retry_config=RetryConfig(
            max_attempts=max_retries,
            multiplier=1.0,
            min_wait=1.0,
            max_wait=8.0,
        ),
    )

    # --- Define the initial plan ---
    plan_steps = [
        "Check the weather forecast for Beijing for the next 3 days using the get_weather tool",
        "Search for popular tourist attractions in Beijing using the search_attractions tool",
        "Search for recommended restaurants in Beijing using the search_restaurants tool",
        "Create a detailed 3-day itinerary based on all gathered information",
    ]

    # --- Define tool schemas ---
    tools = {
        "get_weather": ToolSchema(
            name="get_weather",
            description="Get the weather forecast for a city",
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
        "search_attractions": ToolSchema(
            name="search_attractions",
            description="Search for popular tourist attractions in a city",
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
        "search_restaurants": ToolSchema(
            name="search_restaurants",
            description="Search for recommended restaurants in a city",
            parameters={
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city name, e.g. 'Beijing'",
                    },
                    "cuisine_type": {
                        "type": "string",
                        "description": "Type of cuisine to search for (optional)",
                    },
                },
                "required": ["city"],
            },
        ),
        "check_transport": ToolSchema(
            name="check_transport",
            description="Check transport options between two cities",
            parameters={
                "type": "object",
                "properties": {
                    "from_city": {
                        "type": "string",
                        "description": "Departure city",
                    },
                    "to_city": {
                        "type": "string",
                        "description": "Destination city",
                    },
                },
                "required": ["from_city", "to_city"],
            },
        ),
    }

    # --- Build the ECS World ---
    world = World()

    # Create the agent entity
    main_agent = world.create_entity()

    # Attach components
    world.add_component(main_agent, LLMComponent(provider=provider, model=model))
    world.add_component(
        main_agent,
        ConversationComponent(
            messages=[
                Message(
                    role="user",
                    content="帮我规划一次北京三日游，我想了解天气、景点和美食",
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
                "You are a travel planning assistant using the Plan-and-Execute pattern. "
                "For each step in the plan, use the available tools to gather information. "
                "After gathering data, synthesize it into a detailed travel itinerary. "
                "Consider weather conditions when recommending activities — if it's rainy, "
                "prioritize indoor attractions. Always use tools when a step asks you to."
            ),
        ),
    )
    world.add_component(
        main_agent,
        ToolRegistryComponent(
            tools=tools,
            handlers={
                "get_weather": get_weather,
                "search_attractions": search_attractions,
                "search_restaurants": search_restaurants,
                "check_transport": check_transport,
            },
        ),
    )

    # Register systems (order matters for the Plan-and-Execute loop):
    # 1. PlanningSystem executes one plan step per tick (calls LLM)
    # 2. ToolExecutionSystem runs any tool calls the LLM made
    # 3. ReplanningSystem reviews results and may revise remaining steps
    # 4. MemorySystem truncates conversation if too long
    # 5. ErrorHandlingSystem handles any errors
    world.register_system(PlanningSystem(priority=0), priority=0)
    world.register_system(ToolExecutionSystem(priority=5), priority=5)
    world.register_system(ReplanningSystem(priority=7), priority=7)
    world.register_system(MemorySystem(), priority=10)
    world.register_system(ErrorHandlingSystem(priority=99), priority=99)

    # Subscribe to events for real-time progress
    world.event_bus.subscribe(PlanStepCompletedEvent, on_step_completed)
    world.event_bus.subscribe(PlanRevisedEvent, on_plan_revised)

    # --- Run the Plan-and-Execute loop ---
    print(f"Running Plan-and-Execute agent with {len(plan_steps)}-step initial plan...")
    print("Initial plan:")
    for i, step in enumerate(plan_steps):
        print(f"  {i + 1}. {step}")
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
        status = (
            "COMPLETED"
            if plan.completed
            else f"IN PROGRESS (step {plan.current_step}/{len(plan.steps)})"
        )
        print(f"Plan status: {status}")
        print(f"Final plan ({len(plan.steps)} steps):")
        for i, step in enumerate(plan.steps):
            marker = "✓" if i < plan.current_step else "○"
            print(f"  {marker} {i + 1}. {step}")
        print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
