from __future__ import annotations

import json

import pytest

from ecs_agent.components import (
    ConversationComponent,
    ErrorComponent,
    LLMComponent,
    PlanComponent,
    SystemPromptComponent,
    ToolRegistryComponent,
)
from ecs_agent.core import Runner, World
from ecs_agent.providers.fake_provider import FakeProvider
from ecs_agent.systems.error_handling import ErrorHandlingSystem
from ecs_agent.systems.memory import MemorySystem
from ecs_agent.systems.planning import PlanningSystem
from ecs_agent.systems.replanning import ReplanningSystem
from ecs_agent.systems.tool_execution import ToolExecutionSystem
from ecs_agent.types import (
    CompletionResult,
    Message,
    PlanRevisedEvent,
    ToolCall,
    ToolSchema,
)


def _reply(content: str) -> CompletionResult:
    return CompletionResult(message=Message(role="assistant", content=content))


def _tool_call_reply(tool_name: str, call_id: str, arguments: dict[str, str]) -> CompletionResult:
    return CompletionResult(
        message=Message(
            role="assistant",
            content="",
            tool_calls=[ToolCall(id=call_id, name=tool_name, arguments=arguments)],
        )
    )


def _replan_reply(revised_steps: list[str]) -> CompletionResult:
    content = json.dumps({"revised_steps": revised_steps})
    return CompletionResult(message=Message(role="assistant", content=content))


def _build_plan_execute_world(
    provider: FakeProvider,
    plan_steps: list[str],
    tools: dict[str, ToolSchema] | None = None,
    handlers: dict[str, object] | None = None,
    user_message: str = "Help me with the task",
) -> tuple[World, int]:
    """Build a World with PlanningSystem + ToolExecution + ReplanningSystem."""
    world = World()
    agent_id = world.create_entity()

    world.add_component(agent_id, LLMComponent(provider=provider, model="fake"))
    world.add_component(
        agent_id,
        ConversationComponent(
            messages=[Message(role="user", content=user_message)],
            max_messages=50,
        ),
    )
    world.add_component(agent_id, PlanComponent(steps=plan_steps))
    world.add_component(
        agent_id,
        SystemPromptComponent(content="You are a plan-and-execute agent."),
    )

    if tools is not None and handlers is not None:
        world.add_component(
            agent_id,
            ToolRegistryComponent(tools=tools, handlers=handlers),  # type: ignore[arg-type]
        )

    world.register_system(PlanningSystem(priority=0), priority=0)
    world.register_system(ToolExecutionSystem(priority=5), priority=5)
    world.register_system(ReplanningSystem(priority=7), priority=7)
    world.register_system(MemorySystem(), priority=10)
    world.register_system(ErrorHandlingSystem(priority=99), priority=99)

    return world, agent_id


# ---------------------------------------------------------------------------
# Test 1: Full plan-and-execute with replanning that changes steps
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_plan_and_execute_with_replanning() -> None:
    """3-step plan where replanning after step 1 revises remaining steps.

    Tick 1:
      PlanningSystem step 1 → tool call (search)
      ToolExecutionSystem → skipped (PendingToolCalls just added this tick,
        but ToolExec queries it and runs immediately)
      ReplanningSystem → current_step=1 > last_replanned=0, triggers replan

    Tick 2:
      PlanningSystem step 2 (revised) → tool call
      ToolExecutionSystem → executes step 2 tool
      ReplanningSystem → current_step=2 > last_replanned=1, triggers replan

    Tick 3:
      PlanningSystem step 3 (revised) → final answer
      Plan completes
    """
    call_log: list[str] = []

    async def search(query: str) -> str:
        call_log.append(query)
        return f"Results for: {query}"

    async def analyze(data: str) -> str:
        call_log.append(f"analyze:{data}")
        return f"Analysis of: {data}"

    provider = FakeProvider(
        responses=[
            # Tick 1: PlanningSystem step 1 → tool call
            _tool_call_reply("search", "tc1", {"query": "weather beijing"}),
            # Tick 1: ReplanningSystem after step 1 → revise remaining
            _replan_reply(["Analyze weather data with analyze tool", "Create final itinerary"]),
            # Tick 2: PlanningSystem step 2 (revised) → tool call
            _tool_call_reply("analyze", "tc2", {"data": "weather info"}),
            # Tick 2: ReplanningSystem after step 2 → keep steps
            _replan_reply(["Create final itinerary"]),
            # Tick 3: PlanningSystem step 3 → final answer
            _reply("Here is your Beijing itinerary based on analysis."),
        ]
    )

    plan_steps = [
        "Search for weather",
        "Search for attractions",
        "Create itinerary",
    ]

    world, agent_id = _build_plan_execute_world(
        provider,
        plan_steps,
        tools={
            "search": ToolSchema(
                name="search",
                description="Search for info",
                parameters={"type": "object", "properties": {"query": {"type": "string"}}},
            ),
            "analyze": ToolSchema(
                name="analyze",
                description="Analyze data",
                parameters={"type": "object", "properties": {"data": {"type": "string"}}},
            ),
        },
        handlers={"search": search, "analyze": analyze},
    )

    runner = Runner()
    await runner.run(world, max_ticks=20)

    plan = world.get_component(agent_id, PlanComponent)
    assert plan is not None
    assert plan.completed is True

    # Verify steps were revised: original step2 "Search for attractions" was replaced
    assert plan.steps[0] == "Search for weather"  # preserved
    assert "Analyze" in plan.steps[1]  # revised by replan

    # Tools were actually called
    assert len(call_log) >= 2

    # Final conversation has the itinerary
    conv = world.get_component(agent_id, ConversationComponent)
    assert conv is not None
    assistant_msgs = [m for m in conv.messages if m.role == "assistant" and m.content]
    assert any("itinerary" in m.content.lower() for m in assistant_msgs)

    assert world.get_component(agent_id, ErrorComponent) is None


# ---------------------------------------------------------------------------
# Test 2: Plan completes normally when replanning keeps same steps
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_plan_and_execute_no_replanning_needed() -> None:
    """All replan responses return same remaining steps. Plan completes normally."""

    provider = FakeProvider(
        responses=[
            # Tick 1: PlanningSystem step 1
            _reply("Step 1 done via reasoning."),
            # Tick 1: ReplanningSystem → keep same steps
            _replan_reply(["Do step 2", "Do step 3"]),
            # Tick 2: PlanningSystem step 2
            _reply("Step 2 done."),
            # Tick 2: ReplanningSystem → keep same steps
            _replan_reply(["Do step 3"]),
            # Tick 3: PlanningSystem step 3
            _reply("Step 3 done. All complete."),
        ]
    )

    world, agent_id = _build_plan_execute_world(
        provider, ["Do step 1", "Do step 2", "Do step 3"]
    )

    runner = Runner()
    await runner.run(world, max_ticks=20)

    plan = world.get_component(agent_id, PlanComponent)
    assert plan is not None
    assert plan.completed is True
    assert plan.current_step == 3

    conv = world.get_component(agent_id, ConversationComponent)
    assert conv is not None
    assistant_msgs = [m for m in conv.messages if m.role == "assistant"]
    # 3 planning responses + 2 replan responses = 5 total assistant messages
    # (replan responses are NOT added to conversation by ReplanningSystem)
    # Actually ReplanningSystem calls provider.complete() but does NOT append to conversation
    assert len(assistant_msgs) == 3


# ---------------------------------------------------------------------------
# Test 3: Replanning adds extra steps mid-execution
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_plan_and_execute_replanning_adds_steps() -> None:
    """Replanning adds extra steps. Verify total steps increased and all execute."""

    provider = FakeProvider(
        responses=[
            # Tick 1: PlanningSystem step 1
            _reply("Gathered weather info: rainy."),
            # Tick 1: ReplanningSystem → add extra step
            _replan_reply([
                "Find indoor activities due to rain",
                "Search restaurants",
                "Create itinerary",
            ]),
            # Tick 2: PlanningSystem step 2 ("Find indoor activities")
            _reply("Found museums and galleries."),
            # Tick 2: ReplanningSystem → keep remaining
            _replan_reply(["Search restaurants", "Create itinerary"]),
            # Tick 3: PlanningSystem step 3 ("Search restaurants")
            _reply("Found great restaurants."),
            # Tick 3: ReplanningSystem → keep remaining
            _replan_reply(["Create itinerary"]),
            # Tick 4: PlanningSystem step 4 ("Create itinerary")
            _reply("Here is your 3-day itinerary with indoor activities."),
        ]
    )

    original_steps = [
        "Check weather",
        "Search attractions",
        "Create itinerary",
    ]

    world, agent_id = _build_plan_execute_world(provider, original_steps)

    runner = Runner()
    await runner.run(world, max_ticks=20)

    plan = world.get_component(agent_id, PlanComponent)
    assert plan is not None
    assert plan.completed is True
    # Originally 3 steps, replanning added 1 more ("Find indoor activities")
    assert len(plan.steps) == 4
    assert plan.current_step == 4

    conv = world.get_component(agent_id, ConversationComponent)
    assert conv is not None
    assistant_msgs = [m for m in conv.messages if m.role == "assistant"]
    assert len(assistant_msgs) == 4  # 4 planning step responses


# ---------------------------------------------------------------------------
# Test 4: PlanRevisedEvent fires with correct data
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_plan_revised_event_fired() -> None:
    """PlanRevisedEvent fires with correct old_steps and new_steps."""

    provider = FakeProvider(
        responses=[
            # Tick 1: PlanningSystem step 1
            _reply("Weather is rainy."),
            # Tick 1: ReplanningSystem → revise steps
            _replan_reply(["Find indoor activities", "Create rainy-day itinerary"]),
            # Tick 2: PlanningSystem step 2
            _reply("Found museums."),
            # Tick 2: ReplanningSystem → keep same
            _replan_reply(["Create rainy-day itinerary"]),
            # Tick 3: PlanningSystem step 3
            _reply("Itinerary complete."),
        ]
    )

    original_steps = ["Check weather", "Search attractions", "Create itinerary"]

    events: list[PlanRevisedEvent] = []

    async def on_revised(event: PlanRevisedEvent) -> None:
        events.append(event)

    world, agent_id = _build_plan_execute_world(provider, list(original_steps))
    world.event_bus.subscribe(PlanRevisedEvent, on_revised)

    runner = Runner()
    await runner.run(world, max_ticks=20)

    # First replan changed steps, second didn't
    assert len(events) == 1
    assert events[0].entity_id == agent_id
    assert events[0].old_steps == ["Check weather", "Search attractions", "Create itinerary"]
    assert events[0].new_steps == ["Check weather", "Find indoor activities", "Create rainy-day itinerary"]

    plan = world.get_component(agent_id, PlanComponent)
    assert plan is not None
    assert plan.completed is True
