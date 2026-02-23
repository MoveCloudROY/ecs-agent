from __future__ import annotations


import pytest

from ecs_agent.components import (
    ConversationComponent,
    ErrorComponent,
    LLMComponent,
    PlanComponent,
    PendingToolCallsComponent,
    SystemPromptComponent,
    TerminalComponent,
    ToolRegistryComponent,
)
from ecs_agent.core import Runner, World
from ecs_agent.providers.fake_provider import FakeProvider
from ecs_agent.systems.error_handling import ErrorHandlingSystem
from ecs_agent.systems.memory import MemorySystem
from ecs_agent.systems.planning import PlanningSystem
from ecs_agent.systems.reasoning import ReasoningSystem
from ecs_agent.systems.tool_execution import ToolExecutionSystem
from ecs_agent.types import (
    CompletionResult,
    EntityId,
    Message,
    PlanStepCompletedEvent,
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
            tool_calls=[
                ToolCall(id=call_id, name=tool_name, arguments=arguments)
            ],
        )
    )


def _build_react_world(
    provider: FakeProvider,
    plan_steps: list[str],
    tools: dict[str, ToolSchema] | None = None,
    handlers: dict[str, object] | None = None,
    system_prompt: str = "You are a ReAct agent. Think step by step.",
) -> tuple[World, EntityId]:
    """Build a World with full ReAct system stack."""
    world = World()
    agent_id = world.create_entity()

    world.add_component(agent_id, LLMComponent(provider=provider, model="fake"))
    world.add_component(agent_id, ConversationComponent(messages=[], max_messages=50))
    world.add_component(agent_id, PlanComponent(steps=plan_steps))
    world.add_component(agent_id, SystemPromptComponent(content=system_prompt))

    if tools is not None and handlers is not None:
        world.add_component(
            agent_id,
            ToolRegistryComponent(tools=tools, handlers=handlers),  # type: ignore[arg-type]
        )

    # PlanningSystem runs first (builds step context + calls LLM)
    # ToolExecutionSystem runs next (executes any tool calls)
    # ReasoningSystem is NOT registered — PlanningSystem drives the LLM directly
    # MemorySystem runs after to truncate if needed
    # ErrorHandlingSystem runs last as cleanup
    world.register_system(PlanningSystem(priority=0), priority=0)
    world.register_system(ToolExecutionSystem(priority=5), priority=5)
    world.register_system(MemorySystem(), priority=10)
    world.register_system(ErrorHandlingSystem(priority=99), priority=99)

    return world, agent_id


# ---------------------------------------------------------------------------
# Test 1: Pure reasoning ReAct — plan with multiple steps, no tool calls
# Verifies: PlanningSystem advances through all steps, plan marks completed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_react_pure_reasoning_completes_plan() -> None:
    """Agent follows a 3-step plan with pure reasoning (no tools)."""
    provider = FakeProvider(
        responses=[
            _reply("Step 1: I should gather information about the weather."),
            _reply("Step 2: Based on the data, it will rain tomorrow."),
            _reply("Step 3: I recommend bringing an umbrella."),
        ]
    )

    plan_steps = [
        "Gather weather information",
        "Analyze weather data",
        "Provide recommendation",
    ]

    completed_events: list[PlanStepCompletedEvent] = []

    async def on_step_completed(event: PlanStepCompletedEvent) -> None:
        completed_events.append(event)

    world, agent_id = _build_react_world(provider, plan_steps)
    world.event_bus.subscribe(PlanStepCompletedEvent, on_step_completed)

    runner = Runner()
    await runner.run(world, max_ticks=10)

    # Plan should be fully completed
    plan = world.get_component(agent_id, PlanComponent)
    assert plan is not None
    assert plan.completed is True
    assert plan.current_step == 3

    # All 3 step events should have fired in order
    assert len(completed_events) == 3
    assert completed_events[0].step_index == 0
    assert completed_events[0].step_description == "Gather weather information"
    assert completed_events[1].step_index == 1
    assert completed_events[2].step_index == 2

    # Conversation should contain the 3 assistant replies
    conv = world.get_component(agent_id, ConversationComponent)
    assert conv is not None
    assistant_msgs = [m for m in conv.messages if m.role == "assistant"]
    assert len(assistant_msgs) == 3
    assert "umbrella" in assistant_msgs[2].content

    # No errors
    assert world.get_component(agent_id, ErrorComponent) is None


# ---------------------------------------------------------------------------
# Test 2: ReAct with tool calls — Thought → Action → Observation loop
# Verifies: PlanningSystem produces tool call → ToolExecutionSystem executes
#           → next plan step sees tool result in conversation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_react_thought_action_observation_loop() -> None:
    """Agent executes a 2-step plan where step 1 requires a tool call.

    Tick 1: PlanningSystem processes step 1 → LLM returns tool_call(search)
    Tick 2: ToolExecutionSystem executes search → result appended to conversation
            PlanningSystem processes step 2 → LLM summarizes search result
    Tick 3: Plan completed, provider exhausted → TerminalComponent
    """
    search_results: list[str] = []

    async def search(query: str) -> str:
        search_results.append(query)
        return "Python was created by Guido van Rossum in 1991."

    provider = FakeProvider(
        responses=[
            # Step 1: LLM decides to use the search tool
            _tool_call_reply(
                "search", "tc_search_1", {"query": "Who created Python?"})
            ,
            # Step 2: LLM summarizes the observation
            _reply(
                "Based on the search result, Python was created by Guido van Rossum in 1991."
            ),
        ]
    )

    plan_steps = [
        "Search for information about Python's creator",
        "Summarize findings",
    ]

    world, agent_id = _build_react_world(
        provider,
        plan_steps,
        tools={
            "search": ToolSchema(
                name="search",
                description="Search for information",
                parameters={"type": "object", "properties": {"query": {"type": "string"}}},
            )
        },
        handlers={"search": search},
    )

    runner = Runner()
    await runner.run(world, max_ticks=10)

    # Plan completed
    plan = world.get_component(agent_id, PlanComponent)
    assert plan is not None
    assert plan.completed is True
    assert plan.current_step == 2

    # Tool was actually executed
    assert search_results == ["Who created Python?"]

    # Conversation should contain:
    #   1. assistant message (tool call for step 1)
    #   2. tool result
    #   3. assistant message (summary for step 2)
    conv = world.get_component(agent_id, ConversationComponent)
    assert conv is not None

    tool_msgs = [m for m in conv.messages if m.role == "tool"]
    assert len(tool_msgs) == 1
    assert "Guido van Rossum" in tool_msgs[0].content

    assistant_msgs = [m for m in conv.messages if m.role == "assistant"]
    assert len(assistant_msgs) == 2
    assert "Guido van Rossum" in assistant_msgs[1].content

    # No errors
    assert world.get_component(agent_id, ErrorComponent) is None


# ---------------------------------------------------------------------------
# Test 3: Multi-step ReAct with multiple tool calls across steps
# Verifies: Each plan step can independently issue tool calls
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_react_multiple_steps_with_tools() -> None:
    """Agent executes a 3-step plan: step 1 and step 2 each call a tool,
    step 3 synthesizes all observations into a final answer.
    """
    call_log: list[tuple[str, dict[str, str]]] = []

    async def get_price(symbol: str) -> str:
        call_log.append(("get_price", {"symbol": symbol}))
        prices = {"AAPL": "182.50", "GOOGL": "141.80"}
        return f"${prices.get(symbol, 'N/A')}"

    provider = FakeProvider(
        responses=[
            # Step 1: Look up AAPL price
            _tool_call_reply("get_price", "tc1", {"symbol": "AAPL"}),
            # Step 2: Look up GOOGL price
            _tool_call_reply("get_price", "tc2", {"symbol": "GOOGL"}),
            # Step 3: Synthesize
            _reply("AAPL is $182.50 and GOOGL is $141.80. AAPL has the higher price."),
        ]
    )

    plan_steps = [
        "Look up AAPL stock price",
        "Look up GOOGL stock price",
        "Compare and report which is higher",
    ]

    world, agent_id = _build_react_world(
        provider,
        plan_steps,
        tools={
            "get_price": ToolSchema(
                name="get_price",
                description="Get stock price",
                parameters={"type": "object", "properties": {"symbol": {"type": "string"}}},
            )
        },
        handlers={"get_price": get_price},
    )

    runner = Runner()
    await runner.run(world, max_ticks=15)

    # Plan completed
    plan = world.get_component(agent_id, PlanComponent)
    assert plan is not None
    assert plan.completed is True

    # Both tools were called in order
    assert len(call_log) == 2
    assert call_log[0] == ("get_price", {"symbol": "AAPL"})
    assert call_log[1] == ("get_price", {"symbol": "GOOGL"})

    # Conversation has 2 tool results + 3 assistant messages
    conv = world.get_component(agent_id, ConversationComponent)
    assert conv is not None

    tool_msgs = [m for m in conv.messages if m.role == "tool"]
    assert len(tool_msgs) == 2
    assert "$182.50" in tool_msgs[0].content
    assert "$141.80" in tool_msgs[1].content

    assistant_msgs = [m for m in conv.messages if m.role == "assistant"]
    assert len(assistant_msgs) == 3
    assert "higher" in assistant_msgs[2].content.lower()

    # No errors
    assert world.get_component(agent_id, ErrorComponent) is None


# ---------------------------------------------------------------------------
# Test 4: ReAct handles tool failure gracefully within a plan step
# Verifies: Tool execution error is returned as observation, plan continues
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_react_tool_failure_does_not_break_plan() -> None:
    """When a tool raises an exception during a plan step,
    the error message becomes the observation and the plan continues.
    """

    async def flaky_tool(x: str) -> str:
        raise ValueError(f"Connection timeout for {x}")

    provider = FakeProvider(
        responses=[
            # Step 1: Try to use the flaky tool
            _tool_call_reply("flaky_tool", "tc_flaky", {"x": "data"}),
            # Step 2: LLM sees the error observation and adapts
            _reply("The tool failed with a timeout. I'll provide a fallback answer."),
        ]
    )

    plan_steps = [
        "Fetch data using the tool",
        "Handle the result or error",
    ]

    world, agent_id = _build_react_world(
        provider,
        plan_steps,
        tools={
            "flaky_tool": ToolSchema(
                name="flaky_tool",
                description="A tool that might fail",
                parameters={},
            )
        },
        handlers={"flaky_tool": flaky_tool},
    )

    runner = Runner()
    await runner.run(world, max_ticks=10)

    # Plan still completes despite tool failure
    plan = world.get_component(agent_id, PlanComponent)
    assert plan is not None
    assert plan.completed is True

    # The error observation is in conversation
    conv = world.get_component(agent_id, ConversationComponent)
    assert conv is not None
    tool_msgs = [m for m in conv.messages if m.role == "tool"]
    assert len(tool_msgs) == 1
    assert "Connection timeout" in tool_msgs[0].content

    # Final assistant message acknowledges the failure
    assistant_msgs = [m for m in conv.messages if m.role == "assistant"]
    assert any("fallback" in m.content.lower() for m in assistant_msgs)

    # No ErrorComponent — tool error is handled as observation, not system error
    assert world.get_component(agent_id, ErrorComponent) is None


# ---------------------------------------------------------------------------
# Test 5: PlanStepCompletedEvent fires correctly with tool-call steps
# Verifies: Events fire for every step including those with tool calls
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_react_events_fire_for_all_steps() -> None:
    """PlanStepCompletedEvent fires for every plan step,
    including steps that produce tool calls.
    """
    events: list[PlanStepCompletedEvent] = []

    async def noop_tool(q: str) -> str:
        return "42"

    async def on_step(event: PlanStepCompletedEvent) -> None:
        events.append(event)

    provider = FakeProvider(
        responses=[
            _tool_call_reply("calc", "tc1", {"q": "6*7"}),
            _reply("The answer to life is 42."),
            _reply("All done."),
        ]
    )

    plan_steps = ["Calculate 6*7", "Interpret result", "Report"]

    world, agent_id = _build_react_world(
        provider,
        plan_steps,
        tools={
            "calc": ToolSchema(name="calc", description="Calculate", parameters={})
        },
        handlers={"calc": noop_tool},
    )
    world.event_bus.subscribe(PlanStepCompletedEvent, on_step)

    runner = Runner()
    await runner.run(world, max_ticks=10)

    # All 3 steps should fire events
    assert len(events) == 3
    assert [e.step_index for e in events] == [0, 1, 2]
    assert events[0].step_description == "Calculate 6*7"
    assert events[1].step_description == "Interpret result"
    assert events[2].step_description == "Report"

    # Plan completed
    plan = world.get_component(agent_id, PlanComponent)
    assert plan is not None
    assert plan.completed is True


# ---------------------------------------------------------------------------
# Test 6: ReAct agent terminates correctly after plan completes
# Verifies: After plan is done, provider exhaustion triggers TerminalComponent
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_react_terminates_after_plan_completion() -> None:
    """Once the plan is completed, PlanningSystem stops issuing LLM calls.
    The agent terminates via provider exhaustion on the ReasoningSystem side,
    or simply runs out of work and hits max_ticks.
    """
    provider = FakeProvider(
        responses=[
            _reply("Done with step 1."),
        ]
    )

    world, agent_id = _build_react_world(provider, ["Single step"])

    runner = Runner()
    await runner.run(world, max_ticks=5)

    plan = world.get_component(agent_id, PlanComponent)
    assert plan is not None
    assert plan.completed is True

    # Should hit max_ticks since no ReasoningSystem is registered
    # and plan is already done — no TerminalComponent from provider exhaustion
    terminal = world.get_component(agent_id, TerminalComponent)
    # Either terminal from max_ticks or no terminal (plan just completed)
    # The runner creates a max_ticks terminal on a NEW entity if no terminal exists
    terminals = list(world.query(TerminalComponent))
    assert len(terminals) >= 1  # At least max_ticks terminal exists
