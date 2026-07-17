"""Deterministic prompt-cache prefix audits across all three wire formats.

Provider prompt caches hit only when request N's rendered prompt is a
byte-prefix of request N+1's. Each test drives production systems
(ReasoningSystem, ToolExecutionSystem, PromptContextCollectorSystem) with a
scripted FakeModel, captures every outbound call, renders it through the real
adapters (Anthropic Messages / OpenAI Chat / OpenAI Responses) and audits
consecutive requests. Run with ``-s`` to see the divergence evidence.
"""

from __future__ import annotations

import pytest

from ecs_agent.components import (
    PromptContextQueueComponent,
    UserPromptConfigComponent,
)
from ecs_agent.systems.prompt_context_collector import PromptContextCollectorSystem
from ecs_agent.types import CompletionResult, Message, ToolCall

from tests.cache_audit.harness import (
    WIRE_FORMATS,
    audit_captured_calls,
    build_audit_world,
    run_turn,
    scripted_two_turn_responses,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("wire_format", WIRE_FORMATS)
async def test_default_tool_loop_prefix_is_append_only(wire_format: str) -> None:
    """Default flow (no context pool): every call must extend the previous one."""
    world, entity_id, recorder, systems = build_audit_world(
        responses=scripted_two_turn_responses()
    )
    await run_turn(world, entity_id, systems, "What is the weather in Paris and London?")
    await run_turn(world, entity_id, systems, "And how about tomorrow?")

    assert len(recorder.calls) == 4
    report = audit_captured_calls(recorder.calls, wire_format)
    for divergence in report.divergences:
        print(f"\n[{wire_format}] {divergence.describe()}")
    assert report.clean, (
        f"{wire_format}: default agentic loop broke the cache prefix — "
        + "; ".join(d.describe() for d in report.divergences)
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("wire_format", WIRE_FORMATS)
async def test_context_pool_injection_keeps_prefix_append_only(
    wire_format: str,
) -> None:
    """Context pool (enable_context_pool=True) must not rewrite sent history.

    The pool injects tool results into the *last user message*, which sits in
    the middle of the prompt during an agentic loop. Any per-call variation
    there (entry churn, timestamps) invalidates every subsequent token on
    every call — the worst possible cache shape.
    """
    world, entity_id, recorder, systems = build_audit_world(
        responses=scripted_two_turn_responses()
    )
    world.add_component(entity_id, UserPromptConfigComponent(enable_context_pool=True))
    world.add_component(entity_id, PromptContextQueueComponent())
    collector = PromptContextCollectorSystem(priority=-5)

    await run_turn(
        world,
        entity_id,
        systems,
        "What is the weather in Paris and London?",
        pre_tick_systems=[collector],
    )
    await run_turn(
        world,
        entity_id,
        systems,
        "And how about tomorrow?",
        pre_tick_systems=[collector],
    )

    assert len(recorder.calls) == 4
    report = audit_captured_calls(recorder.calls, wire_format)
    for divergence in report.divergences:
        print(f"\n[{wire_format}] {divergence.describe()}")
    assert report.clean, (
        f"{wire_format}: context-pool injection broke the cache prefix — "
        + "; ".join(d.describe() for d in report.divergences)
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("wire_format", WIRE_FORMATS)
async def test_volatile_suffix_change_cost_is_quantified(wire_format: str) -> None:
    """Changing the volatile system suffix (phase transition / compaction
    refresh) sits *ahead* of the whole conversation, so it forfeits the entire
    history cache. This audit quantifies the retention drop."""
    world, entity_id, recorder, systems = build_audit_world(
        responses=scripted_two_turn_responses(),
        volatile_suffix="## Current phase\nPhase: DRAFT_INTERVIEW — interview the user.",
    )
    await run_turn(world, entity_id, systems, "What is the weather in Paris and London?")

    # Simulate a phase transition: the render system rewrites volatile_text.
    from ecs_agent.components import RenderedSystemPromptComponent

    rendered = world.get_component(entity_id, RenderedSystemPromptComponent)
    assert rendered is not None
    rendered.volatile_text = "## Current phase\nPhase: WRITE_PLAN — produce the plan."

    await run_turn(world, entity_id, systems, "And how about tomorrow?")

    report = audit_captured_calls(recorder.calls, wire_format)
    # Calls 0->1 (within phase) must be clean; the transition call diverges at
    # the volatile system unit by design.
    transition = [d for d in report.divergences if d.call_index == 2]
    within_phase = [d for d in report.divergences if d.call_index != 2]
    for divergence in report.divergences:
        print(f"\n[{wire_format}] {divergence.describe()}")
    assert not within_phase, (
        f"{wire_format}: prefix broke within a stable phase — "
        + "; ".join(d.describe() for d in within_phase)
    )
    assert transition, "expected the volatile flip to surface in the audit"
    print(
        f"\n[{wire_format}] phase transition retention: "
        f"{transition[0].retention:.0%} of the previous prompt stayed cacheable"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("wire_format", WIRE_FORMATS)
async def test_script_trigger_rendered_prompt_stays_stable_across_turns(
    wire_format: str,
) -> None:
    """plan_and_task-shaped flow: a script trigger replaces the slash command
    text with generated dispatch text for the turn's calls. Once the turn
    advances, the rendered bytes must stay in the history — reverting to the
    raw slash text would flip a mid-history message and forfeit the cache for
    everything after it, every turn."""
    from ecs_agent.components import ConversationComponent
    from ecs_agent.prompts.contracts import TriggerSpec
    from ecs_agent.systems.user_prompt_normalization_system import (
        UserPromptNormalizationSystem,
    )
    from ecs_agent.core import World
    from ecs_agent.types import EntityId

    world, entity_id, recorder, systems = build_audit_world(
        responses=scripted_two_turn_responses()
    )

    async def _plan_start(
        world: World, entity_id: EntityId, raw_user_text: str
    ) -> str:
        return (
            "You have a new workflow. Interview the user about the goal, then "
            "draft scope, constraints and acceptance criteria.\n"
            f"Original command: {raw_user_text}"
        )

    world.add_component(
        entity_id,
        UserPromptConfigComponent(
            triggers=[
                TriggerSpec(
                    pattern="/plan:start",
                    match_mode="prefix",
                    action="script",
                    content="plan_start",
                )
            ],
            script_handlers={"plan_start": _plan_start},
        ),
    )
    normalization = UserPromptNormalizationSystem(priority=-10)

    await run_turn(
        world,
        entity_id,
        systems,
        "/plan:start build a todo app",
        pre_tick_systems=[normalization],
    )
    await run_turn(
        world,
        entity_id,
        systems,
        "continue with the next step",
        pre_tick_systems=[normalization],
    )

    assert len(recorder.calls) == 4
    report = audit_captured_calls(recorder.calls, wire_format)
    for divergence in report.divergences:
        print(f"\n[{wire_format}] {divergence.describe()}")
    assert report.clean, (
        f"{wire_format}: script-trigger rendering flipped a sent message — "
        + "; ".join(d.describe() for d in report.divergences)
    )
    # The dispatch text (what the model saw in turn 1) must be what history
    # keeps — not the raw slash command.
    conversation = world.get_component(entity_id, ConversationComponent)
    assert conversation is not None
    assert conversation.messages[0].content.startswith("You have a new workflow.")


@pytest.mark.asyncio
async def test_anthropic_block_growth_per_call_stays_within_lookback() -> None:
    """Anthropic finds the previous cache entry by looking back at most ~20
    content blocks from the (single, trailing) breakpoint. A turn that lands
    more than ~20 blocks between two calls silently misses the whole prefix.

    This audit measures blocks-per-call growth in a wide parallel tool batch
    (as produced by concurrency-enabled tool execution).
    """
    calls_per_turn = 12  # 12 tool_use + 12 tool_result blocks per loop turn
    responses = [
        CompletionResult(
            message=Message(
                role="assistant",
                content="",
                tool_calls=[
                    ToolCall(
                        id=f"call_wide_{i}",
                        name="lookup_weather",
                        arguments={"city": f"City{i}"},
                    )
                    for i in range(calls_per_turn)
                ],
            )
        ),
        CompletionResult(
            message=Message(role="assistant", content="All cities reported.")
        ),
    ]
    world, entity_id, recorder, systems = build_audit_world(responses=responses)
    await run_turn(world, entity_id, systems, "Weather for the twelve cities, please.")

    report = audit_captured_calls(recorder.calls, "anthropic_messages")
    assert report.clean
    assert len(report.block_counts) == 2
    growth = report.block_counts[1] - report.block_counts[0]
    print(
        f"\n[anthropic] blocks per call: {report.block_counts}; "
        f"growth between call 0 and call 1 = {growth} blocks "
        f"(lookback window is ~20)"
    )
    assert growth > 20, (
        "expected this scenario to demonstrate >20-block growth; "
        "adjust calls_per_turn if the message shape changed"
    )
