"""Live smoke tests for Prometheus metrics instrumentation.

Select a profile with --live; see docs/testing.md.
"""

from __future__ import annotations


import pytest

from ecs_agent.components import ConversationComponent, LLMComponent
from ecs_agent.core import Runner, World
from ecs_agent.plugins import install_plugins, uninstall_plugins
from ecs_agent.plugins.prometheus import PrometheusPlugin, render_metrics
from ecs_agent.systems.error_handling import ErrorHandlingSystem
from ecs_agent.systems.reasoning import ReasoningSystem
from ecs_agent.types import ErrorOccurredEvent, Message

from tests.support.models import LiveModelFactory


@pytest.mark.live(
    formats=["openai_chat_completions", "openai_responses", "anthropic_messages"],
    capabilities=["text"],
    dependencies=["prometheus_client"],
)
@pytest.mark.observability
@pytest.mark.asyncio
async def test_live_prometheus_metrics_smoke(
    live_api_key: str, *, live_model_factory: LiveModelFactory
) -> None:
    """A real-model agent run updates and renders the Prometheus metrics registry."""
    world = World()
    plugin = PrometheusPlugin()
    await install_plugins(world, [plugin])
    metrics = plugin.metrics
    assert metrics is not None
    agent = world.create_entity()
    model = live_model_factory.create()
    world.add_component(
        agent,
        LLMComponent(
            model=model,
            system_prompt=(
                "Reply with exactly one short sentence. Do not include secrets, "
                "credentials, or environment values."
            ),
        ),
    )
    world.add_component(
        agent,
        ConversationComponent(
            messages=[Message(role="user", content="Say hello in five words or fewer.")]
        ),
    )
    world.register_system(ReasoningSystem(priority=0), priority=0)
    world.register_system(ErrorHandlingSystem(priority=99), priority=99)

    llm_errors: list[ErrorOccurredEvent] = []

    async def collect_error(event: ErrorOccurredEvent) -> None:
        llm_errors.append(event)

    world.event_bus.subscribe(ErrorOccurredEvent, collect_error)

    try:
        await Runner().run(world, max_ticks=2)
    finally:
        await uninstall_plugins(world)

    # LLM failures never raise through Runner.run: ReasoningSystem swallows
    # them into ErrorComponent and ErrorHandlingSystem removes that component
    # after publishing ErrorOccurredEvent, so the event is the durable signal.
    assert not llm_errors, f"Live agent run failed: {llm_errors[0].error[:200]}"

    conversation = world.get_component(agent, ConversationComponent)
    assert conversation is not None
    assert any(
        message.role == "assistant" and message.content
        for message in conversation.messages
    )

    output = render_metrics(metrics)
    assert b"ecs_agent_runs_total" in output
    assert b"ecs_agent_runner_ticks_total" in output
    assert b"ecs_agent_system_executions_total" in output
    assert b"ecs_agent_llm_invocations_total" in output
    assert b"ecs_agent_llm_invocation_duration_seconds" in output
    assert b'operation="reasoning"' in output
    assert b'status="success"' in output

    assert live_api_key.encode() not in output
    assert b"LLM_API_KEY" not in output
    assert b"entity_id" not in output
    assert b"request_id" not in output
    assert b"correlation_id" not in output
