"""Live smoke tests for Prometheus metrics instrumentation.

Run with real credentials:
    LLM_API_KEY=<key>
    LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
    LLM_MODEL=qwen3.5-flash
    LLM_API_FORMAT=openai_chat_completions  # optional

The suite skips cleanly when ``LLM_API_KEY`` is not set.
"""

from __future__ import annotations

import os

import pytest

from ecs_agent.components import ConversationComponent, LLMComponent
from ecs_agent.core import Runner, World
from ecs_agent.plugins import install_plugins, uninstall_plugins
from ecs_agent.plugins.prometheus import PrometheusPlugin, render_metrics
from ecs_agent.providers import Model
from ecs_agent.providers.config import ApiFormat
from ecs_agent.providers.protocol import LLMModel
from ecs_agent.systems.error_handling import ErrorHandlingSystem
from ecs_agent.systems.reasoning import ReasoningSystem
from ecs_agent.types import ErrorOccurredEvent, Message
from tests.live.api_format import live_transient_error_reason, resolve_live_api_format


def _api_format_from_env() -> ApiFormat:
    api_format = resolve_live_api_format()
    if api_format is None:
        pytest.skip(
            "Unsupported LLM_API_FORMAT for live metrics smoke: "
            f"{os.getenv('LLM_API_FORMAT')!r}"
        )
    return api_format


def _make_live_model(api_key: str) -> LLMModel:
    return Model(
        os.getenv("LLM_MODEL", "qwen3.5-flash"),
        base_url=os.getenv(
            "LLM_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        ),
        api_key=api_key,
        api_format=_api_format_from_env(),
        provider_id="live",
        timeout=30.0,
    )


@pytest.mark.asyncio
async def test_live_prometheus_metrics_smoke(live_api_key: str) -> None:
    """A real-model agent run updates and renders the Prometheus metrics registry."""
    world = World()
    plugin = PrometheusPlugin()
    await install_plugins(world, [plugin])
    metrics = plugin.metrics
    assert metrics is not None
    agent = world.create_entity()
    model = _make_live_model(live_api_key)
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
    if llm_errors:
        hard_failures = [
            event.error
            for event in llm_errors
            if live_transient_error_reason(event.error) is None
        ]
        if hard_failures:
            pytest.fail(f"Live agent run failed: {hard_failures[0][:200]}")
        pytest.skip(
            "Transient live endpoint error during Prometheus metrics smoke: "
            f"{live_transient_error_reason(llm_errors[0].error)}"
        )

    conversation = world.get_component(agent, ConversationComponent)
    assert conversation is not None
    assert any(
        message.role == "assistant" and message.content for message in conversation.messages
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
