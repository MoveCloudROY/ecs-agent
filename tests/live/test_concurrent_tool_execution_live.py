"""Live smoke: concurrency-safe tools batched by a real LLM run concurrently.

Drives an OpenAI-family endpoint (``LLM_BASE_URL``/``LLM_MODEL``/``LLM_API_FORMAT``,
gated on ``LLM_API_KEY``) with two artificially slow, concurrency-safe lookup
tools and asks the model to call both in one response.  Verifies that
ToolExecutionSystem overlaps their execution and still lands tool results in
the model's original tool_calls order.
"""

from __future__ import annotations

import asyncio
import os
import time

import pytest

from ecs_agent.components import (
    ConversationComponent,
    LLMComponent,
    ToolRegistryComponent,
)
from ecs_agent.core import Runner, World
from ecs_agent.providers.config import ApiFormat, ProviderConfig
from ecs_agent.providers.openai_model import OpenAIModel
from ecs_agent.systems.error_handling import ErrorHandlingSystem
from ecs_agent.systems.reasoning import ReasoningSystem
from ecs_agent.systems.tool_execution import ToolExecutionSystem
from ecs_agent.types import ErrorOccurredEvent, Message, ToolSchema
from tests.live.api_format import (
    live_openai_base_url,
    live_openai_model,
    live_transient_error_reason,
    resolve_live_api_format,
)

TOOL_DELAY_SECONDS = 1.0


def _make_model(live_api_key: str) -> OpenAIModel:
    api_format = resolve_live_api_format()
    if api_format is None:
        pytest.skip(
            "Unsupported LLM_API_FORMAT for live concurrency smoke: "
            f"{os.getenv('LLM_API_FORMAT')!r}"
        )
    if api_format is ApiFormat.ANTHROPIC_MESSAGES:
        pytest.skip(
            "Concurrency smoke drives OpenAI-family endpoints; "
            "unset LLM_API_FORMAT or use chat/responses"
        )
    config = ProviderConfig(
        provider_id="live",
        base_url=live_openai_base_url(api_format),
        api_key=live_api_key,
        api_format=api_format,
    )
    return OpenAIModel(config=config, model=live_openai_model())


def _lookup_schema(name: str, param: str, description: str) -> ToolSchema:
    return ToolSchema(
        name=name,
        description=description,
        parameters={
            "type": "object",
            "properties": {param: {"type": "string"}},
            "required": [param],
        },
        concurrency_safe=True,
    )


@pytest.mark.asyncio
async def test_live_batched_safe_tools_overlap_and_land_in_order(
    live_api_key: str,
) -> None:
    spans: dict[str, tuple[float, float]] = {}

    def make_handler(name: str, payload: str):  # noqa: ANN202 - test helper
        async def handler(**kwargs: str) -> str:
            _ = kwargs
            start = time.monotonic()
            await asyncio.sleep(TOOL_DELAY_SECONDS)
            spans[name] = (start, time.monotonic())
            return payload

        return handler

    world = World()
    agent = world.create_entity()
    world.add_component(
        agent,
        LLMComponent(
            model=_make_model(live_api_key),
            system_prompt=(
                "You are a precise assistant. When a request needs several "
                "independent lookups, call all the needed tools in parallel "
                "in a single response instead of one at a time."
            ),
        ),
    )
    world.add_component(
        agent,
        ToolRegistryComponent(
            tools={
                "lookup_weather": _lookup_schema(
                    "lookup_weather",
                    "city",
                    "Look up the current weather for a city.",
                ),
                "lookup_time": _lookup_schema(
                    "lookup_time",
                    "zone",
                    "Look up the current time in a timezone.",
                ),
            },
            handlers={
                "lookup_weather": make_handler("lookup_weather", "sunny, 21C"),
                "lookup_time": make_handler("lookup_time", "10:00"),
            },
        ),
    )
    world.add_component(
        agent,
        ConversationComponent(
            messages=[
                Message(
                    role="user",
                    content=(
                        "Get the current weather in Paris with lookup_weather and "
                        "the current time in UTC with lookup_time. Call both tools "
                        "in parallel in your first response, then summarize both "
                        "results in one sentence."
                    ),
                )
            ]
        ),
    )

    world.register_system(ReasoningSystem(priority=0), priority=0)
    world.register_system(ToolExecutionSystem(), priority=5)
    world.register_system(ErrorHandlingSystem(priority=99), priority=99)

    llm_errors: list[ErrorOccurredEvent] = []

    async def collect_error(event: ErrorOccurredEvent) -> None:
        llm_errors.append(event)

    world.event_bus.subscribe(ErrorOccurredEvent, collect_error)

    await Runner().run(world, max_ticks=4)

    # LLM failures never raise through Runner.run: ReasoningSystem swallows
    # them into ErrorComponent and ErrorHandlingSystem removes it after
    # publishing ErrorOccurredEvent, so the event is the durable signal.
    if llm_errors:
        hard_failures = [
            event.error
            for event in llm_errors
            if live_transient_error_reason(event.error) is None
        ]
        if hard_failures:
            pytest.fail(f"Live agent run failed: {hard_failures[0][:200]}")
        pytest.skip(
            "Transient live endpoint error: "
            f"{live_transient_error_reason(llm_errors[0].error)}"
        )

    conversation = world.get_component(agent, ConversationComponent)
    assert conversation is not None

    batched = next(
        (
            message
            for message in conversation.messages
            if message.role == "assistant"
            and message.tool_calls is not None
            and len(message.tool_calls) >= 2
        ),
        None,
    )
    if batched is None:
        pytest.skip("Model did not batch both tool calls into one response")
    assert batched.tool_calls is not None

    # Both concurrency-safe tools must have executed with overlapping spans.
    assert set(spans) == {"lookup_weather", "lookup_time"}
    (start_a, end_a) = spans["lookup_weather"]
    (start_b, end_b) = spans["lookup_time"]
    overlap = min(end_a, end_b) - max(start_a, start_b)
    assert overlap > 0, (
        "Batched concurrency-safe tools ran serially: "
        f"weather={spans['lookup_weather']}, time={spans['lookup_time']}"
    )

    # Tool results land right after the batching assistant message, one per
    # call id, preserving the model's original tool_calls order.
    batched_index = conversation.messages.index(batched)
    expected_ids = [tool_call.id for tool_call in batched.tool_calls]
    landed = conversation.messages[
        batched_index + 1 : batched_index + 1 + len(expected_ids)
    ]
    assert [message.role for message in landed] == ["tool"] * len(expected_ids)
    assert [message.tool_call_id for message in landed] == expected_ids
