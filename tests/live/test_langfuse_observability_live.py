"""Live smoke tests for Langfuse observability with real LLM calls.

Select a profile with --live; see docs/testing.md.
"""

from __future__ import annotations

import re

import pytest

from ecs_agent.components import ConversationComponent, LLMComponent
from ecs_agent.core import Runner, World
from ecs_agent.plugins import install_plugins
from ecs_agent.plugins.langfuse import LangfuseConfig, LangfusePlugin
from ecs_agent.providers.config import ApiFormat
from ecs_agent.systems.error_handling import ErrorHandlingSystem
from ecs_agent.systems.reasoning import ReasoningSystem
from ecs_agent.types import ErrorOccurredEvent, Message

from tests.support.models import LiveModelFactory


_LANGFUSE_KEY_ENV = ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY")
_LLM_ENV = ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL")
_SECRET_LIKE_PATTERN = re.compile(
    r"(?:sk-lf-|pk-lf-|sk-[A-Za-z0-9_-]{16,}|pk-[A-Za-z0-9_-]{16,}|Bearer\s+[A-Za-z0-9._-]{16,})"
)


async def _run_live_langfuse_agent(
    api_format: ApiFormat, *, live_model_factory: LiveModelFactory
) -> None:
    world = World(name="langfuse-live-test")
    handle = await install_plugins(
        world,
        [
            LangfusePlugin(
                LangfuseConfig(
                    **live_model_factory.settings.service("langfuse"),
                    environment="live-test",
                    tags=["live-test", api_format.value],
                    metadata={"LLM_API_FORMAT": api_format.value},
                    flush_at=1,
                    flush_interval=1.0,
                )
            )
        ],
    )
    agent = world.create_entity()
    world.add_component(
        agent,
        LLMComponent(
            model=live_model_factory.create(),
            system_prompt=(
                "Reply with exactly one short sentence. Do not include secrets, "
                "credentials, tokens, keys, environment values, or URLs."
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
        try:
            await handle.flush()
        finally:
            await handle.shutdown()

    # LLM failures never raise through Runner.run: ReasoningSystem swallows
    # them into ErrorComponent and ErrorHandlingSystem removes that component
    # after publishing ErrorOccurredEvent, so the event is the durable signal.
    assert not llm_errors, f"Live agent run failed: {llm_errors[0].error[:200]}"

    conversation = world.get_component(agent, ConversationComponent)
    assert conversation is not None
    assistant_outputs = [
        message.content.strip()
        for message in conversation.messages
        if message.role == "assistant" and message.content.strip()
    ]
    assert assistant_outputs
    if any(_SECRET_LIKE_PATTERN.search(output) for output in assistant_outputs):
        pytest.fail("Assistant output matched a secret-like pattern")


@pytest.mark.live(
    formats=["openai_chat_completions"],
    capabilities=["text"],
    services=["langfuse"],
    dependencies=["langfuse"],
)
@pytest.mark.observability
@pytest.mark.asyncio
async def test_live_langfuse_openai_chat_agent_run(
    *, live_model_factory: LiveModelFactory
) -> None:
    await _run_live_langfuse_agent(
        ApiFormat.OPENAI_CHAT_COMPLETIONS, live_model_factory=live_model_factory
    )


@pytest.mark.live(
    formats=["openai_responses"],
    capabilities=["text"],
    services=["langfuse"],
    dependencies=["langfuse"],
)
@pytest.mark.observability
@pytest.mark.asyncio
async def test_live_langfuse_openai_responses_agent_run(
    *, live_model_factory: LiveModelFactory
) -> None:
    await _run_live_langfuse_agent(
        ApiFormat.OPENAI_RESPONSES, live_model_factory=live_model_factory
    )


@pytest.mark.live(
    formats=["anthropic_messages"],
    capabilities=["text"],
    services=["langfuse"],
    dependencies=["langfuse"],
)
@pytest.mark.observability
@pytest.mark.asyncio
async def test_live_langfuse_anthropic_messages_agent_run(
    *, live_model_factory: LiveModelFactory
) -> None:
    await _run_live_langfuse_agent(
        ApiFormat.ANTHROPIC_MESSAGES, live_model_factory=live_model_factory
    )
