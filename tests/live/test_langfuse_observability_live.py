"""Live smoke tests for Langfuse observability with real LLM calls.

The suite skips unless RUN_LANGFUSE_LIVE_TESTS is set to 1 and the required
LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST or LANGFUSE_BASE_URL,
LLM_API_KEY, LLM_BASE_URL, and LLM_MODEL variables are present.
"""

from __future__ import annotations

import os
import re

import httpx
import pytest

from ecs_agent.components import ConversationComponent, ErrorComponent, LLMComponent
from ecs_agent.core import Runner, World
from ecs_agent.integrations.langfuse import (
    LangfuseConfig,
    install_langfuse_observability,
)
from ecs_agent.providers import Model
from ecs_agent.providers.config import ApiFormat
from ecs_agent.providers.protocol import LLMModel
from ecs_agent.systems.error_handling import ErrorHandlingSystem
from ecs_agent.systems.reasoning import ReasoningSystem
from ecs_agent.types import Message

_LANGFUSE_KEY_ENV = ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY")
_LLM_ENV = ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL")
_SECRET_LIKE_PATTERN = re.compile(
    r"(?:sk-lf-|pk-lf-|sk-[A-Za-z0-9_-]{16,}|pk-[A-Za-z0-9_-]{16,}|Bearer\s+[A-Za-z0-9._-]{16,})"
)


@pytest.fixture
def langfuse_live_env() -> None:
    """Skip live Langfuse tests unless the explicit live gate is enabled."""
    skip_reason = _langfuse_live_skip_reason()
    if skip_reason is not None:
        pytest.skip(skip_reason)
    pytest.importorskip("langfuse", reason="Langfuse live dependency unavailable")


def _langfuse_live_skip_reason() -> str | None:
    if os.getenv("RUN_LANGFUSE_LIVE_TESTS") != "1":
        return "RUN_LANGFUSE_LIVE_TESTS is not 1"

    missing = [
        name for name in (*_LANGFUSE_KEY_ENV, *_LLM_ENV) if not os.getenv(name)
    ]
    if not (os.getenv("LANGFUSE_HOST") or os.getenv("LANGFUSE_BASE_URL")):
        missing.append("LANGFUSE_HOST or LANGFUSE_BASE_URL")
    if missing:
        return "Missing required live environment variables: " + ", ".join(missing)
    return None


def _make_live_model(api_format: ApiFormat) -> LLMModel:
    return Model(
        os.environ["LLM_MODEL"],
        base_url=os.environ["LLM_BASE_URL"],
        api_key=os.environ["LLM_API_KEY"],
        api_format=api_format,
        provider_id="live",
        timeout=30.0,
    )


async def _run_live_langfuse_agent(api_format: ApiFormat) -> None:
    world = World(name="langfuse-live-test")
    handle = install_langfuse_observability(
        world,
        LangfuseConfig(
            environment="live-test",
            tags=["live-test", api_format.value],
            metadata={"LLM_API_FORMAT": api_format.value},
            flush_at=1,
            flush_interval=1.0,
        ),
    )
    agent = world.create_entity()
    world.add_component(
        agent,
        LLMComponent(
            model=_make_live_model(api_format),
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

    try:
        await Runner().run(world, max_ticks=2)
    except httpx.ReadTimeout:
        pytest.skip("Live LLM endpoint timed out during Langfuse observability smoke")
    finally:
        try:
            await handle.flush()
        finally:
            await handle.shutdown()

    error = world.get_component(agent, ErrorComponent)
    if error is not None:
        if "ReadTimeout" in error.error or "timed out" in error.error.lower():
            pytest.skip("Live LLM endpoint timed out during Langfuse observability smoke")
        pytest.fail("Live Langfuse observability agent run failed")

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


@pytest.mark.asyncio
async def test_live_langfuse_openai_chat_agent_run(
    langfuse_live_env: None,
) -> None:
    await _run_live_langfuse_agent(ApiFormat.OPENAI_CHAT_COMPLETIONS)


@pytest.mark.asyncio
async def test_live_langfuse_openai_responses_agent_run(
    langfuse_live_env: None,
) -> None:
    await _run_live_langfuse_agent(ApiFormat.OPENAI_RESPONSES)


@pytest.mark.asyncio
async def test_live_langfuse_anthropic_messages_agent_run(
    langfuse_live_env: None,
) -> None:
    await _run_live_langfuse_agent(ApiFormat.ANTHROPIC_MESSAGES)
