"""Live tests for opt-in free-form subagent delegation.

These tests skip unless their provider-specific credentials are present.
"""

from __future__ import annotations

import os

import pytest

from ecs_agent.components import LLMComponent, ToolRegistryComponent
from ecs_agent.core import World
from ecs_agent.providers import ApiFormat, ClaudeModel, OpenAIModel, ProviderConfig
from ecs_agent.providers.protocol import LLMModel
from ecs_agent.systems.subagent import SubagentSystem


def _make_openai_model(api_key: str, *, api_format: ApiFormat) -> OpenAIModel:
    return OpenAIModel(
        config=ProviderConfig(
            provider_id="live-free-subagent-openai",
            base_url=os.getenv(
                "LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
            ),
            api_key=api_key,
            api_format=api_format,
            timeout=120.0,
        ),
        model=os.getenv("LLM_MODEL", "qwen3.5-flash"),
    )


def _make_anthropic_model(api_key: str) -> ClaudeModel:
    return ClaudeModel(
        config=ProviderConfig(
            provider_id="live-free-subagent-anthropic",
            base_url=os.environ["ANTHROPIC_LIVE_BASE_URL"],
            api_key=api_key,
            api_format=ApiFormat.ANTHROPIC_MESSAGES,
            timeout=120.0,
        ),
        model=os.environ["ANTHROPIC_LIVE_MODEL"],
        max_tokens=256,
    )


async def _run_free_subagent_live(model: LLMModel) -> str:
    world = World(name="free-subagent-live")
    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(model=model))
    world.add_component(entity_id, ToolRegistryComponent(tools={}, handlers={}))

    system = SubagentSystem(priority=-1, allow_unregistered_subagents=True)
    await system.process(world)

    tools = world.get_component(entity_id, ToolRegistryComponent)
    assert tools is not None
    result = await tools.handlers["subagent"](
        category="ad-hoc-verifier",
        prompt="Reply with exactly: FREE_SUBAGENT_OK",
        timeout=60,
    )
    return result


@pytest.mark.asyncio
async def test_live_openai_chat_free_subagent_allows_unregistered_category() -> None:
    api_key = os.getenv("LLM_API_KEY")
    if not api_key:
        pytest.skip("LLM_API_KEY is not set")

    result = await _run_free_subagent_live(
        _make_openai_model(api_key, api_format=ApiFormat.OPENAI_CHAT_COMPLETIONS)
    )

    assert "FREE_SUBAGENT_OK" in result


@pytest.mark.asyncio
async def test_live_openai_responses_free_subagent_allows_unregistered_category() -> None:
    api_key = os.getenv("LLM_API_KEY")
    if not api_key:
        pytest.skip("LLM_API_KEY is not set")

    result = await _run_free_subagent_live(
        _make_openai_model(api_key, api_format=ApiFormat.OPENAI_RESPONSES)
    )

    assert "FREE_SUBAGENT_OK" in result


@pytest.mark.asyncio
async def test_live_anthropic_free_subagent_allows_unregistered_category() -> None:
    api_key = os.getenv("ANTHROPIC_LIVE_API_KEY")
    if not api_key:
        pytest.skip("ANTHROPIC_LIVE_API_KEY is not set")
    if not os.getenv("ANTHROPIC_LIVE_BASE_URL"):
        pytest.skip("ANTHROPIC_LIVE_BASE_URL is not set")
    if not os.getenv("ANTHROPIC_LIVE_MODEL"):
        pytest.skip("ANTHROPIC_LIVE_MODEL is not set")

    result = await _run_free_subagent_live(_make_anthropic_model(api_key))

    assert "FREE_SUBAGENT_OK" in result
