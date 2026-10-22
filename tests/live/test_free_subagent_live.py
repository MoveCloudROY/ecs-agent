"""Live tests for opt-in free-form subagent delegation.

Select a profile with --live; see docs/testing.md.
"""

from __future__ import annotations


import pytest

from ecs_agent.components import LLMComponent, ToolRegistryComponent
from ecs_agent.core import World
from ecs_agent.providers.protocol import LLMModel
from ecs_agent.systems.subagent import SubagentSystem

from tests.support.models import LiveModelFactory


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


@pytest.mark.live(formats=["openai_chat_completions"], capabilities=["text", "tools"])
@pytest.mark.subagent
@pytest.mark.asyncio
async def test_live_openai_chat_free_subagent_allows_unregistered_category(
    *, live_model_factory: LiveModelFactory
) -> None:

    result = await _run_free_subagent_live(live_model_factory.openai())

    assert "FREE_SUBAGENT_OK" in result


@pytest.mark.live(formats=["openai_responses"], capabilities=["text", "tools"])
@pytest.mark.subagent
@pytest.mark.asyncio
async def test_live_openai_responses_free_subagent_allows_unregistered_category(
    *, live_model_factory: LiveModelFactory
) -> None:

    result = await _run_free_subagent_live(live_model_factory.openai())

    assert "FREE_SUBAGENT_OK" in result


@pytest.mark.live(formats=["anthropic_messages"], capabilities=["text", "tools"])
@pytest.mark.subagent
@pytest.mark.asyncio
async def test_live_anthropic_free_subagent_allows_unregistered_category(
    *, live_model_factory: LiveModelFactory
) -> None:

    result = await _run_free_subagent_live(live_model_factory.claude())

    assert "FREE_SUBAGENT_OK" in result
