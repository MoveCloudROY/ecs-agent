"""Live e2e: phase-driven system prompt swap with a real LLM.

Skipped unless LLM_API_KEY is set (see tests/conftest.py::live_api_key).
Keys live only in .env — never commit them.
"""

from __future__ import annotations

import os

import pytest

from ecs_agent.components import (
    ConversationComponent,
    LLMComponent,
    RenderedSystemPromptComponent,
)
from ecs_agent.core import World
from ecs_agent.phases import PhaseSpec, advance, bind_phase_graph, build_graph
from ecs_agent.prompts.contracts import PromptTemplateSource, SystemPromptConfigSpec
from ecs_agent.systems.reasoning import ReasoningSystem
from ecs_agent.systems.system_prompt_render_system import SystemPromptRenderSystem
from ecs_agent.types import Message

_registry_module = pytest.importorskip("ecs_agent.providers.registry")
ProviderRegistry = _registry_module.ProviderRegistry
get_model = _registry_module.get_model


def _live_registry() -> "ProviderRegistry":
    return ProviderRegistry.from_dict(
        {
            "aliyun": {
                "base_url": os.getenv(
                    "LLM_BASE_URL",
                    "https://dashscope.aliyuncs.com/compatible-mode/v1",
                ),
                "api_format": "openai_chat_completions",
            },
        }
    )


_GRAPH = build_graph(
    "fruit-live",
    initial="APPLE_PHASE",
    phases=[
        PhaseSpec(
            phase_id="APPLE_PHASE",
            prompts={"main": "Reply with exactly one word: APPLE. Nothing else."},
            to=("BANANA_PHASE",),
        ),
        PhaseSpec(
            phase_id="BANANA_PHASE",
            prompts={"main": "Reply with exactly one word: BANANA. Nothing else."},
            terminal=True,
        ),
    ],
)


@pytest.mark.asyncio
async def test_live_phase_transition_swaps_effective_system_prompt(
    live_api_key: str,
) -> None:
    import httpx

    model_name = os.getenv("LLM_MODEL") or "qwen3.5-flash"
    model = get_model(
        f"aliyun/{model_name}", registry=_live_registry(), api_key=live_api_key
    )

    world = World()
    eid = world.create_entity()
    world.add_component(eid, LLMComponent(model=model, system_prompt=""))
    world.add_component(
        eid, ConversationComponent(messages=[Message(role="user", content="Go.")])
    )
    world.add_component(
        eid,
        SystemPromptConfigSpec(
            template_source=PromptTemplateSource(inline="${_phase_prompt}")
        ),
    )
    await bind_phase_graph(world, eid, _GRAPH)

    render = SystemPromptRenderSystem()
    reasoning = ReasoningSystem()

    try:
        await render.process(world)
        rendered = world.get_component(eid, RenderedSystemPromptComponent)
        assert rendered is not None and "APPLE" in rendered.text
        await reasoning.process(world)
        conversation = world.get_component(eid, ConversationComponent)
        assert conversation is not None
        first_reply = conversation.messages[-1].content.upper()

        await advance(world, eid, "BANANA_PHASE", reason="live test")
        conversation.messages.clear()
        conversation.messages.append(Message(role="user", content="Go."))
        await render.process(world)
        await reasoning.process(world)
        second_reply = conversation.messages[-1].content.upper()
    except httpx.ReadTimeout:
        pytest.skip("live endpoint timed out (flaky network)")

    assert "APPLE" in first_reply
    assert "BANANA" in second_reply
