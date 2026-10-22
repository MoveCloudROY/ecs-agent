"""Live e2e: phase-driven system prompt swap with a real LLM.

Select a profile with --live; see docs/testing.md.
"""

from __future__ import annotations


import pytest

from ecs_agent.components import (
    ConversationComponent,
    LLMComponent,
    RenderedSystemPromptComponent,
)
from ecs_agent.core import AgentHandle, World
from ecs_agent.phases import PhaseSpec, advance, bind_phase_graph, build_graph
from ecs_agent.prompts.contracts import PromptTemplateSource, SystemPromptConfigSpec
from ecs_agent.systems.reasoning import ReasoningSystem
from ecs_agent.systems.system_prompt_render_system import SystemPromptRenderSystem

from ecs_agent.providers.registry import ProviderRegistry, get_model

from tests.support.models import LiveModelFactory


def _live_registry(*, live_model_factory: LiveModelFactory) -> "ProviderRegistry":
    return ProviderRegistry.from_dict(
        {
            "live": live_model_factory.registry_entry(),
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


@pytest.mark.live(
    formats=["openai_chat_completions", "openai_responses"],
    capabilities=["text", "tools"],
)
@pytest.mark.workflow
@pytest.mark.asyncio
async def test_live_phase_transition_swaps_effective_system_prompt(
    live_api_key: str, *, live_model_factory: LiveModelFactory
) -> None:
    model_name = live_model_factory.profile.model
    model = live_model_factory.track(
        get_model(
            f"live/{model_name}",
            registry=_live_registry(live_model_factory=live_model_factory),
            api_key=live_api_key,
        )
    )

    world = World()
    eid = world.create_entity()
    world.add_component(eid, LLMComponent(model=model, system_prompt=""))
    world.add_component(eid, ConversationComponent(messages=[]))
    world.add_component(
        eid,
        SystemPromptConfigSpec(
            template_source=PromptTemplateSource(inline="${_phase_prompt}")
        ),
    )
    await bind_phase_graph(world, eid, _GRAPH)

    world.register_system(SystemPromptRenderSystem(), priority=-1)
    world.register_system(ReasoningSystem(), priority=0)
    agent = AgentHandle(world, eid)

    await agent.run_turn("Go.", max_steps=1)
    rendered = world.get_component(eid, RenderedSystemPromptComponent)
    assert rendered is not None and "APPLE" in rendered.text
    conversation = world.get_component(eid, ConversationComponent)
    assert conversation is not None and conversation.messages[-1].role == "assistant"
    first_reply = conversation.messages[-1].content.upper()

    await advance(world, eid, "BANANA_PHASE", reason="live test")
    agent.replace_history([], reason="isolate the new phase prompt")
    await agent.run_turn("Go.", max_steps=1)
    rendered = world.get_component(eid, RenderedSystemPromptComponent)
    assert rendered is not None and "BANANA" in rendered.text
    assert conversation.messages[-1].role == "assistant"
    second_reply = conversation.messages[-1].content.upper()

    assert "APPLE" in first_reply
    assert "BANANA" in second_reply
