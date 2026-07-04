"""Phase prompt provider integration with the system prompt render pipeline."""

from __future__ import annotations

import pytest

from ecs_agent.components import (
    ConversationComponent,
    LLMComponent,
    PhaseComponent,
    RenderedSystemPromptComponent,
)
from ecs_agent.core import World
from ecs_agent.phases.api import PhaseIntegrityError, advance, bind_phase_graph
from ecs_agent.phases.contracts import PhaseSpec, build_graph
from ecs_agent.phases.prompt_provider import PhasePromptPlaceholderProvider
from ecs_agent.prompts.contracts import PromptTemplateSource, SystemPromptConfigSpec
from ecs_agent.prompts.template_render import iter_placeholder_providers
from ecs_agent.providers import FakeModel
from ecs_agent.systems.system_prompt_render_system import (
    VOLATILE_PLACEHOLDER_KEYS,
    SystemPromptRenderSystem,
)
from ecs_agent.types import CompletionResult, Message

_SHARED_REVIEW_PROMPT = "You review."


def _graph():
    return build_graph(
        "writer",
        initial="DRAFT",
        phases=[
            PhaseSpec(phase_id="DRAFT", prompts={"main": "You draft."}, to=("REVIEW",)),
            PhaseSpec(phase_id="REVIEW", prompts={"main": _SHARED_REVIEW_PROMPT}, to=("DONE",)),
            PhaseSpec(phase_id="DONE", prompts={"main": _SHARED_REVIEW_PROMPT}, terminal=True),
        ],
    )


async def _rendered_world():
    world = World()
    eid = world.create_entity()
    world.add_component(
        eid,
        LLMComponent(
            model=FakeModel(
                responses=[CompletionResult(message=Message(role="assistant", content="ok"))]
            ),
            system_prompt="",
        ),
    )
    world.add_component(eid, ConversationComponent(messages=[]))
    world.add_component(
        eid,
        SystemPromptConfigSpec(
            template_source=PromptTemplateSource(inline="${_phase_prompt}")
        ),
    )
    await bind_phase_graph(world, eid, _graph())
    return world, eid


async def test_render_injects_current_phase_prompt() -> None:
    world, eid = await _rendered_world()
    await SystemPromptRenderSystem().process(world)
    rendered = world.get_component(eid, RenderedSystemPromptComponent)
    assert rendered is not None
    assert "You draft." in rendered.text


async def test_transition_changes_rendered_prompt() -> None:
    world, eid = await _rendered_world()
    await SystemPromptRenderSystem().process(world)
    await advance(world, eid, "REVIEW", reason="submitted")
    await SystemPromptRenderSystem().process(world)
    rendered = world.get_component(eid, RenderedSystemPromptComponent)
    assert rendered is not None
    assert "You review." in rendered.text
    assert "You draft." not in rendered.text


async def test_fingerprint_stable_when_prompt_text_shared_across_phases() -> None:
    world, eid = await _rendered_world()
    await advance(world, eid, "REVIEW", reason="r")
    provider = PhasePromptPlaceholderProvider()
    fingerprint_review = provider.provider_fingerprint(world, eid)
    await advance(world, eid, "DONE", reason="d")
    fingerprint_done = provider.provider_fingerprint(world, eid)
    # REVIEW and DONE share identical prompt text: the fingerprint must not churn,
    # preserving the prefix-cache stability contract (README "Anthropic Prompt Caching").
    assert fingerprint_review == fingerprint_done


async def test_provider_absent_for_non_phase_entity() -> None:
    world = World()
    eid = world.create_entity()
    providers = iter_placeholder_providers(world, eid)
    assert not any(isinstance(p, PhasePromptPlaceholderProvider) for p in providers)


async def test_provider_present_for_phase_entity() -> None:
    world, eid = await _rendered_world()
    providers = iter_placeholder_providers(world, eid)
    assert any(isinstance(p, PhasePromptPlaceholderProvider) for p in providers)


async def test_half_bound_entity_render_raises_integrity_error() -> None:
    world = World()
    eid = world.create_entity()
    world.add_component(eid, PhaseComponent(graph_id="writer", phase="DRAFT", graph_hash="x"))
    provider = PhasePromptPlaceholderProvider()
    with pytest.raises(PhaseIntegrityError):
        provider.resolve_placeholders(world, eid)


def test_phase_prompt_is_a_volatile_placeholder() -> None:
    assert "_phase_prompt" in VOLATILE_PLACEHOLDER_KEYS
