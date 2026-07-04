"""Integration: trigger script -> phase advance -> same-tick rendered prompt."""

from __future__ import annotations

from ecs_agent.components import (
    ConversationComponent,
    LLMComponent,
    PhaseComponent,
    RenderedSystemPromptComponent,
    UserPromptConfigComponent,
)
from ecs_agent.core import World
from ecs_agent.phases import PhaseSpec, advance, bind_phase_graph, build_graph
from ecs_agent.prompts.contracts import (
    PromptTemplateSource,
    SystemPromptConfigSpec,
    TriggerSpec,
)
from ecs_agent.providers import FakeModel
from ecs_agent.systems.system_prompt_render_system import SystemPromptRenderSystem
from ecs_agent.systems.user_prompt_normalization_system import (
    UserPromptNormalizationSystem,
)
from ecs_agent.types import CompletionResult, EntityId, Message

_GRAPH = build_graph(
    "trigger-flow",
    initial="DRAFT",
    phases=[
        PhaseSpec(phase_id="DRAFT", prompts={"main": "You are the DRAFT agent."}, to=("REVIEW",)),
        PhaseSpec(phase_id="REVIEW", prompts={"main": "You are the REVIEW agent."}, terminal=True),
    ],
)


async def _build_world() -> tuple[World, EntityId]:
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

    async def handle_go(w: World, entity_id: EntityId, text: str) -> str | None:
        await advance(w, entity_id, "REVIEW", reason="trigger:/go")
        return None

    world.add_component(
        eid,
        ConversationComponent(messages=[Message(role="user", content="/go now")]),
    )
    world.add_component(
        eid,
        SystemPromptConfigSpec(
            template_source=PromptTemplateSource(inline="${_phase_prompt}")
        ),
    )
    world.add_component(
        eid,
        UserPromptConfigComponent(
            triggers=[
                TriggerSpec(pattern="/go", match_mode="prefix", action="script", content="go")
            ],
            script_handlers={"go": handle_go},
        ),
    )
    await bind_phase_graph(world, eid, _GRAPH, agent_key="main")
    return world, eid


async def test_trigger_script_advance_renders_new_phase_prompt_same_tick() -> None:
    world, eid = await _build_world()

    await UserPromptNormalizationSystem().process(world)
    component = world.get_component(eid, PhaseComponent)
    assert component is not None
    assert component.phase == "REVIEW"

    await SystemPromptRenderSystem().process(world)
    rendered = world.get_component(eid, RenderedSystemPromptComponent)
    assert rendered is not None
    assert "You are the REVIEW agent." in rendered.text
    assert "You are the DRAFT agent." not in rendered.text


async def test_no_trigger_keeps_phase_and_prompt() -> None:
    world, eid = await _build_world()
    conversation = world.get_component(eid, ConversationComponent)
    assert conversation is not None
    conversation.messages.clear()
    conversation.messages.append(Message(role="user", content="just chatting"))

    await UserPromptNormalizationSystem().process(world)
    component = world.get_component(eid, PhaseComponent)
    assert component is not None
    assert component.phase == "DRAFT"

    await SystemPromptRenderSystem().process(world)
    rendered = world.get_component(eid, RenderedSystemPromptComponent)
    assert rendered is not None
    assert "You are the DRAFT agent." in rendered.text
