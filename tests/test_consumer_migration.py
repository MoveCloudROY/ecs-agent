from __future__ import annotations

import pytest

from ecs_agent.components import (
    ConversationComponent,
    LLMComponent,
    PlanComponent,
    RenderedSystemPromptComponent,
    RenderedUserPromptComponent,
    SystemPromptComponent,
)
from ecs_agent.core import World
from ecs_agent.prompts.contracts import PromptTemplateSource, SystemPromptConfigSpec
from ecs_agent.providers import FakeModel
from ecs_agent.systems.planning import PlanningSystem
from ecs_agent.systems.reasoning import ReasoningSystem
from ecs_agent.systems.replanning import ReplanningSystem
from ecs_agent.systems.system_prompt_render_system import SystemPromptRenderSystem
from ecs_agent.types import CompletionResult, Message


class RecordingFakeModel(FakeModel):
    def __init__(self, responses: list[CompletionResult]) -> None:
        super().__init__(responses=responses)
        self.calls: list[list[Message]] = []

    async def complete(
        self,
        messages: list[Message],
        tools: list[object] | None = None,
    ) -> CompletionResult:
        _ = tools
        self.calls.append(list(messages))
        return await super().complete(messages, tools=None)


@pytest.mark.asyncio
async def test_reasoning_uses_rendered_system_prompt() -> None:
    world = World()
    model = RecordingFakeModel(
        responses=[CompletionResult(message=Message(role="assistant", content="ok"))]
    )
    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(model=model))
    world.add_component(
        entity_id,
        ConversationComponent(messages=[Message(role="user", content="hello")]),
    )
    world.add_component(entity_id, RenderedSystemPromptComponent(text="Rendered SYS"))

    await ReasoningSystem().process(world)

    sent = model.calls[0]
    assert sent[0] == Message(
        role="system", content="Rendered SYS", cache_control=True
    )


@pytest.mark.asyncio
async def test_reasoning_uses_rendered_user_prompt() -> None:
    world = World()
    model = RecordingFakeModel(
        responses=[CompletionResult(message=Message(role="assistant", content="ok"))]
    )
    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(model=model))
    world.add_component(
        entity_id,
        ConversationComponent(
            messages=[Message(role="user", content="ORIGINAL user text")],
        ),
    )
    world.add_component(
        entity_id,
        RenderedUserPromptComponent(text="NORMALIZED user text"),
    )

    await ReasoningSystem().process(world)

    sent = model.calls[0]
    assert sent[-1] == Message(role="user", content="NORMALIZED user text")


@pytest.mark.asyncio
async def test_planning_uses_rendered_system_prompt() -> None:
    world = World()
    model = RecordingFakeModel(
        responses=[CompletionResult(message=Message(role="assistant", content="ok"))]
    )
    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(model=model))
    world.add_component(
        entity_id,
        ConversationComponent(messages=[Message(role="user", content="plan")]),
    )
    world.add_component(entity_id, PlanComponent(steps=["step one"]))
    world.add_component(entity_id, RenderedSystemPromptComponent(text="Rendered SYS"))

    await PlanningSystem().process(world)

    sent = model.calls[0]
    assert sent[0] == Message(
        role="system", content="Rendered SYS", cache_control=True
    )


@pytest.mark.asyncio
async def test_replanning_uses_rendered_system_prompt() -> None:
    world = World()
    model = RecordingFakeModel(
        responses=[
            CompletionResult(
                message=Message(
                    role="assistant", content='{"revised_steps": ["step 2"]}'
                )
            )
        ]
    )
    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(model=model))
    world.add_component(
        entity_id,
        ConversationComponent(
            messages=[
                Message(role="user", content="objective"),
                Message(role="assistant", content="done"),
            ]
        ),
    )
    world.add_component(
        entity_id, PlanComponent(steps=["step 1", "step 2"], current_step=1)
    )
    world.add_component(entity_id, RenderedSystemPromptComponent(text="Rendered SYS"))

    await ReplanningSystem().process(world)

    sent = model.calls[0]
    assert sent[0] == Message(
        role="system", content="Rendered SYS", cache_control=True
    )


@pytest.mark.asyncio
async def test_no_rendered_component_fallback() -> None:
    world = World()
    model = RecordingFakeModel(
        responses=[CompletionResult(message=Message(role="assistant", content="ok"))]
    )
    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(model=model))
    world.add_component(entity_id, SystemPromptComponent(content="legacy sys"))
    world.add_component(
        entity_id,
        ConversationComponent(messages=[Message(role="user", content="hello")]),
    )

    await ReasoningSystem().process(world)

    sent = model.calls[0]
    assert sent[0] == Message(
        role="system", content="legacy sys", cache_control=True
    )


@pytest.mark.asyncio
async def test_legacy_consumer_reads_bridged_content() -> None:
    world = World()
    entity_id = world.create_entity()
    world.add_component(
        entity_id,
        SystemPromptConfigSpec(
            template_source=PromptTemplateSource(inline="Bridged rendered prompt")
        ),
    )
    world.add_component(entity_id, SystemPromptComponent())

    await SystemPromptRenderSystem().process(world)

    legacy_prompt = world.get_component(entity_id, SystemPromptComponent)
    rendered_prompt = world.get_component(entity_id, RenderedSystemPromptComponent)
    assert legacy_prompt is not None
    assert rendered_prompt is not None
    assert legacy_prompt.content == rendered_prompt.text == "Bridged rendered prompt"
