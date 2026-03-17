from __future__ import annotations

import pytest

from ecs_agent.components import (
    PromptConfigComponent,
    PromptContributionsComponent,
    SystemPromptComponent,
)
from ecs_agent.core import World
from ecs_agent.prompts.contracts import PromptSectionSpec
from ecs_agent.systems.system_prompt_assembly import SystemPromptAssemblySystem


@pytest.mark.asyncio
async def test_opt_in_entity_assembles_system_prompt_from_contributions() -> None:
    world = World()
    entity_id = world.create_entity()
    world.add_component(entity_id, PromptConfigComponent())
    world.add_component(
        entity_id,
        PromptContributionsComponent(
            sections=[
                PromptSectionSpec(title="Safety", lines=["Follow policy"], priority=20),
                PromptSectionSpec(
                    title="Context", lines=["Project: ecs-agent"], priority=10
                ),
            ]
        ),
    )
    world.add_component(entity_id, SystemPromptComponent(content="placeholder"))

    await SystemPromptAssemblySystem().process(world)

    system_prompt = world.get_component(entity_id, SystemPromptComponent)
    assert system_prompt is not None
    assert system_prompt.content == (
        "## Safety\n\nFollow policy\n\n## Context\n\nProject: ecs-agent"
    )


@pytest.mark.asyncio
async def test_assembly_is_deterministic_for_equal_priority_sections() -> None:
    world = World()
    entity_id = world.create_entity()
    world.add_component(entity_id, PromptConfigComponent())
    world.add_component(
        entity_id,
        PromptContributionsComponent(
            sections=[
                PromptSectionSpec(title="Beta", lines=["B"], priority=5),
                PromptSectionSpec(title="Alpha", lines=["A"], priority=5),
                PromptSectionSpec(title="Gamma", lines=["G"], priority=10),
            ]
        ),
    )
    world.add_component(entity_id, SystemPromptComponent(content="placeholder"))

    await SystemPromptAssemblySystem().process(world)

    system_prompt = world.get_component(entity_id, SystemPromptComponent)
    assert system_prompt is not None
    assert system_prompt.content == ("## Gamma\n\nG\n\n## Alpha\n\nA\n\n## Beta\n\nB")


@pytest.mark.asyncio
async def test_non_opt_in_entity_is_unchanged() -> None:
    world = World()
    entity_id = world.create_entity()
    world.add_component(
        entity_id,
        PromptContributionsComponent(
            sections=[PromptSectionSpec(title="Context", lines=["ignored"], priority=1)]
        ),
    )
    world.add_component(entity_id, SystemPromptComponent(content="existing prompt"))

    await SystemPromptAssemblySystem().process(world)

    system_prompt = world.get_component(entity_id, SystemPromptComponent)
    assert system_prompt is not None
    assert system_prompt.content == "existing prompt"
