from __future__ import annotations

import pytest
import ecs_agent.components as component_exports

from ecs_agent.components import (
    PromptConfigComponent,
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
        SystemPromptComponent(
            sections=[
                PromptSectionSpec(title="Safety", lines=["Follow policy"], priority=20),
                PromptSectionSpec(
                    title="Context", lines=["Project: ecs-agent"], priority=10
                ),
            ],
            content="placeholder",
        ),
    )

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
        SystemPromptComponent(
            sections=[
                PromptSectionSpec(title="Beta", lines=["B"], priority=5),
                PromptSectionSpec(title="Alpha", lines=["A"], priority=5),
                PromptSectionSpec(title="Gamma", lines=["G"], priority=10),
            ],
            content="placeholder",
        ),
    )

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
        SystemPromptComponent(
            sections=[
                PromptSectionSpec(title="Context", lines=["ignored"], priority=1)
            ],
            content="existing prompt",
        ),
    )

    await SystemPromptAssemblySystem().process(world)

    system_prompt = world.get_component(entity_id, SystemPromptComponent)
    assert system_prompt is not None
    assert system_prompt.content == "existing prompt"


@pytest.mark.asyncio
async def test_single_component_assembly_rejects_missing_core_placeholders() -> None:
    world = World()
    entity_id = world.create_entity()
    world.add_component(entity_id, PromptConfigComponent())
    world.add_component(
        entity_id,
        SystemPromptComponent(
            template=("Core template\n${toolSelection}\n${exploreSection}"),
            content="",
        ),
    )

    with pytest.raises(
        ValueError, match="toolSelection|exploreSection|librarianSection"
    ):
        await SystemPromptAssemblySystem().process(world)


@pytest.mark.asyncio
async def test_single_component_assembly_renders_template_from_core_sections() -> None:
    world = World()
    entity_id = world.create_entity()
    world.add_component(entity_id, PromptConfigComponent())
    world.add_component(
        entity_id,
        SystemPromptComponent(
            template=(
                "Core Start\n"
                "${toolSelection}\n\n"
                "${exploreSection}\n\n"
                "${librarianSection}\n"
                "Core End"
            ),
            sections=[
                PromptSectionSpec(
                    title="toolSelection",
                    lines=["Use `explore` for broad discovery"],
                    priority=20,
                ),
                PromptSectionSpec(
                    title="exploreSection",
                    lines=["Gather context before edits"],
                    priority=10,
                ),
                PromptSectionSpec(
                    title="librarianSection",
                    lines=["Quote exact lines from source files"],
                    priority=5,
                ),
            ],
            content="",
        ),
    )

    await SystemPromptAssemblySystem().process(world)

    system_prompt = world.get_component(entity_id, SystemPromptComponent)
    assert system_prompt is not None
    assert system_prompt.content == (
        "Core Start\n"
        "## toolSelection\n\n"
        "Use `explore` for broad discovery\n\n"
        "## exploreSection\n\n"
        "Gather context before edits\n\n"
        "## librarianSection\n\n"
        "Quote exact lines from source files\n"
        "Core End"
    )


@pytest.mark.asyncio
async def test_core_placeholder_validation_accepts_braced_and_unbraced_syntax() -> None:
    world = World()
    entity_id = world.create_entity()
    world.add_component(entity_id, PromptConfigComponent())
    world.add_component(
        entity_id,
        SystemPromptComponent(
            template="$toolSelection\n${exploreSection}\n$librarianSection",
            sections=[],
            content="",
        ),
    )

    await SystemPromptAssemblySystem().process(world)

    system_prompt = world.get_component(entity_id, SystemPromptComponent)
    assert system_prompt is not None
    assert system_prompt.content == "\n\n"


@pytest.mark.asyncio
async def test_single_component_assembly_renders_registered_extension_placeholder() -> (
    None
):
    world = World()
    entity_id = world.create_entity()
    world.add_component(entity_id, PromptConfigComponent())
    world.add_component(
        entity_id,
        SystemPromptComponent(
            template=(
                "Core Start\n"
                "${toolSelection}\n\n"
                "${exploreSection}\n\n"
                "${librarianSection}\n\n"
                "${extensionSection}\n"
                "Core End"
            ),
            sections=[
                PromptSectionSpec(
                    title="toolSelection", lines=["Use tools"], priority=30
                ),
                PromptSectionSpec(
                    title="exploreSection", lines=["Explore first"], priority=20
                ),
                PromptSectionSpec(
                    title="librarianSection", lines=["Quote sources"], priority=10
                ),
                PromptSectionSpec(
                    title="extensionSection",
                    lines=["Extension placeholder resolved"],
                    priority=5,
                ),
            ],
            content="",
        ),
    )

    await SystemPromptAssemblySystem().process(world)

    system_prompt = world.get_component(entity_id, SystemPromptComponent)
    assert system_prompt is not None
    assert system_prompt.content == (
        "Core Start\n"
        "## toolSelection\n\n"
        "Use tools\n\n"
        "## exploreSection\n\n"
        "Explore first\n\n"
        "## librarianSection\n\n"
        "Quote sources\n\n"
        "## extensionSection\n\n"
        "Extension placeholder resolved\n"
        "Core End"
    )


@pytest.mark.asyncio
async def test_single_component_assembly_rejects_unknown_extension_placeholder() -> (
    None
):
    world = World()
    entity_id = world.create_entity()
    world.add_component(entity_id, PromptConfigComponent())
    world.add_component(
        entity_id,
        SystemPromptComponent(
            template=(
                "${toolSelection}\n${exploreSection}\n${librarianSection}\n${missingExtension}"
            ),
            sections=[
                PromptSectionSpec(title="toolSelection", lines=["Use tools"]),
                PromptSectionSpec(title="exploreSection", lines=["Explore first"]),
                PromptSectionSpec(title="librarianSection", lines=["Quote sources"]),
            ],
            content="",
        ),
    )

    with pytest.raises(
        ValueError, match="unknown placeholders in template: missingExtension"
    ):
        await SystemPromptAssemblySystem().process(world)


def test_prompt_contributions_component_is_removed_from_exports() -> None:
    assert not hasattr(component_exports, "PromptContributionsComponent")
