"""Tests for TerminalCleanupSystem behavior and defaults."""

import pytest

from ecs_agent.components import OwnerComponent, TerminalComponent
from ecs_agent.core import World
from ecs_agent.systems.terminal_cleanup import TerminalCleanupSystem


def test_terminal_cleanup_system_default_configuration() -> None:
    system = TerminalCleanupSystem()

    assert system.priority == 1
    assert system.clear_reasons == ("reasoning_complete",)
    assert system.include_owned_entities is False


def test_terminal_cleanup_system_exported_from_package() -> None:
    from ecs_agent.systems import TerminalCleanupSystem as Imported

    assert Imported is TerminalCleanupSystem


@pytest.mark.asyncio
async def test_terminal_cleanup_clears_reasoning_complete_by_default() -> None:
    world = World()
    entity_id = world.create_entity()
    world.add_component(entity_id, TerminalComponent(reason="reasoning_complete"))

    await TerminalCleanupSystem().process(world)

    assert world.get_component(entity_id, TerminalComponent) is None


@pytest.mark.asyncio
async def test_terminal_cleanup_preserves_non_default_reason() -> None:
    world = World()
    entity_id = world.create_entity()
    world.add_component(entity_id, TerminalComponent(reason="user_exit_command"))

    await TerminalCleanupSystem().process(world)

    terminal = world.get_component(entity_id, TerminalComponent)
    assert terminal is not None
    assert terminal.reason == "user_exit_command"


@pytest.mark.asyncio
async def test_terminal_cleanup_preserves_user_input_timeout() -> None:
    world = World()
    entity_id = world.create_entity()
    world.add_component(entity_id, TerminalComponent(reason="user_input_timeout"))

    await TerminalCleanupSystem().process(world)

    terminal = world.get_component(entity_id, TerminalComponent)
    assert terminal is not None
    assert terminal.reason == "user_input_timeout"


@pytest.mark.asyncio
async def test_terminal_cleanup_preserves_provider_exhausted() -> None:
    world = World()
    entity_id = world.create_entity()
    world.add_component(entity_id, TerminalComponent(reason="provider_exhausted"))

    await TerminalCleanupSystem().process(world)

    terminal = world.get_component(entity_id, TerminalComponent)
    assert terminal is not None
    assert terminal.reason == "provider_exhausted"


@pytest.mark.asyncio
async def test_terminal_cleanup_skips_owned_entities_by_default() -> None:
    world = World()
    parent_id = world.create_entity()
    child_id = world.create_entity()
    world.add_component(child_id, OwnerComponent(owner_id=parent_id))
    world.add_component(child_id, TerminalComponent(reason="reasoning_complete"))

    await TerminalCleanupSystem().process(world)

    terminal = world.get_component(child_id, TerminalComponent)
    assert terminal is not None
    assert terminal.reason == "reasoning_complete"


@pytest.mark.asyncio
async def test_terminal_cleanup_clears_owned_entities_when_opted_in() -> None:
    world = World()
    parent_id = world.create_entity()
    child_id = world.create_entity()
    world.add_component(child_id, OwnerComponent(owner_id=parent_id))
    world.add_component(child_id, TerminalComponent(reason="reasoning_complete"))

    await TerminalCleanupSystem(include_owned_entities=True).process(world)

    assert world.get_component(child_id, TerminalComponent) is None
