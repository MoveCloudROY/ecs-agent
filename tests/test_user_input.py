"""Tests for UserInputSystem and UserInputComponent."""

import asyncio

import pytest

from ecs_agent.components import (
    ConversationComponent,
    UserInputComponent,
)
from ecs_agent.components.definitions import ErrorComponent, TerminalComponent
from ecs_agent.core import World
from ecs_agent.systems.user_input import UserInputSystem
from ecs_agent.types import Message, UserInputRequestedEvent


@pytest.mark.asyncio
async def test_user_input_system_publishes_event_and_receives_result() -> None:
    """System should create Future, publish event, and store result."""
    world = World()
    entity_id = world.create_entity()
    world.add_component(entity_id, UserInputComponent(prompt="Enter name:"))
    world.add_component(entity_id, ConversationComponent(messages=[]))

    async def provide_input(event: UserInputRequestedEvent) -> None:
        assert event.prompt == "Enter name:"
        assert event.entity_id == entity_id
        event.input_future.set_result("Alice")

    world.event_bus.subscribe(UserInputRequestedEvent, provide_input)

    system = UserInputSystem()
    await system.process(world)

    comp = world.get_component(entity_id, UserInputComponent)
    assert comp is not None
    assert comp.result == "Alice"
    assert comp.future is None  # Cleared after resolution


@pytest.mark.asyncio
async def test_user_input_appends_to_conversation() -> None:
    """Resolved input should be appended as a user message."""
    world = World()
    entity_id = world.create_entity()
    world.add_component(entity_id, UserInputComponent(prompt="Say something:"))
    world.add_component(
        entity_id,
        ConversationComponent(messages=[Message(role="assistant", content="Hi!")]),
    )

    async def provide_input(event: UserInputRequestedEvent) -> None:
        event.input_future.set_result("Hello back!")

    world.event_bus.subscribe(UserInputRequestedEvent, provide_input)

    await UserInputSystem().process(world)

    conv = world.get_component(entity_id, ConversationComponent)
    assert conv is not None
    assert len(conv.messages) == 2
    assert conv.messages[1] == Message(role="user", content="Hello back!")


@pytest.mark.asyncio
async def test_user_input_timeout_creates_error_and_terminal() -> None:
    """When timeout expires, ErrorComponent and TerminalComponent should be added."""
    world = World()
    entity_id = world.create_entity()
    world.add_component(
        entity_id,
        UserInputComponent(prompt="Quick!", timeout=0.01),
    )

    # No handler — future will never resolve → timeout
    await UserInputSystem().process(world)

    error = world.get_component(entity_id, ErrorComponent)
    terminal = world.get_component(entity_id, TerminalComponent)
    assert error is not None
    assert "timeout" in error.error.lower()
    assert error.system_name == "UserInputSystem"
    assert terminal is not None
    assert terminal.reason == "user_input_timeout"


@pytest.mark.asyncio
async def test_user_input_timeout_none_waits_indefinitely() -> None:
    """timeout=None should wait until the future is resolved."""
    world = World()
    entity_id = world.create_entity()
    world.add_component(
        entity_id,
        UserInputComponent(prompt="Take your time:", timeout=None),
    )
    world.add_component(entity_id, ConversationComponent(messages=[]))

    async def delayed_input(event: UserInputRequestedEvent) -> None:
        await asyncio.sleep(0.05)
        event.input_future.set_result("Finally!")

    world.event_bus.subscribe(UserInputRequestedEvent, delayed_input)

    await UserInputSystem().process(world)

    comp = world.get_component(entity_id, UserInputComponent)
    assert comp is not None
    assert comp.result == "Finally!"


@pytest.mark.asyncio
async def test_user_input_skips_already_resolved() -> None:
    """If result is already set, system should skip the entity."""
    world = World()
    entity_id = world.create_entity()
    world.add_component(
        entity_id,
        UserInputComponent(prompt="Done:", result="already-set"),
    )

    events_seen: list[UserInputRequestedEvent] = []

    async def track_events(event: UserInputRequestedEvent) -> None:
        events_seen.append(event)

    world.event_bus.subscribe(UserInputRequestedEvent, track_events)

    await UserInputSystem().process(world)

    assert len(events_seen) == 0  # No event published


@pytest.mark.asyncio
async def test_user_input_without_conversation() -> None:
    """System should work even without ConversationComponent."""
    world = World()
    entity_id = world.create_entity()
    world.add_component(entity_id, UserInputComponent(prompt="No conv:"))

    async def provide_input(event: UserInputRequestedEvent) -> None:
        event.input_future.set_result("standalone")

    world.event_bus.subscribe(UserInputRequestedEvent, provide_input)

    await UserInputSystem().process(world)

    comp = world.get_component(entity_id, UserInputComponent)
    assert comp is not None
    assert comp.result == "standalone"
    # No crash — conversation was None, gracefully skipped


@pytest.mark.asyncio
async def test_user_input_default_timeout_is_none() -> None:
    """Default timeout should be None (infinite wait)."""
    comp = UserInputComponent()
    assert comp.timeout is None
    assert comp.prompt == ""
    assert comp.result is None
    assert comp.future is None


@pytest.mark.asyncio
async def test_user_input_component_exported_from_package() -> None:
    """UserInputComponent should be importable from ecs_agent.components."""
    from ecs_agent.components import UserInputComponent as Imported

    assert Imported is UserInputComponent


@pytest.mark.asyncio
async def test_user_input_system_exported_from_package() -> None:
    """UserInputSystem should be importable from ecs_agent.systems."""
    from ecs_agent.systems import UserInputSystem as Imported

    assert Imported is UserInputSystem


@pytest.mark.asyncio
async def test_user_input_requested_event_exported() -> None:
    """UserInputRequestedEvent should be importable from ecs_agent.types."""
    from ecs_agent.types import UserInputRequestedEvent as Imported

    assert Imported is UserInputRequestedEvent
