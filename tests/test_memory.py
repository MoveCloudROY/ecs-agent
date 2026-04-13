from __future__ import annotations

import pytest

from ecs_agent.components import ConversationComponent, KVStoreComponent
from ecs_agent.core import World
from ecs_agent.systems.memory import MemorySystem
from ecs_agent.types import ConversationTruncatedEvent, EntityId, Message


def _msg(role: str, content: str) -> Message:
    return Message(role=role, content=content)


@pytest.mark.asyncio
async def test_no_truncation_when_messages_under_limit() -> None:
    world = World()
    entity_id = world.create_entity()
    original = [_msg("user", f"u{i}") for i in range(3)]
    world.add_component(
        entity_id, ConversationComponent(messages=list(original), max_messages=5)
    )

    await MemorySystem().process(world)

    conversation = world.get_component(entity_id, ConversationComponent)
    assert conversation is not None
    assert conversation.messages == original


@pytest.mark.asyncio
async def test_truncation_when_messages_over_limit_without_system_message() -> None:
    world = World()
    entity_id = world.create_entity()
    messages = [_msg("user", f"u{i}") for i in range(6)]
    world.add_component(
        entity_id, ConversationComponent(messages=messages, max_messages=4)
    )

    await MemorySystem().process(world)

    conversation = world.get_component(entity_id, ConversationComponent)
    assert conversation is not None
    assert [message.content for message in conversation.messages] == [
        "u2",
        "u3",
        "u4",
        "u5",
    ]


@pytest.mark.asyncio
async def test_truncation_preserves_first_system_message() -> None:
    world = World()
    entity_id = world.create_entity()
    messages = [_msg("system", "s0")] + [_msg("user", f"u{i}") for i in range(6)]
    world.add_component(
        entity_id, ConversationComponent(messages=messages, max_messages=4)
    )

    await MemorySystem().process(world)

    conversation = world.get_component(entity_id, ConversationComponent)
    assert conversation is not None
    assert [message.role for message in conversation.messages] == [
        "system",
        "user",
        "user",
        "user",
    ]
    assert [message.content for message in conversation.messages] == [
        "s0",
        "u3",
        "u4",
        "u5",
    ]


@pytest.mark.asyncio
async def test_event_published_on_truncation_with_removed_count() -> None:
    world = World()
    entity_id = world.create_entity()
    messages = [_msg("user", f"u{i}") for i in range(7)]
    world.add_component(
        entity_id, ConversationComponent(messages=messages, max_messages=3)
    )

    seen: list[ConversationTruncatedEvent] = []

    async def handler(event: ConversationTruncatedEvent) -> None:
        seen.append(event)

    world.event_bus.subscribe(ConversationTruncatedEvent, handler)
    await MemorySystem().process(world)

    assert len(seen) == 1
    assert seen[0].entity_id == entity_id
    assert seen[0].removed_count == 4


@pytest.mark.asyncio
async def test_with_current_summary_component_preserves_all_post_compaction_messages() -> (
    None
):
    """When CurrentCompactionSummaryComponent is present, preserve system message + all remaining conversation."""
    from ecs_agent.components import CurrentCompactionSummaryComponent

    world = World()
    entity_id = world.create_entity()
    messages = [
        _msg("system", "sys"),
        _msg("user", "u0"),
        _msg("assistant", "a0"),
        _msg("user", "u1"),
        _msg("assistant", "a1"),
        _msg("user", "u2"),
    ]
    world.add_component(
        entity_id, ConversationComponent(messages=messages, max_messages=3)
    )
    world.add_component(
        entity_id, CurrentCompactionSummaryComponent(summary="summary-of-first-part")
    )

    await MemorySystem().process(world)

    conversation = world.get_component(entity_id, ConversationComponent)
    assert conversation is not None
    # With summary present, all messages should be preserved (no truncation)
    assert len(conversation.messages) == 6
    assert [message.role for message in conversation.messages] == [
        "system",
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
    ]
    assert [message.content for message in conversation.messages] == [
        "sys",
        "u0",
        "a0",
        "u1",
        "a1",
        "u2",
    ]


@pytest.mark.asyncio
async def test_without_current_summary_component_applies_trailing_window() -> None:
    """When CurrentCompactionSummaryComponent is absent, use trailing-window truncation."""
    world = World()
    entity_id = world.create_entity()
    messages = [
        _msg("system", "sys"),
        _msg("user", "u0"),
        _msg("assistant", "a0"),
        _msg("user", "u1"),
        _msg("assistant", "a1"),
        _msg("user", "u2"),
    ]
    world.add_component(
        entity_id, ConversationComponent(messages=messages, max_messages=3)
    )
    # Deliberately do NOT attach CurrentCompactionSummaryComponent

    await MemorySystem().process(world)

    conversation = world.get_component(entity_id, ConversationComponent)
    assert conversation is not None
    # Without summary, apply trailing-window truncation
    assert [message.content for message in conversation.messages] == [
        "sys",
        "a1",
        "u2",
    ]


@pytest.mark.asyncio
async def test_processes_multiple_entities() -> None:
    world = World()
    first = world.create_entity()
    second = world.create_entity()

    world.add_component(
        first,
        ConversationComponent(
            messages=[_msg("user", f"a{i}") for i in range(5)], max_messages=2
        ),
    )
    world.add_component(
        second,
        ConversationComponent(
            messages=[_msg("system", "sys")]
            + [_msg("user", f"b{i}") for i in range(5)],
            max_messages=3,
        ),
    )

    events: list[tuple[EntityId, int]] = []

    async def handler(event: ConversationTruncatedEvent) -> None:
        events.append((event.entity_id, event.removed_count))

    world.event_bus.subscribe(ConversationTruncatedEvent, handler)
    await MemorySystem().process(world)

    first_conversation = world.get_component(first, ConversationComponent)
    second_conversation = world.get_component(second, ConversationComponent)
    assert first_conversation is not None
    assert second_conversation is not None
    assert [m.content for m in first_conversation.messages] == ["a3", "a4"]
    assert [m.content for m in second_conversation.messages] == ["sys", "b3", "b4"]
    assert sorted(events, key=lambda item: item[0]) == [(first, 3), (second, 3)]


@pytest.mark.asyncio
async def test_entity_without_conversation_component_is_skipped() -> None:
    world = World()
    entity_id = world.create_entity()
    store = {"k": "v"}
    world.add_component(entity_id, KVStoreComponent(store=store))

    await MemorySystem().process(world)

    kv_component = world.get_component(entity_id, KVStoreComponent)
    assert kv_component is not None
    assert kv_component.store == store
