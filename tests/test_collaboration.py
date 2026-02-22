from __future__ import annotations

import pytest

from ecs_agent.components import CollaborationComponent, ConversationComponent
from ecs_agent.core import World
from ecs_agent.systems.collaboration import CollaborationSystem
from ecs_agent.types import Message, MessageDeliveredEvent


def _msg(role: str, content: str) -> Message:
    return Message(role=role, content=content)


@pytest.mark.asyncio
async def test_basic_message_delivery_appends_to_conversation_and_clears_inbox() -> (
    None
):
    world = World()
    sender = world.create_entity()
    receiver = world.create_entity()

    world.add_component(
        receiver,
        CollaborationComponent(
            peers=[sender],
            inbox=[(sender, _msg("assistant", "Hello from sender"))],
        ),
    )
    world.add_component(receiver, ConversationComponent(messages=[]))

    events: list[MessageDeliveredEvent] = []

    async def on_delivered(event: MessageDeliveredEvent) -> None:
        events.append(event)

    world.event_bus.subscribe(MessageDeliveredEvent, on_delivered)
    await CollaborationSystem().process(world)

    conversation = world.get_component(receiver, ConversationComponent)
    collaboration = world.get_component(receiver, CollaborationComponent)

    assert conversation is not None
    assert collaboration is not None
    assert [m.content for m in conversation.messages] == [
        f"From: {sender}: Hello from sender"
    ]
    assert conversation.messages[0].role == "user"
    assert collaboration.inbox == []
    assert len(events) == 1
    assert events[0].from_entity == sender
    assert events[0].to_entity == receiver
    assert events[0].message.content == "Hello from sender"


@pytest.mark.asyncio
async def test_empty_inbox_is_noop() -> None:
    world = World()
    peer = world.create_entity()
    entity_id = world.create_entity()

    existing = [_msg("assistant", "already here")]
    world.add_component(entity_id, CollaborationComponent(peers=[peer], inbox=[]))
    world.add_component(entity_id, ConversationComponent(messages=list(existing)))

    events: list[MessageDeliveredEvent] = []

    async def on_delivered(event: MessageDeliveredEvent) -> None:
        events.append(event)

    world.event_bus.subscribe(MessageDeliveredEvent, on_delivered)
    await CollaborationSystem().process(world)

    conversation = world.get_component(entity_id, ConversationComponent)
    collaboration = world.get_component(entity_id, CollaborationComponent)
    assert conversation is not None
    assert collaboration is not None
    assert conversation.messages == existing
    assert collaboration.inbox == []
    assert events == []


@pytest.mark.asyncio
async def test_multiple_messages_are_processed_in_order() -> None:
    world = World()
    sender = world.create_entity()
    receiver = world.create_entity()

    world.add_component(
        receiver,
        CollaborationComponent(
            peers=[sender],
            inbox=[
                (sender, _msg("assistant", "first")),
                (sender, _msg("assistant", "second")),
                (sender, _msg("assistant", "third")),
            ],
        ),
    )
    world.add_component(receiver, ConversationComponent(messages=[]))

    delivered: list[tuple[int, int, str]] = []

    async def on_delivered(event: MessageDeliveredEvent) -> None:
        delivered.append((event.from_entity, event.to_entity, event.message.content))

    world.event_bus.subscribe(MessageDeliveredEvent, on_delivered)
    await CollaborationSystem().process(world)

    conversation = world.get_component(receiver, ConversationComponent)
    collaboration = world.get_component(receiver, CollaborationComponent)
    assert conversation is not None
    assert collaboration is not None
    assert [m.content for m in conversation.messages] == [
        f"From: {sender}: first",
        f"From: {sender}: second",
        f"From: {sender}: third",
    ]
    assert collaboration.inbox == []
    assert delivered == [
        (sender, receiver, "first"),
        (sender, receiver, "second"),
        (sender, receiver, "third"),
    ]


@pytest.mark.asyncio
async def test_entity_without_conversation_is_skipped() -> None:
    world = World()
    sender = world.create_entity()
    receiver = world.create_entity()
    world.add_component(
        receiver,
        CollaborationComponent(
            peers=[sender],
            inbox=[(sender, _msg("assistant", "pending"))],
        ),
    )

    seen: list[MessageDeliveredEvent] = []

    async def on_delivered(event: MessageDeliveredEvent) -> None:
        seen.append(event)

    world.event_bus.subscribe(MessageDeliveredEvent, on_delivered)
    await CollaborationSystem().process(world)

    collaboration = world.get_component(receiver, CollaborationComponent)
    assert collaboration is not None
    assert [item[1].content for item in collaboration.inbox] == ["pending"]
    assert seen == []


@pytest.mark.asyncio
async def test_multiple_entities_are_processed() -> None:
    world = World()
    sender_a = world.create_entity()
    sender_b = world.create_entity()
    receiver_a = world.create_entity()
    receiver_b = world.create_entity()

    world.add_component(
        receiver_a,
        CollaborationComponent(
            peers=[sender_a],
            inbox=[(sender_a, _msg("assistant", "a-msg"))],
        ),
    )
    world.add_component(
        receiver_b,
        CollaborationComponent(
            peers=[sender_b],
            inbox=[(sender_b, _msg("assistant", "b-msg"))],
        ),
    )
    world.add_component(receiver_a, ConversationComponent(messages=[]))
    world.add_component(receiver_b, ConversationComponent(messages=[]))

    await CollaborationSystem().process(world)

    conv_a = world.get_component(receiver_a, ConversationComponent)
    conv_b = world.get_component(receiver_b, ConversationComponent)
    collab_a = world.get_component(receiver_a, CollaborationComponent)
    collab_b = world.get_component(receiver_b, CollaborationComponent)
    assert conv_a is not None
    assert conv_b is not None
    assert collab_a is not None
    assert collab_b is not None
    assert [m.content for m in conv_a.messages] == [f"From: {sender_a}: a-msg"]
    assert [m.content for m in conv_b.messages] == [f"From: {sender_b}: b-msg"]
    assert collab_a.inbox == []
    assert collab_b.inbox == []
