from __future__ import annotations

import asyncio
from datetime import datetime

import pytest

from ecs_agent.components import MessageBusConversationComponent
from ecs_agent.core import World
from ecs_agent.observability import generate_traceparent
from ecs_agent.systems.message_bus import MessageBusSystem
from ecs_agent.types import MessageBusEnvelope


def _envelope(source: str, content: str) -> MessageBusEnvelope:
    return MessageBusEnvelope(
        id=f"evt-{source}-{content}",
        source=source,
        type="ecs.bus.publish",
        specversion="1.0",
        correlationid=f"corr-{source}-{content}",
        traceparent=generate_traceparent(),
        data=content,
        time=datetime.now(),
    )


@pytest.mark.asyncio
async def test_entity_conversation_retrieval_returns_entity_scoped_messages() -> None:
    world = World()
    sender_a = world.create_entity()
    sender_b = world.create_entity()

    world.add_component(
        sender_a,
        MessageBusConversationComponent(entity_id=sender_a, max_messages=10),
    )
    world.add_component(
        sender_b,
        MessageBusConversationComponent(entity_id=sender_b, max_messages=10),
    )

    bus = MessageBusSystem()
    assert bus.priority == 5

    bus.subscribe(topic="agent.chat", subscriber_id=str(sender_b))
    await bus.process(world)

    delivered = await bus.publish(
        topic="agent.chat",
        envelope=_envelope(source=str(sender_a), content="hello"),
    )

    assert delivered is True
    assert [msg.content for msg in bus.get_conversation(sender_a)] == [
        f"From: {sender_a}: hello"
    ]
    assert bus.get_conversation(sender_a)[0].role == "user"
    assert bus.get_conversation(sender_b) == []


@pytest.mark.asyncio
async def test_publish_without_subscription_safe_failure() -> None:
    world = World()
    sender = world.create_entity()
    world.add_component(
        sender,
        MessageBusConversationComponent(entity_id=sender, max_messages=10),
    )

    bus = MessageBusSystem()
    await bus.process(world)

    delivered = await bus.publish(
        topic="agent.none",
        envelope=_envelope(source=str(sender), content="nobody-listening"),
    )

    assert delivered is False
    assert bus.get_conversation(sender) == []


@pytest.mark.asyncio
async def test_multiple_messages_are_delivered_in_order_for_single_subscriber() -> None:
    world = World()
    sender = world.create_entity()
    receiver = world.create_entity()
    world.add_component(
        sender,
        MessageBusConversationComponent(entity_id=sender, max_messages=10),
    )

    bus = MessageBusSystem()
    receiver_queue = bus.subscribe(topic="agent.chat", subscriber_id=str(receiver))
    await bus.process(world)

    await bus.publish(
        topic="agent.chat",
        envelope=_envelope(source=str(sender), content="first"),
    )
    await bus.publish(
        topic="agent.chat",
        envelope=_envelope(source=str(sender), content="second"),
    )
    await bus.publish(
        topic="agent.chat",
        envelope=_envelope(source=str(sender), content="third"),
    )

    delivered = [
        await asyncio.wait_for(receiver_queue.get(), timeout=0.1) for _ in range(3)
    ]

    assert delivered == ["first", "second", "third"]
    assert [msg.content for msg in bus.get_conversation(sender)] == [
        f"From: {sender}: first",
        f"From: {sender}: second",
        f"From: {sender}: third",
    ]


@pytest.mark.asyncio
async def test_multiple_entities_publish_without_cross_talk() -> None:
    world = World()
    sender_a = world.create_entity()
    sender_b = world.create_entity()
    receiver_a = world.create_entity()
    receiver_b = world.create_entity()

    world.add_component(
        sender_a,
        MessageBusConversationComponent(entity_id=sender_a, max_messages=10),
    )
    world.add_component(
        sender_b,
        MessageBusConversationComponent(entity_id=sender_b, max_messages=10),
    )

    bus = MessageBusSystem()
    queue_a = bus.subscribe(
        topic=f"agent.chat.{receiver_a}", subscriber_id=str(receiver_a)
    )
    queue_b = bus.subscribe(
        topic=f"agent.chat.{receiver_b}", subscriber_id=str(receiver_b)
    )
    await bus.process(world)

    delivered_a = await bus.publish(
        topic=f"agent.chat.{receiver_a}",
        envelope=_envelope(source=str(sender_a), content="a-msg"),
    )
    delivered_b = await bus.publish(
        topic=f"agent.chat.{receiver_b}",
        envelope=_envelope(source=str(sender_b), content="b-msg"),
    )

    payload_a = await asyncio.wait_for(queue_a.get(), timeout=0.1)
    payload_b = await asyncio.wait_for(queue_b.get(), timeout=0.1)

    assert delivered_a is True
    assert delivered_b is True
    assert payload_a == "a-msg"
    assert payload_b == "b-msg"
    assert [msg.content for msg in bus.get_conversation(sender_a)] == [
        f"From: {sender_a}: a-msg"
    ]
    assert [msg.content for msg in bus.get_conversation(sender_b)] == [
        f"From: {sender_b}: b-msg"
    ]


@pytest.mark.asyncio
async def test_publish_with_no_source_conversation_is_safe() -> None:
    world = World()
    sender = world.create_entity()
    receiver = world.create_entity()

    bus = MessageBusSystem()
    queue = bus.subscribe(topic="agent.chat", subscriber_id=str(receiver))
    await bus.process(world)

    delivered = await bus.publish(
        topic="agent.chat",
        envelope=_envelope(source=str(sender), content="pending"),
    )
    payload = await asyncio.wait_for(queue.get(), timeout=0.1)

    assert delivered is True
    assert payload == "pending"
    assert bus.get_conversation(sender) == []
