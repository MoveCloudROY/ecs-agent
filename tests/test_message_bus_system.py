from __future__ import annotations

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
