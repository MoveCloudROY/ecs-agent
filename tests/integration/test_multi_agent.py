from __future__ import annotations

import asyncio
from datetime import datetime

import pytest

from ecs_agent.components import (
    ConversationComponent,
    ErrorComponent,
    LLMComponent,
    MessageBusConfigComponent,
    MessageBusConversationComponent,
    TerminalComponent,
)
from ecs_agent.core import Runner, World
from ecs_agent.observability import generate_traceparent
from ecs_agent.providers import FakeModel
from ecs_agent.systems.error_handling import ErrorHandlingSystem
from ecs_agent.systems.memory import MemorySystem
from ecs_agent.systems.message_bus import MessageBusSystem
from ecs_agent.systems.reasoning import ReasoningSystem
from ecs_agent.types import (
    CompletionResult,
    Message,
    MessageBusDeliveredEvent,
    MessageBusEnvelope,
    MessageBusPublishedEvent,
)


def _assistant_reply(content: str) -> CompletionResult:
    return CompletionResult(message=Message(role="assistant", content=content))


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
async def test_agent_sends_message_to_peer_via_message_bus() -> None:
    world = World()
    agent_a_id = world.create_entity()
    agent_b_id = world.create_entity()

    world.add_component(
        agent_a_id,
        MessageBusConfigComponent(),
    )
    world.add_component(
        agent_a_id,
        MessageBusConversationComponent(entity_id=agent_a_id, max_messages=10),
    )
    world.add_component(
        agent_b_id,
        MessageBusConversationComponent(entity_id=agent_b_id, max_messages=10),
    )

    bus = MessageBusSystem(priority=5)
    queue_b = bus.subscribe(
        topic=f"agent.chat.{agent_b_id}", subscriber_id=str(agent_b_id)
    )
    await bus.process(world)

    published_events: list[MessageBusPublishedEvent] = []
    delivered_events: list[MessageBusDeliveredEvent] = []

    async def on_published(event: MessageBusPublishedEvent) -> None:
        published_events.append(event)

    async def on_delivered(event: MessageBusDeliveredEvent) -> None:
        delivered_events.append(event)

    world.event_bus.subscribe(MessageBusPublishedEvent, on_published)
    world.event_bus.subscribe(MessageBusDeliveredEvent, on_delivered)

    delivered = await bus.publish(
        topic=f"agent.chat.{agent_b_id}",
        envelope=_envelope(source=str(agent_a_id), content="Hello from A"),
    )

    payload = await asyncio.wait_for(queue_b.get(), timeout=0.1)

    assert delivered is True
    assert payload == "Hello from A"
    assert len(published_events) == 1
    assert len(delivered_events) == 1
    assert published_events[0].topic == f"agent.chat.{agent_b_id}"
    assert delivered_events[0].subscriber_id == agent_b_id
    assert [msg.content for msg in bus.get_conversation(agent_a_id)] == [
        f"From: {agent_a_id}: Hello from A"
    ]
    assert bus.get_conversation(agent_b_id) == []


@pytest.mark.asyncio
async def test_bidirectional_communication() -> None:
    world = World()
    agent_a_id = world.create_entity()
    agent_b_id = world.create_entity()

    world.add_component(
        agent_a_id,
        MessageBusConfigComponent(),
    )
    world.add_component(
        agent_a_id,
        MessageBusConversationComponent(entity_id=agent_a_id, max_messages=10),
    )
    world.add_component(
        agent_b_id,
        MessageBusConversationComponent(entity_id=agent_b_id, max_messages=10),
    )

    bus = MessageBusSystem(priority=5)
    queue_a = bus.subscribe(
        topic=f"agent.chat.{agent_a_id}", subscriber_id=str(agent_a_id)
    )
    queue_b = bus.subscribe(
        topic=f"agent.chat.{agent_b_id}", subscriber_id=str(agent_b_id)
    )
    await bus.process(world)

    delivered_to_a = await bus.publish(
        topic=f"agent.chat.{agent_a_id}",
        envelope=_envelope(source=str(agent_b_id), content="B to A"),
    )
    delivered_to_b = await bus.publish(
        topic=f"agent.chat.{agent_b_id}",
        envelope=_envelope(source=str(agent_a_id), content="A to B"),
    )

    payload_a = await asyncio.wait_for(queue_a.get(), timeout=0.1)
    payload_b = await asyncio.wait_for(queue_b.get(), timeout=0.1)

    assert delivered_to_a is True
    assert delivered_to_b is True
    assert payload_a == "B to A"
    assert payload_b == "A to B"
    assert [msg.content for msg in bus.get_conversation(agent_a_id)] == [
        f"From: {agent_a_id}: A to B"
    ]
    assert [msg.content for msg in bus.get_conversation(agent_b_id)] == [
        f"From: {agent_b_id}: B to A"
    ]


@pytest.mark.asyncio
async def test_unknown_peer_graceful_failure_without_subscribers() -> None:
    world = World()
    agent_a_id = world.create_entity()

    world.add_component(agent_a_id, MessageBusConfigComponent())
    world.add_component(
        agent_a_id,
        MessageBusConversationComponent(entity_id=agent_a_id, max_messages=10),
    )

    bus = MessageBusSystem(priority=5)
    await bus.process(world)

    delivered = await bus.publish(
        topic="agent.chat.9999",
        envelope=_envelope(source=str(agent_a_id), content="Can you hear me?"),
    )

    assert delivered is False
    error = world.get_component(agent_a_id, ErrorComponent)
    assert error is None
    assert bus.get_conversation(agent_a_id) == []


@pytest.mark.asyncio
async def test_multi_agent_full_loop() -> None:
    world = World()
    agent_a_id = world.create_entity()
    agent_b_id = world.create_entity()

    model_a = FakeModel(responses=[_assistant_reply("Agent A ready")])
    model_b = FakeModel(responses=[_assistant_reply("Agent B ready")])

    world.add_component(
        agent_a_id,
        LLMComponent(model=model_a, system_prompt=""),
    )
    world.add_component(
        agent_a_id,
        ConversationComponent(messages=[Message(role="user", content="Start A")]),
    )
    world.add_component(
        agent_a_id,
        MessageBusConfigComponent(),
    )
    world.add_component(
        agent_a_id,
        MessageBusConversationComponent(entity_id=agent_a_id, max_messages=10),
    )

    world.add_component(
        agent_b_id,
        LLMComponent(model=model_b, system_prompt=""),
    )
    world.add_component(
        agent_b_id,
        ConversationComponent(messages=[Message(role="user", content="Start B")]),
    )
    world.add_component(
        agent_b_id,
        MessageBusConversationComponent(entity_id=agent_b_id, max_messages=10),
    )

    message_bus = MessageBusSystem(priority=5)
    world.register_system(message_bus, priority=5)

    world.register_system(ReasoningSystem(priority=0), priority=0)
    world.register_system(MemorySystem(), priority=10)
    world.register_system(ErrorHandlingSystem(priority=99), priority=99)

    await Runner().run(world, max_ticks=5)

    terminal_a = world.get_component(agent_a_id, TerminalComponent)
    terminal_b = world.get_component(agent_b_id, TerminalComponent)
    assert terminal_a is not None
    assert terminal_b is not None

    conv_a = world.get_component(agent_a_id, ConversationComponent)
    conv_b = world.get_component(agent_b_id, ConversationComponent)
    assert conv_a is not None
    assert conv_b is not None

    request_task = asyncio.create_task(
        message_bus.request(
            topic=f"agent.rpc.{agent_b_id}",
            message={"from": str(agent_a_id), "content": "Ping from A"},
            timeout=0.2,
        )
    )
    await asyncio.sleep(0)

    pending = getattr(message_bus, "_pending_requests", None)
    assert isinstance(pending, dict)
    assert len(pending) == 1
    correlation_id = next(iter(pending.keys()))

    accepted = await message_bus.respond(
        correlation_id=correlation_id,
        message={"from": str(agent_b_id), "content": "Pong from B"},
    )
    response = await asyncio.wait_for(request_task, timeout=0.2)

    assert accepted is True
    assert response["from"] == str(agent_b_id)
    assert response["content"] == "Pong from B"
    assert list(world.query(ErrorComponent)) == []
