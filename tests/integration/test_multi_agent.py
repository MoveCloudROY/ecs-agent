from __future__ import annotations

# mypy: disable-error-code=import-untyped

import time

import pytest

from ecs_agent.components import (
    CollaborationComponent,
    ConversationComponent,
    ErrorComponent,
    LLMComponent,
    TerminalComponent,
)
from ecs_agent.core import Runner, World
from ecs_agent.providers import FakeProvider
from ecs_agent.systems.collaboration import CollaborationSystem
from ecs_agent.systems.error_handling import ErrorHandlingSystem
from ecs_agent.systems.memory import MemorySystem
from ecs_agent.systems.reasoning import ReasoningSystem
from ecs_agent.types import CompletionResult, EntityId, Message, MessageDeliveredEvent


def _assistant_reply(content: str) -> CompletionResult:
    return CompletionResult(message=Message(role="assistant", content=content))


def _deliver_to_peer(
    world: World,
    sender_id: EntityId,
    target_id: EntityId,
    message: Message,
) -> bool:
    target_collaboration = world.get_component(target_id, CollaborationComponent)
    if target_collaboration is None:
        world.add_component(
            sender_id,
            ErrorComponent(
                error=f"Target peer {target_id} not found",
                system_name="CollaborationSystem",
                timestamp=time.time(),
            ),
        )
        return False

    target_collaboration.inbox.append((sender_id, message))
    return True


@pytest.mark.asyncio
async def test_agent_sends_message_to_peer() -> None:
    world = World()
    agent_a_id = world.create_entity()
    agent_b_id = world.create_entity()

    world.add_component(agent_a_id, ConversationComponent(messages=[]))
    world.add_component(
        agent_a_id,
        CollaborationComponent(peers=[agent_b_id], inbox=[]),
    )

    world.add_component(agent_b_id, ConversationComponent(messages=[]))
    world.add_component(
        agent_b_id,
        CollaborationComponent(peers=[agent_a_id], inbox=[]),
    )

    collab_b = world.get_component(agent_b_id, CollaborationComponent)
    assert collab_b is not None

    outbound = Message(role="user", content="Hello from A")
    collab_b.inbox.append((agent_a_id, outbound))

    assert len(collab_b.inbox) > 0
    sender_id, message = collab_b.inbox[0]
    assert sender_id == agent_a_id
    assert message.content == "Hello from A"

    delivered_events: list[MessageDeliveredEvent] = []

    async def on_delivered(event: MessageDeliveredEvent) -> None:
        delivered_events.append(event)

    world.event_bus.subscribe(MessageDeliveredEvent, on_delivered)
    await CollaborationSystem(priority=5).process(world)

    conversation_b = world.get_component(agent_b_id, ConversationComponent)
    assert conversation_b is not None
    assert conversation_b.messages[-1].content == f"From: {agent_a_id}: Hello from A"
    assert collab_b.inbox == []
    assert len(delivered_events) == 1


@pytest.mark.asyncio
async def test_bidirectional_communication() -> None:
    world = World()
    agent_a_id = world.create_entity()
    agent_b_id = world.create_entity()

    world.add_component(agent_a_id, ConversationComponent(messages=[]))
    world.add_component(
        agent_a_id,
        CollaborationComponent(peers=[agent_b_id], inbox=[]),
    )
    world.add_component(agent_b_id, ConversationComponent(messages=[]))
    world.add_component(
        agent_b_id,
        CollaborationComponent(peers=[agent_a_id], inbox=[]),
    )

    collab_a = world.get_component(agent_a_id, CollaborationComponent)
    collab_b = world.get_component(agent_b_id, CollaborationComponent)
    assert collab_a is not None
    assert collab_b is not None

    collab_b.inbox.append((agent_a_id, Message(role="user", content="A to B")))
    collab_a.inbox.append((agent_b_id, Message(role="user", content="B to A")))

    await CollaborationSystem(priority=5).process(world)

    conv_a = world.get_component(agent_a_id, ConversationComponent)
    conv_b = world.get_component(agent_b_id, ConversationComponent)
    assert conv_a is not None
    assert conv_b is not None
    assert any(msg.content == f"From: {agent_b_id}: B to A" for msg in conv_a.messages)
    assert any(msg.content == f"From: {agent_a_id}: A to B" for msg in conv_b.messages)
    assert collab_a.inbox == []
    assert collab_b.inbox == []


@pytest.mark.asyncio
async def test_unknown_peer_graceful_failure() -> None:
    world = World()
    agent_a_id = world.create_entity()

    unknown_peer = EntityId(9999)
    world.add_component(agent_a_id, ConversationComponent(messages=[]))
    world.add_component(
        agent_a_id,
        CollaborationComponent(peers=[unknown_peer], inbox=[]),
    )

    delivered = _deliver_to_peer(
        world,
        sender_id=agent_a_id,
        target_id=unknown_peer,
        message=Message(role="user", content="Can you hear me?"),
    )

    assert delivered is False
    error = world.get_component(agent_a_id, ErrorComponent)
    assert error is not None
    assert "not found" in error.error


@pytest.mark.asyncio
async def test_multi_agent_full_loop() -> None:
    world = World()
    agent_a_id = world.create_entity()
    agent_b_id = world.create_entity()

    provider_a = FakeProvider(responses=[_assistant_reply("Agent A ready")])
    provider_b = FakeProvider(responses=[_assistant_reply("Agent B ready")])

    world.add_component(
        agent_a_id,
        LLMComponent(provider=provider_a, model="fake", system_prompt=""),
    )
    world.add_component(
        agent_a_id,
        ConversationComponent(messages=[Message(role="user", content="Start A")]),
    )
    world.add_component(
        agent_a_id,
        CollaborationComponent(peers=[agent_b_id], inbox=[]),
    )

    world.add_component(
        agent_b_id,
        LLMComponent(provider=provider_b, model="fake", system_prompt=""),
    )
    world.add_component(
        agent_b_id,
        ConversationComponent(messages=[Message(role="user", content="Start B")]),
    )
    world.add_component(
        agent_b_id,
        CollaborationComponent(peers=[agent_a_id], inbox=[]),
    )

    collab_a = world.get_component(agent_a_id, CollaborationComponent)
    collab_b = world.get_component(agent_b_id, CollaborationComponent)
    assert collab_a is not None
    assert collab_b is not None
    collab_b.inbox.append((agent_a_id, Message(role="user", content="Ping from A")))
    collab_a.inbox.append((agent_b_id, Message(role="user", content="Pong from B")))

    world.register_system(ReasoningSystem(priority=0), priority=0)
    world.register_system(CollaborationSystem(priority=5), priority=5)
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
    assert any(
        msg.content == f"From: {agent_b_id}: Pong from B" for msg in conv_a.messages
    )
    assert any(
        msg.content == f"From: {agent_a_id}: Ping from A" for msg in conv_b.messages
    )
    assert collab_a.inbox == []
    assert collab_b.inbox == []
    assert list(world.query(ErrorComponent)) == []
