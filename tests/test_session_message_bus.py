"""Message-bus history is durable before observers or delivery advance."""

import pytest

from ecs_agent.components import ConversationComponent, MessageBusConversationComponent, SessionRefComponent, SessionPolicyComponent
from ecs_agent.core import World
from ecs_agent.sessions import InMemorySessionStore, SessionService, install_session_recording
from ecs_agent.systems.message_bus import MessageBusSystem
from ecs_agent.types import MessageBusEnvelope, MessageBusPublishedEvent


async def setup():
    world = World()
    entity = world.create_entity()
    world.add_component(entity, ConversationComponent(messages=[]))
    world.add_component(entity, MessageBusConversationComponent(entity_id=entity, messages=[], max_messages=1))
    service = SessionService(InMemorySessionStore())
    install_session_recording(world, entity, service, workspace_id="project", agent_name="main", register_tools=False)
    bus = MessageBusSystem()
    await bus.process(world)
    queue = bus.subscribe("test", str(entity))
    return world, entity, service, bus, queue


def envelope(entity, text, identity):
    return MessageBusEnvelope(id=identity, source=str(entity), type="test", specversion="1.0", correlationid="correlation", traceparent="trace", data=text)


async def test_bus_publication_and_trim_retain_original_evidence():
    world, entity, service, bus, queue = await setup()
    ref = world.get_component(entity, SessionRefComponent)
    observed = []

    async def observe(event):
        observed.append(bool(service.search(ref.session_id, SessionPolicyComponent(), event.envelope.data).items))

    world.event_bus.subscribe(MessageBusPublishedEvent, observe)
    await bus.publish("test", envelope(entity, "first evidence", "one"))
    await bus.publish("test", envelope(entity, "second evidence", "two"))
    assert observed == [True, True]
    assert len(bus.get_conversation(entity)) == 1
    assert "second evidence" in bus.get_conversation(entity)[0].content
    assert len(service.search(ref.session_id, SessionPolicyComponent(), "first evidence").items) == 1
    assert any(record.kind == "bus_trim" for record in service.store.records(ref.session_id))
    assert queue.qsize() == 2
    service.close()


async def test_failed_bus_record_blocks_delivery_and_observation(monkeypatch):
    world, entity, service, bus, queue = await setup()
    observed = []

    async def observe(event):
        observed.append(event)

    def fail(*args, **kwargs):
        raise OSError("no storage")

    world.event_bus.subscribe(MessageBusPublishedEvent, observe)
    monkeypatch.setattr(service.store, "append", fail)
    with pytest.raises(OSError):
        await bus.publish("test", envelope(entity, "must not deliver", "failed"))
    assert queue.empty()
    assert observed == []
    assert bus.get_conversation(entity) == []
    service.close()
