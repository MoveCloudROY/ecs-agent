from __future__ import annotations

from typing import Any

import pytest

from ecs_agent.components import (
    CheckpointComponent,
    ConversationComponent,
    LLMComponent,
    ToolRegistryComponent,
)
from ecs_agent.core import World
from ecs_agent.systems.checkpoint import CheckpointSystem
from ecs_agent.types import (
    CheckpointCreatedEvent,
    CheckpointRestoredEvent,
    Message,
    ToolSchema,
)


class DummyProvider:
    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        stream: bool = False,
        response_format: dict[str, Any] | None = None,
    ) -> Any:
        _ = (messages, tools, stream, response_format)
        raise NotImplementedError


async def async_tool_handler(*args: Any, **kwargs: Any) -> str:
    _ = (args, kwargs)
    return "ok"


@pytest.mark.asyncio
async def test_process_creates_snapshot_each_tick() -> None:
    world = World()
    entity_id = world.create_entity()
    world.add_component(entity_id, CheckpointComponent())
    world.add_component(
        entity_id,
        ConversationComponent(messages=[Message(role="user", content="hello")]),
    )

    system = CheckpointSystem()
    await system.process(world)
    await system.process(world)

    checkpoint = world.get_component(entity_id, CheckpointComponent)
    assert checkpoint is not None
    assert len(checkpoint.snapshots) == 2


@pytest.mark.asyncio
async def test_process_enforces_max_snapshots_limit_by_dropping_oldest() -> None:
    world = World()
    entity_id = world.create_entity()
    world.add_component(entity_id, CheckpointComponent(max_snapshots=3))
    world.add_component(entity_id, ConversationComponent(messages=[]))

    system = CheckpointSystem()
    for index in range(5):
        conversation = world.get_component(entity_id, ConversationComponent)
        assert conversation is not None
        conversation.messages.append(Message(role="user", content=f"m{index}"))
        await system.process(world)

    checkpoint = world.get_component(entity_id, CheckpointComponent)
    assert checkpoint is not None
    assert len(checkpoint.snapshots) == 3
    first_kept = checkpoint.snapshots[0]["entities"]["1"]["ConversationComponent"][
        "messages"
    ][-1]["content"]
    last_kept = checkpoint.snapshots[-1]["entities"]["1"]["ConversationComponent"][
        "messages"
    ][-1]["content"]
    assert first_kept == "m2"
    assert last_kept == "m4"


@pytest.mark.asyncio
async def test_process_publishes_checkpoint_created_event() -> None:
    world = World()
    entity_id = world.create_entity()
    world.add_component(entity_id, CheckpointComponent())

    seen: list[CheckpointCreatedEvent] = []

    async def handler(event: CheckpointCreatedEvent) -> None:
        seen.append(event)

    world.event_bus.subscribe(CheckpointCreatedEvent, handler)

    await CheckpointSystem().process(world)

    assert len(seen) == 1
    assert seen[0].entity_id == entity_id
    assert seen[0].checkpoint_id == 0
    assert seen[0].timestamp > 0


@pytest.mark.asyncio
async def test_undo_restores_previous_conversation_state() -> None:
    world = World()
    entity_id = world.create_entity()
    world.add_component(entity_id, CheckpointComponent())
    world.add_component(
        entity_id,
        ConversationComponent(messages=[Message(role="user", content="v1")]),
    )

    system = CheckpointSystem()
    await system.process(world)

    conversation = world.get_component(entity_id, ConversationComponent)
    assert conversation is not None
    conversation.messages.append(Message(role="assistant", content="v2"))
    await system.process(world)

    await CheckpointSystem.undo(world, providers={}, tool_handlers={})

    restored = world.get_component(entity_id, ConversationComponent)
    assert restored is not None
    assert [message.content for message in restored.messages] == ["v1"]


@pytest.mark.asyncio
async def test_undo_raises_value_error_when_no_snapshots() -> None:
    world = World()
    entity_id = world.create_entity()
    world.add_component(entity_id, CheckpointComponent())

    with pytest.raises(ValueError, match="No checkpoint snapshots"):
        await CheckpointSystem.undo(world, providers={}, tool_handlers={})


@pytest.mark.asyncio
async def test_undo_publishes_checkpoint_restored_event() -> None:
    world = World()
    entity_id = world.create_entity()
    world.add_component(entity_id, CheckpointComponent())

    await CheckpointSystem().process(world)

    seen: list[CheckpointRestoredEvent] = []

    async def handler(event: CheckpointRestoredEvent) -> None:
        seen.append(event)

    world.event_bus.subscribe(CheckpointRestoredEvent, handler)

    await CheckpointSystem.undo(world, providers={}, tool_handlers={})

    assert len(seen) == 1
    assert seen[0].entity_id == entity_id
    assert seen[0].checkpoint_id == 0
    assert seen[0].timestamp > 0


@pytest.mark.asyncio
async def test_undo_reinjects_provider_and_tool_handlers() -> None:
    provider = DummyProvider()
    handlers = {"ping": async_tool_handler}

    world = World()
    entity_id = world.create_entity()
    world.add_component(entity_id, CheckpointComponent())
    world.add_component(entity_id, LLMComponent(provider=provider, model="gpt-4"))
    world.add_component(
        entity_id,
        ToolRegistryComponent(
            tools={"ping": ToolSchema(name="ping", description="Ping", parameters={})},
            handlers=handlers,
        ),
    )

    await CheckpointSystem().process(world)
    await CheckpointSystem.undo(
        world,
        providers={"default": provider, "gpt-4": provider},
        tool_handlers=handlers,
    )

    restored_llm = world.get_component(entity_id, LLMComponent)
    restored_tools = world.get_component(entity_id, ToolRegistryComponent)
    assert restored_llm is not None
    assert restored_tools is not None
    assert restored_llm.provider is provider
    assert restored_tools.handlers is handlers


@pytest.mark.asyncio
async def test_process_skips_entities_without_checkpoint_component() -> None:
    world = World()
    entity_id = world.create_entity()
    world.add_component(
        entity_id,
        ConversationComponent(messages=[Message(role="user", content="x")]),
    )

    seen: list[CheckpointCreatedEvent] = []

    async def handler(event: CheckpointCreatedEvent) -> None:
        seen.append(event)

    world.event_bus.subscribe(CheckpointCreatedEvent, handler)

    await CheckpointSystem().process(world)

    assert seen == []
