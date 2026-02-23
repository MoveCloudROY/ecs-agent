from __future__ import annotations

from typing import Any

from ecs_agent.components import (
    CollaborationComponent,
    ConversationComponent,
    ErrorComponent,
    KVStoreComponent,
    LLMComponent,
    OwnerComponent,
    PendingToolCallsComponent,
    PlanComponent,
    SystemPromptComponent,
    TerminalComponent,
    ToolRegistryComponent,
    ToolResultsComponent,
)
from ecs_agent.core.world import World
from ecs_agent.serialization import NON_SERIALIZABLE_PLACEHOLDER, WorldSerializer
from ecs_agent.types import EntityId, Message, ToolCall, ToolSchema


class DummyProvider:
    async def complete(self, messages, tools=None, stream=False, response_format=None):
        _ = (messages, tools, stream, response_format)
        raise NotImplementedError


async def async_tool_handler(*args: Any, **kwargs: Any) -> str:
    _ = (args, kwargs)
    return "ok"


def test_to_dict_with_simple_components() -> None:
    world = World()
    entity = world.create_entity()
    world.add_component(
        entity,
        ConversationComponent(
            messages=[Message(role="user", content="hello")], max_messages=50
        ),
    )
    world.add_component(entity, KVStoreComponent(store={"k": "v"}))

    data = WorldSerializer.to_dict(world)

    assert data["next_entity_id"] == 2
    assert data["entities"]["1"]["ConversationComponent"] == {
        "messages": [
            {
                "role": "user",
                "content": "hello",
                "tool_calls": None,
                "tool_call_id": None,
            }
        ],
        "max_messages": 50,
    }
    assert data["entities"]["1"]["KVStoreComponent"] == {"store": {"k": "v"}}


def test_to_dict_skips_non_serializable_fields() -> None:
    provider = DummyProvider()
    world = World()
    entity = world.create_entity()
    world.add_component(
        entity, LLMComponent(provider=provider, model="gpt-4", system_prompt="sys")
    )
    world.add_component(
        entity,
        ToolRegistryComponent(
            tools={"ping": ToolSchema(name="ping", description="Ping", parameters={})},
            handlers={"ping": async_tool_handler},
        ),
    )

    data = WorldSerializer.to_dict(world)

    assert (
        data["entities"]["1"]["LLMComponent"]["provider"]
        == NON_SERIALIZABLE_PLACEHOLDER
    )
    assert (
        data["entities"]["1"]["ToolRegistryComponent"]["handlers"]
        == NON_SERIALIZABLE_PLACEHOLDER
    )


def test_from_dict_reconstructs_world_correctly() -> None:
    provider = DummyProvider()
    providers = {"default": provider, "gpt-4": provider}
    handlers = {"ping": async_tool_handler}
    data = {
        "next_entity_id": 5,
        "entities": {
            "1": {
                "LLMComponent": {
                    "provider": NON_SERIALIZABLE_PLACEHOLDER,
                    "model": "gpt-4",
                    "system_prompt": "sys",
                },
                "ConversationComponent": {
                    "messages": [
                        {
                            "role": "user",
                            "content": "hi",
                            "tool_calls": None,
                            "tool_call_id": None,
                        }
                    ],
                    "max_messages": 10,
                },
                "ToolRegistryComponent": {
                    "tools": {
                        "ping": {
                            "name": "ping",
                            "description": "Ping",
                            "parameters": {},
                        }
                    },
                    "handlers": NON_SERIALIZABLE_PLACEHOLDER,
                },
            }
        },
    }

    world = WorldSerializer.from_dict(data, providers=providers, tool_handlers=handlers)

    llm = world.get_component(EntityId(1), LLMComponent)
    conv = world.get_component(EntityId(1), ConversationComponent)
    tool_registry = world.get_component(EntityId(1), ToolRegistryComponent)

    assert llm is not None
    assert llm.provider is provider
    assert llm.model == "gpt-4"
    assert conv is not None
    assert isinstance(conv.messages[0], Message)
    assert conv.messages[0].content == "hi"
    assert tool_registry is not None
    assert isinstance(tool_registry.tools["ping"], ToolSchema)
    assert tool_registry.handlers is handlers
    assert world.create_entity() == EntityId(5)


def test_round_trip_preserves_state() -> None:
    provider = DummyProvider()
    providers = {"default": provider, "gpt-4": provider}
    handlers = {"ping": async_tool_handler}

    world = World()
    entity = world.create_entity()
    world.add_component(
        entity, LLMComponent(provider=provider, model="gpt-4", system_prompt="sys")
    )
    world.add_component(
        entity, ConversationComponent(messages=[Message(role="user", content="hello")])
    )
    world.add_component(
        entity, PlanComponent(steps=["a", "b"], current_step=1, completed=False)
    )
    world.add_component(
        entity,
        PendingToolCallsComponent(
            tool_calls=[ToolCall(id="tc1", name="ping", arguments={"x": 1})]
        ),
    )
    world.add_component(entity, ToolResultsComponent(results={"tc1": "ok"}))
    world.add_component(entity, KVStoreComponent(store={"foo": "bar"}))

    data = WorldSerializer.to_dict(world)
    restored = WorldSerializer.from_dict(
        data, providers=providers, tool_handlers=handlers
    )

    assert WorldSerializer.to_dict(restored) == data


def test_save_and_load_to_file(tmp_path) -> None:
    provider = DummyProvider()
    providers = {"default": provider, "gpt-4": provider}
    handlers = {"ping": async_tool_handler}

    world = World()
    entity = world.create_entity()
    world.add_component(entity, LLMComponent(provider=provider, model="gpt-4"))
    world.add_component(entity, KVStoreComponent(store={"a": 1}))

    path = tmp_path / "world.json"
    WorldSerializer.save(world, path)

    loaded = WorldSerializer.load(path, providers=providers, tool_handlers=handlers)
    loaded_llm = loaded.get_component(EntityId(1), LLMComponent)
    loaded_kv = loaded.get_component(EntityId(1), KVStoreComponent)

    assert path.exists()
    assert loaded_llm is not None
    assert loaded_llm.provider is provider
    assert loaded_kv == KVStoreComponent(store={"a": 1})


def test_serialization_with_all_component_types() -> None:
    provider = DummyProvider()
    providers = {"default": provider, "gpt-4": provider}
    handlers = {"ping": async_tool_handler}

    world = World()
    entity = world.create_entity()
    world.add_component(
        entity, LLMComponent(provider=provider, model="gpt-4", system_prompt="sys")
    )
    world.add_component(
        entity, ConversationComponent(messages=[Message(role="user", content="hello")])
    )
    world.add_component(
        entity, PlanComponent(steps=["step1", "step2"], current_step=1, completed=False)
    )
    world.add_component(
        entity,
        ToolRegistryComponent(
            tools={"ping": ToolSchema(name="ping", description="Ping", parameters={})},
            handlers=handlers,
        ),
    )
    world.add_component(
        entity,
        PendingToolCallsComponent(
            tool_calls=[ToolCall(id="1", name="ping", arguments={})]
        ),
    )
    world.add_component(entity, ToolResultsComponent(results={"1": "ok"}))
    world.add_component(entity, KVStoreComponent(store={"memory": "value"}))
    world.add_component(
        entity,
        CollaborationComponent(
            peers=[EntityId(2)],
            inbox=[(EntityId(2), Message(role="assistant", content="x"))],
        ),
    )
    world.add_component(entity, OwnerComponent(owner_id=EntityId(99)))
    world.add_component(
        entity, ErrorComponent(error="err", system_name="planner", timestamp=1.5)
    )
    world.add_component(entity, TerminalComponent(reason="done"))
    world.add_component(entity, SystemPromptComponent(content="be concise"))

    serialized = WorldSerializer.to_dict(world)
    component_names = set(serialized["entities"]["1"].keys())
    expected = {
        "LLMComponent",
        "ConversationComponent",
        "PlanComponent",
        "ToolRegistryComponent",
        "PendingToolCallsComponent",
        "ToolResultsComponent",
        "KVStoreComponent",
        "CollaborationComponent",
        "OwnerComponent",
        "ErrorComponent",
        "TerminalComponent",
        "SystemPromptComponent",
    }
    assert component_names == expected

    restored = WorldSerializer.from_dict(
        serialized, providers=providers, tool_handlers=handlers
    )
    assert restored.has_component(EntityId(1), LLMComponent)
    assert restored.has_component(EntityId(1), ConversationComponent)
    assert restored.has_component(EntityId(1), PlanComponent)
    assert restored.has_component(EntityId(1), ToolRegistryComponent)
    assert restored.has_component(EntityId(1), PendingToolCallsComponent)
    assert restored.has_component(EntityId(1), ToolResultsComponent)
    assert restored.has_component(EntityId(1), KVStoreComponent)
    assert restored.has_component(EntityId(1), CollaborationComponent)
    assert restored.has_component(EntityId(1), OwnerComponent)
    assert restored.has_component(EntityId(1), ErrorComponent)
    assert restored.has_component(EntityId(1), TerminalComponent)
    assert restored.has_component(EntityId(1), SystemPromptComponent)
