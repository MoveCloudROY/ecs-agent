from __future__ import annotations

from typing import Any

from ecs_agent.components import (
    CollaborationComponent,
    ConversationComponent,
    EmbeddingComponent,
    ErrorComponent,
    KVStoreComponent,
    LLMComponent,
    OwnerComponent,
    PendingToolCallsComponent,
    PlanComponent,
    PlanSearchComponent,
    RAGTriggerComponent,
    SandboxConfigComponent,
    SystemPromptComponent,
    TerminalComponent,
    ToolApprovalComponent,
    ToolRegistryComponent,
    ToolResultsComponent,
    VectorStoreComponent,
    VectorStoreComponent,
    CheckpointComponent,
    CompactionConfigComponent,
    ConversationArchiveComponent,
    RunnerStateComponent,
    StreamingComponent,
)
from ecs_agent.core.world import World
from ecs_agent.serialization import NON_SERIALIZABLE_PLACEHOLDER, WorldSerializer
from ecs_agent.types import ApprovalPolicy, EntityId, Message, ToolCall, ToolSchema


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


def test_serialization_roundtrip_with_tool_approval_component() -> None:
    """Test that ToolApprovalComponent with ApprovalPolicy enum roundtrips correctly."""
    world = World()
    entity = world.create_entity()
    world.add_component(
        entity,
        ToolApprovalComponent(
            policy=ApprovalPolicy.REQUIRE_APPROVAL,
            timeout=45.0,
            approved_calls=["call1"],
            denied_calls=["call2"],
        ),
    )

    serialized = WorldSerializer.to_dict(world)
    restored = WorldSerializer.from_dict(
        serialized, providers={}, tool_handlers={}
    )

    restored_comp = restored.get_component(entity, ToolApprovalComponent)
    assert restored_comp is not None
    assert restored_comp.policy == ApprovalPolicy.REQUIRE_APPROVAL
    assert restored_comp.timeout == 45.0
    assert restored_comp.approved_calls == ["call1"]
    assert restored_comp.denied_calls == ["call2"]


def test_serialization_roundtrip_with_sandbox_config() -> None:
    """Test that SandboxConfigComponent roundtrips correctly."""
    world = World()
    entity = world.create_entity()
    world.add_component(
        entity,
        SandboxConfigComponent(timeout=60.0, max_output_size=50_000),
    )

    serialized = WorldSerializer.to_dict(world)
    restored = WorldSerializer.from_dict(
        serialized, providers={}, tool_handlers={}
    )

    restored_comp = restored.get_component(entity, SandboxConfigComponent)
    assert restored_comp is not None
    assert restored_comp.timeout == 60.0
    assert restored_comp.max_output_size == 50_000


def test_serialization_roundtrip_with_plan_search() -> None:
    """Test that PlanSearchComponent roundtrips correctly."""
    world = World()
    entity = world.create_entity()
    world.add_component(
        entity,
        PlanSearchComponent(
            max_depth=10,
            max_branching=5,
            exploration_weight=2.0,
            best_plan=["a", "b", "c"],
            search_active=True,
        ),
    )

    serialized = WorldSerializer.to_dict(world)
    restored = WorldSerializer.from_dict(
        serialized, providers={}, tool_handlers={}
    )

    restored_comp = restored.get_component(entity, PlanSearchComponent)
    assert restored_comp is not None
    assert restored_comp.max_depth == 10
    assert restored_comp.max_branching == 5
    assert restored_comp.exploration_weight == 2.0
    assert restored_comp.best_plan == ["a", "b", "c"]
    assert restored_comp.search_active is True


def test_serialization_roundtrip_with_rag_trigger() -> None:
    """Test that RAGTriggerComponent roundtrips correctly."""
    world = World()
    entity = world.create_entity()
    world.add_component(
        entity,
        RAGTriggerComponent(
            query="search query",
            top_k=10,
            retrieved_docs=["doc1", "doc2"],
        ),
    )

    serialized = WorldSerializer.to_dict(world)
    restored = WorldSerializer.from_dict(
        serialized, providers={}, tool_handlers={}
    )

    restored_comp = restored.get_component(entity, RAGTriggerComponent)
    assert restored_comp is not None
    assert restored_comp.query == "search query"
    assert restored_comp.top_k == 10
    assert restored_comp.retrieved_docs == ["doc1", "doc2"]


def test_serialization_embedding_component_uses_placeholder() -> None:
    """Test that EmbeddingComponent.provider is serialized as placeholder."""
    from unittest.mock import Mock

    provider = Mock()
    world = World()
    entity = world.create_entity()
    world.add_component(
        entity,
        EmbeddingComponent(provider=provider, dimension=768),
    )

    data = WorldSerializer.to_dict(world)
    assert (
        data["entities"]["1"]["EmbeddingComponent"]["provider"]
        == NON_SERIALIZABLE_PLACEHOLDER
    )
    assert data["entities"]["1"]["EmbeddingComponent"]["dimension"] == 768


def test_serialization_vector_store_component_uses_placeholder() -> None:
    """Test that VectorStoreComponent.store is serialized as placeholder."""
    from unittest.mock import Mock

    store = Mock()
    world = World()
    entity = world.create_entity()
    world.add_component(
        entity,
        VectorStoreComponent(store=store),
    )

    data = WorldSerializer.to_dict(world)
    assert (
        data["entities"]["1"]["VectorStoreComponent"]["store"]
        == NON_SERIALIZABLE_PLACEHOLDER
    )


def test_serialization_roundtrip_mixed_new_components() -> None:
    """Test roundtrip with multiple new components together."""
    world = World()
    entity = world.create_entity()
    world.add_component(
        entity,
        ToolApprovalComponent(policy=ApprovalPolicy.ALWAYS_APPROVE),
    )
    world.add_component(
        entity,
        SandboxConfigComponent(timeout=30.0),
    )
    world.add_component(
        entity,
        PlanSearchComponent(max_depth=3),
    )
    world.add_component(
        entity,
        RAGTriggerComponent(query="test", top_k=5),
    )

    serialized = WorldSerializer.to_dict(world)
    component_names = set(serialized["entities"]["1"].keys())
    expected = {
        "ToolApprovalComponent",
        "SandboxConfigComponent",
        "PlanSearchComponent",
        "RAGTriggerComponent",
    }
    assert component_names == expected

    restored = WorldSerializer.from_dict(
        serialized, providers={}, tool_handlers={}
    )
    assert restored.has_component(entity, ToolApprovalComponent)
    assert restored.has_component(entity, SandboxConfigComponent)
    assert restored.has_component(entity, PlanSearchComponent)
    assert restored.has_component(entity, RAGTriggerComponent)


def test_serialization_roundtrip_streaming_component() -> None:
    """Test that StreamingComponent roundtrips correctly."""
    world = World()
    entity = world.create_entity()
    world.add_component(entity, StreamingComponent(enabled=True))

    serialized = WorldSerializer.to_dict(world)
    restored = WorldSerializer.from_dict(serialized, providers={}, tool_handlers={})

    restored_comp = restored.get_component(entity, StreamingComponent)
    assert restored_comp is not None
    assert restored_comp.enabled is True


def test_serialization_roundtrip_checkpoint_component() -> None:
    """Test that CheckpointComponent with snapshots roundtrips correctly."""
    world = World()
    entity = world.create_entity()
    snapshot1 = {"entities": {"1": {"ConversationComponent": {"messages": []}}}, "next_entity_id": 2}
    snapshot2 = {"entities": {"1": {"KVStoreComponent": {"store": {"k": "v"}}}}, "next_entity_id": 3}
    world.add_component(
        entity,
        CheckpointComponent(snapshots=[snapshot1, snapshot2], max_snapshots=5),
    )

    serialized = WorldSerializer.to_dict(world)
    restored = WorldSerializer.from_dict(serialized, providers={}, tool_handlers={})

    restored_comp = restored.get_component(entity, CheckpointComponent)
    assert restored_comp is not None
    assert len(restored_comp.snapshots) == 2
    assert restored_comp.snapshots[0] == snapshot1
    assert restored_comp.snapshots[1] == snapshot2
    assert restored_comp.max_snapshots == 5


def test_serialization_roundtrip_compaction_config_component() -> None:
    """Test that CompactionConfigComponent roundtrips correctly."""
    world = World()
    entity = world.create_entity()
    world.add_component(
        entity,
        CompactionConfigComponent(threshold_tokens=5000, summary_model="gpt-4"),
    )

    serialized = WorldSerializer.to_dict(world)
    restored = WorldSerializer.from_dict(serialized, providers={}, tool_handlers={})

    restored_comp = restored.get_component(entity, CompactionConfigComponent)
    assert restored_comp is not None
    assert restored_comp.threshold_tokens == 5000
    assert restored_comp.summary_model == "gpt-4"


def test_serialization_roundtrip_conversation_archive_component() -> None:
    """Test that ConversationArchiveComponent roundtrips correctly."""
    world = World()
    entity = world.create_entity()
    world.add_component(
        entity,
        ConversationArchiveComponent(archived_summaries=["summary1", "summary2"]),
    )

    serialized = WorldSerializer.to_dict(world)
    restored = WorldSerializer.from_dict(serialized, providers={}, tool_handlers={})

    restored_comp = restored.get_component(entity, ConversationArchiveComponent)
    assert restored_comp is not None
    assert restored_comp.archived_summaries == ["summary1", "summary2"]


def test_serialization_roundtrip_runner_state_component() -> None:
    """Test that RunnerStateComponent roundtrips correctly."""
    world = World()
    entity = world.create_entity()
    world.add_component(
        entity,
        RunnerStateComponent(
            current_tick=5, is_paused=True, checkpoint_path="/tmp/checkpoint.json"
        ),
    )

    serialized = WorldSerializer.to_dict(world)
    restored = WorldSerializer.from_dict(serialized, providers={}, tool_handlers={})

    restored_comp = restored.get_component(entity, RunnerStateComponent)
    assert restored_comp is not None
    assert restored_comp.current_tick == 5
    assert restored_comp.is_paused is True
    assert restored_comp.checkpoint_path == "/tmp/checkpoint.json"


def test_serialization_backward_compatibility_without_new_components() -> None:
    """Test that world without new components deserializes successfully."""
    # Old serialized data without new components
    old_data = {
        "next_entity_id": 2,
        "entities": {
            "1": {
                "ConversationComponent": {
                    "messages": [{"role": "user", "content": "hi", "tool_calls": None, "tool_call_id": None}],
                    "max_messages": 100,
                }
            }
        },
    }

    # Should not raise KeyError or other errors
    restored = WorldSerializer.from_dict(old_data, providers={}, tool_handlers={})
    assert restored is not None
    conv = restored.get_component(EntityId(1), ConversationComponent)
    assert conv is not None
    assert conv.messages[0].content == "hi"


def test_serialization_full_world_with_all_new_components() -> None:
    """Test full world round-trip with all 5 new components together."""
    world = World()
    entity = world.create_entity()

    # Add all 5 new components
    world.add_component(entity, StreamingComponent(enabled=True))
    world.add_component(
        entity,
        CheckpointComponent(
            snapshots=[{"test": "data"}], max_snapshots=15
        ),
    )
    world.add_component(
        entity,
        CompactionConfigComponent(threshold_tokens=3000, summary_model="gpt-3.5"),
    )
    world.add_component(
        entity,
        ConversationArchiveComponent(archived_summaries=["archive1"]),
    )
    world.add_component(
        entity, RunnerStateComponent(current_tick=10, is_paused=False)
    )

    serialized = WorldSerializer.to_dict(world)
    component_names = set(serialized["entities"]["1"].keys())
    expected = {
        "StreamingComponent",
        "CheckpointComponent",
        "CompactionConfigComponent",
        "ConversationArchiveComponent",
        "RunnerStateComponent",
    }
    assert component_names == expected

    restored = WorldSerializer.from_dict(serialized, providers={}, tool_handlers={})

    # Verify all components survived round-trip
    assert restored.has_component(entity, StreamingComponent)
    assert restored.has_component(entity, CheckpointComponent)
    assert restored.has_component(entity, CompactionConfigComponent)
    assert restored.has_component(entity, ConversationArchiveComponent)
    assert restored.has_component(entity, RunnerStateComponent)

    # Verify field values
    streaming = restored.get_component(entity, StreamingComponent)
    assert streaming is not None
    assert streaming.enabled is True

    checkpoint = restored.get_component(entity, CheckpointComponent)
    assert checkpoint is not None
    assert checkpoint.snapshots == [{"test": "data"}]
    assert checkpoint.max_snapshots == 15

    compaction = restored.get_component(entity, CompactionConfigComponent)
    assert compaction is not None
    assert compaction.threshold_tokens == 3000
    assert compaction.summary_model == "gpt-3.5"

    archive = restored.get_component(entity, ConversationArchiveComponent)
    assert archive is not None
    assert archive.archived_summaries == ["archive1"]

    runner_state = restored.get_component(entity, RunnerStateComponent)
    assert runner_state is not None
    assert runner_state.current_tick == 10
    assert runner_state.is_paused is False
    assert runner_state.checkpoint_path is None