from __future__ import annotations

from typing import Any

from ecs_agent.components import (
    CheckpointComponent,
    CompactionConfigComponent,
    ContextTrimConfig,
    ContextCacheComponent,
    ConversationArchiveComponent,
    ConversationComponent,
    EmbeddingComponent,
    ErrorComponent,
    SubagentRegistryComponent,
    KVStoreComponent,
    LLMComponent,
    MessageBusConfigComponent,
    MessageBusConversationComponent,
    MessageBusSubscriptionComponent,
    OwnerComponent,
    PendingToolCallsComponent,
    PlanComponent,
    PlanSearchComponent,
    RAGTriggerComponent,
    ResponsesAPIStateComponent,
    RunnerStateComponent,
    SandboxConfigComponent,
    StreamingComponent,
    SubagentNotificationQueueComponent,
    SubagentSessionTableComponent,
    SubagentWaitComponent,
    SystemPromptComponent,
    TerminalComponent,
    ToolApprovalComponent,
    ToolRegistryComponent,
    ToolResultsComponent,
    VectorStoreComponent,
)
from ecs_agent.core.world import World
from ecs_agent.serialization import NON_SERIALIZABLE_PLACEHOLDER, WorldSerializer
from ecs_agent.types import (
    ApprovalPolicy,
    CachedToolResultRef,
    EntityId,
    FileRefPart,
    FreeSubagentConfig,
    InheritancePolicy,
    ImageUrlPart,
    Message,
    SubagentConfig,
    SubagentNotificationRecord,
    SubagentSessionRecord,
    ToolCall,
    ToolSchema,
)


class DummyProvider:
    model_id: str = "default"

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


def test_to_dict_message_with_multimodal_parts_is_deterministic() -> None:
    world = World()
    entity = world.create_entity()
    world.add_component(
        entity,
        ConversationComponent(
            messages=[
                Message(
                    role="user",
                    content="hello",
                    parts=[
                        ImageUrlPart(url="https://example.com/a.png", detail="low"),
                        FileRefPart(file_id="file_123", filename="doc.txt"),
                    ],
                )
            ]
        ),
    )

    data = WorldSerializer.to_dict(world)

    assert data["entities"]["1"]["ConversationComponent"]["messages"][0] == {
        "role": "user",
        "content": "hello",
        "parts": [
            {
                "type": "image_url",
                "url": "https://example.com/a.png",
                "detail": "low",
            },
            {"type": "file_ref", "file_id": "file_123", "filename": "doc.txt"},
        ],
        "tool_calls": None,
        "tool_call_id": None,
    }


def test_serialization_roundtrip_message_with_multimodal_parts() -> None:
    world = World()
    entity = world.create_entity()
    world.add_component(
        entity,
        ConversationComponent(
            messages=[
                Message(
                    role="user",
                    content="what is in this file?",
                    parts=[
                        FileRefPart(file_id="file_abc", filename="report.pdf"),
                    ],
                )
            ]
        ),
    )

    serialized = WorldSerializer.to_dict(world)
    restored = WorldSerializer.from_dict(serialized, providers={}, tool_handlers={})
    restored_conv = restored.get_component(entity, ConversationComponent)

    assert restored_conv is not None
    restored_message = restored_conv.messages[0]
    assert restored_message.content == "what is in this file?"
    assert restored_message.parts is not None
    assert len(restored_message.parts) == 1
    assert isinstance(restored_message.parts[0], FileRefPart)
    assert restored_message.parts[0].file_id == "file_abc"
    assert restored_message.parts[0].filename == "report.pdf"


def test_serialization_roundtrip_preserves_reasoning_replay_metadata() -> None:
    world = World()
    entity = world.create_entity()
    world.add_component(
        entity,
        ConversationComponent(
            messages=[
                Message(
                    role="assistant",
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="call_123",
                            name="get_weather",
                            arguments={"city": "San Francisco"},
                        )
                    ],
                    reasoning_content="",
                    reasoning_signature="sig_123",
                )
            ]
        ),
    )

    serialized = WorldSerializer.to_dict(world)
    restored = WorldSerializer.from_dict(serialized, providers={}, tool_handlers={})
    restored_conv = restored.get_component(entity, ConversationComponent)

    assert restored_conv is not None
    restored_message = restored_conv.messages[0]
    assert restored_message.reasoning_content == ""
    assert restored_message.reasoning_signature == "sig_123"
    assert restored_message.tool_calls == [
        ToolCall(
            id="call_123",
            name="get_weather",
            arguments={"city": "San Francisco"},
        )
    ]


def test_to_dict_skips_non_serializable_fields() -> None:
    model = DummyProvider()
    world = World()
    entity = world.create_entity()
    world.add_component(
        entity, LLMComponent(model=model, system_prompt="sys")
    )
    world.add_component(
        entity,
        ToolRegistryComponent(
            tools={"ping": ToolSchema(name="ping", description="Ping", parameters={})},
            handlers={"ping": async_tool_handler},
        ),
    )

    data = WorldSerializer.to_dict(world)

    # LLMComponent.model is serialized as model_id string (not a placeholder)
    assert data["entities"]["1"]["LLMComponent"]["model"] == "default"
    assert (
        data["entities"]["1"]["ToolRegistryComponent"]["handlers"]
        == NON_SERIALIZABLE_PLACEHOLDER
    )


def test_from_dict_reconstructs_world_correctly() -> None:
    model = DummyProvider()
    providers = {"default": model, "gpt-4": model}
    handlers = {"ping": async_tool_handler}
    data = {
        "next_entity_id": 5,
        "entities": {
            "1": {
                "LLMComponent": {
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
    assert llm.model is model
    assert conv is not None
    assert isinstance(conv.messages[0], Message)
    assert conv.messages[0].content == "hi"
    assert tool_registry is not None
    assert isinstance(tool_registry.tools["ping"], ToolSchema)
    assert tool_registry.handlers is handlers
    assert world.create_entity() == EntityId(5)


def test_round_trip_preserves_state() -> None:
    model = DummyProvider()
    providers = {"default": model, "gpt-4": model}
    handlers = {"ping": async_tool_handler}

    world = World()
    entity = world.create_entity()
    world.add_component(
        entity, LLMComponent(model=model, system_prompt="sys")
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
    model = DummyProvider()
    providers = {"default": model, "gpt-4": model}
    handlers = {"ping": async_tool_handler}

    world = World()
    entity = world.create_entity()
    world.add_component(entity, LLMComponent(model=model))
    world.add_component(entity, KVStoreComponent(store={"a": 1}))

    path = tmp_path / "world.json"
    WorldSerializer.save(world, path)

    loaded = WorldSerializer.load(path, providers=providers, tool_handlers=handlers)
    loaded_llm = loaded.get_component(EntityId(1), LLMComponent)
    loaded_kv = loaded.get_component(EntityId(1), KVStoreComponent)

    assert path.exists()
    assert loaded_llm is not None
    assert loaded_llm.model is model
    assert loaded_kv == KVStoreComponent(store={"a": 1})


def test_serialization_with_all_component_types() -> None:
    model = DummyProvider()
    providers = {"default": model, "gpt-4": model}
    handlers = {"ping": async_tool_handler}

    world = World()
    entity = world.create_entity()
    world.add_component(
        entity, LLMComponent(model=model, system_prompt="sys")
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
        MessageBusConversationComponent(
            entity_id=entity,
            messages=[Message(role="assistant", content="x")],
            max_messages=100,
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
        "MessageBusConversationComponent",
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
    assert restored.has_component(EntityId(1), MessageBusConversationComponent)
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
    restored = WorldSerializer.from_dict(serialized, providers={}, tool_handlers={})

    restored_comp = restored.get_component(entity, ToolApprovalComponent)
    assert restored_comp is not None
    assert restored_comp.policy == ApprovalPolicy.REQUIRE_APPROVAL
    assert restored_comp.timeout == 45.0
    assert restored_comp.approved_calls == ["call1"]
    assert restored_comp.denied_calls == ["call2"]


def test_serialization_roundtrip_with_context_budget_config_component() -> None:
    world = World()
    entity = world.create_entity()
    world.add_component(
        entity,
        ContextTrimConfig(
            max_tokens=2048,
            trim_tool_results=False,
            trim_reasoning=True,
            token_estimation_chars_per_token=3.5,
            overflow_behavior="warn",
        ),
    )

    serialized = WorldSerializer.to_dict(world)
    restored = WorldSerializer.from_dict(serialized, providers={}, tool_handlers={})

    assert serialized["entities"]["1"]["ContextTrimConfig"] == {
        "max_tokens": 2048,
        "trim_tool_results": False,
        "trim_reasoning": True,
        "token_estimation_chars_per_token": 3.5,
        "overflow_behavior": "warn",
        "protect_recent_turns": 100,
    }

    restored_component = restored.get_component(entity, ContextTrimConfig)
    assert restored_component is not None
    assert restored_component.max_tokens == 2048
    assert restored_component.trim_tool_results is False
    assert restored_component.trim_reasoning is True
    assert restored_component.token_estimation_chars_per_token == 3.5
    assert restored_component.overflow_behavior == "warn"


def test_serialization_roundtrip_with_context_cache_component() -> None:
    world = World()
    entity = world.create_entity()
    world.add_component(
        entity,
        ContextCacheComponent(
            cached_tool_results=[
                CachedToolResultRef(
                    tool_call_id="tool-call-1",
                    artifact_path="scratchbook/records/tool/tool_123",
                    summary="cached tool result",
                )
            ]
        ),
    )

    serialized = WorldSerializer.to_dict(world)
    restored = WorldSerializer.from_dict(serialized, providers={}, tool_handlers={})

    assert serialized["entities"]["1"]["ContextCacheComponent"] == {
        "cached_tool_results": [
            {
                "tool_call_id": "tool-call-1",
                "artifact_path": "scratchbook/records/tool/tool_123",
                "summary": "cached tool result",
                "original_content": None,
            }
        ]
    }

    restored_component = restored.get_component(entity, ContextCacheComponent)
    assert restored_component is not None
    assert restored_component.cached_tool_results == [
        CachedToolResultRef(
            tool_call_id="tool-call-1",
            artifact_path="scratchbook/records/tool/tool_123",
            summary="cached tool result",
            original_content=None,
        )
    ]


def test_context_cache_component_round_trips_multiple_entries() -> None:
    world = World()
    entity = world.create_entity()
    world.add_component(
        entity,
        ContextCacheComponent(
            cached_tool_results=[
                CachedToolResultRef(
                    tool_call_id="tool-call-1",
                    artifact_path="scratchbook/records/tool/tool_123",
                    summary="cached tool result",
                ),
                CachedToolResultRef(
                    tool_call_id="tool-call-2",
                    artifact_path="scratchbook/records/tool/tool_456",
                    summary=None,
                ),
            ]
        ),
    )

    serialized = WorldSerializer.to_dict(world)
    restored = WorldSerializer.from_dict(serialized, providers={}, tool_handlers={})

    assert serialized["entities"]["1"]["ContextCacheComponent"] == {
        "cached_tool_results": [
            {
                "tool_call_id": "tool-call-1",
                "artifact_path": "scratchbook/records/tool/tool_123",
                "summary": "cached tool result",
                "original_content": None,
            },
            {
                "tool_call_id": "tool-call-2",
                "artifact_path": "scratchbook/records/tool/tool_456",
                "summary": None,
                "original_content": None,
            },
        ]
    }

    restored_component = restored.get_component(entity, ContextCacheComponent)
    assert restored_component is not None
    assert restored_component.cached_tool_results == [
        CachedToolResultRef(
            tool_call_id="tool-call-1",
            artifact_path="scratchbook/records/tool/tool_123",
            summary="cached tool result",
            original_content=None,
        ),
        CachedToolResultRef(
            tool_call_id="tool-call-2",
            artifact_path="scratchbook/records/tool/tool_456",
            summary=None,
            original_content=None,
        ),
    ]


def test_serialization_roundtrip_preserves_subagent_session_table_records() -> None:
    world = World()
    entity = world.create_entity()
    world.add_component(
        entity,
        SubagentSessionTableComponent(
            sessions={
                "queued-b": SubagentSessionRecord(
                    session_id="queued-b",
                    category="queued-agent",
                    prompt="Second queued task",
                    parent_entity_id=entity,
                    created_at="2026-04-05T10:01:00Z",
                    updated_at="2026-04-05T10:01:00Z",
                    status="queued",
                    load_skills=["skill-b"],
                    background=True,
                ),
                "running-a": SubagentSessionRecord(
                    session_id="running-a",
                    category="running-agent",
                    prompt="Running during checkpoint",
                    parent_entity_id=entity,
                    created_at="2026-04-05T10:00:00Z",
                    updated_at="2026-04-05T10:02:00Z",
                    status="running",
                    background=True,
                    started_at="2026-04-05T10:00:30Z",
                ),
                "done-c": SubagentSessionRecord(
                    session_id="done-c",
                    category="done-agent",
                    prompt="Completed before checkpoint",
                    parent_entity_id=entity,
                    created_at="2026-04-05T09:50:00Z",
                    updated_at="2026-04-05T09:55:00Z",
                    status="succeeded",
                    background=True,
                    result_excerpt="already done",
                    started_at="2026-04-05T09:50:10Z",
                    finished_at="2026-04-05T09:55:00Z",
                ),
            }
        ),
    )

    serialized = WorldSerializer.to_dict(world)
    restored = WorldSerializer.from_dict(serialized, providers={}, tool_handlers={})

    restored_table = restored.get_component(entity, SubagentSessionTableComponent)
    assert restored_table is not None
    assert list(restored_table.sessions) == ["queued-b", "running-a", "done-c"]
    assert restored_table.sessions["queued-b"].status == "queued"
    assert restored_table.sessions["queued-b"].load_skills == ["skill-b"]
    assert restored_table.sessions["running-a"].status == "running"
    assert restored_table.sessions["running-a"].started_at == "2026-04-05T10:00:30Z"
    assert restored_table.sessions["done-c"].status == "succeeded"
    assert restored_table.sessions["done-c"].result_excerpt == "already done"


def test_serialization_subagent_notification_queue_roundtrip() -> None:
    world = World()
    entity = world.create_entity()
    world.add_component(
        entity,
        SubagentNotificationQueueComponent(
            notifications=[
                SubagentNotificationRecord(
                    notification_id="session-1:succeeded",
                    session_id="session-1",
                    parent_entity_id=1,
                    terminal_status="succeeded",
                    summary="Background summary",
                    error=None,
                    created_at="2026-04-06T12:00:00Z",
                    delivered_at=None,
                ),
                SubagentNotificationRecord(
                    notification_id="session-2:failed",
                    session_id="session-2",
                    parent_entity_id=1,
                    terminal_status="failed",
                    summary=None,
                    error="child failed",
                    created_at="2026-04-06T12:05:00Z",
                    delivered_at="2026-04-06T12:06:00Z",
                ),
            ]
        ),
    )

    serialized = WorldSerializer.to_dict(world)
    restored = WorldSerializer.from_dict(serialized, providers={}, tool_handlers={})

    assert serialized["entities"]["1"]["SubagentNotificationQueueComponent"] == {
        "notifications": [
            {
                "notification_id": "session-1:succeeded",
                "session_id": "session-1",
                "parent_entity_id": 1,
                "terminal_status": "succeeded",
                "summary": "Background summary",
                "error": None,
                "created_at": "2026-04-06T12:00:00Z",
                "delivered_at": None,
            },
            {
                "notification_id": "session-2:failed",
                "session_id": "session-2",
                "parent_entity_id": 1,
                "terminal_status": "failed",
                "summary": None,
                "error": "child failed",
                "created_at": "2026-04-06T12:05:00Z",
                "delivered_at": "2026-04-06T12:06:00Z",
            },
        ]
    }

    restored_queue = restored.get_component(entity, SubagentNotificationQueueComponent)
    assert restored_queue is not None
    assert len(restored_queue.notifications) == 2
    assert restored_queue.notifications[0].summary == "Background summary"
    assert restored_queue.notifications[1].error == "child failed"


def test_serialization_subagent_wait_component_excludes_future_runtime_state() -> None:
    import asyncio

    world = World()
    entity = world.create_entity()
    loop = asyncio.new_event_loop()
    future: asyncio.Future[None] = loop.create_future()
    world.add_component(
        entity,
        SubagentWaitComponent(
            session_ids=["session-1", "session-2"],
            timeout=15.0,
            future=future,
            started_at="2026-04-06T12:00:00Z",
        ),
    )

    serialized = WorldSerializer.to_dict(world)
    restored = WorldSerializer.from_dict(serialized, providers={}, tool_handlers={})

    assert serialized["entities"]["1"]["SubagentWaitComponent"] == {
        "session_ids": ["session-1", "session-2"],
        "timeout": 15.0,
        "future": None,
        "started_at": "2026-04-06T12:00:00Z",
        "resolved_session_ids": None,
        "auto_restart_budget": 0,
        "restart_counts": {},
    }

    restored_wait = restored.get_component(entity, SubagentWaitComponent)
    assert restored_wait is not None
    assert restored_wait.session_ids == ["session-1", "session-2"]
    assert restored_wait.timeout == 15.0
    assert restored_wait.future is None
    assert restored_wait.started_at == "2026-04-06T12:00:00Z"

    loop.close()


def test_serialization_roundtrip_preserves_notification_delivery_state_and_wait_future_reset() -> (
    None
):
    world = World()
    entity = world.create_entity()
    world.add_component(
        entity,
        ConversationComponent(messages=[Message(role="user", content="resume")]),
    )
    world.add_component(
        entity,
        SubagentNotificationQueueComponent(
            notifications=[
                SubagentNotificationRecord(
                    notification_id="session-unread:succeeded",
                    session_id="session-unread",
                    parent_entity_id=entity,
                    terminal_status="succeeded",
                    summary="keep unread",
                    error=None,
                    created_at="2026-04-06T12:00:00Z",
                    delivered_at=None,
                ),
                SubagentNotificationRecord(
                    notification_id="session-read:failed",
                    session_id="session-read",
                    parent_entity_id=entity,
                    terminal_status="failed",
                    summary=None,
                    error="already read",
                    created_at="2026-04-06T12:01:00Z",
                    delivered_at="2026-04-06T12:02:00Z",
                ),
            ]
        ),
    )
    world.add_component(
        entity,
        SubagentWaitComponent(
            session_ids=["session-unread"],
            timeout=30.0,
            started_at="2026-04-06T12:00:00Z",
        ),
    )

    serialized = WorldSerializer.to_dict(world)
    restored = WorldSerializer.from_dict(serialized, providers={}, tool_handlers={})

    restored_queue = restored.get_component(entity, SubagentNotificationQueueComponent)
    assert restored_queue is not None
    assert restored_queue.notifications[0].delivered_at is None
    assert restored_queue.notifications[1].delivered_at == "2026-04-06T12:02:00Z"

    restored_wait = restored.get_component(entity, SubagentWaitComponent)
    assert restored_wait is not None
    assert restored_wait.session_ids == ["session-unread"]
    assert restored_wait.future is None


def test_serialization_roundtrip_with_sandbox_config() -> None:
    """Test that SandboxConfigComponent roundtrips correctly."""
    world = World()
    entity = world.create_entity()
    world.add_component(
        entity,
        SandboxConfigComponent(timeout=60.0, max_output_size=50_000),
    )

    serialized = WorldSerializer.to_dict(world)
    restored = WorldSerializer.from_dict(serialized, providers={}, tool_handlers={})

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
    restored = WorldSerializer.from_dict(serialized, providers={}, tool_handlers={})

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
    restored = WorldSerializer.from_dict(serialized, providers={}, tool_handlers={})

    restored_comp = restored.get_component(entity, RAGTriggerComponent)
    assert restored_comp is not None
    assert restored_comp.query == "search query"
    assert restored_comp.top_k == 10
    assert restored_comp.retrieved_docs == ["doc1", "doc2"]


def test_serialization_roundtrip_with_responses_api_state_component() -> None:
    world = World()
    entity = world.create_entity()
    world.add_component(
        entity,
        ResponsesAPIStateComponent(previous_response_id="resp_state_001"),
    )

    serialized = WorldSerializer.to_dict(world)
    restored = WorldSerializer.from_dict(serialized, providers={}, tool_handlers={})

    restored_comp = restored.get_component(entity, ResponsesAPIStateComponent)
    assert restored_comp is not None
    assert restored_comp.previous_response_id == "resp_state_001"


def test_serialization_embedding_component_uses_placeholder() -> None:
    """Test that EmbeddingComponent.provider is serialized as placeholder."""
    from unittest.mock import Mock

    model = Mock()
    world = World()
    entity = world.create_entity()
    world.add_component(
        entity,
        EmbeddingComponent(provider=model, dimension=768),
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

    restored = WorldSerializer.from_dict(serialized, providers={}, tool_handlers={})
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
    snapshot1 = {
        "entities": {"1": {"ConversationComponent": {"messages": []}}},
        "next_entity_id": 2,
    }
    snapshot2 = {
        "entities": {"1": {"KVStoreComponent": {"store": {"k": "v"}}}},
        "next_entity_id": 3,
    }
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
                    "messages": [
                        {
                            "role": "user",
                            "content": "hi",
                            "tool_calls": None,
                            "tool_call_id": None,
                        }
                    ],
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
        CheckpointComponent(snapshots=[{"test": "data"}], max_snapshots=15),
    )
    world.add_component(
        entity,
        CompactionConfigComponent(threshold_tokens=3000, summary_model="gpt-3.5"),
    )
    world.add_component(
        entity,
        ConversationArchiveComponent(archived_summaries=["archive1"]),
    )
    world.add_component(entity, RunnerStateComponent(current_tick=10, is_paused=False))

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


def test_new_components_in_registry() -> None:
    """Verify ResponsesAPIStateComponent, ConversationTreeComponent, SubagentRegistryComponent, and MessageBus components are in COMPONENT_REGISTRY."""
    from ecs_agent.serialization import COMPONENT_REGISTRY

    # Check all six new components are registered
    assert "ResponsesAPIStateComponent" in COMPONENT_REGISTRY, (
        "ResponsesAPIStateComponent missing from registry"
    )
    assert "ConversationTreeComponent" in COMPONENT_REGISTRY, (
        "ConversationTreeComponent missing from registry"
    )
    assert "SubagentRegistryComponent" in COMPONENT_REGISTRY, (
        "SubagentRegistryComponent missing from registry"
    )
    assert "MessageBusConfigComponent" in COMPONENT_REGISTRY, (
        "MessageBusConfigComponent missing from registry"
    )
    assert "MessageBusSubscriptionComponent" in COMPONENT_REGISTRY, (
        "MessageBusSubscriptionComponent missing from registry"
    )
    assert "MessageBusConversationComponent" in COMPONENT_REGISTRY, (
        "MessageBusConversationComponent missing from registry"
    )

    assert len(COMPONENT_REGISTRY) >= 28, (
        f"Registry has {len(COMPONENT_REGISTRY)} components, expected at least 28"
    )


def test_subagent_registry_free_config_roundtrip() -> None:
    model = DummyProvider()
    world = World()
    entity = world.create_entity()
    world.add_component(
        entity,
        SubagentRegistryComponent(
            subagents={
                "registered": SubagentConfig(name="registered", model=model),
            },
            free_subagent_config=FreeSubagentConfig(
                enabled=True,
                system_prompt_template='Worker {name}; JSON example {"ok": true}.',
                skills=["read-file"],
                max_ticks=7,
                inheritance_policy=InheritancePolicy(
                    inherit_system_prompt=False,
                    inherit_tools=["read_file"],
                    inherit_permissions=True,
                ),
            ),
        ),
    )

    serialized = WorldSerializer.to_dict(world)
    restored = WorldSerializer.from_dict(
        serialized,
        providers={"default": model},
        tool_handlers={},
    )

    restored_registry = restored.get_component(entity, SubagentRegistryComponent)
    assert restored_registry is not None
    assert restored_registry.free_subagent_config.enabled is True
    assert restored_registry.free_subagent_config.system_prompt_template == (
        'Worker {name}; JSON example {"ok": true}.'
    )
    assert restored_registry.free_subagent_config.skills == ["read-file"]
    assert restored_registry.free_subagent_config.max_ticks == 7
    assert restored_registry.free_subagent_config.inheritance_policy.inherit_tools == [
        "read_file"
    ]
    assert restored_registry.free_subagent_config.inheritance_policy.inherit_permissions is True


def test_subagent_registry_free_config_defaults_when_legacy_checkpoint_missing_field() -> None:
    model = DummyProvider()
    entity_id = "1"
    data = {
        "next_entity_id": 2,
        "entities": {
            entity_id: {
                "SubagentRegistryComponent": {
                    "subagents": {},
                }
            }
        },
    }

    restored = WorldSerializer.from_dict(data, providers={"default": model}, tool_handlers={})

    restored_registry = restored.get_component(EntityId(1), SubagentRegistryComponent)
    assert restored_registry is not None
    assert restored_registry.free_subagent_config.enabled is False


def test_message_bus_config_roundtrip() -> None:
    """Test that MessageBusConfigComponent roundtrips correctly."""
    world = World()
    entity = world.create_entity()
    world.add_component(
        entity,
        MessageBusConfigComponent(
            max_queue_size=500,
            publish_timeout=1.5,
            request_timeout=20.0,
            cleanup_interval=45.0,
            max_pending_requests=5000,
        ),
    )

    serialized = WorldSerializer.to_dict(world)
    restored = WorldSerializer.from_dict(serialized, providers={}, tool_handlers={})

    restored_comp = restored.get_component(entity, MessageBusConfigComponent)
    assert restored_comp is not None
    assert restored_comp.max_queue_size == 500
    assert restored_comp.publish_timeout == 1.5
    assert restored_comp.request_timeout == 20.0
    assert restored_comp.cleanup_interval == 45.0
    assert restored_comp.max_pending_requests == 5000


def test_message_bus_subscription_roundtrip() -> None:
    """Test that MessageBusSubscriptionComponent roundtrips correctly with set-to-list conversion."""
    world = World()
    entity = world.create_entity()
    world.add_component(
        entity,
        MessageBusSubscriptionComponent(
            subscriptions={
                "topic1": {"sub1", "sub2", "sub3"},
                "topic2": {"sub4"},
            }
        ),
    )

    serialized = WorldSerializer.to_dict(world)

    # Verify serialized format uses lists (JSON-compatible)
    assert isinstance(
        serialized["entities"]["1"]["MessageBusSubscriptionComponent"]["subscriptions"][
            "topic1"
        ],
        list,
    )

    restored = WorldSerializer.from_dict(serialized, providers={}, tool_handlers={})

    restored_comp = restored.get_component(entity, MessageBusSubscriptionComponent)
    assert restored_comp is not None
    assert restored_comp.subscriptions["topic1"] == {"sub1", "sub2", "sub3"}
    assert restored_comp.subscriptions["topic2"] == {"sub4"}


def test_message_bus_conversation_roundtrip() -> None:
    """Test that MessageBusConversationComponent roundtrips correctly with EntityId and Message conversion."""
    world = World()
    entity = world.create_entity()
    world.add_component(
        entity,
        MessageBusConversationComponent(
            entity_id=EntityId(42),
            messages=[
                Message(role="user", content="Hello"),
                Message(role="assistant", content="Hi there"),
            ],
            max_messages=100,
        ),
    )

    serialized = WorldSerializer.to_dict(world)

    # Verify EntityId is serialized as int
    assert (
        serialized["entities"]["1"]["MessageBusConversationComponent"]["entity_id"]
        == 42
    )

    restored = WorldSerializer.from_dict(serialized, providers={}, tool_handlers={})

    restored_comp = restored.get_component(entity, MessageBusConversationComponent)
    assert restored_comp is not None
    assert restored_comp.entity_id == EntityId(42)
    assert len(restored_comp.messages) == 2
    assert isinstance(restored_comp.messages[0], Message)
    assert restored_comp.messages[0].content == "Hello"
    assert restored_comp.messages[1].content == "Hi there"
    assert restored_comp.max_messages == 100


def test_message_bus_no_runtime_state_serialized() -> None:
    """Test serialization boundary: runtime state (queues, futures, pending requests) not serialized."""
    world = World()
    entity = world.create_entity()

    # Add all three MessageBus components
    world.add_component(
        entity,
        MessageBusConfigComponent(max_queue_size=1000),
    )
    world.add_component(
        entity,
        MessageBusSubscriptionComponent(subscriptions={"topic": {"sub1"}}),
    )
    world.add_component(
        entity,
        MessageBusConversationComponent(
            entity_id=EntityId(1),
            messages=[Message(role="user", content="test")],
        ),
    )

    serialized = WorldSerializer.to_dict(world)

    # Verify ONLY config, subscriptions, and conversation are serialized
    entity_data = serialized["entities"]["1"]
    component_names = set(entity_data.keys())

    # These SHOULD be present
    assert "MessageBusConfigComponent" in component_names
    assert "MessageBusSubscriptionComponent" in component_names
    assert "MessageBusConversationComponent" in component_names

    # Runtime state should NOT be present (no queue fields, no futures, no pending_requests)
    config_data = entity_data["MessageBusConfigComponent"]
    assert "queue" not in config_data
    assert "futures" not in config_data
    assert "_pending_requests" not in config_data


def test_message_bus_mixed_roundtrip() -> None:
    """Test full world round-trip with all MessageBus components together."""
    world = World()
    entity = world.create_entity()

    world.add_component(
        entity,
        MessageBusConfigComponent(
            max_queue_size=800,
            publish_timeout=2.5,
        ),
    )
    world.add_component(
        entity,
        MessageBusSubscriptionComponent(
            subscriptions={
                "orders": {"order-processor", "analytics"},
                "notifications": {"email-service"},
            }
        ),
    )
    world.add_component(
        entity,
        MessageBusConversationComponent(
            entity_id=EntityId(99),
            messages=[
                Message(role="user", content="First"),
                Message(role="assistant", content="Second"),
            ],
            max_messages=50,
        ),
    )

    serialized = WorldSerializer.to_dict(world)
    component_names = set(serialized["entities"]["1"].keys())
    expected = {
        "MessageBusConfigComponent",
        "MessageBusSubscriptionComponent",
        "MessageBusConversationComponent",
    }
    assert component_names == expected

    restored = WorldSerializer.from_dict(serialized, providers={}, tool_handlers={})

    # Verify all components survived round-trip
    assert restored.has_component(entity, MessageBusConfigComponent)
    assert restored.has_component(entity, MessageBusSubscriptionComponent)
    assert restored.has_component(entity, MessageBusConversationComponent)

    # Verify field values
    config = restored.get_component(entity, MessageBusConfigComponent)
    assert config is not None
    assert config.max_queue_size == 800
    assert config.publish_timeout == 2.5

    subscription = restored.get_component(entity, MessageBusSubscriptionComponent)
    assert subscription is not None
    assert subscription.subscriptions["orders"] == {"order-processor", "analytics"}
    assert subscription.subscriptions["notifications"] == {"email-service"}

    conversation = restored.get_component(entity, MessageBusConversationComponent)
    assert conversation is not None
    assert conversation.entity_id == EntityId(99)
    assert len(conversation.messages) == 2
    assert conversation.max_messages == 50


def test_serialization_roundtrip_entity_registry() -> None:
    """Test that entity registry (_entity_registry, _entity_tags) roundtrips correctly."""
    world = World()
    entity1 = world.create_entity()
    entity2 = world.create_entity()

    # Register entities with names and tags
    world.register_entity(entity1, "agent-main", {"agent", "primary"})
    world.register_entity(entity2, "agent-helper", {"agent", "secondary"})

    # Add some component to make entities visible
    world.add_component(entity1, ConversationComponent(messages=[]))
    world.add_component(entity2, ConversationComponent(messages=[]))

    serialized = WorldSerializer.to_dict(world)
    restored = WorldSerializer.from_dict(serialized, providers={}, tool_handlers={})

    # Verify names are preserved
    assert restored.resolve_entity("agent-main") == entity1
    assert restored.resolve_entity("agent-helper") == entity2

    # Verify tags are preserved
    agent_entities = set(restored.list_entities_by_tag("agent"))
    assert agent_entities == {entity1, entity2}
    assert restored.list_entities_by_tag("primary") == [entity1]
    assert restored.list_entities_by_tag("secondary") == [entity2]


def test_serialization_backward_compatibility_no_registry_fields() -> None:
    """Test that snapshots without registry fields load successfully with safe defaults."""
    # Old serialized data without _entity_registry or _entity_tags
    old_data = {
        "next_entity_id": 2,
        "entities": {
            "1": {
                "ConversationComponent": {
                    "messages": [
                        {
                            "role": "user",
                            "content": "test",
                            "tool_calls": None,
                            "tool_call_id": None,
                        }
                    ],
                    "max_messages": 100,
                }
            }
        },
        # Note: _entity_registry and _entity_tags are missing
    }

    # Should not raise KeyError or other errors
    restored = WorldSerializer.from_dict(old_data, providers={}, tool_handlers={})
    assert restored is not None

    # Verify registry defaults to empty
    assert restored.resolve_entity("any-name") is None
    assert restored.list_entities_by_tag("any-tag") == []

    # Verify existing components still work
    conv = restored.get_component(EntityId(1), ConversationComponent)
    assert conv is not None
    assert conv.messages[0].content == "test"


def test_checkpoint_preserves_entity_registry_through_undo() -> None:
    """Test that entity registry survives checkpoint save and restore."""
    # This test verifies CheckpointSystem automatically handles registry
    # through WorldSerializer.to_dict/from_dict
    world = World()
    entity = world.create_entity()

    # Register with name and tags
    world.register_entity(entity, "agent-1", {"test", "main"})

    # Add checkpoint component and create snapshot
    world.add_component(entity, CheckpointComponent())
    world.add_component(entity, ConversationComponent(messages=[]))

    # Take snapshot (will use WorldSerializer.to_dict)
    import asyncio
    from ecs_agent.systems.checkpoint import CheckpointSystem

    asyncio.run(CheckpointSystem().process(world))

    # Modify registry state
    entity2 = world.create_entity()
    world.register_entity(entity2, "agent-2", {"test"})

    # Restore (will use WorldSerializer.from_dict)
    asyncio.run(CheckpointSystem.undo(world, providers={}, tool_handlers={}))

    # Verify registry state was restored
    assert world.resolve_entity("agent-1") == entity
    assert world.resolve_entity("agent-2") is None  # Should not exist after undo
    assert set(world.list_entities_by_tag("test")) == {entity}
    assert world.list_entities_by_tag("main") == [entity]


def test_subagent_session_table_component_roundtrip() -> None:
    """Test that SubagentSessionTableComponent with SubagentSessionRecord roundtrips correctly."""
    from ecs_agent.components import SubagentSessionTableComponent
    from ecs_agent.types import SubagentSessionRecord, EntityId

    world = World()
    entity = world.create_entity()

    # Create session records with all fields
    session1 = SubagentSessionRecord(
        session_id="sess_001",
        category="quick",
        prompt="Do something",
        load_skills=["skill1"],
        background=True,
        status="Working",
        correlation_id="corr_123",
        traceparent="00-trace-span-00",
        launch_trace_id="trace-launch-123",
        launch_run_id="run-launch-123",
        launch_parent_observation_id="obs-launch-123",
        parent_entity_id=EntityId(42),
        created_at="2026-03-10T10:00:00Z",
        updated_at="2026-03-10T10:05:00Z",
        timeout_seconds=60.0,
        deadline_at="2026-03-10T11:00:00Z",
        result_excerpt="Success",
        error=None,
        started_at="2026-03-10T10:00:30Z",
        finished_at="2026-03-10T10:04:59Z",
    )
    session2 = SubagentSessionRecord(
        session_id="sess_002",
        category="artistry",
        prompt="Create design",
        load_skills=[],
        background=False,
        status="Dead",
        correlation_id="corr_456",
        traceparent="00-trace-span-01",
        parent_entity_id=EntityId(43),
        created_at="2026-03-10T09:00:00Z",
        updated_at="2026-03-10T09:30:00Z",
        timeout_seconds=None,
        deadline_at=None,
        result_excerpt=None,
        error="Timeout exceeded",
        started_at="2026-03-10T09:00:10Z",
        finished_at="2026-03-10T09:30:00Z",
    )
    session3 = SubagentSessionRecord(
        session_id="sess_003",
        category="quick",
        prompt="Wait in queue",
        load_skills=["skill2"],
        background=True,
        status="queued",
        correlation_id="corr_789",
        traceparent="00-trace-span-02",
        parent_entity_id=EntityId(44),
        created_at="2026-03-10T08:00:00Z",
        updated_at="2026-03-10T08:00:00Z",
        timeout_seconds=120.0,
        deadline_at="2026-03-10T08:02:00Z",
        result_excerpt=None,
        error=None,
    )

    world.add_component(
        entity,
        SubagentSessionTableComponent(
            sessions={
                "sess_001": session1,
                "sess_002": session2,
                "sess_003": session3,
            }
        ),
    )

    serialized = WorldSerializer.to_dict(world)
    restored = WorldSerializer.from_dict(serialized, providers={}, tool_handlers={})

    restored_comp = restored.get_component(entity, SubagentSessionTableComponent)
    assert restored_comp is not None
    assert len(restored_comp.sessions) == 3

    # Verify session1
    restored_session1 = restored_comp.sessions["sess_001"]
    assert restored_session1.session_id == "sess_001"
    assert restored_session1.category == "quick"
    assert restored_session1.prompt == "Do something"
    assert restored_session1.load_skills == ["skill1"]
    assert restored_session1.background is True
    assert restored_session1.timeout_seconds == 60.0
    assert restored_session1.status == "running"
    assert restored_session1.correlation_id == "corr_123"
    assert restored_session1.traceparent == "00-trace-span-00"
    assert restored_session1.launch_trace_id == "trace-launch-123"
    assert restored_session1.launch_run_id == "run-launch-123"
    assert restored_session1.launch_parent_observation_id == "obs-launch-123"
    assert restored_session1.parent_entity_id == EntityId(42)
    assert restored_session1.created_at == "2026-03-10T10:00:00Z"
    assert restored_session1.updated_at == "2026-03-10T10:05:00Z"
    assert restored_session1.deadline_at == "2026-03-10T11:00:00Z"
    assert restored_session1.result_excerpt == "Success"
    assert restored_session1.error is None
    assert restored_session1.started_at == "2026-03-10T10:00:30Z"
    assert restored_session1.finished_at == "2026-03-10T10:04:59Z"

    # Verify session2
    restored_session2 = restored_comp.sessions["sess_002"]
    assert restored_session2.session_id == "sess_002"
    assert restored_session2.status == "failed"
    assert restored_session2.parent_entity_id == EntityId(43)
    assert restored_session2.error == "Timeout exceeded"
    assert restored_session2.result_excerpt is None
    assert restored_session2.started_at == "2026-03-10T09:00:10Z"
    assert restored_session2.finished_at == "2026-03-10T09:30:00Z"

    restored_session3 = restored_comp.sessions["sess_003"]
    assert restored_session3.session_id == "sess_003"
    assert restored_session3.status == "queued"
    assert restored_session3.parent_entity_id == EntityId(44)
    assert restored_session3.started_at is None
    assert restored_session3.finished_at is None


def test_subagent_session_table_ignores_unknown_legacy_session_fields() -> None:
    """Old checkpoints with removed session keys still restore."""
    data = {
        "next_entity_id": 2,
        "entities": {
            "1": {
                "SubagentSessionTableComponent": {
                    "sessions": {
                        "sess_legacy": {
                            "session_id": "sess_legacy",
                            "category": "quick",
                            "prompt": "restore me",
                            "parent_entity_id": 1,
                            "created_at": "2026-03-10T10:00:00Z",
                            "updated_at": "2026-03-10T10:00:00Z",
                            "traceparent": "00-trace-span-00",
                            "legacy_task_handle": "not serializable anymore",
                        }
                    }
                }
            }
        },
    }

    restored = WorldSerializer.from_dict(data, providers={}, tool_handlers={})

    restored_comp = restored.get_component(
        EntityId(1),
        SubagentSessionTableComponent,
    )
    assert restored_comp is not None
    restored_session = restored_comp.sessions["sess_legacy"]
    assert restored_session.session_id == "sess_legacy"
    assert restored_session.parent_entity_id == EntityId(1)
    assert restored_session.traceparent == "00-trace-span-00"


def test_subagent_session_table_rejects_runtime_handles() -> None:
    """Test that SubagentSessionRecord cannot store asyncio.Task or Future handles."""
    from ecs_agent.types import SubagentSessionRecord, EntityId

    # SubagentSessionRecord should only have serializable fields
    # This test verifies that the dataclass definition does NOT allow runtime handles

    # Create a valid session record
    session = SubagentSessionRecord(
        session_id="sess_003",
        category="quick",
        prompt="Test",
        parent_entity_id=EntityId(1),
        created_at="2026-03-10T10:00:00Z",
        updated_at="2026-03-10T10:00:00Z",
    )

    # Verify that the session record has no fields for Task or Future
    # (this is a structural test - the field should not exist in the dataclass definition)
    assert not hasattr(session, "task_handle")
    assert not hasattr(session, "future_handle")
    assert not hasattr(session, "_task")
    assert not hasattr(session, "_future")

    # Verify serialization does not fail (would fail if non-serializable objects were present)
    from ecs_agent.serialization import WorldSerializer
    from ecs_agent.components import SubagentSessionTableComponent

    world = World()
    entity = world.create_entity()
    world.add_component(
        entity,
        SubagentSessionTableComponent(sessions={"sess_003": session}),
    )

    # This should succeed without errors
    serialized = WorldSerializer.to_dict(world)
    assert "SubagentSessionTableComponent" in serialized["entities"]["1"]

    # Verify roundtrip works
    restored = WorldSerializer.from_dict(serialized, providers={}, tool_handlers={})
    restored_comp = restored.get_component(entity, SubagentSessionTableComponent)
    assert restored_comp is not None
    assert "sess_003" in restored_comp.sessions


def test_serialization_preserves_world_name() -> None:
    """Test that WorldSerializer persists and restores the world name."""
    world = World(name="my-world")
    data = WorldSerializer.to_dict(world)
    assert data["world_name"] == "my-world"
    restored = WorldSerializer.from_dict(data, providers={}, tool_handlers={})
    assert restored.name == "my-world"


def test_serialization_preserves_none_world_name() -> None:
    """Test that WorldSerializer correctly handles a World with no name."""
    world = World()
    data = WorldSerializer.to_dict(world)
    assert data["world_name"] is None
    restored = WorldSerializer.from_dict(data, providers={}, tool_handlers={})
    assert restored.name is None


def test_serialization_backward_compat_missing_world_name() -> None:
    """Old serialized data without world_name key deserializes to name=None."""
    data = {
        "next_entity_id": 1,
        "entities": {},
        "_entity_registry": {},
        "_entity_tags": {},
    }
    restored = WorldSerializer.from_dict(data, providers={}, tool_handlers={})
    assert restored.name is None


def test_serialization_roundtrip_todo_list_component() -> None:
    from ecs_agent.components import TodoItem, TodoListComponent

    world = World()
    entity = world.create_entity()
    world.add_component(
        entity,
        TodoListComponent(
            items=[
                TodoItem(content="Split utils into io/format modules", status="completed"),
                TodoItem(content="Add unit tests for both new modules", status="in_progress"),
                TodoItem(content="Update docs and README"),
            ]
        ),
    )

    serialized = WorldSerializer.to_dict(world)
    restored = WorldSerializer.from_dict(serialized, providers={}, tool_handlers={})

    component = restored.get_component(entity, TodoListComponent)
    assert component is not None
    assert all(isinstance(item, TodoItem) for item in component.items)
    assert [(item.content, item.status) for item in component.items] == [
        ("Split utils into io/format modules", "completed"),
        ("Add unit tests for both new modules", "in_progress"),
        ("Update docs and README", "pending"),
    ]
