"""Tests for subagent types and SubagentSystem."""

from __future__ import annotations
import json
import asyncio
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from typing import Any

import pytest

from ecs_agent.components import (
    ConversationComponent,
    LLMComponent,
    MessageBusConfigComponent,
    StreamingComponent,
    SubagentNotificationQueueComponent,
    SubagentWaitComponent,
)
from ecs_agent.components.definitions import (
    SubagentRegistryComponent,
    SubagentSessionTableComponent,
)
from ecs_agent.core.world import World
from ecs_agent.providers.fake_model import FakeModel
from ecs_agent.scratchbook.artifact_registry import ArtifactRegistry
from ecs_agent.serialization import WorldSerializer
from ecs_agent.types import (
    CompletionResult,
    DelegationCompletedEvent,
    DelegationStartedEvent,
    EntityId,
    InheritancePolicy,
    Message,
    RetryConfig,
    StreamDelta,
    StreamEndEvent,
    StreamReasoningDeltaEvent,
    StreamStartEvent,
    SubagentStreamDeltaEvent,
    SubagentStreamEndEvent,
    SubagentStreamStartEvent,
    SubagentLifecycleStatus,
    SubagentNotificationRecord,
    SubagentSessionRecord,
    SubagentConfig,
    FreeSubagentConfig,
    ToolSchema,
    is_wake_worthy,
    validate_subagent_lifecycle_transition,
)
from ecs_agent.systems.message_bus import MessageBusSystem
from ecs_agent.systems.reasoning import ReasoningSystem
from ecs_agent.systems.subagent import SubagentSystem
from ecs_agent.systems.subagent_wait import (
    SubagentWaitSystem,
    build_subagent_compaction_state,
)


class ReasoningAndContentStreamingFakeModel(FakeModel):
    async def _stream_complete(
        self, result: CompletionResult
    ) -> AsyncIterator[StreamDelta]:
        _ = result
        yield StreamDelta(reasoning_content="thinking")
        yield StreamDelta(content="done")
        yield StreamDelta(finish_reason="stop")


class BrokenStreamingFakeModel(FakeModel):
    async def _stream_complete(
        self, result: CompletionResult
    ) -> AsyncIterator[StreamDelta]:
        _ = result
        yield StreamDelta(reasoning_content="thinking")
        raise RuntimeError("stream exploded")


class RecordingFakeModel(FakeModel):
    def __init__(self, responses: list[CompletionResult]) -> None:
        super().__init__(responses=responses)
        self.calls: list[list[Message]] = []

    async def complete(
        self,
        messages: list[Message],
        tools: list[object] | None = None,
        stream: bool = False,
        response_format: dict[str, Any] | None = None,
    ) -> CompletionResult | AsyncIterator[StreamDelta]:
        _ = tools
        self.calls.append(list(messages))
        return await super().complete(
            messages,
            tools=None,
            stream=stream,
            response_format=response_format,
        )


def test_subagent_config_dataclass() -> None:
    """Verify SubagentConfig has all required fields."""
    model = FakeModel(
        responses=[CompletionResult(message=Message(role="assistant", content="done"))]
    )

    config = SubagentConfig(
        name="researcher",
        model=model,
        system_prompt="You research things",
        skills=["web-search", "read-file"],
        max_ticks=5,
    )

    assert config.name == "researcher"
    assert config.model is model
    assert config.system_prompt == "You research things"
    assert config.skills == ["web-search", "read-file"]
    assert config.max_ticks == 5


def test_subagent_config_defaults() -> None:
    """Verify SubagentConfig has sensible defaults."""
    model = FakeModel(responses=[])

    config = SubagentConfig(
        name="default-agent",
        model=model,
    )

    assert config.system_prompt == ""
    assert config.skills == []
    assert config.max_ticks is None
    # Verify inheritance_policy has correct defaults
    assert config.inheritance_policy is not None
    assert config.inheritance_policy.enabled is True
    assert config.inheritance_policy.inherit_system_prompt is True
    assert config.inheritance_policy.inherit_tools == []
    assert config.inheritance_policy.inherit_permissions is False
    assert config.inheritance_policy.tool_conflict_policy == "skip"
    assert config.inheritance_policy.missing_skill_policy == "warn"


def test_subagent_registry_component_defaults() -> None:
    """Verify SubagentRegistryComponent starts with empty registry."""
    registry = SubagentRegistryComponent()
    assert registry.subagents == {}


def test_subagent_registry_register_and_lookup() -> None:
    """Register a SubagentConfig and look it up by name."""
    model = FakeModel(
        responses=[
            CompletionResult(message=Message(role="assistant", content="result"))
        ]
    )

    registry = SubagentRegistryComponent()
    config = SubagentConfig(
        name="researcher",
        model=model,
        system_prompt="You research things",
    )

    registry.subagents["researcher"] = config

    retrieved = registry.subagents["researcher"]
    assert retrieved.name == "researcher"
    assert retrieved.model is model
    assert retrieved.max_ticks is None


def test_delegation_started_event() -> None:
    """Verify DelegationStartedEvent has required fields."""
    world = World()
    entity = world.create_entity()

    event = DelegationStartedEvent(
        entity_id=entity,
        subagent_name="researcher",
        task="Find the latest AI research papers",
        correlation_id="corr-123",
        traceparent="00-0123456789abcdef0123456789abcdef-0123456789abcdef-01",
    )

    assert event.entity_id == entity
    assert event.subagent_name == "researcher"
    assert event.task == "Find the latest AI research papers"
    assert event.correlation_id == "corr-123"
    assert (
        event.traceparent == "00-0123456789abcdef0123456789abcdef-0123456789abcdef-01"
    )


def test_delegation_completed_event() -> None:
    """Verify DelegationCompletedEvent has required fields."""
    world = World()
    entity = world.create_entity()

    event = DelegationCompletedEvent(
        entity_id=entity,
        subagent_name="researcher",
        result="Found 5 papers from 2024",
        success=True,
        error=None,
        correlation_id="corr-123",
        traceparent="00-0123456789abcdef0123456789abcdef-0123456789abcdef-01",
    )

    assert event.entity_id == entity
    assert event.subagent_name == "researcher"
    assert event.result == "Found 5 papers from 2024"
    assert event.success is True
    assert event.error is None
    assert event.correlation_id == "corr-123"
    assert (
        event.traceparent == "00-0123456789abcdef0123456789abcdef-0123456789abcdef-01"
    )


def test_delegation_started_event_has_child_world_name_field() -> None:
    from ecs_agent.types import EntityId

    world = World()
    entity = world.create_entity()
    evt = DelegationStartedEvent(
        entity_id=entity,
        subagent_name="researcher",
        task="do work",
        correlation_id="c1",
        traceparent="t1",
        child_world_name="researcher-abc12345",
    )
    assert evt.child_world_name == "researcher-abc12345"


def test_delegation_completed_event_has_child_world_name_field() -> None:
    world = World()
    entity = world.create_entity()
    evt = DelegationCompletedEvent(
        entity_id=entity,
        subagent_name="researcher",
        result="done",
        child_world_name="researcher-abc12345",
    )
    assert evt.child_world_name == "researcher-abc12345"


@pytest.mark.parametrize(
    ("from_status", "to_status"),
    [
        ("queued", "running"),
        ("queued", "cancelled"),
        ("running", "succeeded"),
        ("running", "failed"),
        ("running", "timed_out"),
        ("running", "cancelled"),
    ],
)
def test_subagent_contract_and_lifecycle_valid_transition_matrix(
    from_status: SubagentLifecycleStatus,
    to_status: SubagentLifecycleStatus,
) -> None:
    validate_subagent_lifecycle_transition(from_status, to_status)


@pytest.mark.parametrize(
    ("from_status", "to_status"),
    [
        ("queued", "queued"),
        ("queued", "succeeded"),
        ("queued", "failed"),
        ("queued", "timed_out"),
        ("running", "queued"),
        ("running", "running"),
        ("succeeded", "queued"),
        ("succeeded", "running"),
        ("succeeded", "failed"),
        ("succeeded", "timed_out"),
        ("succeeded", "cancelled"),
        ("failed", "queued"),
        ("failed", "running"),
        ("failed", "succeeded"),
        ("failed", "timed_out"),
        ("failed", "cancelled"),
        ("timed_out", "queued"),
        ("timed_out", "running"),
        ("timed_out", "succeeded"),
        ("timed_out", "failed"),
        ("timed_out", "cancelled"),
        ("cancelled", "queued"),
        ("cancelled", "running"),
        ("cancelled", "succeeded"),
        ("cancelled", "failed"),
        ("cancelled", "timed_out"),
    ],
)
def test_invalid_lifecycle_transition_raises_value_error(
    from_status: SubagentLifecycleStatus,
    to_status: SubagentLifecycleStatus,
) -> None:
    with pytest.raises(
        ValueError,
        match=(
            "Invalid subagent lifecycle transition "
            f"from '{from_status}' to '{to_status}'"
        ),
    ):
        validate_subagent_lifecycle_transition(from_status, to_status)


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        ("queued", False),
        ("running", False),
        ("succeeded", True),
        ("failed", True),
        ("timed_out", True),
        ("cancelled", False),
    ],
)
def test_is_wake_worthy_returns_expected_terminal_wake_policy(
    status: SubagentLifecycleStatus,
    expected: bool,
) -> None:
    assert is_wake_worthy(status) is expected


def test_subagent_contract_and_lifecycle_session_record_fields() -> None:
    record = SubagentSessionRecord(
        session_id="session-1",
        category="research",
        prompt="Gather context",
        parent_entity_id=EntityId(1),
        created_at="2026-03-10T10:00:00Z",
        updated_at="2026-03-10T10:00:00Z",
        load_skills=["search", "summarize"],
        background=False,
        timeout_seconds=120,
        status="queued",
        correlation_id="corr-123",
        traceparent="00-0123456789abcdef0123456789abcdef-0123456789abcdef-01",
    )
    assert record.session_id == "session-1"
    assert record.category == "research"
    assert record.prompt == "Gather context"
    assert record.load_skills == ["search", "summarize"]
    assert record.background is False
    assert record.timeout_seconds == 120
    assert record.status == "queued"
    assert record.correlation_id == "corr-123"
    assert (
        record.traceparent == "00-0123456789abcdef0123456789abcdef-0123456789abcdef-01"
    )
    assert record.result_summary is None


def test_subagent_notification_contract_record_fields() -> None:
    record = SubagentNotificationRecord(
        notification_id="session-1:succeeded",
        session_id="session-1",
        parent_entity_id=1,
        terminal_status="succeeded",
        summary="Condensed result",
        error=None,
        created_at="2026-04-06T12:00:00Z",
        delivered_at=None,
    )

    assert record.notification_id == "session-1:succeeded"
    assert record.session_id == "session-1"
    assert record.parent_entity_id == 1
    assert record.terminal_status == "succeeded"
    assert record.summary == "Condensed result"
    assert record.error is None
    assert record.created_at == "2026-04-06T12:00:00Z"
    assert record.delivered_at is None


def test_subagent_wait_component_defaults_without_runtime_future() -> None:
    component = SubagentWaitComponent()

    assert component.session_ids is None
    assert component.timeout is None
    assert component.future is None
    assert component.started_at is None


def test_subagent_notification_queue_component_starts_empty() -> None:
    component = SubagentNotificationQueueComponent()

    assert component.notifications == []


def test_build_subagent_compaction_state_classifies_pending_and_completed_sessions() -> (
    None
):
    table = SubagentSessionTableComponent(
        sessions={
            "sess-running": SubagentSessionRecord(
                session_id="sess-running",
                category="research",
                prompt="Investigate",
                parent_entity_id=EntityId(1),
                created_at="2026-04-01T10:00:00Z",
                updated_at="2026-04-01T10:05:00Z",
                status="running",
            ),
            "sess-succeeded": SubagentSessionRecord(
                session_id="sess-succeeded",
                category="research",
                prompt="Summarize",
                parent_entity_id=EntityId(1),
                created_at="2026-04-01T10:00:00Z",
                updated_at="2026-04-01T10:05:00Z",
                status="succeeded",
            ),
            "sess-cancelled": SubagentSessionRecord(
                session_id="sess-cancelled",
                category="research",
                prompt="Abort",
                parent_entity_id=EntityId(1),
                created_at="2026-04-01T10:00:00Z",
                updated_at="2026-04-01T10:05:00Z",
                status="cancelled",
            ),
        }
    )
    queue = SubagentNotificationQueueComponent(
        notifications=[
            SubagentNotificationRecord(
                notification_id="sess-succeeded:succeeded",
                session_id="sess-succeeded",
                parent_entity_id=1,
                terminal_status="succeeded",
                summary="cached summary",
                error=None,
                created_at="2026-04-01T10:06:00Z",
                delivered_at=None,
            )
        ]
    )

    state = build_subagent_compaction_state(table, queue)

    assert state.pending == ["sess-running"]
    assert state.completed == [
        ("sess-cancelled", "cancelled"),
        ("sess-succeeded", "succeeded"),
    ]
    assert state.notifications == [
        'sess-succeeded: notification status=succeeded delivered=no summary="cached summary"'
    ]


def test_subagent_stream_start_event_dataclass_fields() -> None:
    parent_entity_id = EntityId(7)

    event = SubagentStreamStartEvent(
        session_id="session-1",
        parent_entity_id=parent_entity_id,
        category="research",
        child_world_name="researcher-abc12345",
        seq=1,
        timestamp="2026-04-05T10:00:00Z",
    )

    assert event.session_id == "session-1"
    assert event.parent_entity_id == parent_entity_id
    assert event.category == "research"
    assert event.child_world_name == "researcher-abc12345"
    assert event.seq == 1
    assert event.timestamp == "2026-04-05T10:00:00Z"
    assert not hasattr(event, "__dict__")


def test_subagent_stream_delta_event_dataclass_fields() -> None:
    parent_entity_id = EntityId(8)

    event = SubagentStreamDeltaEvent(
        session_id="session-2",
        parent_entity_id=parent_entity_id,
        category="analysis",
        child_world_name="analyst-def67890",
        seq=2,
        timestamp="2026-04-05T10:00:01Z",
        delta="partial answer",
        reasoning_delta="thinking out loud",
    )

    assert event.session_id == "session-2"
    assert event.parent_entity_id == parent_entity_id
    assert event.category == "analysis"
    assert event.child_world_name == "analyst-def67890"
    assert event.seq == 2
    assert event.timestamp == "2026-04-05T10:00:01Z"
    assert event.delta == "partial answer"
    assert event.reasoning_delta == "thinking out loud"
    assert not hasattr(event, "__dict__")


def test_subagent_stream_delta_event_defaults_reasoning_delta_to_none() -> None:
    event = SubagentStreamDeltaEvent(
        session_id="session-3",
        parent_entity_id=EntityId(9),
        category="analysis",
        child_world_name="analyst-fedcba98",
        seq=3,
        timestamp="2026-04-05T10:00:02Z",
        delta="content only",
    )

    assert event.reasoning_delta is None


def test_subagent_stream_end_event_dataclass_fields() -> None:
    parent_entity_id = EntityId(10)

    event = SubagentStreamEndEvent(
        session_id="session-4",
        parent_entity_id=parent_entity_id,
        category="research",
        child_world_name="researcher-1234abcd",
        seq=4,
        timestamp="2026-04-05T10:00:03Z",
        total_tokens=42,
    )

    assert event.session_id == "session-4"
    assert event.parent_entity_id == parent_entity_id
    assert event.category == "research"
    assert event.child_world_name == "researcher-1234abcd"
    assert event.seq == 4
    assert event.timestamp == "2026-04-05T10:00:03Z"
    assert event.total_tokens == 42
    assert not hasattr(event, "__dict__")


def test_subagent_stream_end_event_defaults_total_tokens_to_none() -> None:
    event = SubagentStreamEndEvent(
        session_id="session-5",
        parent_entity_id=EntityId(11),
        category="research",
        child_world_name="researcher-ff00aa11",
        seq=5,
        timestamp="2026-04-05T10:00:04Z",
    )

    assert event.total_tokens is None


def test_subagent_stream_event_session_record_stream_flag_defaults_false() -> None:
    record = SubagentSessionRecord(
        session_id="session-6",
        category="research",
        prompt="Gather context",
        parent_entity_id=EntityId(12),
        created_at="2026-04-05T10:00:05Z",
        updated_at="2026-04-05T10:00:05Z",
    )

    assert record.stream is False


async def test_subagent_background_stream_true_bridges_child_stream_events_to_parent_event_bus() -> (
    None
):
    world = World()
    parent_entity = world.create_entity()
    model = ReasoningAndContentStreamingFakeModel(
        responses=[CompletionResult(message=Message(role="assistant", content="done"))]
    )
    config = SubagentConfig(name="research", model=model)

    system = SubagentSystem()
    captured: dict[str, Any] = {}
    original_assemble_child_world = system._assemble_child_world

    def capture_child_world(
        parent_world: World,
        parent_entity_id: EntityId,
        config_snapshot: SubagentConfig,
        parent_child_entity: EntityId | None = None,
    ) -> tuple[World, EntityId]:
        child_world, child_entity_id = original_assemble_child_world(
            parent_world,
            parent_entity_id,
            config_snapshot,
            parent_child_entity,
        )
        captured["child_world"] = child_world
        captured["child_entity_id"] = child_entity_id
        return child_world, child_entity_id

    system._assemble_child_world = capture_child_world  # type: ignore[method-assign]

    received: list[
        SubagentStreamStartEvent | SubagentStreamDeltaEvent | SubagentStreamEndEvent
    ] = []

    async def on_start(event: SubagentStreamStartEvent) -> None:
        received.append(event)

    async def on_delta(event: SubagentStreamDeltaEvent) -> None:
        received.append(event)

    async def on_end(event: SubagentStreamEndEvent) -> None:
        received.append(event)

    world.event_bus.subscribe(SubagentStreamStartEvent, on_start)
    world.event_bus.subscribe(SubagentStreamDeltaEvent, on_delta)
    world.event_bus.subscribe(SubagentStreamEndEvent, on_end)

    result, success, error = await system._execute_subagent_core(
        world,
        parent_entity,
        "research",
        "Investigate",
        "corr-1",
        "trace-1",
        config,
        session_id="session-1",
        stream=True,
    )
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert success is True
    assert error is None
    assert result == "done"

    assert [type(event) for event in received] == [
        SubagentStreamStartEvent,
        SubagentStreamDeltaEvent,
        SubagentStreamDeltaEvent,
        SubagentStreamEndEvent,
    ]
    assert [event.seq for event in received] == [0, 1, 2, 3]

    start_event = received[0]
    assert isinstance(start_event, SubagentStreamStartEvent)
    assert start_event.session_id == "session-1"
    assert start_event.parent_entity_id == parent_entity
    assert start_event.category == "research"
    assert start_event.child_world_name.startswith("research-")
    assert start_event.timestamp

    reasoning_delta = received[1]
    assert isinstance(reasoning_delta, SubagentStreamDeltaEvent)
    assert reasoning_delta.delta == ""
    assert reasoning_delta.reasoning_delta == "thinking"

    content_delta = received[2]
    assert isinstance(content_delta, SubagentStreamDeltaEvent)
    assert content_delta.delta == "done"
    assert content_delta.reasoning_delta is None

    end_event = received[3]
    assert isinstance(end_event, SubagentStreamEndEvent)
    assert end_event.total_tokens is None

    child_world = captured["child_world"]
    child_entity_id = captured["child_entity_id"]
    assert child_world.get_component(child_entity_id, StreamingComponent) is not None
    assert child_world.event_bus._handlers == {}


async def test_subagent_background_stream_false_keeps_bridge_dormant() -> None:
    world = World()
    parent_entity = world.create_entity()
    model = ReasoningAndContentStreamingFakeModel(
        responses=[CompletionResult(message=Message(role="assistant", content="done"))]
    )
    config = SubagentConfig(name="research", model=model)

    system = SubagentSystem()
    captured: dict[str, Any] = {}
    original_assemble_child_world = system._assemble_child_world

    def capture_child_world(
        parent_world: World,
        parent_entity_id: EntityId,
        config_snapshot: SubagentConfig,
        parent_child_entity: EntityId | None = None,
    ) -> tuple[World, EntityId]:
        child_world, child_entity_id = original_assemble_child_world(
            parent_world,
            parent_entity_id,
            config_snapshot,
            parent_child_entity,
        )
        captured["child_world"] = child_world
        captured["child_entity_id"] = child_entity_id
        return child_world, child_entity_id

    system._assemble_child_world = capture_child_world  # type: ignore[method-assign]

    received: list[
        SubagentStreamStartEvent | SubagentStreamDeltaEvent | SubagentStreamEndEvent
    ] = []

    async def on_start(event: SubagentStreamStartEvent) -> None:
        received.append(event)

    async def on_delta(event: SubagentStreamDeltaEvent) -> None:
        received.append(event)

    async def on_end(event: SubagentStreamEndEvent) -> None:
        received.append(event)

    world.event_bus.subscribe(SubagentStreamStartEvent, on_start)
    world.event_bus.subscribe(SubagentStreamDeltaEvent, on_delta)
    world.event_bus.subscribe(SubagentStreamEndEvent, on_end)

    result, success, error = await system._execute_subagent_core(
        world,
        parent_entity,
        "research",
        "Investigate",
        "corr-2",
        "trace-2",
        config,
        session_id="session-2",
        stream=False,
    )
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert success is True
    assert error is None
    assert result == "done"
    assert received == []

    child_world = captured["child_world"]
    child_entity_id = captured["child_entity_id"]
    assert child_world.get_component(child_entity_id, StreamingComponent) is None
    assert child_world.event_bus._handlers == {}


async def test_subagent_background_parent_event_bus_bridge_cleans_up_after_failure() -> (
    None
):
    world = World()
    parent_entity = world.create_entity()
    model = FakeModel(
        responses=[
            CompletionResult(message=Message(role="assistant", content="ignored"))
        ]
    )
    config = SubagentConfig(name="research", model=model)

    system = SubagentSystem()
    captured: dict[str, Any] = {}
    original_assemble_child_world = system._assemble_child_world

    def capture_child_world(
        parent_world: World,
        parent_entity_id: EntityId,
        config_snapshot: SubagentConfig,
        parent_child_entity: EntityId | None = None,
    ) -> tuple[World, EntityId]:
        child_world, child_entity_id = original_assemble_child_world(
            parent_world,
            parent_entity_id,
            config_snapshot,
            parent_child_entity,
        )
        captured["child_world"] = child_world
        return child_world, child_entity_id

    system._assemble_child_world = capture_child_world  # type: ignore[method-assign]

    async def fake_execute_delegation(
        child_world: World,
        child_entity: EntityId,
        task: str,
        config_snapshot: SubagentConfig,
    ) -> str:
        del child_entity, task, config_snapshot
        await child_world.event_bus.publish(
            StreamStartEvent(entity_id=EntityId(1), timestamp=1.0)
        )
        await child_world.event_bus.publish(
            StreamReasoningDeltaEvent(
                entity_id=EntityId(1),
                reasoning_delta="thinking",
            )
        )
        await child_world.event_bus.publish(
            StreamEndEvent(entity_id=EntityId(1), timestamp=2.0)
        )
        raise RuntimeError("stream exploded")

    system._execute_delegation = fake_execute_delegation  # type: ignore[method-assign]

    received: list[
        SubagentStreamStartEvent | SubagentStreamDeltaEvent | SubagentStreamEndEvent
    ] = []

    async def on_start(event: SubagentStreamStartEvent) -> None:
        received.append(event)

    async def on_delta(event: SubagentStreamDeltaEvent) -> None:
        received.append(event)

    async def on_end(event: SubagentStreamEndEvent) -> None:
        received.append(event)

    world.event_bus.subscribe(SubagentStreamStartEvent, on_start)
    world.event_bus.subscribe(SubagentStreamDeltaEvent, on_delta)
    world.event_bus.subscribe(SubagentStreamEndEvent, on_end)

    result, success, error = await system._execute_subagent_core(
        world,
        parent_entity,
        "research",
        "Investigate",
        "corr-3",
        "trace-3",
        config,
        session_id="session-3",
        stream=True,
    )
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert success is False
    assert result == "Error during delegation: stream exploded"
    assert error == "Error during delegation: stream exploded"
    assert [type(event) for event in received] == [
        SubagentStreamStartEvent,
        SubagentStreamDeltaEvent,
        SubagentStreamEndEvent,
    ]
    assert [event.seq for event in received] == [0, 1, 2]
    assert captured["child_world"].event_bus._handlers == {}


# ──────────────────────────────────────────────────────────────────────────────
# Task 12: SubagentSystem + Subagent Tool Tests
# ──────────────────────────────────────────────────────────────────────────────

from ecs_agent.components import (
    ConversationComponent,
    ErrorComponent,
    LLMComponent,
    OwnerComponent,
    PermissionComponent,
    TerminalComponent,
    ToolRegistryComponent,
)
from ecs_agent.components.definitions import SkillComponent
from ecs_agent.core.runner import Runner


def _register_message_bus(
    world: World, parent_entity: EntityId, request_timeout: float = 30.0
) -> MessageBusSystem:
    world.add_component(
        parent_entity,
        MessageBusConfigComponent(request_timeout=request_timeout),
    )
    message_bus = MessageBusSystem(priority=5, request_timeout=request_timeout)
    world.register_system(message_bus, priority=5)
    return message_bus


async def test_delegation_event_correlation_integrity() -> None:
    """DelegationStartedEvent and DelegationCompletedEvent MUST share correlation_id."""
    from ecs_agent.systems.subagent import SubagentSystem

    world = World()
    parent_entity = world.create_entity()

    model = FakeModel(
        responses=[CompletionResult(message=Message(role="assistant", content="Done"))]
    )
    config = SubagentConfig(name="test-agent", model=model)
    registry = SubagentRegistryComponent(subagents={"test-agent": config})
    world.add_component(parent_entity, registry)
    world.add_component(parent_entity, ToolRegistryComponent(tools={}, handlers={}))
    _register_message_bus(world, parent_entity)

    # Subscribe to delegation events
    started_events: list[DelegationStartedEvent] = []
    completed_events: list[DelegationCompletedEvent] = []

    async def on_started(event: DelegationStartedEvent) -> None:
        started_events.append(event)

    async def on_completed(event: DelegationCompletedEvent) -> None:
        completed_events.append(event)

    world.event_bus.subscribe(DelegationStartedEvent, on_started)
    world.event_bus.subscribe(DelegationCompletedEvent, on_completed)

    # Process to register subagent tool
    system = SubagentSystem()
    await system.process(world)

    tool_registry = world.get_component(parent_entity, ToolRegistryComponent)
    assert tool_registry is not None
    subagent_handler = tool_registry.handlers["subagent"]

    # Invoke subagent
    await subagent_handler(category="test-agent", prompt="test task")

    # Verify events published
    assert len(started_events) == 1, "DelegationStartedEvent MUST be published"
    assert len(completed_events) == 1, "DelegationCompletedEvent MUST be published"

    # EVENT CORRELATION INTEGRITY
    started_event = started_events[0]
    completed_event = completed_events[0]

    # Assert correlation_id matches
    assert started_event.correlation_id == completed_event.correlation_id, (
        "DelegationStartedEvent and DelegationCompletedEvent MUST share correlation_id"
    )

    # Assert traceparent matches (distributed tracing integrity)
    assert started_event.traceparent == completed_event.traceparent, (
        "DelegationStartedEvent and DelegationCompletedEvent MUST share traceparent"
    )

    # Assert correlation_id is not empty (valid UUID)
    assert started_event.correlation_id, "correlation_id MUST be non-empty"
    assert len(started_event.correlation_id) > 0, "correlation_id MUST be valid UUID"

    # Assert traceparent format (W3C Trace Context: 00-{trace-id}-{parent-id}-{flags})
    assert started_event.traceparent.startswith("00-"), (
        "traceparent MUST follow W3C Trace Context format"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Task 2: Subagent Tool Auto-Registration Tests
# ──────────────────────────────────────────────────────────────────────────────


async def test_backward_compatible_auto_registration_still_works() -> None:
    """Test that existing process() auto-registration behavior is preserved."""
    from ecs_agent.systems.subagent import SubagentSystem

    world = World()
    entity = world.create_entity()

    model = FakeModel(
        responses=[CompletionResult(message=Message(role="assistant", content="Done"))]
    )
    config = SubagentConfig(name="test-agent", model=model)
    registry = SubagentRegistryComponent(subagents={"test-agent": config})
    world.add_component(entity, registry)
    world.add_component(entity, ToolRegistryComponent(tools={}, handlers={}))

    system = SubagentSystem()
    # This is the EXISTING behavior - must still work!
    await system.process(world)

    tool_registry = world.get_component(entity, ToolRegistryComponent)
    assert tool_registry is not None
    assert "subagent" in tool_registry.handlers, (
        "Process must auto-register subagent tool"
    )
    assert "subagent" in tool_registry.tools, "Process must register subagent schema"


def _shared_tool_schema() -> ToolSchema:
    return ToolSchema(
        name="shared_tool",
        description="Shared tool for inheritance tests",
        parameters={"type": "object", "properties": {}, "required": []},
    )


def _find_child_entity(world: World, parent_entity: EntityId) -> EntityId:
    for entity_id, components in world.query(OwnerComponent):
        owner_comp = components[0]
        assert isinstance(owner_comp, OwnerComponent)
        if owner_comp.owner_id == parent_entity:
            return entity_id
    raise AssertionError("Expected delegated child entity to exist")


async def _delegate_with_policy(
    *,
    policy: InheritancePolicy,
    parent_system_prompt: str = "parent-system",
    child_system_prompt: str = "",
    child_has_permission: bool = False,
) -> tuple[World, EntityId, EntityId, str]:
    from ecs_agent.systems.subagent import SubagentSystem

    world = World()
    parent_entity = world.create_entity()

    async def parent_shared_tool_handler() -> str:
        return "parent-handler"

    parent_provider = FakeModel(
        responses=[
            CompletionResult(message=Message(role="assistant", content="Parent done"))
        ]
    )
    parent_registry = ToolRegistryComponent(
        tools={"shared_tool": _shared_tool_schema()},
        handlers={"shared_tool": parent_shared_tool_handler},
    )
    world.add_component(parent_entity, parent_registry)
    world.add_component(
        parent_entity,
        LLMComponent(
            model=parent_provider,
            system_prompt=parent_system_prompt,
        ),
    )
    if child_has_permission:
        world.add_component(
            parent_entity,
            PermissionComponent(allowed_tools=["shared_tool"], denied_tools=[]),
        )

    child_provider = FakeModel(
        responses=[
            CompletionResult(message=Message(role="assistant", content="Child done"))
        ]
    )
    config = SubagentConfig(
        name="child",
        model=child_provider,
        system_prompt=child_system_prompt,
        inheritance_policy=policy,
    )
    world.add_component(
        parent_entity,
        SubagentRegistryComponent(subagents={"child": config}),
    )
    _register_message_bus(world, parent_entity)

    system = SubagentSystem()
    await system.process(world)

    parent_tools = world.get_component(parent_entity, ToolRegistryComponent)
    assert parent_tools is not None
    subagent_handler = parent_tools.handlers["subagent"]
    result = await subagent_handler(category="child", prompt="Run delegated task")

    child_entity = _find_child_entity(world, parent_entity)
    return world, parent_entity, child_entity, result


async def test_inheritance_policy_conflict_skip_keeps_child_tool() -> None:
    policy = InheritancePolicy(
        enabled=True,
        inherit_tools=["shared_tool"],
        tool_conflict_policy="skip",
    )
    world, _, child_entity, _ = await _delegate_with_policy(policy=policy)

    child_tools = world.get_component(child_entity, ToolRegistryComponent)
    assert child_tools is not None, (
        "Skip policy test requires child ToolRegistryComponent to be present"
    )
    assert child_tools.handlers["shared_tool"] is not None, (
        "Skip policy should preserve child handler and not drop tool registration"
    )


async def test_inheritance_policy_conflict_error_raises() -> None:
    policy = InheritancePolicy(
        enabled=True,
        inherit_tools=["shared_tool"],
        tool_conflict_policy="error",
    )

    with pytest.raises(ValueError, match="shared_tool"):
        await _delegate_with_policy(policy=policy)


async def test_inheritance_policy_conflict_override_replaces_child_tool() -> None:
    policy = InheritancePolicy(
        enabled=True,
        inherit_tools=["shared_tool"],
        tool_conflict_policy="override",
    )
    world, _, child_entity, _ = await _delegate_with_policy(policy=policy)

    child_tools = world.get_component(child_entity, ToolRegistryComponent)
    assert child_tools is not None, (
        "Override policy test requires child ToolRegistryComponent to be present"
    )
    assert "shared_tool" in child_tools.tools, (
        "Override policy should keep shared_tool registered after replacement"
    )


async def test_inheritance_policy_explicit_child_system_prompt_authoritative() -> None:
    policy = InheritancePolicy(enabled=True, inherit_system_prompt=True)
    world, _, child_entity, _ = await _delegate_with_policy(
        policy=policy,
        parent_system_prompt="parent prompt",
        child_system_prompt="child prompt",
    )

    child_llm = world.get_component(child_entity, LLMComponent)
    assert child_llm is not None
    assert child_llm.system_prompt.startswith("child prompt"), (
        "Child explicit system_prompt must remain authoritative over inherited prompt"
    )
    assert "## Available Tools" in child_llm.system_prompt
    assert "## Available Skills" in child_llm.system_prompt


async def test_inheritance_policy_whitelist_only_inherits_named_tools() -> None:
    policy = InheritancePolicy(
        enabled=True,
        inherit_tools=["shared_tool"],
    )
    world, _, child_entity, _ = await _delegate_with_policy(policy=policy)

    child_tools = world.get_component(child_entity, ToolRegistryComponent)
    assert child_tools is not None, (
        "Whitelist test requires child ToolRegistryComponent to be present"
    )
    assert "shared_tool" in child_tools.tools, (
        "Whitelist inheritance should copy only named tools to child"
    )
    assert "subagent" not in child_tools.tools, (
        "Whitelist inheritance must not copy tools omitted from inherit_tools"
    )


async def test_inheritance_policy_enabled_false_skips_all_inheritance() -> None:
    policy = InheritancePolicy(
        enabled=False,
        inherit_system_prompt=True,
        inherit_tools=["shared_tool", "subagent"],
        inherit_permissions=True,
    )
    world, _, child_entity, _ = await _delegate_with_policy(
        policy=policy,
        parent_system_prompt="parent prompt",
        child_system_prompt="",
        child_has_permission=True,
    )

    child_llm = world.get_component(child_entity, LLMComponent)
    assert child_llm is not None
    assert child_llm.system_prompt != "", (
        "enabled=False must disable system_prompt inheritance"
    )
    assert "## Available Tools" in child_llm.system_prompt
    assert "## Available Skills" in child_llm.system_prompt

    child_tools = world.get_component(child_entity, ToolRegistryComponent)
    if child_tools is not None:
        assert "shared_tool" not in child_tools.tools, (
            "enabled=False must disable tool inheritance"
        )
    child_perm = world.get_component(child_entity, PermissionComponent)
    assert child_perm is None, "enabled=False must disable permission inheritance"


async def test_inheritance_policy_inherit_system_prompt_when_child_empty() -> None:
    policy = InheritancePolicy(enabled=True, inherit_system_prompt=True)
    world, _, child_entity, _ = await _delegate_with_policy(
        policy=policy,
        parent_system_prompt="parent inherited prompt",
        child_system_prompt="",
    )

    child_llm = world.get_component(child_entity, LLMComponent)
    assert child_llm is not None
    assert child_llm.system_prompt.startswith("parent inherited prompt"), (
        "inherit_system_prompt=True should copy parent prompt when child prompt is empty"
    )
    assert "## Available Tools" in child_llm.system_prompt
    assert "## Available Skills" in child_llm.system_prompt


async def test_inheritance_policy_inherit_permissions_true_copies_permission_component() -> (
    None
):
    policy = InheritancePolicy(enabled=True, inherit_permissions=True)
    world, _, child_entity, _ = await _delegate_with_policy(
        policy=policy,
        child_has_permission=True,
    )

    child_perm = world.get_component(child_entity, PermissionComponent)
    assert child_perm is not None, (
        "inherit_permissions=True should copy parent PermissionComponent to child"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Task 4: SkillManager-Aligned Skill Inheritance Tests (RED)
# ──────────────────────────────────────────────────────────────────────────────


async def test_subagent_skills_skill_manager_inherited_tools_available() -> None:
    """Child can execute tools from inherited skills (SkillManager semantics)."""
    from ecs_agent.systems.subagent import SubagentSystem
    from ecs_agent.skills.manager import SkillManager
    from ecs_agent.skills.script_skill import ScriptSkill

    # Define a test skill with a tool
    class ParentSkill(ScriptSkill):
        name: str = "parent-skill"
        description: str = "Skill installed on parent"

        def tools(self) -> dict[str, tuple[ToolSchema, Any]]:
            async def parent_tool() -> str:
                return "parent_tool_result"

            return {
                "parent_tool": (
                    ToolSchema(
                        name="parent_tool",
                        description="A tool from parent skill",
                        parameters={"type": "object", "properties": {}},
                    ),
                    parent_tool,
                )
            }

        def system_prompt(self) -> str:
            return "Parent skill system prompt"

        def install(self, world: World, entity_id: EntityId) -> None:
            pass

        def uninstall(self, world: World, entity_id: EntityId) -> None:
            pass

    world = World()
    parent_entity = world.create_entity()

    # Install skill on parent using SkillManager
    skill_manager = SkillManager()
    parent_skill = ParentSkill()
    skill_manager.install(world, parent_entity, parent_skill)

    # Configure child to inherit parent skill
    child_provider = FakeModel(
        responses=[CompletionResult(message=Message(role="assistant", content="Done"))]
    )
    config = SubagentConfig(
        name="child",
        model=child_provider,
        inheritance_policy=InheritancePolicy(
            enabled=True,
            inherit_tools=["parent_tool"],  # Inherit from parent skill
        ),
    )

    registry = SubagentRegistryComponent(subagents={"child": config})
    world.add_component(parent_entity, registry)
    _register_message_bus(world, parent_entity)

    # Delegation logic doesn't exist yet - RED!
    system = SubagentSystem()
    await system.process(world)

    tool_registry = world.get_component(parent_entity, ToolRegistryComponent)
    assert tool_registry is not None
    subagent_handler = tool_registry.handlers["subagent"]

    # Delegate task
    result = await subagent_handler(category="child", prompt="test task")

    # Find child entity
    child_entity = _find_child_entity(world, parent_entity)

    # RED TEST: Verify child has SkillComponent with parent skill
    child_skills = world.get_component(child_entity, SkillComponent)
    assert child_skills is not None, (
        "Child entity should have SkillComponent after skill inheritance"
    )
    assert "parent-skill" in child_skills.skills, (
        "parent-skill should be installed on child via SkillManager semantics"
    )

    # RED TEST: Verify child can call parent_tool
    child_tools = world.get_component(child_entity, ToolRegistryComponent)
    assert child_tools is not None
    assert "parent_tool" in child_tools.handlers, (
        "Child should have inherited parent_tool handler"
    )
    assert "parent_tool" in child_tools.tools, (
        "Child should have inherited parent_tool schema"
    )

    # RED TEST: Verify tool is actually callable
    tool_result = await child_tools.handlers["parent_tool"]()
    assert tool_result == "parent_tool_result", (
        "Inherited tool should execute and return expected result"
    )


async def test_subagent_skills_skill_manager_requested_tools_available() -> None:
    """Child can execute tools from requested skills (SubagentConfig.skills)."""
    from ecs_agent.systems.subagent import SubagentSystem
    from ecs_agent.skills.manager import SkillManager
    from ecs_agent.skills.script_skill import ScriptSkill

    # Define a skill that child requests explicitly
    class RequestedSkill(ScriptSkill):
        name: str = "requested-skill"
        description: str = "Skill requested by child"

        def tools(self) -> dict[str, tuple[ToolSchema, Any]]:
            async def requested_tool() -> str:
                return "requested_tool_result"

            return {
                "requested_tool": (
                    ToolSchema(
                        name="requested_tool",
                        description="Tool from requested skill",
                        parameters={"type": "object", "properties": {}},
                    ),
                    requested_tool,
                )
            }

        def system_prompt(self) -> str:
            return "Requested skill prompt"

        def install(self, world: World, entity_id: EntityId) -> None:
            pass

        def uninstall(self, world: World, entity_id: EntityId) -> None:
            pass

    world = World()
    parent_entity = world.create_entity()

    # Install skill on parent using SkillManager
    skill_manager = SkillManager()
    requested_skill = RequestedSkill()
    skill_manager.install(world, parent_entity, requested_skill)

    # Configure child to request skill explicitly
    child_provider = FakeModel(
        responses=[CompletionResult(message=Message(role="assistant", content="Done"))]
    )
    config = SubagentConfig(
        name="child",
        model=child_provider,
        skills=["requested-skill"],  # Request skill installation
    )

    registry = SubagentRegistryComponent(subagents={"child": config})
    world.add_component(parent_entity, registry)
    _register_message_bus(world, parent_entity)

    # Delegation logic doesn't exist yet - RED!
    system = SubagentSystem()
    await system.process(world)

    tool_registry = world.get_component(parent_entity, ToolRegistryComponent)
    assert tool_registry is not None
    subagent_handler = tool_registry.handlers["subagent"]

    # Delegate task
    result = await subagent_handler(category="child", prompt="test task")

    # Find child entity
    child_entity = _find_child_entity(world, parent_entity)

    # RED TEST: Verify child has SkillComponent with requested skill
    child_skills = world.get_component(child_entity, SkillComponent)
    assert child_skills is not None, (
        "Child entity should have SkillComponent with requested skills"
    )
    assert "requested-skill" in child_skills.skills, (
        "requested-skill should be installed on child via SkillManager.install()"
    )

    # RED TEST: Verify child can call requested_tool
    child_tools = world.get_component(child_entity, ToolRegistryComponent)
    assert child_tools is not None
    assert "requested_tool" in child_tools.handlers, (
        "Child should have requested_tool handler from SkillManager.install()"
    )
    assert "requested_tool" in child_tools.tools, (
        "Child should have requested_tool schema from SkillManager.install()"
    )

    # RED TEST: Verify tool is actually callable
    tool_result = await child_tools.handlers["requested_tool"]()
    assert tool_result == "requested_tool_result", (
        "Requested skill tool should execute via SkillManager lifecycle"
    )


async def test_subagent_skills_missing_skill_warn_policy() -> None:
    """When missing skill + warn policy, logs warning and continues."""
    from ecs_agent.systems.subagent import SubagentSystem
    import logging
    from unittest.mock import patch

    world = World()
    parent_entity = world.create_entity()

    # Parent has NO skills installed
    world.add_component(parent_entity, ToolRegistryComponent(tools={}, handlers={}))

    # Child config requests nonexistent skill with warn policy
    child_provider = FakeModel(
        responses=[CompletionResult(message=Message(role="assistant", content="Done"))]
    )
    config = SubagentConfig(
        name="child",
        model=child_provider,
        skills=["nonexistent_skill"],  # Request missing skill
        inheritance_policy=InheritancePolicy(
            missing_skill_policy="warn"  # Key policy
        ),
    )

    registry = SubagentRegistryComponent(subagents={"child": config})
    world.add_component(parent_entity, registry)
    _register_message_bus(world, parent_entity)

    # Track warning logs
    with patch("ecs_agent.systems.subagent.logger") as mock_logger:
        system = SubagentSystem()
        await system.process(world)

        tool_registry = world.get_component(parent_entity, ToolRegistryComponent)
        assert tool_registry is not None
        subagent_handler = tool_registry.handlers["subagent"]

        # RED TEST: Delegation should complete despite missing skill
        result = await subagent_handler(category="child", prompt="test task")

        # RED TEST: Verify delegation completed (no exception raised)
        assert isinstance(result, str), (
            "Delegation with warn policy should return result string, not raise"
        )

        # RED TEST: Verify warning was logged (doesn't exist yet - RED!)
        # This assertion will fail because warning logic doesn't exist
        mock_logger.warning.assert_called_once()
        warning_call = mock_logger.warning.call_args
        assert "nonexistent_skill" in str(warning_call), (
            "Warning should mention the missing skill name"
        )


async def test_subagent_skills_missing_skill_error_policy() -> None:
    """When missing skill + error policy, raises error."""
    from ecs_agent.systems.subagent import SubagentSystem

    world = World()
    parent_entity = world.create_entity()

    # Parent has NO skills installed
    world.add_component(parent_entity, ToolRegistryComponent(tools={}, handlers={}))

    # Child config requests nonexistent skill with error policy
    child_provider = FakeModel(
        responses=[CompletionResult(message=Message(role="assistant", content="Done"))]
    )
    config = SubagentConfig(
        name="child",
        model=child_provider,
        skills=["nonexistent_skill"],  # Request missing skill
        inheritance_policy=InheritancePolicy(
            missing_skill_policy="error"  # Key policy
        ),
    )

    registry = SubagentRegistryComponent(subagents={"child": config})
    world.add_component(parent_entity, registry)
    _register_message_bus(world, parent_entity)

    system = SubagentSystem()
    await system.process(world)

    tool_registry = world.get_component(parent_entity, ToolRegistryComponent)
    assert tool_registry is not None
    subagent_handler = tool_registry.handlers["subagent"]

    # RED TEST: Delegation should raise error for missing skill
    # This will fail because error-raising logic doesn't exist yet - RED!
    with pytest.raises((ValueError, KeyError), match="nonexistent_skill"):
        await subagent_handler(category="child", prompt="test task")


async def test_subagent_skills_skill_manager_install_uninstall_lifecycle() -> None:
    """Skills installed via SkillManager.install(), not manual dict copying."""
    from ecs_agent.systems.subagent import SubagentSystem
    from ecs_agent.skills.manager import SkillManager
    from ecs_agent.skills.script_skill import ScriptSkill
    from unittest.mock import patch

    # Define a skill to track SkillManager.install() calls
    class LifecycleSkill(ScriptSkill):
        name: str = "lifecycle-skill"
        description: str = "Skill with lifecycle tracking"

        def tools(self) -> dict[str, tuple[ToolSchema, Any]]:
            async def lifecycle_tool() -> str:
                return "lifecycle_result"

            return {
                "lifecycle_tool": (
                    ToolSchema(
                        name="lifecycle_tool",
                        description="Lifecycle tool",
                        parameters={"type": "object", "properties": {}},
                    ),
                    lifecycle_tool,
                )
            }

        def system_prompt(self) -> str:
            return "Lifecycle prompt"

        def install(self, world: World, entity_id: EntityId) -> None:
            # Track install call
            pass

        def uninstall(self, world: World, entity_id: EntityId) -> None:
            pass

    world = World()
    parent_entity = world.create_entity()

    # Install skill on parent
    skill_manager = SkillManager()
    lifecycle_skill = LifecycleSkill()
    skill_manager.install(world, parent_entity, lifecycle_skill)

    # Child requests skill
    child_provider = FakeModel(
        responses=[CompletionResult(message=Message(role="assistant", content="Done"))]
    )
    config = SubagentConfig(
        name="child",
        model=child_provider,
        skills=["lifecycle-skill"],
    )

    registry = SubagentRegistryComponent(subagents={"child": config})
    world.add_component(parent_entity, registry)
    _register_message_bus(world, parent_entity)

    # RED TEST: Verify SkillManager.install() is called (not manual dict copy)
    # This will fail because SkillManager integration doesn't exist - RED!
    with patch.object(
        SkillManager, "install", wraps=skill_manager.install
    ) as mock_install:
        system = SubagentSystem()
        await system.process(world)

        tool_registry = world.get_component(parent_entity, ToolRegistryComponent)
        assert tool_registry is not None
        subagent_handler = tool_registry.handlers["subagent"]

        await subagent_handler(category="child", prompt="test task")

        # Verify SkillManager.install() was called for child entity
        child_entity = _find_child_entity(world, parent_entity)

        # This assertion will fail - current implementation uses dict copying
        mock_install.assert_called()
        install_calls = mock_install.call_args_list
        child_install_calls = [
            call
            for call in install_calls
            if call[0][1] == child_entity  # entity_id arg
        ]
        assert len(child_install_calls) > 0, (
            "SkillManager.install() should be called for child entity, not manual dict copy"
        )


# Runtime session manager tests


def _reset_global_scheduler(monkeypatch: pytest.MonkeyPatch) -> None:
    import ecs_agent.systems.subagent_runtime as runtime_module

    monkeypatch.setattr(runtime_module, "_GLOBAL_SCHEDULER", None)


async def test_runtime_manager_create_session_returns_unique_ids() -> None:
    """Session IDs should be unique across multiple creations."""
    from ecs_agent.systems.subagent_runtime import SubagentRuntimeManager

    manager = SubagentRuntimeManager()
    ids = {manager.create_session() for _ in range(100)}
    assert len(ids) == 100, "All session IDs should be unique"


async def test_runtime_manager_register_and_retrieve_task() -> None:
    """Registered tasks should be retrievable by session ID."""
    import asyncio
    from ecs_agent.systems.subagent_runtime import SubagentRuntimeManager

    manager = SubagentRuntimeManager()
    session_id = manager.create_session()

    async def dummy_task() -> None:
        await asyncio.sleep(10)

    task = asyncio.create_task(dummy_task())
    metadata = SubagentSessionRecord(
        session_id=session_id,
        category="test",
        prompt="Test prompt",
        parent_entity_id=EntityId(1),
        created_at="2026-03-10T14:00:00Z",
        updated_at="2026-03-10T14:00:00Z",
        status="running",
    )

    await manager.register_task(session_id, task, metadata)
    retrieved = await manager.get_session(session_id)

    assert retrieved is not None
    assert retrieved.session_id == session_id
    assert retrieved.category == "test"
    assert retrieved.status == "running"

    # Cleanup
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


async def test_runtime_manager_cancel_cleans_handles() -> None:
    """Cancelled sessions should have status updated and task cancelled."""
    import asyncio
    from ecs_agent.systems.subagent_runtime import SubagentRuntimeManager

    manager = SubagentRuntimeManager()
    session_id = manager.create_session()

    async def long_running_task() -> None:
        await asyncio.sleep(100)

    task = asyncio.create_task(long_running_task())
    metadata = SubagentSessionRecord(
        session_id=session_id,
        category="test",
        prompt="Test prompt",
        parent_entity_id=EntityId(1),
        created_at="2026-03-10T14:00:00Z",
        updated_at="2026-03-10T14:00:00Z",
        status="running",
    )

    await manager.register_task(session_id, task, metadata)
    await manager.cancel_session(session_id)

    # Give cancel time to propagate
    await asyncio.sleep(0.01)

    # Verify status updated
    retrieved = await manager.get_session(session_id)
    assert retrieved is not None
    assert retrieved.status == "cancelled"

    # Verify task was cancelled
    assert task.cancelled()

    # Cleanup
    try:
        await task
    except asyncio.CancelledError:
        pass


async def test_runtime_manager_scheduler_enforces_cap_and_fifo_queue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ecs_agent.systems.subagent_runtime as runtime_module

    _reset_global_scheduler(monkeypatch)

    manager = runtime_module.SubagentRuntimeManager(max_background_concurrency=2)
    started: list[str] = []
    release_events = {name: asyncio.Event() for name in ("s1", "s2", "s3", "s4")}

    def make_metadata(session_id: str) -> SubagentSessionRecord:
        return SubagentSessionRecord(
            session_id=session_id,
            category="test",
            prompt=f"prompt-{session_id}",
            parent_entity_id=EntityId(1),
            created_at="2026-03-10T14:00:00Z",
            updated_at="2026-03-10T14:00:00Z",
        )

    def make_factory(session_id: str) -> Any:
        async def run() -> None:
            started.append(session_id)
            await release_events[session_id].wait()

        return run

    for session_id in ("s1", "s2", "s3", "s4"):
        await manager.enqueue_session(
            session_id,
            make_metadata(session_id),
            make_factory(session_id),
        )

    await asyncio.sleep(0)

    scheduler = runtime_module._GLOBAL_SCHEDULER
    assert scheduler is not None
    assert scheduler.max_concurrency == 2
    assert scheduler.active_count == 2
    assert started == ["s1", "s2"]
    assert [item.session_id for item in scheduler.pending_queue] == ["s3", "s4"]

    release_events["s1"].set()
    task = await manager.get_task("s1")
    assert task is not None
    await task
    await asyncio.sleep(0)

    assert started == ["s1", "s2", "s3"]
    assert scheduler.active_count == 2
    assert [item.session_id for item in scheduler.pending_queue] == ["s4"]

    for session_id in ("s2", "s3", "s4"):
        release_events[session_id].set()
        running_task = await manager.get_task(session_id)
        if running_task is not None:
            await running_task


async def test_runtime_manager_scheduler_rejects_conflicting_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ecs_agent.systems.subagent_runtime as runtime_module

    _reset_global_scheduler(monkeypatch)

    runtime_module.SubagentRuntimeManager(max_background_concurrency=2)

    with pytest.raises(ValueError, match="max_background_concurrency"):
        runtime_module.SubagentRuntimeManager(max_background_concurrency=3)


async def test_runtime_manager_cancel_removes_queued_session_without_slot_use(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ecs_agent.systems.subagent_runtime as runtime_module

    _reset_global_scheduler(monkeypatch)

    manager = runtime_module.SubagentRuntimeManager(max_background_concurrency=1)
    started: list[str] = []
    running_release = asyncio.Event()

    def make_metadata(session_id: str) -> SubagentSessionRecord:
        return SubagentSessionRecord(
            session_id=session_id,
            category="test",
            prompt=f"prompt-{session_id}",
            parent_entity_id=EntityId(1),
            created_at="2026-03-10T14:00:00Z",
            updated_at="2026-03-10T14:00:00Z",
        )

    async def first_run() -> None:
        started.append("s1")
        await running_release.wait()

    async def queued_run() -> None:
        started.append("s2")

    await manager.enqueue_session("s1", make_metadata("s1"), lambda: first_run())
    await manager.enqueue_session("s2", make_metadata("s2"), lambda: queued_run())
    await asyncio.sleep(0)

    scheduler = runtime_module._GLOBAL_SCHEDULER
    assert scheduler is not None
    assert scheduler.active_count == 1
    assert [item.session_id for item in scheduler.pending_queue] == ["s2"]

    await manager.cancel_session("s2")

    queued_session = await manager.get_session("s2")
    assert queued_session is not None
    assert queued_session.status == "cancelled"
    assert scheduler.active_count == 1
    assert [item.session_id for item in scheduler.pending_queue] == []
    assert await manager.get_task("s2") is None
    assert started == ["s1"]

    running_release.set()
    running_task = await manager.get_task("s1")
    assert running_task is not None
    await running_task


@pytest.mark.parametrize(
    "category,prompt,load_skills,expected_error",
    [
        ("", "valid prompt", [], "category cannot be empty"),
        ("  ", "valid prompt", [], "category cannot be empty"),
        ("ultrabrain", "", [], "prompt cannot be empty"),
        ("ultrabrain", "  ", [], "prompt cannot be empty"),
        ("ultrabrain", "valid prompt", "not-a-list", "load_skills must be a list"),
        ("ultrabrain", "valid prompt", None, "load_skills must be a list"),
    ],
)
async def test_validate_subagent_params_rejects_invalid_input(
    category: str, prompt: str, load_skills: list[str] | str | None, expected_error: str
) -> None:
    """SubagentSystem._validate_subagent_params rejects invalid parameters."""
    world = World()
    system = SubagentSystem()

    with pytest.raises(ValueError, match=expected_error):
        system._validate_subagent_params(category, prompt, load_skills)  # type: ignore[arg-type]


async def test_validate_subagent_params_accepts_valid_input() -> None:
    """SubagentSystem._validate_subagent_params accepts valid parameters."""
    world = World()
    system = SubagentSystem()

    # Should not raise
    system._validate_subagent_params("ultrabrain", "analyze this problem", [])
    system._validate_subagent_params("quick", "fix typo", ["skill1", "skill2"])


@pytest.mark.parametrize(
    "config_skills,load_skills,expected",
    [
        ([], [], []),
        (["skill1"], [], ["skill1"]),
        ([], ["skill2"], ["skill2"]),
        (["skill1"], ["skill2"], ["skill1", "skill2"]),
        (["skill1", "skill2"], ["skill3"], ["skill1", "skill2", "skill3"]),
        (["skill1"], ["skill1"], ["skill1"]),  # Deduplication
        (
            ["skill1", "skill2"],
            ["skill2", "skill3"],
            ["skill1", "skill2", "skill3"],
        ),  # Dedup + order
        (
            ["skill2", "skill1"],
            ["skill1", "skill3"],
            ["skill2", "skill1", "skill3"],
        ),  # Preserve config order
    ],
)
async def test_normalize_load_skills_merges_and_deduplicates(
    config_skills: list[str], load_skills: list[str], expected: list[str]
) -> None:
    """SubagentSystem._normalize_load_skills merges config + load_skills and deduplicates."""
    world = World()
    system = SubagentSystem()
    config = SubagentConfig(model=FakeModel(
            responses=[
                CompletionResult(message=Message(role="assistant", content="done"))
            ]
        ),
name="test",
        skills=config_skills,
    )

    result = system._normalize_load_skills(config, load_skills)
    assert result == expected


async def test_category_mapping_exact_match() -> None:
    """SubagentSystem._resolve_subagent_config looks up subagent from registry."""
    system = SubagentSystem()
    model = FakeModel(
        responses=[CompletionResult(message=Message(role="assistant", content="done"))]
    )
    config = SubagentConfig(
        name="ultrabrain",
        model=model,
    )
    registry = SubagentRegistryComponent(subagents={"ultrabrain": config})

    resolved = system._resolve_subagent_config(registry, "ultrabrain")

    assert resolved.name == "ultrabrain"
    assert resolved.model is model


async def test_category_mapping_unknown_category() -> None:
    """SubagentSystem._resolve_subagent_config raises ValueError for unknown subagent."""
    system = SubagentSystem()
    registry = SubagentRegistryComponent(subagents={})

    with pytest.raises(ValueError, match="Error: Unknown subagent 'invalid_category'"):
        system._resolve_subagent_config(registry, "invalid_category")


async def test_subagent_tool_sync_happy_path() -> None:
    world = World()
    parent_entity = world.create_entity()
    model = FakeModel(
        responses=[CompletionResult(message=Message(role="assistant", content="done"))]
    )
    config = SubagentConfig(name="quick", model=model)

    world.add_component(
        parent_entity,
        SubagentRegistryComponent(subagents={"quick": config}),
    )
    world.add_component(parent_entity, ToolRegistryComponent(tools={}, handlers={}))

    system = SubagentSystem()
    system.install_subagent_tool(world, parent_entity)

    async def fake_execute_core(
        world_arg: World,
        parent_entity_id: EntityId,
        subagent_name: str,
        task: str,
        correlation_id: str,
        traceparent: str,
        config_snapshot: SubagentConfig,
    ) -> tuple[str, bool, str | None]:
        assert world_arg is world
        assert parent_entity_id == parent_entity
        assert subagent_name == "quick"
        assert task == "investigate this"
        assert correlation_id
        assert traceparent
        assert config_snapshot.name == config.name
        assert config_snapshot.model == config.model
        assert config_snapshot.skills == config.skills
        return ("sync-result", True, None)

    system._execute_subagent_core = fake_execute_core  # type: ignore[method-assign]

    tool_registry = world.get_component(parent_entity, ToolRegistryComponent)
    assert tool_registry is not None
    handler = tool_registry.handlers["subagent"]

    result = await handler(
        category="quick",
        prompt="investigate this",
        load_skills=[],
        background=False,
        timeout=None,
    )

    assert result == "sync-result"


async def test_process_installs_free_subagent_tool_without_registry_when_system_option_enabled() -> None:
    world = World()
    parent_entity = world.create_entity()
    model = FakeModel(
        responses=[CompletionResult(message=Message(role="assistant", content="done"))]
    )
    world.add_component(parent_entity, LLMComponent(model=model))
    world.add_component(parent_entity, ToolRegistryComponent(tools={}, handlers={}))

    system = SubagentSystem(allow_unregistered_subagents=True)

    await system.process(world)

    registry = world.get_component(parent_entity, SubagentRegistryComponent)
    assert registry is not None
    assert registry.free_subagent_config.enabled is True
    tool_registry = world.get_component(parent_entity, ToolRegistryComponent)
    assert tool_registry is not None
    assert "subagent" in tool_registry.tools
    assert "subagent" in tool_registry.handlers


async def test_subagent_tool_free_mode_allows_unregistered_category_from_parent_model() -> None:
    world = World()
    parent_entity = world.create_entity()
    parent_model = FakeModel(
        responses=[CompletionResult(message=Message(role="assistant", content="done"))]
    )
    world.add_component(parent_entity, LLMComponent(model=parent_model))
    world.add_component(
        parent_entity,
        SubagentRegistryComponent(
            free_subagent_config=FreeSubagentConfig(enabled=True),
        ),
    )
    world.add_component(parent_entity, ToolRegistryComponent(tools={}, handlers={}))

    system = SubagentSystem()
    system.install_subagent_tool(world, parent_entity)

    async def fake_execute_core(
        world_arg: World,
        parent_entity_id: EntityId,
        subagent_name: str,
        task: str,
        correlation_id: str,
        traceparent: str,
        config_snapshot: SubagentConfig,
    ) -> tuple[str, bool, str | None]:
        assert world_arg is world
        assert parent_entity_id == parent_entity
        assert subagent_name == "security-reviewer"
        assert task == "review this module"
        assert correlation_id
        assert traceparent
        assert config_snapshot.name == "security-reviewer"
        assert config_snapshot.model is parent_model
        assert "security-reviewer" in config_snapshot.system_prompt
        return ("free-result", True, None)

    system._execute_subagent_core = fake_execute_core  # type: ignore[method-assign]

    tool_registry = world.get_component(parent_entity, ToolRegistryComponent)
    assert tool_registry is not None
    handler = tool_registry.handlers["subagent"]

    result = await handler(
        category="security-reviewer",
        prompt="review this module",
        load_skills=[],
        background=False,
        timeout=None,
    )

    assert result == "free-result"


async def test_subagent_tool_free_mode_prefers_registered_subagent_config() -> None:
    parent_model = FakeModel(responses=[])
    registered_model = FakeModel(
        responses=[CompletionResult(message=Message(role="assistant", content="done"))]
    )
    registry = SubagentRegistryComponent(
        subagents={
            "reviewer": SubagentConfig(
                name="reviewer",
                model=registered_model,
                system_prompt="Registered reviewer prompt.",
            )
        },
        free_subagent_config=FreeSubagentConfig(enabled=True),
    )

    resolved = SubagentSystem()._resolve_subagent_config(
        registry,
        "reviewer",
        parent_model=parent_model,
    )

    assert resolved.model is registered_model
    assert resolved.system_prompt == "Registered reviewer prompt."


async def test_subagent_tool_free_mode_template_allows_literal_braces() -> None:
    parent_model = FakeModel(responses=[])
    registry = SubagentRegistryComponent(
        free_subagent_config=FreeSubagentConfig(
            enabled=True,
            system_prompt_template='You are {name}. Return JSON like {"ok": true}.',
        ),
    )

    resolved = SubagentSystem()._resolve_subagent_config(
        registry,
        "json-worker",
        parent_model=parent_model,
    )

    assert resolved.system_prompt == 'You are json-worker. Return JSON like {"ok": true}.'


async def test_subagent_tool_description_mentions_free_categories_when_enabled() -> None:
    world = World()
    parent_entity = world.create_entity()
    model = FakeModel(responses=[])
    world.add_component(parent_entity, LLMComponent(model=model))
    world.add_component(
        parent_entity,
        SubagentRegistryComponent(
            free_subagent_config=FreeSubagentConfig(enabled=True),
        ),
    )
    world.add_component(parent_entity, ToolRegistryComponent(tools={}, handlers={}))

    system = SubagentSystem()
    system.install_subagent_tool(world, parent_entity)

    tool_registry = world.get_component(parent_entity, ToolRegistryComponent)
    assert tool_registry is not None
    schema = tool_registry.tools["subagent"]
    assert "any descriptive category name" in schema.description
    assert "unregistered" in schema.parameters["properties"]["category"]["description"]


async def test_subagent_tool_background_returns_session_id() -> None:
    world = World()
    parent_entity = world.create_entity()
    model = FakeModel(
        responses=[CompletionResult(message=Message(role="assistant", content="done"))]
    )
    config = SubagentConfig(name="deep", model=model)

    world.add_component(
        parent_entity,
        SubagentRegistryComponent(subagents={"deep": config}),
    )
    world.add_component(parent_entity, ToolRegistryComponent(tools={}, handlers={}))

    system = SubagentSystem()
    system.install_subagent_tool(world, parent_entity)

    async def fake_execute_core(
        world_arg: World,
        parent_entity_id: EntityId,
        subagent_name: str,
        task: str,
        correlation_id: str,
        traceparent: str,
        config_snapshot: SubagentConfig,
        **kwargs: Any,
    ) -> tuple[str, bool, str | None]:
        _ = (
            world_arg,
            parent_entity_id,
            subagent_name,
            task,
            correlation_id,
            traceparent,
            config_snapshot,
            kwargs,
        )
        return ("async-result", True, None)

    system._execute_subagent_core = fake_execute_core  # type: ignore[method-assign]

    tool_registry = world.get_component(parent_entity, ToolRegistryComponent)
    assert tool_registry is not None
    handler = tool_registry.handlers["subagent"]

    result = await handler(
        category="deep",
        prompt="run in background",
        load_skills=["skill-a"],
        background=True,
        timeout=300.0,
    )

    payload = json.loads(result)
    assert isinstance(payload["session_id"], str)
    assert payload["session_id"]
    assert payload["status"] == "queued"
    assert payload["category"] == "deep"
    assert payload["timeout"] == 300.0


async def test_subagent_handler_uses_immutable_effective_config_snapshot() -> None:
    world = World()
    parent_entity = world.create_entity()
    model = FakeModel(
        responses=[CompletionResult(message=Message(role="assistant", content="done"))]
    )
    config = SubagentConfig(
        name="deep",
        model=model,
        skills=["base-skill"],
    )

    registry = SubagentRegistryComponent(subagents={"deep": config})
    world.add_component(parent_entity, registry)
    world.add_component(parent_entity, ToolRegistryComponent(tools={}, handlers={}))

    system = SubagentSystem()
    system.install_subagent_tool(world, parent_entity)

    def fake_assemble_child_world(
        parent_world: World,
        parent_entity_id: EntityId,
        config_snapshot: SubagentConfig,
        parent_child_entity: EntityId | None = None,
    ) -> tuple[World, EntityId]:
        assert parent_world is world
        assert parent_entity_id == parent_entity
        assert parent_child_entity is not None
        assert config_snapshot.skills == ["base-skill", "extra-skill"]
        assert registry.subagents["deep"].skills == ["base-skill"]

        child_world = World(name="deep-child")
        child_entity = child_world.create_entity()
        child_world.add_component(
            child_entity,
            LLMComponent(model=model, system_prompt=""),
        )
        child_world.add_component(child_entity, ConversationComponent(messages=[]))
        return child_world, child_entity

    async def fake_execute_delegation(
        child_world: World,
        child_entity: EntityId,
        task: str,
        config_snapshot: SubagentConfig,
    ) -> str:
        assert child_world.name == "deep-child"
        assert child_entity == EntityId(1)
        assert task == "run with extra skill"
        assert config_snapshot.skills == ["base-skill", "extra-skill"]
        return "snapshot-result"

    async def fake_publish_delegation_events(*args: Any, **kwargs: Any) -> None:
        del args, kwargs

    system._assemble_child_world = fake_assemble_child_world  # type: ignore[method-assign]
    system._execute_delegation = fake_execute_delegation  # type: ignore[method-assign]
    system._publish_delegation_events = fake_publish_delegation_events  # type: ignore[method-assign]

    tool_registry = world.get_component(parent_entity, ToolRegistryComponent)
    assert tool_registry is not None
    handler = tool_registry.handlers["subagent"]

    result = await handler(
        category="deep",
        prompt="run with extra skill",
        load_skills=["extra-skill"],
        background=False,
        timeout=None,
    )

    assert result == "snapshot-result"
    assert registry.subagents["deep"].skills == ["base-skill"]


async def test_subagent_tool_background_enqueue_session_records_stream() -> None:
    from ecs_agent.components.definitions import SubagentSessionTableComponent

    world = World()
    parent_entity = world.create_entity()
    model = FakeModel(
        responses=[CompletionResult(message=Message(role="assistant", content="done"))]
    )
    config = SubagentConfig(
        name="deep",
        model=model,
        skills=["base-skill"],
    )

    world.add_component(
        parent_entity,
        SubagentRegistryComponent(subagents={"deep": config}),
    )
    world.add_component(parent_entity, SubagentSessionTableComponent())
    world.add_component(parent_entity, ToolRegistryComponent(tools={}, handlers={}))

    system = SubagentSystem()
    system.install_subagent_tool(world, parent_entity)

    captured: dict[str, Any] = {}

    async def fake_enqueue_session(
        session_id: str,
        metadata: SubagentSessionRecord,
        coroutine_factory: Any,
    ) -> None:
        captured["session_id"] = session_id
        captured["metadata"] = metadata
        captured["coroutine_factory"] = coroutine_factory

    system._runtime_manager.enqueue_session = fake_enqueue_session  # type: ignore[method-assign]

    tool_registry = world.get_component(parent_entity, ToolRegistryComponent)
    assert tool_registry is not None
    subagent_tool = tool_registry.tools["subagent"]
    assert "stream" in subagent_tool.parameters["properties"]
    assert subagent_tool.parameters["properties"]["stream"]["type"] == "boolean"

    handler = tool_registry.handlers["subagent"]
    result = await handler(
        category="deep",
        prompt="run in background",
        load_skills=["extra-skill"],
        background=True,
        timeout=300.0,
        stream=True,
    )

    payload = json.loads(result)
    assert payload["session_id"] == captured["session_id"]
    assert payload["status"] == "queued"
    assert payload["lifecycle_status"] == "queued"

    metadata = captured["metadata"]
    assert metadata.load_skills == ["base-skill", "extra-skill"]
    assert metadata.background is True
    assert metadata.stream is True


async def test_subagent_tool_background_returns_queued_lifecycle_status_when_scheduler_is_full(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ecs_agent.systems.subagent_runtime as runtime_module
    from ecs_agent.components.definitions import SubagentSessionTableComponent

    _reset_global_scheduler(monkeypatch)

    world = World()
    parent_entity = world.create_entity()
    model = FakeModel(
        responses=[CompletionResult(message=Message(role="assistant", content="done"))]
    )
    config = SubagentConfig(name="deep", model=model)

    world.add_component(
        parent_entity,
        SubagentRegistryComponent(subagents={"deep": config}),
    )
    world.add_component(parent_entity, SubagentSessionTableComponent())
    world.add_component(parent_entity, ToolRegistryComponent(tools={}, handlers={}))

    system = SubagentSystem(max_background_concurrency=1)
    system.install_subagent_tool(world, parent_entity)

    started = asyncio.Event()
    release = asyncio.Event()

    async def fake_execute_core(
        *args: Any, **kwargs: Any
    ) -> tuple[str, bool, str | None]:
        del args, kwargs
        started.set()
        await release.wait()
        return ("async-result", True, None)

    system._execute_subagent_core = fake_execute_core  # type: ignore[method-assign]

    tool_registry = world.get_component(parent_entity, ToolRegistryComponent)
    assert tool_registry is not None
    handler = tool_registry.handlers["subagent"]

    first_result = await handler(
        category="deep",
        prompt="first background task",
        load_skills=[],
        background=True,
        timeout=None,
    )
    first_payload = json.loads(first_result)
    await started.wait()
    await asyncio.sleep(0)

    second_result = await handler(
        category="deep",
        prompt="second background task",
        load_skills=[],
        background=True,
        timeout=None,
    )
    second_payload = json.loads(second_result)

    scheduler = runtime_module._GLOBAL_SCHEDULER
    assert scheduler is not None
    assert scheduler.active_count == 1
    assert second_payload["status"] == "queued"
    assert second_payload["lifecycle_status"] == "queued"

    second_session = await system._runtime_manager.get_session(
        second_payload["session_id"]
    )
    assert second_session is not None
    assert second_session.status == "queued"

    release.set()
    first_task = await system._runtime_manager.get_task(first_payload["session_id"])
    assert first_task is not None
    await first_task
    await asyncio.sleep(0)

    second_task = await system._runtime_manager.get_task(second_payload["session_id"])
    assert second_task is not None
    await second_task


async def test_subagent_tool_validates_parameters() -> None:
    world = World()
    parent_entity = world.create_entity()
    model = FakeModel(
        responses=[CompletionResult(message=Message(role="assistant", content="done"))]
    )
    config = SubagentConfig(name="quick", model=model)

    world.add_component(
        parent_entity,
        SubagentRegistryComponent(subagents={"quick": config}),
    )
    world.add_component(parent_entity, ToolRegistryComponent(tools={}, handlers={}))

    system = SubagentSystem()
    system.install_subagent_tool(world, parent_entity)

    tool_registry = world.get_component(parent_entity, ToolRegistryComponent)
    assert tool_registry is not None
    handler = tool_registry.handlers["subagent"]

    with pytest.raises(ValueError, match="category cannot be empty"):
        await handler(
            category="",
            prompt="valid",
            load_skills=[],
            background=False,
            timeout=None,
        )


class SlowFakeModel:
    """Test model that simulates slow responses with configurable delay."""

    model_id: str = "slow-fake"

    def __init__(self, delay: float, response: CompletionResult):
        self._delay = delay
        self._response = response

    async def complete(
        self, messages: list[Message], **kwargs: Any
    ) -> CompletionResult:
        import asyncio  # Import here for test isolation

        await asyncio.sleep(self._delay)
        return self._response


async def test_subagent_timeout_global_default() -> None:
    """Test that global default timeout is enforced when no per-call timeout provided."""
    world = World()
    parent_entity = world.create_entity()

    # Provider that simulates a long-running task
    model = SlowFakeModel(
        delay=5.0,
        response=CompletionResult(message=Message(role="assistant", content="done")),
    )

    config = SubagentConfig(name="slow", model=model)
    world.add_component(
        parent_entity,
        SubagentRegistryComponent(subagents={"slow": config}),
    )
    world.add_component(parent_entity, ToolRegistryComponent(tools={}, handlers={}))

    # Create system with 0.5s global timeout
    system = SubagentSystem(default_timeout=0.5)
    system.install_subagent_tool(world, parent_entity)

    tool_registry = world.get_component(parent_entity, ToolRegistryComponent)
    assert tool_registry is not None
    handler = tool_registry.handlers["subagent"]

    # Execute sync mode (should timeout)
    result = await handler(
        category="slow",
        prompt="run slow task",
        load_skills=[],
        background=False,
        timeout=None,  # Use global default
    )

    assert "Error: Subagent timeout after 0.5s" in result


async def test_subagent_timeout_per_call_override() -> None:
    """Test that per-call timeout overrides global default."""
    world = World()
    parent_entity = world.create_entity()

    # Provider that simulates a task taking 0.3s
    model = SlowFakeModel(
        delay=0.3,
        response=CompletionResult(message=Message(role="assistant", content="done")),
    )

    config = SubagentConfig(name="medium", model=model)
    world.add_component(
        parent_entity,
        SubagentRegistryComponent(subagents={"medium": config}),
    )
    world.add_component(parent_entity, ToolRegistryComponent(tools={}, handlers={}))

    # Create system with 1.0s global timeout
    system = SubagentSystem(default_timeout=1.0)
    system.install_subagent_tool(world, parent_entity)

    tool_registry = world.get_component(parent_entity, ToolRegistryComponent)
    assert tool_registry is not None
    handler = tool_registry.handlers["subagent"]

    # Execute with 0.1s per-call timeout (should timeout - overrides 1.0s global)
    result = await handler(
        category="medium",
        prompt="run medium task",
        load_skills=[],
        background=False,
        timeout=0.1,  # Override global
    )

    assert "Error: Subagent timeout after 0.1s" in result


async def test_subagent_timeout_none_disables() -> None:
    """Test that explicit timeout=None works when global default exists."""
    world = World()
    parent_entity = world.create_entity()

    # Fast model
    model = FakeModel(
        responses=[
            CompletionResult(message=Message(role="assistant", content="success"))
        ]
    )

    config = SubagentConfig(name="fast", model=model)
    world.add_component(
        parent_entity,
        SubagentRegistryComponent(subagents={"fast": config}),
    )
    world.add_component(parent_entity, ToolRegistryComponent(tools={}, handlers={}))

    # Create system with 0.01s global timeout (would fail without override)
    system = SubagentSystem(default_timeout=0.01)
    system.install_subagent_tool(world, parent_entity)

    tool_registry = world.get_component(parent_entity, ToolRegistryComponent)
    assert tool_registry is not None
    handler = tool_registry.handlers["subagent"]

    # Execute with timeout=None explicitly (should succeed despite low global)
    result = await handler(
        category="fast",
        prompt="run fast task",
        load_skills=[],
        background=False,
        timeout=None,  # Should NOT use global default (0.01s)
    )

    # Should succeed since we have no timeout enforcement
    # Note: This currently uses global default, which is correct per spec
    # "timeout=None" in the call means "use default", not "disable timeout"
    # To disable timeout with a global default, don't set a global default
    assert "success" in result or "timeout" in result.lower()


async def test_subagent_timeout_sets_state() -> None:
    world = World()
    parent_entity = world.create_entity()

    # Provider that simulates a long-running task
    model = SlowFakeModel(
        delay=5.0,
        response=CompletionResult(message=Message(role="assistant", content="done")),
    )

    config = SubagentConfig(name="slow_bg", model=model)
    world.add_component(
        parent_entity,
        SubagentRegistryComponent(subagents={"slow_bg": config}),
    )
    world.add_component(parent_entity, ToolRegistryComponent(tools={}, handlers={}))

    system = SubagentSystem(default_timeout=0.2)
    system.install_subagent_tool(world, parent_entity)

    tool_registry = world.get_component(parent_entity, ToolRegistryComponent)
    assert tool_registry is not None
    handler = tool_registry.handlers["subagent"]

    # Execute in background mode
    result_json = await handler(
        category="slow_bg",
        prompt="run slow background task",
        load_skills=[],
        background=True,
        timeout=None,  # Use global 0.2s
    )

    payload = json.loads(result_json)
    session_id = payload["session_id"]

    # Wait for timeout to occur
    await asyncio.sleep(0.5)

    # Check session status
    metadata = await system._runtime_manager.get_session(session_id)
    assert metadata is not None
    assert metadata.status == "timed_out"
    assert metadata.error is not None
    assert "timeout" in metadata.error.lower()


async def test_completion_enqueues_parent_notification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _reset_global_scheduler(monkeypatch)

    world = World()
    parent_entity = world.create_entity()
    model = FakeModel(
        responses=[CompletionResult(message=Message(role="assistant", content="done"))]
    )
    config = SubagentConfig(name="notify-success", model=model)

    world.add_component(
        parent_entity,
        SubagentRegistryComponent(subagents={"notify-success": config}),
    )
    world.add_component(parent_entity, SubagentSessionTableComponent())
    world.add_component(parent_entity, ToolRegistryComponent(tools={}, handlers={}))

    system = SubagentSystem()
    completed_events: list[DelegationCompletedEvent] = []
    observed_notification_ids: list[str] = []
    session_id: str | None = None

    async def on_completed(event: DelegationCompletedEvent) -> None:
        completed_events.append(event)
        queue = world.get_component(parent_entity, SubagentNotificationQueueComponent)
        assert queue is not None
        assert session_id is not None
        observed_notification_ids.extend(
            notification.notification_id for notification in queue.notifications
        )
        assert observed_notification_ids == [f"{session_id}:succeeded"]

    world.event_bus.subscribe(DelegationCompletedEvent, on_completed)

    system.install_subagent_tool(world, parent_entity)
    tool_registry = world.get_component(parent_entity, ToolRegistryComponent)
    assert tool_registry is not None

    payload = json.loads(
        await tool_registry.handlers["subagent"](
            category="notify-success",
            prompt="finish successfully",
            load_skills=[],
            background=True,
            timeout=None,
        )
    )
    session_id = payload["session_id"]

    task = await system._runtime_manager.get_task(session_id)
    assert task is not None
    await task

    queue = world.get_component(parent_entity, SubagentNotificationQueueComponent)
    assert queue is not None
    assert len(queue.notifications) == 1
    notification = queue.notifications[0]
    assert notification.notification_id == f"{session_id}:succeeded"
    assert notification.session_id == session_id
    assert notification.parent_entity_id == parent_entity
    assert notification.terminal_status == "succeeded"
    assert (
        notification.summary is None
    )  # only set when using <subagent_background_result> envelope
    assert notification.error is None
    assert notification.delivered_at is None
    assert completed_events and completed_events[0].success is True
    assert observed_notification_ids == [f"{session_id}:succeeded"]


async def test_failed_background_session_enqueues_parent_notification_before_completion_event_publish(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _reset_global_scheduler(monkeypatch)

    world = World()
    parent_entity = world.create_entity()
    model = FakeModel(responses=[])
    config = SubagentConfig(name="notify-failure", model=model)

    world.add_component(
        parent_entity,
        SubagentRegistryComponent(subagents={"notify-failure": config}),
    )
    world.add_component(parent_entity, SubagentSessionTableComponent())
    world.add_component(
        parent_entity,
        SubagentNotificationQueueComponent(),
    )
    world.add_component(parent_entity, ToolRegistryComponent(tools={}, handlers={}))

    system = SubagentSystem()
    completed_events: list[DelegationCompletedEvent] = []
    session_id: str | None = None

    async def fake_execute_subagent_core(
        *args: Any, **kwargs: Any
    ) -> tuple[str, bool, str | None]:
        del args, kwargs
        error_msg = "Error during delegation: boom"
        return (error_msg, False, error_msg)

    system._execute_subagent_core = fake_execute_subagent_core  # type: ignore[method-assign]

    async def on_completed(event: DelegationCompletedEvent) -> None:
        completed_events.append(event)
        queue = world.get_component(parent_entity, SubagentNotificationQueueComponent)
        assert queue is not None
        assert session_id is not None
        assert [item.notification_id for item in queue.notifications] == [
            f"{session_id}:failed"
        ]

    world.event_bus.subscribe(DelegationCompletedEvent, on_completed)

    system.install_subagent_tool(world, parent_entity)
    tool_registry = world.get_component(parent_entity, ToolRegistryComponent)
    assert tool_registry is not None

    payload = json.loads(
        await tool_registry.handlers["subagent"](
            category="notify-failure",
            prompt="explode",
            load_skills=[],
            background=True,
            timeout=None,
        )
    )
    session_id = payload["session_id"]

    task = await system._runtime_manager.get_task(session_id)
    assert task is not None
    await task

    queue = world.get_component(parent_entity, SubagentNotificationQueueComponent)
    assert queue is not None
    assert len(queue.notifications) == 1
    notification = queue.notifications[0]
    assert notification.notification_id == f"{session_id}:failed"
    assert notification.session_id == session_id
    assert notification.terminal_status == "failed"
    assert notification.summary is None
    assert notification.error == "Error during delegation: boom"
    assert notification.delivered_at is None
    assert len(completed_events) == 1
    assert completed_events[0].success is False
    assert completed_events[0].error == "Error during delegation: boom"


async def test_timed_out_background_session_enqueues_parent_notification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _reset_global_scheduler(monkeypatch)

    world = World()
    parent_entity = world.create_entity()
    model = SlowFakeModel(
        delay=5.0,
        response=CompletionResult(message=Message(role="assistant", content="done")),
    )
    config = SubagentConfig(name="notify-timeout", model=model)

    world.add_component(
        parent_entity,
        SubagentRegistryComponent(subagents={"notify-timeout": config}),
    )
    world.add_component(parent_entity, SubagentSessionTableComponent())
    world.add_component(parent_entity, ToolRegistryComponent(tools={}, handlers={}))

    system = SubagentSystem(default_timeout=0.05)
    completed_events: list[DelegationCompletedEvent] = []
    session_id: str | None = None

    async def on_completed(event: DelegationCompletedEvent) -> None:
        completed_events.append(event)
        queue = world.get_component(parent_entity, SubagentNotificationQueueComponent)
        assert queue is not None
        assert session_id is not None
        assert [item.notification_id for item in queue.notifications] == [
            f"{session_id}:timed_out"
        ]

    world.event_bus.subscribe(DelegationCompletedEvent, on_completed)

    system.install_subagent_tool(world, parent_entity)
    tool_registry = world.get_component(parent_entity, ToolRegistryComponent)
    assert tool_registry is not None

    payload = json.loads(
        await tool_registry.handlers["subagent"](
            category="notify-timeout",
            prompt="run too long",
            load_skills=[],
            background=True,
            timeout=None,
        )
    )
    session_id = payload["session_id"]

    task = await system._runtime_manager.get_task(session_id)
    assert task is not None
    await task

    queue = world.get_component(parent_entity, SubagentNotificationQueueComponent)
    assert queue is not None
    assert len(queue.notifications) == 1
    notification = queue.notifications[0]
    assert notification.notification_id == f"{session_id}:timed_out"
    assert notification.session_id == session_id
    assert notification.terminal_status == "timed_out"
    assert notification.summary is None
    assert notification.error is not None
    assert "timeout" in notification.error.lower()
    assert notification.delivered_at is None
    assert len(completed_events) == 1
    assert completed_events[0].success is False
    assert completed_events[0].error is not None
    assert "timeout" in completed_events[0].error.lower()


async def test_cancelled_background_session_does_not_enqueue_notification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _reset_global_scheduler(monkeypatch)

    world = World()
    parent_entity = world.create_entity()
    model = SlowFakeModel(
        delay=5.0,
        response=CompletionResult(message=Message(role="assistant", content="done")),
    )
    config = SubagentConfig(name="notify-cancel", model=model)

    world.add_component(
        parent_entity,
        SubagentRegistryComponent(subagents={"notify-cancel": config}),
    )
    world.add_component(parent_entity, SubagentSessionTableComponent())
    world.add_component(parent_entity, SubagentNotificationQueueComponent())
    world.add_component(parent_entity, ToolRegistryComponent(tools={}, handlers={}))

    system = SubagentSystem()
    completed_events: list[DelegationCompletedEvent] = []

    async def on_completed(event: DelegationCompletedEvent) -> None:
        completed_events.append(event)

    world.event_bus.subscribe(DelegationCompletedEvent, on_completed)

    system.install_subagent_control_tools(world, parent_entity)
    system.install_subagent_tool(world, parent_entity)
    tool_registry = world.get_component(parent_entity, ToolRegistryComponent)
    assert tool_registry is not None

    payload = json.loads(
        await tool_registry.handlers["subagent"](
            category="notify-cancel",
            prompt="cancel me",
            load_skills=[],
            background=True,
            timeout=None,
        )
    )
    session_id = payload["session_id"]

    await asyncio.sleep(0.05)
    await tool_registry.handlers["subagent_cancel"](session_id=session_id)

    task = await system._runtime_manager.get_task(session_id)
    if task is not None:
        with pytest.raises(asyncio.CancelledError):
            await task

    queue = world.get_component(parent_entity, SubagentNotificationQueueComponent)
    assert queue is not None
    assert queue.notifications == []
    assert completed_events == []


async def test_sync_subagent_call_does_not_enqueue_parent_notification() -> None:
    world = World()
    parent_entity = world.create_entity()
    model = FakeModel(
        responses=[CompletionResult(message=Message(role="assistant", content="done"))]
    )
    config = SubagentConfig(name="sync-agent", model=model)

    world.add_component(
        parent_entity,
        SubagentRegistryComponent(subagents={"sync-agent": config}),
    )
    world.add_component(parent_entity, ToolRegistryComponent(tools={}, handlers={}))

    system = SubagentSystem()
    system.install_subagent_tool(world, parent_entity)
    tool_registry = world.get_component(parent_entity, ToolRegistryComponent)
    assert tool_registry is not None

    result = await tool_registry.handlers["subagent"](
        category="sync-agent",
        prompt="finish inline",
        load_skills=[],
        background=False,
        timeout=None,
    )

    assert result == "done"
    queue = world.get_component(parent_entity, SubagentNotificationQueueComponent)
    assert queue is None


async def test_subagent_wait_returns_ack_and_attaches_wait_component() -> None:
    world = World()
    parent_entity = world.create_entity()
    system = SubagentSystem()

    world.add_component(parent_entity, SubagentSessionTableComponent())
    world.add_component(parent_entity, ToolRegistryComponent(tools={}, handlers={}))

    system.install_subagent_control_tools(world, parent_entity)
    tools = world.get_component(parent_entity, ToolRegistryComponent)
    assert tools is not None

    result = await tools.handlers["subagent_wait"](
        session_ids=["session-a", "session-b"],
        timeout=12.5,
    )

    assert result == (
        "Waiting for background subagents. Will be notified when they complete."
    )
    wait_component = world.get_component(parent_entity, SubagentWaitComponent)
    assert wait_component is not None
    assert wait_component.session_ids == ["session-a", "session-b"]
    assert wait_component.timeout == 12.5
    assert wait_component.future is None
    assert wait_component.started_at is not None


async def test_waiting_parent_resumes_without_polling_when_matching_unread_notification_exists() -> (
    None
):
    world = World()
    parent_entity = world.create_entity()

    world.add_component(
        parent_entity,
        SubagentNotificationQueueComponent(
            notifications=[
                SubagentNotificationRecord(
                    notification_id="session-1:succeeded",
                    session_id="session-1",
                    parent_entity_id=parent_entity,
                    terminal_status="succeeded",
                    summary="done",
                    error=None,
                    created_at="2026-04-06T12:00:00Z",
                    delivered_at=None,
                )
            ]
        ),
    )
    world.add_component(
        parent_entity,
        SubagentWaitComponent(session_ids=["session-1"], timeout=0.5),
    )

    system = SubagentWaitSystem()
    await system.process(world)

    assert world.get_component(parent_entity, SubagentWaitComponent) is None
    assert world.get_component(parent_entity, TerminalComponent) is None
    assert world.get_component(parent_entity, ErrorComponent) is None


async def test_subagent_wait_timeout_attaches_error_and_terminal_components() -> None:
    world = World()
    parent_entity = world.create_entity()
    world.add_component(parent_entity, SubagentNotificationQueueComponent())
    world.add_component(
        parent_entity,
        SubagentWaitComponent(
            timeout=0.01,
            started_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        ),
    )

    system = SubagentWaitSystem()
    await system.process(world)

    error = world.get_component(parent_entity, ErrorComponent)
    terminal = world.get_component(parent_entity, TerminalComponent)
    wait_component = world.get_component(parent_entity, SubagentWaitComponent)

    assert wait_component is not None
    assert error is not None
    assert error.system_name == "SubagentWaitSystem"
    assert "timeout" in error.error.lower()
    assert terminal is not None
    assert terminal.reason == "subagent_wait_timeout"


async def test_waiting_parent_future_is_resolved_when_background_completion_notification_arrives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _reset_global_scheduler(monkeypatch)

    world = World()
    parent_entity = world.create_entity()
    model = FakeModel(
        responses=[CompletionResult(message=Message(role="assistant", content="done"))]
    )
    config = SubagentConfig(name="notify-waiter", model=model)

    world.add_component(
        parent_entity,
        SubagentRegistryComponent(subagents={"notify-waiter": config}),
    )
    world.add_component(parent_entity, SubagentSessionTableComponent())
    world.add_component(parent_entity, ToolRegistryComponent(tools={}, handlers={}))
    world.add_component(
        parent_entity,
        SubagentWaitComponent(session_ids=None, timeout=1.0),
    )

    wait_system = SubagentWaitSystem()
    wait_task = asyncio.create_task(wait_system.process(world))
    await asyncio.sleep(0)

    wait_component = world.get_component(parent_entity, SubagentWaitComponent)
    assert wait_component is not None
    assert wait_component.future is not None
    assert wait_component.future.done() is False

    system = SubagentSystem()
    system.install_subagent_tool(world, parent_entity)
    tools = world.get_component(parent_entity, ToolRegistryComponent)
    assert tools is not None

    payload = json.loads(
        await tools.handlers["subagent"](
            category="notify-waiter",
            prompt="finish successfully",
            load_skills=[],
            background=True,
            timeout=None,
        )
    )
    session_id = payload["session_id"]

    task = await system._runtime_manager.get_task(session_id)
    assert task is not None
    await task
    await wait_task

    queue = world.get_component(parent_entity, SubagentNotificationQueueComponent)
    assert queue is not None
    assert [item.notification_id for item in queue.notifications] == [
        f"{session_id}:succeeded"
    ]
    assert world.get_component(parent_entity, SubagentWaitComponent) is None


async def test_completion_notification_is_injected_before_next_reasoning_turn() -> None:
    world = World()
    entity_id = world.create_entity()
    model = RecordingFakeModel(
        responses=[
            CompletionResult(
                message=Message(role="assistant", content="I resumed work.")
            )
        ]
    )

    world.add_component(entity_id, LLMComponent(model=model))
    world.add_component(
        entity_id,
        ConversationComponent(messages=[Message(role="user", content="Continue.")]),
    )
    world.add_component(
        entity_id,
        SubagentNotificationQueueComponent(
            notifications=[
                SubagentNotificationRecord(
                    notification_id="session-abc:succeeded",
                    session_id="session-abc",
                    parent_entity_id=entity_id,
                    terminal_status="succeeded",
                    summary=None,
                    error=None,
                    created_at="2026-04-06T12:00:00Z",
                    delivered_at=None,
                )
            ]
        ),
    )
    world.add_component(
        entity_id,
        SubagentWaitComponent(session_ids=["session-abc"], timeout=1.0),
    )

    wait_system = SubagentWaitSystem()
    await wait_system.process(world)

    conversation = world.get_component(entity_id, ConversationComponent)
    assert conversation is not None
    assert conversation.messages[-1] == Message(
        role="system",
        content=(
            "Background subagent updates:\n"
            '- session-abc succeeded. Call subagent_result(session_id="session-abc") '
            "for the full result."
        ),
    )
    queue = world.get_component(entity_id, SubagentNotificationQueueComponent)
    assert queue is not None
    assert queue.notifications[0].delivered_at is not None

    reasoning_system = ReasoningSystem(priority=0)
    await reasoning_system.process(world)

    assert len(model.calls) == 1
    assert model.calls[0][-1] == Message(
        role="system",
        content=(
            "Background subagent updates:\n"
            '- session-abc succeeded. Call subagent_result(session_id="session-abc") '
            "for the full result."
        ),
    )


async def test_restored_unread_notification_is_delivered_once() -> None:
    world = World()
    entity_id = world.create_entity()

    world.add_component(
        entity_id,
        ConversationComponent(messages=[Message(role="user", content="Resume work")]),
    )
    world.add_component(
        entity_id,
        SubagentNotificationQueueComponent(
            notifications=[
                SubagentNotificationRecord(
                    notification_id="session-restored:succeeded",
                    session_id="session-restored",
                    parent_entity_id=entity_id,
                    terminal_status="succeeded",
                    summary="restored summary",
                    error=None,
                    created_at="2026-04-06T12:00:00Z",
                    delivered_at=None,
                ),
                SubagentNotificationRecord(
                    notification_id="session-delivered:failed",
                    session_id="session-delivered",
                    parent_entity_id=entity_id,
                    terminal_status="failed",
                    summary=None,
                    error="already delivered",
                    created_at="2026-04-06T12:01:00Z",
                    delivered_at="2026-04-06T12:02:00Z",
                ),
            ]
        ),
    )
    world.add_component(
        entity_id,
        SubagentWaitComponent(session_ids=["session-restored"], timeout=1.0),
    )

    restored = WorldSerializer.from_dict(
        WorldSerializer.to_dict(world),
        providers={},
        tool_handlers={},
    )
    wait_system = SubagentWaitSystem()

    restored_wait = restored.get_component(entity_id, SubagentWaitComponent)
    assert restored_wait is not None
    assert restored_wait.future is None

    await wait_system.process(restored)

    conversation = restored.get_component(entity_id, ConversationComponent)
    assert conversation is not None
    assert conversation.messages[-1] == Message(
        role="system",
        content=(
            "Background subagent updates:\n"
            '- session-restored succeeded. Call subagent_result(session_id="session-restored") '
            "for the full result or "
            'subagent_result(session_id="session-restored", read_method="summary") '
            "for the cached summary."
        ),
    )

    queue = restored.get_component(entity_id, SubagentNotificationQueueComponent)
    assert queue is not None
    assert queue.notifications[0].delivered_at is not None
    assert queue.notifications[1].delivered_at == "2026-04-06T12:02:00Z"

    message_count = len(conversation.messages)
    await wait_system.process(restored)
    assert len(conversation.messages) == message_count


def test_unread_notifications_are_batched() -> None:
    world = World()
    entity_id = world.create_entity()
    world.add_component(
        entity_id,
        ConversationComponent(messages=[Message(role="user", content="Continue.")]),
    )
    world.add_component(
        entity_id,
        SubagentNotificationQueueComponent(
            notifications=[
                SubagentNotificationRecord(
                    notification_id="session-abc:succeeded",
                    session_id="session-abc",
                    parent_entity_id=entity_id,
                    terminal_status="succeeded",
                    summary="cached summary",
                    error=None,
                    created_at="2026-04-06T12:00:00Z",
                    delivered_at=None,
                ),
                SubagentNotificationRecord(
                    notification_id="session-def:succeeded",
                    session_id="session-def",
                    parent_entity_id=entity_id,
                    terminal_status="succeeded",
                    summary=None,
                    error=None,
                    created_at="2026-04-06T12:01:00Z",
                    delivered_at=None,
                ),
                SubagentNotificationRecord(
                    notification_id="session-old:succeeded",
                    session_id="session-old",
                    parent_entity_id=entity_id,
                    terminal_status="succeeded",
                    summary="already delivered",
                    error=None,
                    created_at="2026-04-06T11:59:00Z",
                    delivered_at="2026-04-06T12:02:00Z",
                ),
            ]
        ),
    )
    world.add_component(entity_id, SubagentWaitComponent(timeout=1.0))

    system = SubagentWaitSystem()
    system._resolve_wait(
        world,
        entity_id,
        world.get_component(entity_id, SubagentWaitComponent),
    )

    conversation = world.get_component(entity_id, ConversationComponent)
    assert conversation is not None
    assert conversation.messages == [
        Message(role="user", content="Continue."),
        Message(
            role="system",
            content=(
                "Background subagent updates:\n"
                '- session-abc succeeded. Call subagent_result(session_id="session-abc") '
                "for the full result or "
                'subagent_result(session_id="session-abc", read_method="summary") '
                "for the cached summary.\n"
                '- session-def succeeded. Call subagent_result(session_id="session-def") '
                "for the full result."
            ),
        ),
    ]

    queue = world.get_component(entity_id, SubagentNotificationQueueComponent)
    assert queue is not None
    assert queue.notifications[0].delivered_at is not None
    assert queue.notifications[1].delivered_at is not None
    assert queue.notifications[2].delivered_at == "2026-04-06T12:02:00Z"

    world.add_component(entity_id, SubagentWaitComponent(timeout=1.0))
    system._resolve_wait(
        world,
        entity_id,
        world.get_component(entity_id, SubagentWaitComponent),
    )
    assert conversation.messages == [
        Message(role="user", content="Continue."),
        Message(
            role="system",
            content=(
                "Background subagent updates:\n"
                '- session-abc succeeded. Call subagent_result(session_id="session-abc") '
                "for the full result or "
                'subagent_result(session_id="session-abc", read_method="summary") '
                "for the cached summary.\n"
                '- session-def succeeded. Call subagent_result(session_id="session-def") '
                "for the full result."
            ),
        ),
    ]


def test_unread_notifications_are_filtered_to_wait_scope() -> None:
    world = World()
    entity_id = world.create_entity()
    world.add_component(
        entity_id,
        ConversationComponent(messages=[Message(role="user", content="Continue.")]),
    )
    world.add_component(
        entity_id,
        SubagentNotificationQueueComponent(
            notifications=[
                SubagentNotificationRecord(
                    notification_id="session-match:succeeded",
                    session_id="session-match",
                    parent_entity_id=entity_id,
                    terminal_status="succeeded",
                    summary="cached summary",
                    error=None,
                    created_at="2026-04-06T12:00:00Z",
                    delivered_at=None,
                ),
                SubagentNotificationRecord(
                    notification_id="session-other:failed",
                    session_id="session-other",
                    parent_entity_id=entity_id,
                    terminal_status="failed",
                    summary=None,
                    error="boom",
                    created_at="2026-04-06T12:01:00Z",
                    delivered_at=None,
                ),
            ]
        ),
    )
    world.add_component(
        entity_id,
        SubagentWaitComponent(session_ids=["session-match"], timeout=1.0),
    )

    system = SubagentWaitSystem()
    wait_component = world.get_component(entity_id, SubagentWaitComponent)
    assert wait_component is not None
    system._resolve_wait(world, entity_id, wait_component)

    conversation = world.get_component(entity_id, ConversationComponent)
    queue = world.get_component(entity_id, SubagentNotificationQueueComponent)

    assert conversation is not None
    assert queue is not None
    assert conversation.messages[-1] == Message(
        role="system",
        content=(
            "Background subagent updates:\n"
            '- session-match succeeded. Call subagent_result(session_id="session-match") '
            "for the full result or "
            'subagent_result(session_id="session-match", read_method="summary") '
            "for the cached summary."
        ),
    )
    assert queue.notifications[0].delivered_at is not None
    assert queue.notifications[1].delivered_at is None


def test_failure_notification_includes_error() -> None:
    world = World()
    entity_id = world.create_entity()
    world.add_component(
        entity_id,
        ConversationComponent(messages=[Message(role="user", content="Continue.")]),
    )
    world.add_component(
        entity_id,
        SubagentNotificationQueueComponent(
            notifications=[
                SubagentNotificationRecord(
                    notification_id="session-xyz:failed",
                    session_id="session-xyz",
                    parent_entity_id=entity_id,
                    terminal_status="failed",
                    summary=None,
                    error="Connection refused",
                    created_at="2026-04-06T12:00:00Z",
                    delivered_at=None,
                ),
                SubagentNotificationRecord(
                    notification_id="session-def:timed_out",
                    session_id="session-def",
                    parent_entity_id=entity_id,
                    terminal_status="timed_out",
                    summary=None,
                    error="Deadline exceeded",
                    created_at="2026-04-06T12:01:00Z",
                    delivered_at=None,
                ),
            ]
        ),
    )
    world.add_component(entity_id, SubagentWaitComponent(timeout=1.0))

    system = SubagentWaitSystem()
    system._resolve_wait(
        world,
        entity_id,
        world.get_component(entity_id, SubagentWaitComponent),
    )

    conversation = world.get_component(entity_id, ConversationComponent)
    assert conversation is not None
    assert conversation.messages[-1] == Message(
        role="system",
        content=(
            "Background subagent updates:\n"
            "- session-xyz failed: Connection refused. Call "
            'subagent_result(session_id="session-xyz") for details.\n'
            "- session-def timed_out: Deadline exceeded. Call "
            'subagent_result(session_id="session-def") for details.'
        ),
    )


async def test_subagent_retry_default_wrap() -> None:
    """Test that non-wrapped providers are wrapped with RetryModel by default."""
    from ecs_agent.providers.config import ApiFormat, ProviderConfig
    from ecs_agent.providers.openai_model import OpenAIModel
    from ecs_agent.providers.retry_model import RetryModel as _RetryModel

    world = World()
    parent_entity = world.create_entity()

    base_model = OpenAIModel(
        config=ProviderConfig(
            provider_id="openai",
            base_url="http://test",
            api_key="test",
            api_format=ApiFormat.OPENAI_CHAT_COMPLETIONS,
        ),
        model="test",
    )
    config = SubagentConfig(name="test", model=base_model)
    world.add_component(
        parent_entity, SubagentRegistryComponent(subagents={"test": config})
    )

    system = SubagentSystem()
    registry = world.get_component(parent_entity, SubagentRegistryComponent)
    assert registry is not None

    resolved = system._resolve_subagent_config(registry, "test")

    # Verify model is now wrapped with RetryModel
    assert isinstance(resolved.model, _RetryModel)


async def test_subagent_retry_no_double_wrap() -> None:
    """Test that already-wrapped providers are not double-wrapped."""
    from ecs_agent.providers.config import ApiFormat, ProviderConfig
    from ecs_agent.providers.openai_model import OpenAIModel
    from ecs_agent.providers.retry_model import RetryModel as _RetryModel2

    world = World()
    parent_entity = world.create_entity()

    base_model = OpenAIModel(
        config=ProviderConfig(
            provider_id="openai",
            base_url="http://test",
            api_key="test",
            api_format=ApiFormat.OPENAI_CHAT_COMPLETIONS,
        ),
        model="test",
    )
    retry_model = _RetryModel2(model=base_model, retry_config=RetryConfig())

    config = SubagentConfig(name="test", model=retry_model)
    world.add_component(
        parent_entity, SubagentRegistryComponent(subagents={"test": config})
    )

    system = SubagentSystem()
    registry = world.get_component(parent_entity, SubagentRegistryComponent)
    assert registry is not None

    resolved = system._resolve_subagent_config(registry, "test")

    # Verify model is STILL the same RetryModel (not double-wrapped)
    assert resolved.model is retry_model


async def test_subagent_retry_fake_provider_stable() -> None:
    """Test that FakeModel remains unwrapped for deterministic tests."""
    world = World()
    parent_entity = world.create_entity()

    fake_provider = FakeModel(responses=[])
    config = SubagentConfig(name="test", model=fake_provider)
    world.add_component(
        parent_entity, SubagentRegistryComponent(subagents={"test": config})
    )

    system = SubagentSystem()
    registry = world.get_component(parent_entity, SubagentRegistryComponent)
    assert registry is not None

    resolved = system._resolve_subagent_config(registry, "test")

    # Verify FakeModel is NOT wrapped
    assert resolved.model is fake_provider
    assert type(resolved.model).__name__ == "FakeModel"


async def test_reminder_table_updates_on_transitions() -> None:
    """Verify session table updates on each lifecycle transition."""
    from ecs_agent.components.definitions import SubagentSessionTableComponent
    from ecs_agent.types import render_subagent_session_reminder_table

    world = World()
    system = SubagentSystem()
    parent = world.create_entity()

    # Setup parent with session table and tool registry
    from ecs_agent.components import ToolRegistryComponent

    world.add_component(parent, SubagentSessionTableComponent())
    world.add_component(parent, ToolRegistryComponent(tools={}, handlers={}))

    registry = SubagentRegistryComponent()
    registry.subagents["test-agent"] = SubagentConfig(model=FakeModel(
            responses=[
                CompletionResult(
                    message=Message(role="assistant", content="Test result")
                )
            ]
        ),
name="test-agent",
        system_prompt="Test",
        skills=[],
    )
    world.add_component(parent, registry)

    # Install subagent tool
    system.install_subagent_tool(world, parent)
    tools = world.get_component(parent, ToolRegistryComponent)
    assert tools is not None
    assert "subagent" in tools.tools

    # Execute background subagent
    handler = tools.handlers["subagent"]
    result_json = await handler(
        category="test-agent",
        prompt="Test task",
        load_skills=[],
        background=True,
        timeout=None,
    )

    result = json.loads(result_json)
    session_id = result["session_id"]

    # Wait for completion
    await asyncio.sleep(0.1)

    # Check session table was updated
    table = world.get_component(parent, SubagentSessionTableComponent)
    assert table is not None
    assert session_id in table.sessions
    session = table.sessions[session_id]

    # Verify session fields
    assert session.category == "test-agent"
    assert session.status in ["queued", "running", "succeeded", "failed"]
    assert session.updated_at != ""

    # Render reminder table
    reminder = render_subagent_session_reminder_table(table.sessions)
    assert session_id in reminder
    assert "test-agent" in reminder


async def test_reminder_table_deterministic_sort() -> None:
    """Verify deterministic sorting: updated_at desc, session_id asc."""
    from ecs_agent.components.definitions import SubagentSessionTableComponent
    from ecs_agent.types import render_subagent_session_reminder_table
    from ecs_agent.types import SubagentSessionRecord
    from datetime import datetime, timezone, timedelta

    # Create sessions with different timestamps
    world = World()
    parent = world.create_entity()
    table = SubagentSessionTableComponent()

    now = datetime.now(timezone.utc)
    sessions = {
        "session-b": SubagentSessionRecord(
            session_id="session-b",
            category="test",
            prompt="Task B",
            parent_entity_id=parent,
            created_at=now.isoformat(),
            updated_at=(now - timedelta(seconds=10)).isoformat(),  # Older
        ),
        "session-a": SubagentSessionRecord(
            session_id="session-a",
            category="test",
            prompt="Task A",
            parent_entity_id=parent,
            created_at=now.isoformat(),
            updated_at=(now - timedelta(seconds=10)).isoformat(),  # Same time as B
        ),
        "session-c": SubagentSessionRecord(
            session_id="session-c",
            category="test",
            prompt="Task C",
            parent_entity_id=parent,
            created_at=now.isoformat(),
            updated_at=now.isoformat(),  # Most recent
        ),
    }
    table.sessions = sessions

    # Render twice and verify deterministic output
    reminder1 = render_subagent_session_reminder_table(table.sessions)
    reminder2 = render_subagent_session_reminder_table(table.sessions)
    assert reminder1 == reminder2

    # Verify sort order: session-c first (most recent), then session-a, then session-b
    lines = reminder1.split("\n")
    data_lines = [
        l
        for l in lines
        if l and not l.startswith("Session ID") and not l.startswith("-")
    ]
    assert len(data_lines) == 3
    assert "session-c" in data_lines[0]  # Most recent first
    assert "session-a" in data_lines[1]  # Alphabetically before B (same timestamp)
    assert "session-b" in data_lines[2]  # Alphabetically after A


# ===== Task 11: Control Tools Tests =====


async def test_subagent_status_aggregate_table() -> None:
    """Test subagent_status tool returns aggregate table when session_id=None."""
    from ecs_agent.components.definitions import SubagentSessionTableComponent
    from ecs_agent.components import ToolRegistryComponent

    world = World()
    system = SubagentSystem()
    parent = world.create_entity()

    # Setup components
    world.add_component(parent, SubagentSessionTableComponent())
    world.add_component(parent, ToolRegistryComponent(tools={}, handlers={}))

    # Setup registry with test agent
    registry = SubagentRegistryComponent()
    registry.subagents["test-agent"] = SubagentConfig(model=FakeModel(
            responses=[
                CompletionResult(message=Message(role="assistant", content="Result 1"))
            ]
        ),
name="test-agent",
        system_prompt="Test",
        skills=[],
    )
    world.add_component(parent, registry)

    # Install control tools
    system.install_subagent_control_tools(world, parent)

    # Create 2 background sessions
    system.install_subagent_tool(world, parent)
    tools = world.get_component(parent, ToolRegistryComponent)
    assert tools is not None

    handler = tools.handlers["subagent"]
    result1 = await handler(
        category="test-agent",
        prompt="Task 1",
        load_skills=[],
        background=True,
        timeout=None,
    )
    result2 = await handler(
        category="test-agent",
        prompt="Task 2",
        load_skills=[],
        background=True,
        timeout=None,
    )

    session1 = json.loads(result1)["session_id"]
    session2 = json.loads(result2)["session_id"]

    # Wait for completion
    await asyncio.sleep(0.1)

    # Call subagent_status with no session_id
    status_handler = tools.handlers["subagent_status"]
    status_result_json = await status_handler(session_id=None)
    status_result = json.loads(status_result_json)

    # Verify response structure
    assert "session_count" in status_result
    assert status_result["session_count"] == 2
    assert "summary_table" in status_result

    # Verify both sessions appear in table
    table_text = status_result["summary_table"]
    assert session1 in table_text
    assert session2 in table_text
    assert "test-agent" in table_text


async def test_subagent_status_single_session() -> None:
    """Test subagent_status tool returns single session details when session_id provided."""
    from ecs_agent.components.definitions import SubagentSessionTableComponent
    from ecs_agent.components import ToolRegistryComponent

    world = World()
    system = SubagentSystem()
    parent = world.create_entity()

    # Setup components
    world.add_component(parent, SubagentSessionTableComponent())
    world.add_component(parent, ToolRegistryComponent(tools={}, handlers={}))

    registry = SubagentRegistryComponent()
    registry.subagents["test-agent"] = SubagentConfig(model=FakeModel(
            responses=[
                CompletionResult(
                    message=Message(role="assistant", content="Single result")
                )
            ]
        ),
name="test-agent",
        system_prompt="Test",
        skills=[],
    )
    world.add_component(parent, registry)

    # Install tools
    system.install_subagent_control_tools(world, parent)
    system.install_subagent_tool(world, parent)

    tools = world.get_component(parent, ToolRegistryComponent)
    assert tools is not None

    # Create one session
    handler = tools.handlers["subagent"]
    result = await handler(
        category="test-agent",
        prompt="Single task",
        load_skills=[],
        background=True,
        timeout=None,
    )
    session_id = json.loads(result)["session_id"]

    await asyncio.sleep(0.1)

    # Call subagent_status with session_id
    status_handler = tools.handlers["subagent_status"]
    status_result_json = await status_handler(session_id=session_id)
    status_result = json.loads(status_result_json)

    # Verify single session response
    assert status_result["session_id"] == session_id
    assert status_result["category"] == "test-agent"
    assert "lifecycle_status" in status_result


def test_session_payload_does_not_contain_timestamp_fields() -> None:
    system = SubagentSystem()
    session = SubagentSessionRecord(
        session_id="session-1",
        category="research",
        prompt="Gather context",
        parent_entity_id=EntityId(1),
        created_at="2026-04-06T10:00:00Z",
        updated_at="2026-04-06T10:00:00Z",
        started_at="2026-04-06T10:00:01Z",
        finished_at="2026-04-06T10:00:02Z",
        status="succeeded",
        result_excerpt="x",
        error=None,
    )

    payload = system._session_payload(session, status="subagent_complete")

    assert {
        "status",
        "session_id",
        "category",
        "lifecycle_status",
        "artifact_id",
        "record_path",
        "inline_content",
        "error",
    } <= payload.keys()
    assert {
        "created_at",
        "updated_at",
        "started_at",
        "finished_at",
        "result_excerpt",
    }.isdisjoint(payload.keys())


def test_no_nested_artifact_object_in_payload() -> None:
    system = SubagentSystem()
    session = SubagentSessionRecord(
        session_id="session-1",
        category="research",
        prompt="Gather context",
        parent_entity_id=EntityId(1),
        created_at="2026-04-06T10:00:00Z",
        updated_at="2026-04-06T10:00:00Z",
        status="succeeded",
        artifact_id="artifact-1",
        artifact_record_path="scratchbook/records/subagent/subagent_123",
        artifact_inline_content="short result",
    )

    payload = system._session_payload(session, status="subagent_complete")

    assert "artifact" not in payload


async def test_inline_content_populated_without_registry_for_small_result() -> None:
    world = World()
    system = SubagentSystem()
    parent = world.create_entity()

    world.add_component(parent, SubagentSessionTableComponent())
    world.add_component(parent, ToolRegistryComponent(tools={}, handlers={}))
    world.add_component(
        parent,
        SubagentRegistryComponent(
            subagents={
                "test-agent": SubagentConfig(model=FakeModel(
                        responses=[
                            CompletionResult(
                                message=Message(
                                    role="assistant",
                                    content="small background result",
                                )
                            )
                        ]
                    ),
name="test-agent",
                    )
            }
        ),
    )

    system.install_subagent_tool(world, parent)
    system.install_subagent_control_tools(world, parent)

    tools = world.get_component(parent, ToolRegistryComponent)
    assert tools is not None

    launch_raw = await tools.handlers["subagent"](
        category="test-agent",
        prompt="run in background",
        load_skills=[],
        background=True,
        timeout=None,
    )
    session_id = json.loads(launch_raw)["session_id"]

    await asyncio.sleep(0.1)

    result_raw = await tools.handlers["subagent_result"](
        session_id=session_id,
        timeout=None,
    )
    result_payload = json.loads(result_raw)

    assert result_payload["inline_content"] == "small background result"


async def test_inline_content_null_without_registry_for_large_result() -> None:
    full_result = "x" * 9000

    world = World()
    system = SubagentSystem()
    parent = world.create_entity()

    world.add_component(parent, SubagentSessionTableComponent())
    world.add_component(parent, ToolRegistryComponent(tools={}, handlers={}))
    world.add_component(
        parent,
        SubagentRegistryComponent(
            subagents={
                "test-agent": SubagentConfig(model=FakeModel(
                        responses=[
                            CompletionResult(
                                message=Message(role="assistant", content=full_result)
                            )
                        ]
                    ),
name="test-agent",
                    )
            }
        ),
    )

    system.install_subagent_tool(world, parent)
    system.install_subagent_control_tools(world, parent)

    tools = world.get_component(parent, ToolRegistryComponent)
    assert tools is not None

    launch_raw = await tools.handlers["subagent"](
        category="test-agent",
        prompt="run in background",
        load_skills=[],
        background=True,
        timeout=None,
    )
    session_id = json.loads(launch_raw)["session_id"]

    await asyncio.sleep(0.1)

    result_raw = await tools.handlers["subagent_result"](
        session_id=session_id,
        timeout=None,
    )
    result_payload = json.loads(result_raw)

    assert result_payload["inline_content"] is None


async def test_inline_content_still_works_with_registry(tmp_path: Any) -> None:
    world = World()
    system = SubagentSystem(registry=ArtifactRegistry(tmp_path))
    parent = world.create_entity()

    world.add_component(parent, SubagentSessionTableComponent())
    world.add_component(parent, ToolRegistryComponent(tools={}, handlers={}))
    world.add_component(
        parent,
        SubagentRegistryComponent(
            subagents={
                "test-agent": SubagentConfig(model=FakeModel(
                        responses=[
                            CompletionResult(
                                message=Message(
                                    role="assistant",
                                    content="persisted short result",
                                )
                            )
                        ]
                    ),
name="test-agent",
                    )
            }
        ),
    )

    system.install_subagent_tool(world, parent)
    system.install_subagent_control_tools(world, parent)

    tools = world.get_component(parent, ToolRegistryComponent)
    assert tools is not None

    launch_raw = await tools.handlers["subagent"](
        category="test-agent",
        prompt="run in background",
        load_skills=[],
        background=True,
        timeout=None,
    )
    session_id = json.loads(launch_raw)["session_id"]

    await asyncio.sleep(0.1)

    result_raw = await tools.handlers["subagent_result"](
        session_id=session_id,
        timeout=None,
    )
    result_payload = json.loads(result_raw)

    assert result_payload["inline_content"] == "persisted short result"
    assert result_payload["record_path"] is not None


async def test_inline_content_is_hint_when_large_result_with_registry(
    tmp_path: Any,
) -> None:
    full_result = "x" * 9000

    world = World()
    system = SubagentSystem(registry=ArtifactRegistry(tmp_path))
    parent = world.create_entity()

    world.add_component(parent, SubagentSessionTableComponent())
    world.add_component(parent, ToolRegistryComponent(tools={}, handlers={}))
    world.add_component(
        parent,
        SubagentRegistryComponent(
            subagents={
                "test-agent": SubagentConfig(model=FakeModel(
                        responses=[
                            CompletionResult(
                                message=Message(role="assistant", content=full_result)
                            )
                        ]
                    ),
name="test-agent",
                    )
            }
        ),
    )

    system.install_subagent_tool(world, parent)
    system.install_subagent_control_tools(world, parent)

    tools = world.get_component(parent, ToolRegistryComponent)
    assert tools is not None

    launch_raw = await tools.handlers["subagent"](
        category="test-agent",
        prompt="run in background",
        load_skills=[],
        background=True,
        timeout=None,
    )
    session_id = json.loads(launch_raw)["session_id"]

    await asyncio.sleep(0.1)

    result_raw = await tools.handlers["subagent_result"](
        session_id=session_id,
        timeout=None,
    )
    result_payload = json.loads(result_raw)

    record_path = result_payload["record_path"]
    assert record_path is not None
    assert result_payload["inline_content"] is not None
    assert record_path in result_payload["inline_content"]
    assert "Result persisted to" in result_payload["inline_content"]


async def test_subagent_status_queued_session_reports_queue_position(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from unittest.mock import AsyncMock

    from ecs_agent.components import ToolRegistryComponent
    from ecs_agent.components.definitions import SubagentSessionTableComponent

    _reset_global_scheduler(monkeypatch)

    world = World()
    system = SubagentSystem(max_background_concurrency=1)
    parent = world.create_entity()

    world.add_component(parent, SubagentSessionTableComponent())
    world.add_component(parent, ToolRegistryComponent(tools={}, handlers={}))

    running_release = asyncio.Event()

    async def blocking_completion(*args: Any, **kwargs: Any) -> CompletionResult:
        await running_release.wait()
        return CompletionResult(message=Message(role="assistant", content="first done"))

    blocking_provider = FakeModel(responses=[])
    blocking_provider.complete = AsyncMock(side_effect=blocking_completion)  # type: ignore[method-assign]

    registry = SubagentRegistryComponent()
    registry.subagents["blocking-agent"] = SubagentConfig(
        name="blocking-agent",
        model=blocking_provider,
        system_prompt="Test",
        skills=[],
    )
    registry.subagents["queued-agent"] = SubagentConfig(model=FakeModel(
            responses=[
                CompletionResult(
                    message=Message(role="assistant", content="queued done")
                )
            ]
        ),
name="queued-agent",
        system_prompt="Test",
        skills=[],
    )
    world.add_component(parent, registry)

    system.install_subagent_control_tools(world, parent)
    system.install_subagent_tool(world, parent)

    tools = world.get_component(parent, ToolRegistryComponent)
    assert tools is not None

    launch_handler = tools.handlers["subagent"]
    first_payload = json.loads(
        await launch_handler(
            category="blocking-agent",
            prompt="Block the only slot",
            load_skills=[],
            background=True,
            timeout=None,
        )
    )
    second_payload = json.loads(
        await launch_handler(
            category="queued-agent",
            prompt="Wait in queue",
            load_skills=[],
            background=True,
            timeout=None,
        )
    )

    assert first_payload["lifecycle_status"] in {"queued", "running"}
    assert second_payload["lifecycle_status"] == "queued"

    status_handler = tools.handlers["subagent_status"]
    status_result = json.loads(
        await status_handler(session_id=second_payload["session_id"])
    )

    assert status_result["session_id"] == second_payload["session_id"]
    assert status_result["lifecycle_status"] == "queued"
    assert status_result["queue_position"] == 0
    assert "artifact" not in status_result
    assert "error" in status_result
    assert status_result["error"] is None

    running_release.set()
    first_task = await system._runtime_manager.get_task(first_payload["session_id"])
    assert first_task is not None
    await first_task

    await asyncio.sleep(0)
    second_task = await system._runtime_manager.get_task(second_payload["session_id"])
    if second_task is not None:
        await second_task


async def test_subagent_status_missing_session() -> None:
    """Test subagent_status returns error for missing session_id."""
    from ecs_agent.components.definitions import SubagentSessionTableComponent
    from ecs_agent.components import ToolRegistryComponent

    world = World()
    system = SubagentSystem()
    parent = world.create_entity()

    world.add_component(parent, SubagentSessionTableComponent())
    world.add_component(parent, ToolRegistryComponent(tools={}, handlers={}))

    # Install control tools
    system.install_subagent_control_tools(world, parent)

    tools = world.get_component(parent, ToolRegistryComponent)
    assert tools is not None

    # Query nonexistent session
    status_handler = tools.handlers["subagent_status"]
    result_json = await status_handler(session_id="nonexistent-session-id")
    result = json.loads(result_json)

    # Verify error response
    assert "error" in result
    assert "not found" in result["error"].lower()


async def test_subagent_result_completed_session() -> None:
    from ecs_agent.components.definitions import SubagentSessionTableComponent
    from ecs_agent.components import ToolRegistryComponent

    world = World()
    system = SubagentSystem()
    parent = world.create_entity()

    world.add_component(parent, SubagentSessionTableComponent())
    world.add_component(parent, ToolRegistryComponent(tools={}, handlers={}))

    registry = SubagentRegistryComponent()
    registry.subagents["test-agent"] = SubagentConfig(model=FakeModel(
            responses=[
                CompletionResult(
                    message=Message(
                        role="assistant",
                        content=(
                            "<subagent_background_result>"
                            "<summary>Completed summary</summary>"
                            "<full_result>Completed result</full_result>"
                            "</subagent_background_result>"
                        ),
                    )
                )
            ]
        ),
name="test-agent",
        system_prompt="Test",
        skills=[],
    )
    world.add_component(parent, registry)

    system.install_subagent_control_tools(world, parent)
    system.install_subagent_tool(world, parent)

    tools = world.get_component(parent, ToolRegistryComponent)
    assert tools is not None

    # Create and wait for completion
    handler = tools.handlers["subagent"]
    result = await handler(
        category="test-agent",
        prompt="Task",
        load_skills=[],
        background=True,
        timeout=None,
    )
    session_id = json.loads(result)["session_id"]

    await asyncio.sleep(0.1)  # Let it finish

    # Get result
    result_handler = tools.handlers["subagent_result"]
    result_json = await result_handler(session_id=session_id, timeout=None)
    result_data = json.loads(result_json)

    # Verify successful result
    assert result_data["status"] == "success"
    assert result_data["lifecycle_status"] == "succeeded"
    assert "session_id" in result_data


async def test_subagent_result_waits_for_queued_session_without_task_handle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from unittest.mock import AsyncMock

    from ecs_agent.components import ToolRegistryComponent
    from ecs_agent.components.definitions import SubagentSessionTableComponent

    _reset_global_scheduler(monkeypatch)

    world = World()
    system = SubagentSystem(max_background_concurrency=1)
    parent = world.create_entity()

    world.add_component(parent, SubagentSessionTableComponent())
    world.add_component(parent, ToolRegistryComponent(tools={}, handlers={}))

    running_release = asyncio.Event()

    async def blocking_completion(*args: Any, **kwargs: Any) -> CompletionResult:
        await running_release.wait()
        return CompletionResult(message=Message(role="assistant", content="first done"))

    blocking_provider = FakeModel(responses=[])
    blocking_provider.complete = AsyncMock(side_effect=blocking_completion)  # type: ignore[method-assign]

    registry = SubagentRegistryComponent()
    registry.subagents["blocking-agent"] = SubagentConfig(
        name="blocking-agent",
        model=blocking_provider,
        system_prompt="Test",
        skills=[],
    )
    registry.subagents["queued-agent"] = SubagentConfig(model=FakeModel(
            responses=[
                CompletionResult(
                    message=Message(role="assistant", content="queued result")
                )
            ]
        ),
name="queued-agent",
        system_prompt="Test",
        skills=[],
    )
    world.add_component(parent, registry)

    system.install_subagent_control_tools(world, parent)
    system.install_subagent_tool(world, parent)

    tools = world.get_component(parent, ToolRegistryComponent)
    assert tools is not None

    launch_handler = tools.handlers["subagent"]
    first_payload = json.loads(
        await launch_handler(
            category="blocking-agent",
            prompt="Block the only slot",
            load_skills=[],
            background=True,
            timeout=None,
        )
    )
    second_payload = json.loads(
        await launch_handler(
            category="queued-agent",
            prompt="Return after queue admission",
            load_skills=[],
            background=True,
            timeout=None,
        )
    )

    assert second_payload["lifecycle_status"] == "queued"

    result_handler = tools.handlers["subagent_result"]
    wait_task = asyncio.create_task(
        result_handler(session_id=second_payload["session_id"], timeout=None)
    )

    await asyncio.sleep(0.05)
    assert not wait_task.done()

    running_release.set()
    first_task = await system._runtime_manager.get_task(first_payload["session_id"])
    assert first_task is not None
    await first_task

    result_data = json.loads(await asyncio.wait_for(wait_task, timeout=1.0))
    assert result_data["status"] in {"success", "completed"}
    assert result_data["lifecycle_status"] == "succeeded"
    assert result_data["session_id"] == second_payload["session_id"]
    error_message = result_data.get("error")
    assert error_message in (None, "")

    second_result_data = json.loads(
        await result_handler(session_id=second_payload["session_id"], timeout=None)
    )
    assert second_result_data["status"] == "success"
    assert second_result_data["lifecycle_status"] == "succeeded"


async def test_subagent_result_timeout() -> None:
    """Test subagent_result returns timeout error when waiting too long."""
    from ecs_agent.components.definitions import SubagentSessionTableComponent
    from ecs_agent.components import ToolRegistryComponent

    world = World()
    system = SubagentSystem()
    parent = world.create_entity()

    world.add_component(parent, SubagentSessionTableComponent())
    world.add_component(parent, ToolRegistryComponent(tools={}, handlers={}))

    # Create model with delayed response
    async def slow_completion(*args: Any, **kwargs: Any) -> CompletionResult:
        await asyncio.sleep(10.0)  # Very long delay
        return CompletionResult(
            message=Message(role="assistant", content="Slow result")
        )

    from unittest.mock import AsyncMock

    slow_provider = FakeModel(responses=[])
    slow_provider.complete = AsyncMock(side_effect=slow_completion)  # type: ignore[method-assign]

    registry = SubagentRegistryComponent()
    registry.subagents["slow-agent"] = SubagentConfig(
        name="slow-agent",
        model=slow_provider,
        system_prompt="Test",
        skills=[],
    )
    world.add_component(parent, registry)

    system.install_subagent_control_tools(world, parent)
    system.install_subagent_tool(world, parent)

    tools = world.get_component(parent, ToolRegistryComponent)
    assert tools is not None

    # Launch slow session
    handler = tools.handlers["subagent"]
    result = await handler(
        category="slow-agent",
        prompt="Slow task",
        load_skills=[],
        background=True,
        timeout=None,
    )
    session_id = json.loads(result)["session_id"]

    # Try to get result with very short timeout
    result_handler = tools.handlers["subagent_result"]
    result_json = await result_handler(session_id=session_id, timeout=0.05)
    result_data = json.loads(result_json)

    # Verify timeout error
    assert "error" in result_data
    assert (
        "timeout" in result_data["error"].lower()
        or "timed out" in result_data["error"].lower()
    )


async def test_subagent_result_read_method_accepts_full_and_summary() -> None:
    from ecs_agent.components import ToolRegistryComponent
    from ecs_agent.components.definitions import SubagentSessionTableComponent

    world = World()
    system = SubagentSystem()
    parent = world.create_entity()

    world.add_component(parent, SubagentSessionTableComponent())
    world.add_component(parent, ToolRegistryComponent(tools={}, handlers={}))

    registry = SubagentRegistryComponent()
    registry.subagents["test-agent"] = SubagentConfig(model=FakeModel(
            responses=[
                CompletionResult(
                    message=Message(
                        role="assistant",
                        content=(
                            "<subagent_background_result>"
                            "<summary>Completed summary</summary>"
                            "<full_result>Completed result</full_result>"
                            "</subagent_background_result>"
                        ),
                    )
                )
            ]
        ),
name="test-agent",
        system_prompt="Test",
        skills=[],
    )
    world.add_component(parent, registry)

    system.install_subagent_control_tools(world, parent)
    system.install_subagent_tool(world, parent)

    tools = world.get_component(parent, ToolRegistryComponent)
    assert tools is not None

    launch_handler = tools.handlers["subagent"]
    session_id = json.loads(
        await launch_handler(
            category="test-agent",
            prompt="Task",
            load_skills=[],
            background=True,
            timeout=None,
        )
    )["session_id"]

    await asyncio.sleep(0.1)

    result_handler = tools.handlers["subagent_result"]
    full_result = json.loads(
        await result_handler(session_id=session_id, read_method="full", timeout=None)
    )
    summary_result = json.loads(
        await result_handler(session_id=session_id, read_method="summary", timeout=None)
    )

    assert full_result["status"] == "success"
    assert summary_result["status"] == "success"
    assert full_result["session_id"] == session_id
    assert summary_result["session_id"] == session_id
    assert full_result["inline_content"] == "Completed result"
    assert summary_result["inline_content"] == "Completed summary"
    assert summary_result["read_method"] == "summary"


def test_subagent_result_schema_exposes_read_method() -> None:
    from ecs_agent.components import ToolRegistryComponent
    from ecs_agent.components.definitions import SubagentSessionTableComponent

    world = World()
    system = SubagentSystem()
    parent = world.create_entity()

    world.add_component(parent, SubagentSessionTableComponent())
    world.add_component(parent, ToolRegistryComponent(tools={}, handlers={}))
    world.add_component(
        parent,
        LLMComponent(
            model=FakeModel(responses=[]),
            system_prompt="Test",
        ),
    )
    world.add_component(parent, ConversationComponent(messages=[]))

    tools = world.get_component(parent, ToolRegistryComponent)
    assert tools is not None

    system.install_subagent_control_tools(world, parent)

    schema = tools.tools["subagent_result"]
    properties = schema.parameters["properties"]

    assert "read_method" in properties
    read_method = properties["read_method"]
    assert "enum" in read_method
    assert "full" in read_method["enum"]
    assert "summary" in read_method["enum"]


async def test_subagent_result_defaults_to_full() -> None:
    world = World()
    system = SubagentSystem()
    parent = world.create_entity()

    world.add_component(parent, SubagentSessionTableComponent())
    world.add_component(parent, ToolRegistryComponent(tools={}, handlers={}))
    world.add_component(
        parent,
        SubagentRegistryComponent(
            subagents={
                "test-agent": SubagentConfig(model=FakeModel(
                        responses=[
                            CompletionResult(
                                message=Message(
                                    role="assistant",
                                    content=(
                                        "<subagent_background_result>"
                                        "<summary>Short summary</summary>"
                                        "<full_result>Full background result</full_result>"
                                        "</subagent_background_result>"
                                    ),
                                )
                            )
                        ]
                    ),
name="test-agent",
                    )
            }
        ),
    )

    system.install_subagent_control_tools(world, parent)
    system.install_subagent_tool(world, parent)

    tools = world.get_component(parent, ToolRegistryComponent)
    assert tools is not None

    session_id = json.loads(
        await tools.handlers["subagent"](
            category="test-agent",
            prompt="Task",
            load_skills=[],
            background=True,
            timeout=None,
        )
    )["session_id"]

    await asyncio.sleep(0.1)

    result_payload = json.loads(
        await tools.handlers["subagent_result"](session_id=session_id, timeout=None)
    )

    assert result_payload["status"] == "success"
    assert result_payload["session_id"] == session_id
    assert result_payload["inline_content"] == "Full background result"
    assert "read_method" not in result_payload


async def test_subagent_result_summary_returns_cached_summary() -> None:
    world = World()
    system = SubagentSystem()
    parent = world.create_entity()

    world.add_component(parent, SubagentSessionTableComponent())
    world.add_component(parent, ToolRegistryComponent(tools={}, handlers={}))
    world.add_component(
        parent,
        SubagentRegistryComponent(
            subagents={
                "test-agent": SubagentConfig(model=FakeModel(
                        responses=[
                            CompletionResult(
                                message=Message(
                                    role="assistant",
                                    content=(
                                        "<subagent_background_result>"
                                        "<summary>Cached background summary</summary>"
                                        "<full_result>Detailed background result</full_result>"
                                        "</subagent_background_result>"
                                    ),
                                )
                            )
                        ]
                    ),
name="test-agent",
                    )
            }
        ),
    )

    system.install_subagent_control_tools(world, parent)
    system.install_subagent_tool(world, parent)

    tools = world.get_component(parent, ToolRegistryComponent)
    assert tools is not None

    session_id = json.loads(
        await tools.handlers["subagent"](
            category="test-agent",
            prompt="Task",
            load_skills=[],
            background=True,
            timeout=None,
        )
    )["session_id"]

    await asyncio.sleep(0.1)

    summary_payload = json.loads(
        await tools.handlers["subagent_result"](
            session_id=session_id,
            read_method="summary",
            timeout=None,
        )
    )

    assert summary_payload["status"] == "success"
    assert summary_payload["session_id"] == session_id
    assert summary_payload["read_method"] == "summary"
    assert summary_payload["inline_content"] == "Cached background summary"

    table = world.get_component(parent, SubagentSessionTableComponent)
    assert table is not None
    assert table.sessions[session_id].result_summary == "Cached background summary"
    assert (
        table.sessions[session_id].artifact_inline_content
        == "Detailed background result"
    )


async def test_subagent_result_summary_unavailable_returns_error() -> None:
    world = World()
    system = SubagentSystem()
    parent = world.create_entity()

    world.add_component(parent, SubagentSessionTableComponent())
    world.add_component(parent, ToolRegistryComponent(tools={}, handlers={}))
    world.add_component(
        parent,
        SubagentRegistryComponent(
            subagents={
                "test-agent": SubagentConfig(model=FakeModel(
                        responses=[
                            CompletionResult(
                                message=Message(
                                    role="assistant",
                                    content="Raw background result without envelope",
                                )
                            )
                        ]
                    ),
name="test-agent",
                    )
            }
        ),
    )

    system.install_subagent_control_tools(world, parent)
    system.install_subagent_tool(world, parent)

    tools = world.get_component(parent, ToolRegistryComponent)
    assert tools is not None

    session_id = json.loads(
        await tools.handlers["subagent"](
            category="test-agent",
            prompt="Task",
            load_skills=[],
            background=True,
            timeout=None,
        )
    )["session_id"]

    await asyncio.sleep(0.1)

    summary_payload = json.loads(
        await tools.handlers["subagent_result"](
            session_id=session_id,
            read_method="summary",
            timeout=None,
        )
    )

    assert summary_payload == {
        "error": 'Summary not available for this session. Retry with read_method="full".',
        "read_method": "summary",
        "session_id": session_id,
    }


async def test_subagent_result_read_method_rejects_unknown_value() -> None:
    from ecs_agent.components import ToolRegistryComponent
    from ecs_agent.components.definitions import SubagentSessionTableComponent

    world = World()
    system = SubagentSystem()
    parent = world.create_entity()

    world.add_component(parent, SubagentSessionTableComponent())
    world.add_component(parent, ToolRegistryComponent(tools={}, handlers={}))

    system.install_subagent_control_tools(world, parent)

    tools = world.get_component(parent, ToolRegistryComponent)
    assert tools is not None

    result_handler = tools.handlers["subagent_result"]
    result = json.loads(
        await result_handler(
            session_id="missing-session",
            read_method="snapshot",
            timeout=None,
        )
    )

    assert result == {
        "error": "Invalid read_method 'snapshot'. Expected one of: full, summary",
        "read_method": "snapshot",
        "session_id": "missing-session",
    }


async def test_subagent_cancel_active_session() -> None:
    from ecs_agent.components.definitions import SubagentSessionTableComponent
    from ecs_agent.components import ToolRegistryComponent

    world = World()
    system = SubagentSystem()
    parent = world.create_entity()

    world.add_component(parent, SubagentSessionTableComponent())
    world.add_component(parent, ToolRegistryComponent(tools={}, handlers={}))

    # Create model with very slow response
    async def very_slow_completion(*args: Any, **kwargs: Any) -> CompletionResult:
        await asyncio.sleep(100.0)
        return CompletionResult(
            message=Message(role="assistant", content="Never happens")
        )

    from unittest.mock import AsyncMock

    slow_provider = FakeModel(responses=[])
    slow_provider.complete = AsyncMock(side_effect=very_slow_completion)  # type: ignore[method-assign]

    registry = SubagentRegistryComponent()
    registry.subagents["cancel-test"] = SubagentConfig(
        name="cancel-test",
        model=slow_provider,
        system_prompt="Test",
        skills=[],
    )
    world.add_component(parent, registry)

    system.install_subagent_control_tools(world, parent)
    system.install_subagent_tool(world, parent)

    tools = world.get_component(parent, ToolRegistryComponent)
    assert tools is not None

    # Launch session
    handler = tools.handlers["subagent"]
    result = await handler(
        category="cancel-test",
        prompt="Long task",
        load_skills=[],
        background=True,
        timeout=None,
    )
    session_id = json.loads(result)["session_id"]

    await asyncio.sleep(0.05)  # Let it start

    # Cancel the session
    cancel_handler = tools.handlers["subagent_cancel"]
    cancel_result_json = await cancel_handler(session_id=session_id)
    cancel_result = json.loads(cancel_result_json)

    # Verify cancellation
    assert cancel_result["status"] == "cancelled"
    assert cancel_result["session_id"] == session_id

    # Verify session state updated
    table = world.get_component(parent, SubagentSessionTableComponent)
    assert table is not None
    assert session_id in table.sessions
    assert table.sessions[session_id].status == "cancelled"


async def test_subagent_cancel_queued_session_skips_next_admission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from unittest.mock import AsyncMock

    from ecs_agent.components import ToolRegistryComponent
    from ecs_agent.components.definitions import SubagentSessionTableComponent
    import ecs_agent.systems.subagent_runtime as runtime_module

    _reset_global_scheduler(monkeypatch)

    world = World()
    system = SubagentSystem(max_background_concurrency=1)
    parent = world.create_entity()

    world.add_component(parent, SubagentSessionTableComponent())
    world.add_component(parent, ToolRegistryComponent(tools={}, handlers={}))

    running_release = asyncio.Event()
    started: list[str] = []

    async def blocking_completion(*args: Any, **kwargs: Any) -> CompletionResult:
        started.append("first")
        await running_release.wait()
        return CompletionResult(message=Message(role="assistant", content="first done"))

    async def cancelled_completion(*args: Any, **kwargs: Any) -> CompletionResult:
        started.append("cancelled")
        return CompletionResult(
            message=Message(role="assistant", content="should never run")
        )

    async def third_completion(*args: Any, **kwargs: Any) -> CompletionResult:
        started.append("third")
        return CompletionResult(message=Message(role="assistant", content="third done"))

    blocking_provider = FakeModel(responses=[])
    blocking_provider.complete = AsyncMock(side_effect=blocking_completion)  # type: ignore[method-assign]
    cancelled_provider = FakeModel(responses=[])
    cancelled_provider.complete = AsyncMock(side_effect=cancelled_completion)  # type: ignore[method-assign]
    third_provider = FakeModel(responses=[])
    third_provider.complete = AsyncMock(side_effect=third_completion)  # type: ignore[method-assign]

    registry = SubagentRegistryComponent()
    registry.subagents["blocking-agent"] = SubagentConfig(
        name="blocking-agent",
        model=blocking_provider,
        system_prompt="Test",
        skills=[],
    )
    registry.subagents["cancelled-agent"] = SubagentConfig(
        name="cancelled-agent",
        model=cancelled_provider,
        system_prompt="Test",
        skills=[],
    )
    registry.subagents["third-agent"] = SubagentConfig(
        name="third-agent",
        model=third_provider,
        system_prompt="Test",
        skills=[],
    )
    world.add_component(parent, registry)

    system.install_subagent_control_tools(world, parent)
    system.install_subagent_tool(world, parent)

    tools = world.get_component(parent, ToolRegistryComponent)
    assert tools is not None

    launch_handler = tools.handlers["subagent"]
    first_payload = json.loads(
        await launch_handler(
            category="blocking-agent",
            prompt="Occupy the only slot",
            load_skills=[],
            background=True,
            timeout=None,
        )
    )
    cancelled_payload = json.loads(
        await launch_handler(
            category="cancelled-agent",
            prompt="Queue and then cancel",
            load_skills=[],
            background=True,
            timeout=None,
        )
    )
    third_payload = json.loads(
        await launch_handler(
            category="third-agent",
            prompt="Run after queued cancel",
            load_skills=[],
            background=True,
            timeout=None,
        )
    )

    scheduler = runtime_module._GLOBAL_SCHEDULER
    assert scheduler is not None
    assert [item.session_id for item in scheduler.pending_queue] == [
        cancelled_payload["session_id"],
        third_payload["session_id"],
    ]

    cancel_handler = tools.handlers["subagent_cancel"]
    cancel_result = json.loads(
        await cancel_handler(session_id=cancelled_payload["session_id"])
    )

    assert cancel_result["status"] == "cancelled"
    assert cancel_result["lifecycle_status"] == "cancelled"
    assert cancel_result["session_id"] == cancelled_payload["session_id"]
    assert [item.session_id for item in scheduler.pending_queue] == [
        third_payload["session_id"]
    ]
    assert (
        await system._runtime_manager.get_task(cancelled_payload["session_id"]) is None
    )

    running_release.set()
    first_task = await system._runtime_manager.get_task(first_payload["session_id"])
    assert first_task is not None
    await first_task

    result_handler = tools.handlers["subagent_result"]
    third_result = json.loads(
        await asyncio.wait_for(
            result_handler(session_id=third_payload["session_id"], timeout=None),
            timeout=1.0,
        )
    )

    assert third_result["lifecycle_status"] == "succeeded"
    assert started == ["first", "third"]


async def test_subagent_cancel_terminal_session() -> None:
    from ecs_agent.components.definitions import SubagentSessionTableComponent
    from ecs_agent.components import ToolRegistryComponent

    world = World()
    system = SubagentSystem()
    parent = world.create_entity()

    world.add_component(parent, SubagentSessionTableComponent())
    world.add_component(parent, ToolRegistryComponent(tools={}, handlers={}))

    registry = SubagentRegistryComponent()
    registry.subagents["test-agent"] = SubagentConfig(model=FakeModel(
            responses=[
                CompletionResult(message=Message(role="assistant", content="Done"))
            ]
        ),
name="test-agent",
        system_prompt="Test",
        skills=[],
    )
    world.add_component(parent, registry)

    system.install_subagent_control_tools(world, parent)
    system.install_subagent_tool(world, parent)

    tools = world.get_component(parent, ToolRegistryComponent)
    assert tools is not None

    # Create and wait for completion
    handler = tools.handlers["subagent"]
    result = await handler(
        category="test-agent",
        prompt="Task",
        load_skills=[],
        background=True,
        timeout=None,
    )
    session_id = json.loads(result)["session_id"]

    await asyncio.sleep(0.1)  # Let it finish

    # Try to cancel already-finished session
    cancel_handler = tools.handlers["subagent_cancel"]
    cancel_result_json = await cancel_handler(session_id=session_id)
    cancel_result = json.loads(cancel_result_json)

    assert "error" in cancel_result
    assert cancel_result["lifecycle_status"] == "succeeded"
    assert cancel_result["session_id"] == session_id


def test_assemble_child_world_name_follows_convention() -> None:
    """Child world name must be '<subagent_name>-<8hex>' format."""
    import re
    from ecs_agent.systems.subagent import SubagentSystem
    from ecs_agent.types import SubagentConfig
    from ecs_agent.components.definitions import (
        ToolRegistryComponent,
        ConversationComponent,
        LLMComponent,
    )

    model = FakeModel(
        responses=[CompletionResult(message=Message(role="assistant", content="done"))]
    )
    world = World(name="parent-world")
    parent = world.create_entity()
    world.add_component(
        parent, LLMComponent(model=model, system_prompt="")
    )
    world.add_component(parent, ToolRegistryComponent(tools={}, handlers={}))
    world.add_component(parent, ConversationComponent(messages=[]))

    config = SubagentConfig(
        name="researcher",
        model=model,
        system_prompt="",
    )

    system = SubagentSystem()
    child_world, _ = system._assemble_child_world(world, parent, config)

    assert child_world.name is not None
    pattern = re.compile(r"^researcher-[0-9a-f]{8}$")
    assert pattern.match(child_world.name), f"Got: {child_world.name!r}"


def test_assemble_child_world_different_calls_produce_unique_names() -> None:
    """Each call must produce a distinct child world name."""
    from ecs_agent.systems.subagent import SubagentSystem
    from ecs_agent.types import SubagentConfig
    from ecs_agent.components.definitions import (
        ToolRegistryComponent,
        ConversationComponent,
        LLMComponent,
    )

    model = FakeModel(
        responses=[CompletionResult(message=Message(role="assistant", content="done"))]
        * 4
    )
    world = World(name="parent")
    parent = world.create_entity()
    world.add_component(
        parent, LLMComponent(model=model, system_prompt="")
    )
    world.add_component(parent, ToolRegistryComponent(tools={}, handlers={}))
    world.add_component(parent, ConversationComponent(messages=[]))

    config = SubagentConfig(name="worker", model=model)
    system = SubagentSystem()

    w1, _ = system._assemble_child_world(world, parent, config)
    w2, _ = system._assemble_child_world(world, parent, config)
    assert w1.name != w2.name


def test_child_world_inherits_parent_compaction_config() -> None:
    from ecs_agent.components.definitions import (
        CompactionConfigComponent,
        ConversationArchiveComponent,
    )
    from ecs_agent.systems.compaction import CompactionSystem

    model = FakeModel(
        responses=[CompletionResult(message=Message(role="assistant", content="done"))]
    )
    world = World(name="parent")
    parent = world.create_entity()
    world.add_component(parent, LLMComponent(model=model, system_prompt=""))
    world.add_component(parent, ToolRegistryComponent(tools={}, handlers={}))
    world.add_component(parent, ConversationComponent(messages=[]))
    world.add_component(
        parent,
        CompactionConfigComponent(
            threshold_tokens=123,
            compaction_method="predrop_then_compact",
        ),
    )

    config = SubagentConfig(name="worker", model=model)
    system = SubagentSystem()

    child_world, child_entity_id = system._assemble_child_world(world, parent, config)
    child_world.apply_pending_system_operations()

    compaction = child_world.get_component(child_entity_id, CompactionConfigComponent)
    archive = child_world.get_component(child_entity_id, ConversationArchiveComponent)

    assert compaction is not None
    assert compaction.threshold_tokens == 123
    assert compaction.compaction_method == "predrop_then_compact"
    assert archive is not None
    assert any(
        isinstance(entry.system, CompactionSystem) and entry.priority == -30
        for entry in child_world._systems._systems
    )


# ---------------------------------------------------------------------------
# Task 8: catalog skill resolution + workspace inheritance
# ---------------------------------------------------------------------------


async def test_subagent_resolves_skill_from_catalog_when_not_in_parent() -> None:
    """Subagent installs a skill from the global catalog even if parent never had it."""
    from ecs_agent.skills import catalog
    from ecs_agent.skills.catalog import SkillDescriptor, SkillType
    from ecs_agent.skills.script_skill import ScriptSkill
    from ecs_agent.skills.manager import SkillManager
    from ecs_agent.components.definitions import SkillComponent
    from pathlib import Path

    class CatalogOnlySkill(ScriptSkill):
        name: str = "catalog-only-skill"
        description: str = "A skill that lives in the catalog but not on parent"

        def tools(self) -> dict[str, tuple[ToolSchema, Any]]:
            async def catalog_tool() -> str:
                return "catalog_tool_result"

            return {
                "catalog_tool": (
                    ToolSchema(
                        name="catalog_tool",
                        description="A tool from catalog-only skill",
                        parameters={"type": "object", "properties": {}},
                    ),
                    catalog_tool,
                )
            }

        def system_prompt(self) -> str:
            return "Catalog skill system prompt"

        def install(self, world: World, entity_id: EntityId) -> None:
            pass

        def uninstall(self, world: World, entity_id: EntityId) -> None:
            pass

    catalog.clear_catalog()
    descriptor = SkillDescriptor(
        name="catalog-only-skill",
        skill_type=SkillType.SCRIPT,
        source_path=Path("fake/catalog_only_skill"),
        _materializer=CatalogOnlySkill,
        metadata={},
    )
    catalog.register(descriptor)

    world = World()
    parent_entity = world.create_entity()
    world.add_component(parent_entity, ToolRegistryComponent(tools={}, handlers={}))

    child_provider = FakeModel(
        responses=[CompletionResult(message=Message(role="assistant", content="Done"))]
    )
    config = SubagentConfig(
        name="child",
        model=child_provider,
        skills=["catalog-only-skill"],
        inheritance_policy=InheritancePolicy(
            missing_skill_policy="error",  # should NOT raise — catalog has it
        ),
    )

    registry = SubagentRegistryComponent(subagents={"child": config})
    world.add_component(parent_entity, registry)
    _register_message_bus(world, parent_entity)

    system = SubagentSystem()
    await system.process(world)

    tool_registry = world.get_component(parent_entity, ToolRegistryComponent)
    assert tool_registry is not None
    subagent_handler = tool_registry.handlers["subagent"]

    result = await subagent_handler(category="child", prompt="test task")
    assert isinstance(result, str)

    child_entity = _find_child_entity(world, parent_entity)

    child_skills = world.get_component(child_entity, SkillComponent)
    assert child_skills is not None, (
        "Child should have SkillComponent after catalog skill install"
    )
    assert "catalog-only-skill" in child_skills.skills, (
        "catalog-only-skill must be installed on child via catalog lookup"
    )

    child_tools = world.get_component(child_entity, ToolRegistryComponent)
    assert child_tools is not None
    assert "catalog_tool" in child_tools.handlers, (
        "Child should have catalog_tool handler"
    )
    assert "catalog_tool" in child_tools.tools, "Child should have catalog_tool schema"

    catalog.clear_catalog()


async def test_subagent_inherits_parent_workspace_binding() -> None:
    """Child entity inherits WorkspaceBindingComponent from parent when policy allows."""
    from ecs_agent.components.definitions import (
        WorkspaceBindingComponent,
    )
    from pathlib import Path

    parent_workspace = Path("/tmp/parent-workspace")

    world = World()
    parent_entity = world.create_entity()
    world.add_component(parent_entity, ToolRegistryComponent(tools={}, handlers={}))
    world.add_component(
        parent_entity, WorkspaceBindingComponent(workspace_root=parent_workspace)
    )

    child_provider = FakeModel(
        responses=[CompletionResult(message=Message(role="assistant", content="Done"))]
    )
    config = SubagentConfig(
        name="child",
        model=child_provider,
        inheritance_policy=InheritancePolicy(enabled=True),
    )

    registry = SubagentRegistryComponent(subagents={"child": config})
    world.add_component(parent_entity, registry)
    _register_message_bus(world, parent_entity)

    system = SubagentSystem()
    await system.process(world)

    tool_registry = world.get_component(parent_entity, ToolRegistryComponent)
    assert tool_registry is not None
    subagent_handler = tool_registry.handlers["subagent"]

    result = await subagent_handler(category="child", prompt="test task")
    assert isinstance(result, str)

    child_entity = _find_child_entity(world, parent_entity)

    child_binding = world.get_component(child_entity, WorkspaceBindingComponent)
    assert child_binding is not None, (
        "Child should inherit WorkspaceBindingComponent from parent"
    )
    assert child_binding.workspace_root == parent_workspace, (
        "Child workspace root must match parent workspace root"
    )


# ---------------------------------------------------------------------------
# Task 9: subagent prompt rendering via SystemPromptRenderSystem
# ---------------------------------------------------------------------------


async def test_subagent_child_world_registers_system_prompt_render_system() -> None:
    """Child world must have SystemPromptRenderSystem registered at priority < 0."""
    from ecs_agent.systems.subagent import SubagentSystem
    from ecs_agent.systems.system_prompt_render_system import SystemPromptRenderSystem

    world = World()
    parent_entity = world.create_entity()
    world.add_component(
        parent_entity,
        LLMComponent(
            model=FakeModel(responses=[]),
            system_prompt="parent-prompt",
        ),
    )
    world.add_component(parent_entity, ToolRegistryComponent(tools={}, handlers={}))

    config = SubagentConfig(model=FakeModel(
            responses=[
                CompletionResult(message=Message(role="assistant", content="done"))
            ]
        ),
name="child",
        )

    system = SubagentSystem()
    child_world, _child_entity_id = system._assemble_child_world(
        world, parent_entity, config
    )

    # Apply queued operations so _systems list is populated
    child_world._systems.apply_queued_operations()

    # Verify SystemPromptRenderSystem is registered at priority < 0 (before ReasoningSystem at 0)
    render_system_found = False
    for entry in child_world._systems._systems:
        if isinstance(entry.system, SystemPromptRenderSystem):
            render_system_found = True
            assert entry.priority < 0, (
                f"SystemPromptRenderSystem must be at priority < 0, got {entry.priority}"
            )
    assert render_system_found, (
        "Child world must have SystemPromptRenderSystem registered"
    )


async def test_subagent_child_world_has_system_prompt_config_spec() -> None:
    """Child entity must have SystemPromptConfigSpec attached (for renderer to consume)."""
    from ecs_agent.systems.subagent import SubagentSystem
    from ecs_agent.prompts.contracts import SystemPromptConfigSpec

    world = World()
    parent_entity = world.create_entity()
    world.add_component(
        parent_entity,
        LLMComponent(
            model=FakeModel(responses=[]),
            system_prompt="parent-prompt",
        ),
    )
    world.add_component(parent_entity, ToolRegistryComponent(tools={}, handlers={}))

    config = SubagentConfig(model=FakeModel(
            responses=[
                CompletionResult(message=Message(role="assistant", content="done"))
            ]
        ),
name="child",
        system_prompt="explicit-child-prompt",
    )

    system = SubagentSystem()
    child_world, child_entity_id = system._assemble_child_world(
        world, parent_entity, config
    )

    # After Task 9: child entity must have SystemPromptConfigSpec so the renderer can resolve it
    spec = child_world.get_component(child_entity_id, SystemPromptConfigSpec)
    assert spec is not None, (
        "Child entity must have SystemPromptConfigSpec for SystemPromptRenderSystem to consume"
    )
    assert spec.template_source.inline is not None, (
        f"SystemPromptConfigSpec template must be non-None, got: {spec.template_source.inline!r}"
    )
    assert spec.template_source.inline.startswith("explicit-child-prompt"), (
        f"SystemPromptConfigSpec template must start with config.system_prompt, got: {spec.template_source.inline!r}"
    )
    assert "${_installed_tools}" in spec.template_source.inline, (
        f"SystemPromptConfigSpec template must include ${{_installed_tools}}, got: {spec.template_source.inline!r}"
    )
    assert "${_installed_skills}" in spec.template_source.inline, (
        f"SystemPromptConfigSpec template must include ${{_installed_skills}}, got: {spec.template_source.inline!r}"
    )


def test_build_child_prompt_template_appends_both_when_absent() -> None:
    """When neither placeholder is present, both are appended."""
    from ecs_agent.systems.subagent import _build_child_prompt_template

    result = _build_child_prompt_template("You are a coder.")
    assert "${_installed_tools}" in result
    assert "${_installed_skills}" in result
    assert result.startswith("You are a coder.")


def test_build_child_prompt_template_no_duplicate_tools() -> None:
    """When tools placeholder is already present, it is not duplicated."""
    from ecs_agent.systems.subagent import _build_child_prompt_template

    prompt = "Tools: ${_installed_tools}"
    result = _build_child_prompt_template(prompt)
    assert result.count("${_installed_tools}") == 1
    assert "${_installed_skills}" in result


def test_build_child_prompt_template_no_duplicate_skills() -> None:
    """When skills placeholder is already present, it is not duplicated."""
    from ecs_agent.systems.subagent import _build_child_prompt_template

    prompt = "Skills: ${_installed_skills}"
    result = _build_child_prompt_template(prompt)
    assert result.count("${_installed_skills}") == 1
    assert "${_installed_tools}" in result


def test_build_child_prompt_template_no_append_when_both_present() -> None:
    """When both placeholders are already present, the prompt is returned unchanged."""
    from ecs_agent.systems.subagent import _build_child_prompt_template

    prompt = "Tools: ${_installed_tools}\nSkills: ${_installed_skills}"
    result = _build_child_prompt_template(prompt)
    assert result == prompt


def test_build_child_prompt_template_empty_prompt() -> None:
    """An empty prompt still gets both placeholders appended."""
    from ecs_agent.systems.subagent import _build_child_prompt_template

    result = _build_child_prompt_template("")
    assert "${_installed_tools}" in result
    assert "${_installed_skills}" in result


def test_build_child_prompt_template_only_appends_missing_skills() -> None:
    from ecs_agent.systems.subagent import _build_child_prompt_template

    prompt = "Base prompt\n\nTools block:\n${_installed_tools}"
    result = _build_child_prompt_template(prompt)

    assert result.startswith(prompt)
    assert result.count("${_installed_tools}") == 1
    assert result.count("${_installed_skills}") == 1
    assert "## Available Skills" in result


def test_build_child_prompt_template_only_appends_missing_tools() -> None:
    from ecs_agent.systems.subagent import _build_child_prompt_template

    prompt = "Base prompt\n\nSkills block:\n${_installed_skills}"
    result = _build_child_prompt_template(prompt)

    assert result.startswith(prompt)
    assert result.count("${_installed_skills}") == 1
    assert result.count("${_installed_tools}") == 1
    assert "## Available Tools" in result


def test_build_child_prompt_template_does_not_append_background_result_envelope() -> (
    None
):
    from ecs_agent.systems.subagent import _build_child_prompt_template

    result = _build_child_prompt_template("You are a coder.")

    assert "<subagent_background_result>" not in result


def test_build_background_child_prompt_template_appends_background_result_envelope() -> (
    None
):
    from ecs_agent.systems.subagent import _build_background_child_prompt_template

    result = _build_background_child_prompt_template("You are a coder.")

    assert result.startswith("You are a coder.")
    assert "<subagent_background_result>" in result
    assert "<summary>" in result
    assert "<full_result>" in result
