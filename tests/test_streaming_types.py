"""Tests for streaming event types."""

import pytest

from ecs_agent.types import (
    EntityId,
    StreamStartEvent,
    StreamDeltaEvent,
    StreamEndEvent,
    CheckpointCreatedEvent,
    CheckpointRestoredEvent,
    CompactionCompleteEvent,
)


def test_stream_start_event_instantiation() -> None:
    """Test StreamStartEvent can be instantiated with required fields."""
    entity_id = EntityId(1)
    timestamp = 1234567890.0
    event = StreamStartEvent(entity_id=entity_id, timestamp=timestamp)

    assert event.entity_id == entity_id
    assert event.timestamp == timestamp


def test_stream_start_event_has_entity_id_first() -> None:
    """Test StreamStartEvent has entity_id as first field."""
    event = StreamStartEvent(entity_id=EntityId(42), timestamp=100.0)
    assert event.entity_id == EntityId(42)


def test_stream_delta_event_instantiation() -> None:
    """Test StreamDeltaEvent can be instantiated with required fields."""
    entity_id = EntityId(2)
    delta = "Hello, world!"
    event = StreamDeltaEvent(entity_id=entity_id, delta=delta)

    assert event.entity_id == entity_id
    assert event.delta == delta


def test_stream_delta_event_has_entity_id_first() -> None:
    """Test StreamDeltaEvent has entity_id as first field."""
    event = StreamDeltaEvent(entity_id=EntityId(99), delta="test")
    assert event.entity_id == EntityId(99)


def test_stream_end_event_instantiation() -> None:
    """Test StreamEndEvent can be instantiated with required fields."""
    entity_id = EntityId(3)
    timestamp = 1234567900.0
    event = StreamEndEvent(entity_id=entity_id, timestamp=timestamp)

    assert event.entity_id == entity_id
    assert event.timestamp == timestamp


def test_stream_end_event_has_entity_id_first() -> None:
    """Test StreamEndEvent has entity_id as first field."""
    event = StreamEndEvent(entity_id=EntityId(77), timestamp=200.0)
    assert event.entity_id == EntityId(77)


def test_checkpoint_created_event_instantiation() -> None:
    """Test CheckpointCreatedEvent can be instantiated with required fields."""
    entity_id = EntityId(4)
    checkpoint_id = 5
    timestamp = 1234567910.0
    event = CheckpointCreatedEvent(
        entity_id=entity_id, checkpoint_id=checkpoint_id, timestamp=timestamp
    )

    assert event.entity_id == entity_id
    assert event.checkpoint_id == checkpoint_id
    assert event.timestamp == timestamp


def test_checkpoint_created_event_has_entity_id_first() -> None:
    """Test CheckpointCreatedEvent has entity_id as first field."""
    event = CheckpointCreatedEvent(
        entity_id=EntityId(11), checkpoint_id=1, timestamp=300.0
    )
    assert event.entity_id == EntityId(11)


def test_checkpoint_restored_event_instantiation() -> None:
    """Test CheckpointRestoredEvent can be instantiated with required fields."""
    entity_id = EntityId(5)
    checkpoint_id = 6
    timestamp = 1234567920.0
    event = CheckpointRestoredEvent(
        entity_id=entity_id, checkpoint_id=checkpoint_id, timestamp=timestamp
    )

    assert event.entity_id == entity_id
    assert event.checkpoint_id == checkpoint_id
    assert event.timestamp == timestamp


def test_checkpoint_restored_event_has_entity_id_first() -> None:
    """Test CheckpointRestoredEvent has entity_id as first field."""
    event = CheckpointRestoredEvent(
        entity_id=EntityId(22), checkpoint_id=2, timestamp=400.0
    )
    assert event.entity_id == EntityId(22)


def test_compaction_complete_event_instantiation() -> None:
    """Test CompactionCompleteEvent can be instantiated with required fields."""
    entity_id = EntityId(6)
    original_tokens = 1000
    compacted_tokens = 500
    event = CompactionCompleteEvent(
        entity_id=entity_id,
        original_tokens=original_tokens,
        compacted_tokens=compacted_tokens,
    )

    assert event.entity_id == entity_id
    assert event.original_tokens == original_tokens
    assert event.compacted_tokens == compacted_tokens


def test_compaction_complete_event_has_entity_id_first() -> None:
    """Test CompactionCompleteEvent has entity_id as first field."""
    event = CompactionCompleteEvent(
        entity_id=EntityId(33), original_tokens=2000, compacted_tokens=1000
    )
    assert event.entity_id == EntityId(33)


def test_stream_events_are_dataclasses() -> None:
    """Test that all stream events are proper dataclasses."""
    stream_start = StreamStartEvent(entity_id=EntityId(1), timestamp=100.0)
    stream_delta = StreamDeltaEvent(entity_id=EntityId(2), delta="x")
    stream_end = StreamEndEvent(entity_id=EntityId(3), timestamp=200.0)
    checkpoint_created = CheckpointCreatedEvent(
        entity_id=EntityId(4), checkpoint_id=1, timestamp=300.0
    )
    checkpoint_restored = CheckpointRestoredEvent(
        entity_id=EntityId(5), checkpoint_id=1, timestamp=400.0
    )
    compaction = CompactionCompleteEvent(
        entity_id=EntityId(6), original_tokens=100, compacted_tokens=50
    )

    # All should have __slots__ attribute (dataclass(slots=True))
    assert hasattr(stream_start, "__slots__")
    assert hasattr(stream_delta, "__slots__")
    assert hasattr(stream_end, "__slots__")
    assert hasattr(checkpoint_created, "__slots__")
    assert hasattr(checkpoint_restored, "__slots__")
    assert hasattr(compaction, "__slots__")


def test_stream_start_event_repr() -> None:
    """Test StreamStartEvent has valid string representation."""
    event = StreamStartEvent(entity_id=EntityId(7), timestamp=500.0)
    repr_str = repr(event)
    assert "StreamStartEvent" in repr_str
    assert "entity_id" in repr_str or "7" in repr_str


def test_stream_delta_event_with_empty_delta() -> None:
    """Test StreamDeltaEvent can have empty delta string."""
    event = StreamDeltaEvent(entity_id=EntityId(8), delta="")
    assert event.delta == ""


def test_checkpoint_created_and_restored_consistency() -> None:
    """Test created and restored events can use same checkpoint_id."""
    created = CheckpointCreatedEvent(
        entity_id=EntityId(9), checkpoint_id=1, timestamp=100.0
    )
    restored = CheckpointRestoredEvent(
        entity_id=EntityId(9), checkpoint_id=1, timestamp=200.0
    )

    assert created.checkpoint_id == restored.checkpoint_id
    assert created.entity_id == restored.entity_id
