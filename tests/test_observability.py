"""Tests for observability.py W3C TraceContext utilities."""

from pathlib import Path

import pytest

from ecs_agent.observability import (
    extract_parent_id,
    extract_trace_id,
    generate_traceparent,
    propagate_trace_context,
)


def test_generate_traceparent_format() -> None:
    """Test that generated traceparent follows W3C format."""
    traceparent = generate_traceparent()
    parts = traceparent.split("-")

    # W3C format: {version}-{trace-id}-{parent-id}-{trace-flags}
    assert len(parts) == 4, f"Expected 4 parts, got {len(parts)}"

    version, trace_id, parent_id, trace_flags = parts

    # Version is always "00"
    assert version == "00", f"Expected version '00', got '{version}'"

    # Trace ID is 32 hex chars (16 bytes)
    assert len(trace_id) == 32, f"Expected 32 hex chars, got {len(trace_id)}"
    assert all(c in "0123456789abcdef" for c in trace_id), (
        "Trace ID contains non-hex chars"
    )

    # Parent ID is 16 hex chars (8 bytes)
    assert len(parent_id) == 16, f"Expected 16 hex chars, got {len(parent_id)}"
    assert all(c in "0123456789abcdef" for c in parent_id), (
        "Parent ID contains non-hex chars"
    )

    # Trace flags is 2 hex chars (1 byte)
    assert len(trace_flags) == 2, f"Expected 2 hex chars, got {len(trace_flags)}"
    assert trace_flags in ["00", "01"], f"Expected '00' or '01', got '{trace_flags}'"


def test_generate_traceparent_sampled() -> None:
    """Test that sampled flag is set correctly."""
    sampled = generate_traceparent(sampled=True)
    not_sampled = generate_traceparent(sampled=False)

    assert sampled.endswith("-01"), "Sampled traceparent should end with '01'"
    assert not_sampled.endswith("-00"), "Not sampled traceparent should end with '00'"


def test_generate_traceparent_uniqueness() -> None:
    """Test that generated traceparents are unique."""
    traceparents = [generate_traceparent() for _ in range(100)]
    assert len(set(traceparents)) == 100, "Generated traceparents should be unique"


def test_extract_trace_id_valid() -> None:
    """Test extracting trace ID from valid traceparent."""
    traceparent = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
    trace_id = extract_trace_id(traceparent)
    assert trace_id == "4bf92f3577b34da6a3ce929d0e0e4736"


def test_extract_trace_id_invalid_parts() -> None:
    """Test that invalid traceparent format raises ValueError."""
    with pytest.raises(
        ValueError, match="Invalid traceparent format: expected 4 parts"
    ):
        extract_trace_id("00-4bf92f3577b34da6a3ce929d0e0e4736")


def test_extract_trace_id_invalid_length() -> None:
    """Test that invalid trace ID length raises ValueError."""
    with pytest.raises(ValueError, match="Invalid trace-id length"):
        extract_trace_id("00-tooshort-00f067aa0ba902b7-01")


def test_extract_parent_id_valid() -> None:
    """Test extracting parent ID from valid traceparent."""
    traceparent = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
    parent_id = extract_parent_id(traceparent)
    assert parent_id == "00f067aa0ba902b7"


def test_extract_parent_id_invalid_parts() -> None:
    """Test that invalid traceparent format raises ValueError."""
    with pytest.raises(
        ValueError, match="Invalid traceparent format: expected 4 parts"
    ):
        extract_parent_id("00-4bf92f3577b34da6a3ce929d0e0e4736")


def test_extract_parent_id_invalid_length() -> None:
    """Test that invalid parent ID length raises ValueError."""
    with pytest.raises(ValueError, match="Invalid parent-id length"):
        extract_parent_id("00-4bf92f3577b34da6a3ce929d0e0e4736-tooshort-01")


def test_propagate_trace_context_preserves_trace_id() -> None:
    """Test that propagation preserves trace ID."""
    parent = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
    child = propagate_trace_context(parent)

    parent_trace_id = extract_trace_id(parent)
    child_trace_id = extract_trace_id(child)

    assert parent_trace_id == child_trace_id, "Trace ID should be preserved"


def test_propagate_trace_context_changes_parent_id() -> None:
    """Test that propagation generates new parent ID."""
    parent = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
    child = propagate_trace_context(parent)

    parent_parent_id = extract_parent_id(parent)
    child_parent_id = extract_parent_id(child)

    assert parent_parent_id != child_parent_id, "Parent ID should change"


def test_propagate_trace_context_preserves_version_and_flags() -> None:
    """Test that propagation preserves version and trace flags."""
    parent_sampled = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
    parent_not_sampled = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-00"

    child_sampled = propagate_trace_context(parent_sampled)
    child_not_sampled = propagate_trace_context(parent_not_sampled)

    assert child_sampled.startswith("00-"), "Version should be preserved"
    assert child_sampled.endswith("-01"), "Sampled flag should be preserved"
    assert child_not_sampled.endswith("-00"), "Not sampled flag should be preserved"


def test_propagate_trace_context_invalid_format() -> None:
    """Test that propagation rejects invalid traceparent."""
    with pytest.raises(ValueError, match="Invalid traceparent format"):
        propagate_trace_context("invalid-format")


def test_extract_trace_id_from_generated() -> None:
    """Test extracting trace ID from generated traceparent."""
    traceparent = generate_traceparent()
    trace_id = extract_trace_id(traceparent)

    # Verify it's a valid hex string with correct length
    assert len(trace_id) == 32
    assert all(c in "0123456789abcdef" for c in trace_id)


def test_propagate_trace_context_chain() -> None:
    """Test chaining multiple propagations preserves trace ID."""
    original = generate_traceparent()
    child1 = propagate_trace_context(original)
    child2 = propagate_trace_context(child1)
    child3 = propagate_trace_context(child2)

    # All should have the same trace ID
    original_trace_id = extract_trace_id(original)
    assert extract_trace_id(child1) == original_trace_id
    assert extract_trace_id(child2) == original_trace_id
    assert extract_trace_id(child3) == original_trace_id

    # But different parent IDs
    parent_ids = [
        extract_parent_id(original),
        extract_parent_id(child1),
        extract_parent_id(child2),
        extract_parent_id(child3),
    ]
    assert len(set(parent_ids)) == 4, "All parent IDs should be unique"


def test_observability_is_package() -> None:
    import ecs_agent.observability as observability

    assert Path(observability.__file__).name == "__init__.py"


def test_user_input_received_event_legacy_observability_events_import() -> None:
    """Legacy observability.events import remains an alias to the core event type."""
    from ecs_agent.types import UserInputReceivedEvent as CoreUserInputReceivedEvent
    from ecs_agent.observability.events import UserInputReceivedEvent

    assert UserInputReceivedEvent is CoreUserInputReceivedEvent
