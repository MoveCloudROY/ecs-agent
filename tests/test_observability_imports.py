from __future__ import annotations

from pathlib import Path

import pytest

from ecs_agent.observability import (
    extract_parent_id,
    extract_trace_id,
    generate_traceparent,
    propagate_trace_context,
)


def test_observability_is_package() -> None:
    import ecs_agent.observability as observability

    assert Path(observability.__file__).name == "__init__.py"


def test_generate_traceparent_has_expected_shape() -> None:
    traceparent = generate_traceparent()
    parts = traceparent.split("-")

    assert len(parts) == 4
    assert parts[0] == "00"
    assert len(parts[1]) == 32
    assert len(parts[2]) == 16
    assert parts[3] == "01"


def test_extract_trace_id_returns_32_hex_chars() -> None:
    traceparent = generate_traceparent()

    assert len(extract_trace_id(traceparent)) == 32


def test_extract_parent_id_returns_16_hex_chars() -> None:
    traceparent = generate_traceparent()

    assert len(extract_parent_id(traceparent)) == 16


def test_propagate_trace_context_preserves_trace_and_flags() -> None:
    parent = generate_traceparent(sampled=False)
    child = propagate_trace_context(parent)

    parent_parts = parent.split("-")
    child_parts = child.split("-")

    assert child_parts[0] == parent_parts[0]
    assert child_parts[1] == parent_parts[1]
    assert child_parts[2] != parent_parts[2]
    assert len(child_parts[2]) == 16
    assert child_parts[3] == parent_parts[3]


@pytest.mark.parametrize(
    ("traceparent", "message"),
    [
        ("00-abc-1234567890abcdef-01", "Invalid trace-id length"),
        ("00-1234567890abcdef1234567890abcdef-1234567890abcdef", "Invalid traceparent format"),
    ],
)
def test_invalid_traceparent_raises_value_error(traceparent: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        extract_trace_id(traceparent)


def test_extract_parent_id_rejects_short_parent_id() -> None:
    with pytest.raises(ValueError, match="Invalid parent-id length"):
        extract_parent_id("00-1234567890abcdef1234567890abcdef-abc-01")
