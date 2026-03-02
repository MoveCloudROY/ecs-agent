"""W3C TraceContext utilities for distributed tracing.

Implements W3C Trace Context format for traceparent header:
https://www.w3.org/TR/trace-context/

Format: {version}-{trace-id}-{parent-id}-{trace-flags}
Example: 00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01

- version: Always "00"
- trace-id: 32 hex chars (16 bytes)
- parent-id: 16 hex chars (8 bytes)
- trace-flags: 2 hex chars (1 byte, "01" = sampled, "00" = not sampled)
"""

import secrets


def generate_traceparent(sampled: bool = True) -> str:
    """Generate a new W3C TraceContext traceparent header.

    Args:
        sampled: Whether the trace is sampled (default True).

    Returns:
        W3C traceparent string in format: {version}-{trace-id}-{parent-id}-{trace-flags}
    """
    version = "00"
    trace_id = secrets.token_hex(16)  # 16 bytes = 32 hex chars
    parent_id = secrets.token_hex(8)  # 8 bytes = 16 hex chars
    trace_flags = "01" if sampled else "00"
    return f"{version}-{trace_id}-{parent_id}-{trace_flags}"


def extract_trace_id(traceparent: str) -> str:
    """Extract the trace ID from a W3C traceparent header.

    Args:
        traceparent: W3C traceparent string.

    Returns:
        Trace ID (32 hex chars).

    Raises:
        ValueError: If traceparent format is invalid.
    """
    parts = traceparent.split("-")
    if len(parts) != 4:
        raise ValueError(
            f"Invalid traceparent format: expected 4 parts, got {len(parts)}"
        )
    if len(parts[1]) != 32:
        raise ValueError(
            f"Invalid trace-id length: expected 32 hex chars, got {len(parts[1])}"
        )
    return parts[1]


def extract_parent_id(traceparent: str) -> str:
    """Extract the parent ID from a W3C traceparent header.

    Args:
        traceparent: W3C traceparent string.

    Returns:
        Parent ID (16 hex chars).

    Raises:
        ValueError: If traceparent format is invalid.
    """
    parts = traceparent.split("-")
    if len(parts) != 4:
        raise ValueError(
            f"Invalid traceparent format: expected 4 parts, got {len(parts)}"
        )
    if len(parts[2]) != 16:
        raise ValueError(
            f"Invalid parent-id length: expected 16 hex chars, got {len(parts[2])}"
        )
    return parts[2]


def propagate_trace_context(parent_traceparent: str) -> str:
    """Generate a child traceparent from a parent traceparent.

    Preserves the trace ID but generates a new parent ID.

    Args:
        parent_traceparent: Parent W3C traceparent string.

    Returns:
        Child W3C traceparent string with same trace ID, new parent ID.

    Raises:
        ValueError: If parent_traceparent format is invalid.
    """
    parts = parent_traceparent.split("-")
    if len(parts) != 4:
        raise ValueError(
            f"Invalid traceparent format: expected 4 parts, got {len(parts)}"
        )
    version = parts[0]
    trace_id = parts[1]
    trace_flags = parts[3]
    new_parent_id = secrets.token_hex(8)  # Generate new parent ID
    return f"{version}-{trace_id}-{new_parent_id}-{trace_flags}"
