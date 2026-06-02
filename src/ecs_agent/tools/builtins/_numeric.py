"""Shared numeric parsing helpers for built-in tools."""

from __future__ import annotations


def parse_bounded_integer(
    value: int | str,
    *,
    minimum: int,
    invalid_message: str,
    range_message: str,
) -> int:
    """Parse an integer or decimal string and enforce a minimum value."""
    if type(value) is int:
        parsed = value
    elif isinstance(value, str) and value.isdecimal():
        parsed = int(value)
    else:
        raise ValueError(invalid_message)

    if parsed < minimum:
        raise ValueError(range_message)
    return parsed


__all__ = ["parse_bounded_integer"]
