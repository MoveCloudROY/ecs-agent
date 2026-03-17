"""Deterministic renderers for prompt components like tables and strings."""

from __future__ import annotations


def render_table(headers: list[str], rows: list[list[str]]) -> str:
    """Render a markdown table deterministically.

    Args:
        headers: List of column headers.
        rows: List of rows, where each row is a list of cell values.

    Returns:
        A markdown formatted table string.
    """
    if not headers:
        return ""

    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "|" + "|".join(["---"] * len(headers)) + "|"

    lines = [header_line, separator_line]

    for row in rows:
        # Pad row if it has fewer columns than headers
        padded_row = row + [""] * (len(headers) - len(row))
        # Truncate row if it has more columns than headers
        padded_row = padded_row[: len(headers)]
        lines.append("| " + " | ".join(padded_row) + " |")

    return "\n".join(lines)


def render_string(template: str, variables: dict[str, str]) -> str:
    """Render a string template with variables, handling missing keys gracefully.

    Args:
        template: The string template with {placeholder} syntax.
        variables: Dictionary of values to inject.

    Returns:
        The rendered string.
    """
    # Use a custom formatter or just simple replace to avoid KeyError
    result = template
    for key, value in variables.items():
        result = result.replace(f"{{{key}}}", value)
    return result
