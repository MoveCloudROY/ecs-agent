"""Tests for prompt section and table rendering."""

import pytest
from ecs_agent.prompts.contracts import PromptSectionSpec
from ecs_agent.prompts.renderers import render_table, render_string
from ecs_agent.prompts.sections import render_sections


def test_render_table_basic() -> None:
    headers = ["Col1", "Col2"]
    rows = [
        ["Val1", "Val2"],
        ["Val3", "Val4"],
    ]
    expected = "| Col1 | Col2 |\n|---|---|\n| Val1 | Val2 |\n| Val3 | Val4 |"
    assert render_table(headers, rows) == expected


def test_render_table_empty_rows() -> None:
    headers = ["Col1", "Col2"]
    rows: list[list[str]] = []
    expected = "| Col1 | Col2 |\n|---|---|"
    assert render_table(headers, rows) == expected


def test_render_string_basic() -> None:
    template = "Hello {name}!"
    variables = {"name": "World"}
    assert render_string(template, variables) == "Hello World!"


def test_render_string_missing_variable() -> None:
    template = "Hello {name}! {missing}"
    variables = {"name": "World"}
    # Should gracefully handle missing variables by leaving them as is or empty?
    # The task says "graceful sparse-data fallback (missing fields do NOT crash)"
    # Let's assume it leaves the placeholder if missing, or replaces with empty string.
    # Usually, leaving it as is is better for debugging, but "sparse-data fallback" might mean empty.
    # Let's check if there's a preference. AGENTS.md doesn't specify.
    # I'll assume it replaces with empty string for now, or just doesn't crash.
    assert render_string(template, variables) == "Hello World! {missing}"


def test_render_sections_priority() -> None:
    sections = [
        PromptSectionSpec(title="Low", lines=["Low priority"], priority=0),
        PromptSectionSpec(title="High", lines=["High priority"], priority=10),
        PromptSectionSpec(title="Mid", lines=["Mid priority"], priority=5),
    ]
    # Higher priority rendered earlier
    rendered = render_sections(sections)
    assert "## High" in rendered
    assert "## Mid" in rendered
    assert "## Low" in rendered

    high_idx = rendered.find("## High")
    mid_idx = rendered.find("## Mid")
    low_idx = rendered.find("## Low")

    assert high_idx < mid_idx < low_idx


def test_render_sections_content() -> None:
    sections = [
        PromptSectionSpec(title="Section 1", lines=["Line 1", "Line 2"]),
    ]
    expected = "## Section 1\n\nLine 1\nLine 2"
    assert render_sections(sections) == expected
