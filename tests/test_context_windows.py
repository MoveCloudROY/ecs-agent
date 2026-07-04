"""Tests for the model context-window catalog (ISSUE-5)."""

from __future__ import annotations

from ecs_agent.context_windows import (
    DEFAULT_OUTPUT_RESERVE,
    resolve_context_budget,
    resolve_context_window,
)


def test_resolve_window_by_family_prefix() -> None:
    assert resolve_context_window("claude-opus-4-8") == 200_000
    assert resolve_context_window("gpt-4o-2024-08-06") == 128_000
    assert resolve_context_window("deepseek-v4-flash") == 65_536
    assert resolve_context_window("qwen3.5-flash") == 131_072


def test_longest_prefix_wins() -> None:
    # "gpt-4.1" is more specific than "gpt-4".
    assert resolve_context_window("gpt-4.1-mini") == 1_047_576
    assert resolve_context_window("gpt-4-0613") == 8_192


def test_unknown_model_returns_none() -> None:
    assert resolve_context_window("some-unknown-model") is None
    assert resolve_context_window("") is None
    assert resolve_context_budget("some-unknown-model") is None


def test_budget_reserves_output_space() -> None:
    assert resolve_context_budget("claude-opus-4-8") == 200_000 - DEFAULT_OUTPUT_RESERVE
    assert (
        resolve_context_budget("deepseek-v4-flash", output_reserve=1_000)
        == 65_536 - 1_000
    )


def test_budget_floored_at_one() -> None:
    # An absurd reserve larger than the window floors to 1, never negative.
    assert resolve_context_budget("gpt-4", output_reserve=1_000_000) == 1


def test_case_insensitive() -> None:
    assert resolve_context_window("CLAUDE-OPUS-4-8") == 200_000
