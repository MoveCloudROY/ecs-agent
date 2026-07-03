"""Tests for centralized token counting (ISSUE-8).

The real tiktoken path is exercised only when tiktoken (and its offline BPE
ranks) are available; the CJK-aware fallback is always tested and must be a
strict improvement over the old ``len(text.split()) * 1.3`` word heuristic.
"""

from __future__ import annotations

import math

import pytest

from ecs_agent import token_counting
from ecs_agent.token_counting import (
    count_messages_tokens,
    count_tokens,
    tokenizer_available,
)
from ecs_agent.types import Message


def _old_word_heuristic(text: str) -> int:
    return int(math.ceil(len(text.split()) * 1.3))


def test_empty_text_is_zero() -> None:
    assert count_tokens("") == 0


def test_ascii_fallback_matches_chars_per_token() -> None:
    # With no CJK, the fallback reduces to ceil(len / chars_per_token). This is
    # the invariant that keeps existing char-based budget behavior unchanged.
    if tokenizer_available():
        pytest.skip("tiktoken active; heuristic invariant not exercised")
    text = "abcdefgh"  # 8 ASCII chars
    assert count_tokens(text, fallback_chars_per_token=4.0) == 2
    assert count_tokens("a" * 10, fallback_chars_per_token=4.0) == math.ceil(10 / 4)


def test_cjk_beats_word_heuristic() -> None:
    # A whole Chinese sentence has ~1 "word" -> the old heuristic returns ~1,
    # catastrophically under-counting. The new counter must be much larger.
    cjk = "你好世界，这是一个用于测试的中文句子。"
    assert _old_word_heuristic(cjk) <= 2
    assert count_tokens(cjk) >= len(cjk) // 2
    # Fallback path specifically: each CJK char ~= 1 token.
    if not tokenizer_available():
        assert count_tokens(cjk) >= 15


def test_monotonic_in_length() -> None:
    short = count_tokens("hello world")
    long = count_tokens("hello world " * 50)
    assert long > short


def test_count_messages_sums_content() -> None:
    messages = [
        Message(role="user", content="alpha beta"),
        Message(role="assistant", content="gamma delta"),
    ]
    combined = count_tokens("alpha beta" + "gamma delta")
    # count_messages_tokens sums per-message; for the fallback this equals the
    # concatenated count when there is no rounding boundary difference.
    total = count_messages_tokens(messages)
    assert total >= 1
    if not tokenizer_available():
        # Fallback is additive per message here (all ASCII, exact division).
        assert total == count_tokens("alpha beta") + count_tokens("gamma delta")
        assert combined >= 1


def test_none_content_message_is_safe() -> None:
    messages = [Message(role="tool", content="", tool_call_id="t1")]
    assert count_messages_tokens(messages) == 0


@pytest.mark.skipif(
    not tokenizer_available(), reason="tiktoken (with offline ranks) not available"
)
def test_tiktoken_real_counts() -> None:
    # Real BPE: a known short English phrase is a handful of tokens, and CJK is
    # counted per the real tokenizer (not word count).
    assert 1 <= count_tokens("Hello, world!") <= 6
    assert count_tokens("你好世界") >= 2


def test_module_exports() -> None:
    assert hasattr(token_counting, "count_tokens")
    assert hasattr(token_counting, "count_messages_tokens")
    assert hasattr(token_counting, "tokenizer_available")
