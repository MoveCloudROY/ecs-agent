"""Centralized token counting.

Prefers a real BPE tokenizer (``tiktoken``, an optional dependency) so that
context-budget and compaction thresholds reflect actual token usage. When
``tiktoken`` is not installed, falls back to a CJK-aware character heuristic that
is far closer to reality than a word count — a Chinese sentence has almost no
whitespace, so ``len(text.split())`` catastrophically under-counts it (ISSUE-8).

Usage::

    from ecs_agent.token_counting import count_tokens, count_messages_tokens
    n = count_tokens("你好世界")               # real BPE count when tiktoken present
    n = count_messages_tokens(conversation)    # sum over message .content

The counter is provider-neutral: ``cl100k_base`` is a stable, offline encoding
that approximates modern Claude/OpenAI/others well enough for budget decisions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from ecs_agent.types import Message

_DEFAULT_ENCODING = "cl100k_base"
_DEFAULT_CHARS_PER_TOKEN = 4.0

# Lazily-loaded tiktoken encoder. ``_encoder_loaded`` distinguishes "not tried
# yet" from "tried and unavailable" so we only pay the import/lookup cost once.
_encoder: object | None = None
_encoder_loaded = False


def _get_encoder() -> object | None:
    global _encoder, _encoder_loaded
    if not _encoder_loaded:
        _encoder_loaded = True
        try:
            import tiktoken

            _encoder = tiktoken.get_encoding(_DEFAULT_ENCODING)
        except Exception:
            # tiktoken not installed, or encoding data unavailable offline.
            _encoder = None
    return _encoder


def tokenizer_available() -> bool:
    """Return True when a real BPE tokenizer backs :func:`count_tokens`."""
    return _get_encoder() is not None


def count_tokens(
    text: str,
    *,
    fallback_chars_per_token: float = _DEFAULT_CHARS_PER_TOKEN,
) -> int:
    """Count tokens in *text*.

    Uses ``tiktoken`` when available; otherwise a CJK-aware character heuristic
    where each CJK character counts as roughly one token and other characters
    are divided by ``fallback_chars_per_token``.
    """
    if not text:
        return 0

    encoder = _get_encoder()
    if encoder is not None:
        # disallowed_special=() prevents ValueErrors on texts that happen to
        # contain special-token markup (e.g. "<|endoftext|>").
        return len(encoder.encode(text, disallowed_special=()))  # type: ignore[attr-defined]

    return _heuristic_tokens(text, fallback_chars_per_token)


def count_messages_tokens(
    messages: Iterable[Message],
    *,
    fallback_chars_per_token: float = _DEFAULT_CHARS_PER_TOKEN,
) -> int:
    """Sum token counts across message ``.content`` bodies."""
    return sum(
        count_tokens(
            message.content or "",
            fallback_chars_per_token=fallback_chars_per_token,
        )
        for message in messages
    )


def _heuristic_tokens(text: str, chars_per_token: float) -> int:
    from math import ceil

    safe_chars_per_token = chars_per_token if chars_per_token > 0 else _DEFAULT_CHARS_PER_TOKEN
    cjk = 0
    other = 0
    for char in text:
        if _is_cjk(char):
            cjk += 1
        else:
            other += 1
    return ceil(cjk + other / safe_chars_per_token)


def _is_cjk(char: str) -> bool:
    """Whether *char* is a CJK/Kana/Hangul character (roughly one token each)."""
    code = ord(char)
    return (
        0x2E80 <= code <= 0x9FFF  # CJK radicals, Kangxi, Hiragana/Katakana, CJK ideographs
        or 0xA960 <= code <= 0xA97F  # Hangul Jamo Extended-A
        or 0xAC00 <= code <= 0xD7FF  # Hangul syllables
        or 0xF900 <= code <= 0xFAFF  # CJK compatibility ideographs
        or 0xFF00 <= code <= 0xFFEF  # Halfwidth/Fullwidth forms
        or 0x20000 <= code <= 0x3FFFF  # CJK extensions B–G
    )


__all__ = [
    "count_tokens",
    "count_messages_tokens",
    "tokenizer_available",
]
