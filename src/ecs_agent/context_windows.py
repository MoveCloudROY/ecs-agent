"""Model context-window catalog.

Maps a model id to its input context window (in tokens) so context management
(trimming / compaction) can derive a token budget from the real window instead
of requiring a hard-coded number. Matching is by id prefix, so family entries
(``claude-``, ``gpt-4o``, ...) cover their variants. The table is intentionally
conservative and easily extended.
"""

from __future__ import annotations

# Prefix -> context window (input tokens). Longest matching prefix wins, so more
# specific entries can override a family default.
CONTEXT_WINDOWS: dict[str, int] = {
    # Anthropic Claude
    "claude-": 200_000,
    # OpenAI
    "gpt-4o": 128_000,
    "gpt-4.1": 1_047_576,
    "gpt-4-turbo": 128_000,
    "gpt-4": 8_192,
    "gpt-3.5": 16_385,
    "o1": 200_000,
    "o3": 200_000,
    # DeepSeek
    "deepseek": 65_536,
    # Alibaba Qwen
    "qwen": 131_072,
    # Google Gemini
    "gemini-1.5": 1_000_000,
    "gemini-2": 1_000_000,
    "gemini": 32_768,
}

# Tokens held back from the window for the model's own response.
DEFAULT_OUTPUT_RESERVE = 8_192


def resolve_context_window(model_id: str) -> int | None:
    """Return the context window for *model_id*, or None when unknown.

    Uses the longest matching prefix in :data:`CONTEXT_WINDOWS`.
    """
    if not model_id:
        return None
    normalized = model_id.strip().lower()
    best_prefix = ""
    best_window: int | None = None
    for prefix, window in CONTEXT_WINDOWS.items():
        if normalized.startswith(prefix) and len(prefix) > len(best_prefix):
            best_prefix = prefix
            best_window = window
    return best_window


def resolve_context_budget(
    model_id: str, *, output_reserve: int = DEFAULT_OUTPUT_RESERVE
) -> int | None:
    """Return a usable input-token budget for *model_id*, or None when unknown.

    The budget is the context window minus ``output_reserve`` (space kept for the
    model's response), floored at 1. Callers treat None as "no derived budget"
    and fall back to an explicit ``max_tokens`` (or skip budget-based trimming).
    """
    window = resolve_context_window(model_id)
    if window is None:
        return None
    return max(1, window - max(0, output_reserve))


__all__ = [
    "CONTEXT_WINDOWS",
    "DEFAULT_OUTPUT_RESERVE",
    "resolve_context_window",
    "resolve_context_budget",
]
