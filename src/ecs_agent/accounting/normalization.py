"""Normalization utilities for provider usage payloads."""

from __future__ import annotations

from typing import Any

from ecs_agent.accounting.models import PromptCacheStats, UsageRecord


def _extract_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    return value if isinstance(value, int) else None


def normalize_openai_usage(raw: dict[str, Any]) -> UsageRecord:
    """Normalize OpenAI-compatible usage payloads into UsageRecord."""

    prompt_tokens = _extract_int(raw.get("prompt_tokens", raw.get("input_tokens")))
    completion_tokens = _extract_int(
        raw.get("completion_tokens", raw.get("output_tokens"))
    )
    total_tokens = _extract_int(raw.get("total_tokens"))

    prompt_details = raw.get("prompt_tokens_details")
    cached_input_tokens: int | None = None
    if isinstance(prompt_details, dict):
        cached_input_tokens = _extract_int(prompt_details.get("cached_tokens"))

    return UsageRecord(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        cached_input_tokens=cached_input_tokens,
    )


def normalize_anthropic_usage(raw: dict[str, Any]) -> UsageRecord:
    """Normalize Anthropic-compatible usage payloads into UsageRecord."""

    prompt_tokens = _extract_int(raw.get("input_tokens"))
    completion_tokens = _extract_int(raw.get("output_tokens"))
    total_tokens = _extract_int(raw.get("total_tokens"))

    if (
        total_tokens is None
        and prompt_tokens is not None
        and completion_tokens is not None
    ):
        total_tokens = prompt_tokens + completion_tokens

    return UsageRecord(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        cache_creation_tokens=_extract_int(raw.get("cache_creation_input_tokens")),
        cache_read_tokens=_extract_int(raw.get("cache_read_input_tokens")),
    )


def compute_cache_stats(usage: UsageRecord) -> PromptCacheStats | None:
    """Compute prompt-cache metrics from canonical usage fields."""

    has_cache_fields = any(
        token is not None
        for token in (
            usage.cached_input_tokens,
            usage.cache_creation_tokens,
            usage.cache_read_tokens,
        )
    )
    if not has_cache_fields:
        return None

    openai_cached = usage.cached_input_tokens or 0
    cache_read_tokens = usage.cache_read_tokens
    if cache_read_tokens is None:
        cache_read_tokens = openai_cached

    cache_creation_tokens = usage.cache_creation_tokens
    if cache_creation_tokens is None and usage.cached_input_tokens is not None:
        cache_creation_tokens = 0

    prompt_tokens = usage.prompt_tokens or 0
    total_prompt_tokens = (
        (prompt_tokens - openai_cached)
        + (cache_creation_tokens or 0)
        + cache_read_tokens
    )

    hit_rate: float | None = None
    if total_prompt_tokens > 0:
        hit_rate = cache_read_tokens / total_prompt_tokens

    return PromptCacheStats(
        cache_read_tokens=cache_read_tokens,
        total_prompt_tokens=total_prompt_tokens,
        hit_rate=hit_rate,
    )
