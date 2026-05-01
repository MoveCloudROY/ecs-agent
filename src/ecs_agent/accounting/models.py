"""Canonical accounting models for LLM usage, cache metrics, and cost events."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal


LLMInvocationStatus = Literal["success", "error", "cancelled", "unknown"]


class StreamCompleteness(Enum):
    """Completeness state of usage reported for an invocation."""

    COMPLETE = "complete"
    PARTIAL = "partial"
    UNKNOWN = "unknown"


@dataclass(slots=True)
class UsageRecord:
    """Canonical usage schema across provider implementations."""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    cached_input_tokens: int | None = None
    cache_creation_tokens: int | None = None
    cache_read_tokens: int | None = None
    image_count: int | None = None
    audio_seconds: float | None = None
    provider_id: str | None = None
    model: str | None = None
    stream_completeness: StreamCompleteness = StreamCompleteness.COMPLETE


@dataclass(slots=True)
class PromptCacheStats:
    """Per-request prompt cache hit-rate metrics."""

    cache_read_tokens: int
    total_prompt_tokens: int
    hit_rate: float | None


@dataclass(slots=True)
class CostRecord:
    """Computed cost for one LLM invocation."""

    input_cost: float | None = None
    cached_input_cost: float | None = None
    output_cost: float | None = None
    total_cost: float | None = None
    currency: str = "USD"
    is_estimated: bool = False


@dataclass(slots=True)
class LLMInvocationEvent:
    """Single canonical terminal event emitted once per LLM invocation."""

    entity_id: int
    provider_id: str
    model: str
    usage: UsageRecord
    cost: CostRecord | None = None
    cache_stats: PromptCacheStats | None = None
    request_id: str | None = None
    operation: str = "completion"
    status: LLMInvocationStatus = "success"
    streaming: bool = False
    duration_seconds: float | None = None

    def __post_init__(self) -> None:
        interrupted = self.usage.stream_completeness is not StreamCompleteness.COMPLETE
        if interrupted and self.cost is not None and not self.cost.is_estimated:
            raise ValueError(
                "interrupted stream policy violation: cost must be None or estimated"
            )


@dataclass(slots=True)
class LLMRetryEvent:
    """Retry attempt metadata emitted separately from logical LLM invocations."""

    provider_id: str
    model: str
    reason: str
    attempt: int
    operation: str = "completion"
