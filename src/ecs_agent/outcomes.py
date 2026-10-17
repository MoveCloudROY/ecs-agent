"""Generation outcomes, independent of transport success and token accounting."""

from enum import StrEnum
from typing import Literal


InvocationStatus = Literal[
    "success", "error", "cancelled", "unknown", "incomplete", "filtered"
]

# Interactive hosts may clear these after reporting the outcome, then wait for
# fresh user input. Child runners retain them as their task's terminal result.
RECOVERABLE_REASONING_TERMINALS = (
    "reasoning_complete", "reasoning_incomplete", "reasoning_filtered",
    "reasoning_cancelled", "reasoning_unknown", "reasoning_error", "budget_exhausted",
)


class GenerationStatus(StrEnum):
    """Semantic outcome of one model generation, not of the entire agent run."""

    COMPLETED = "completed"
    TOOL_CALLS = "tool_calls"
    INCOMPLETE = "incomplete"
    FILTERED = "filtered"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"
    ERROR = "error"

    @property
    def invocation_status(self) -> InvocationStatus:
        """Map generation outcomes to the public invocation status vocabulary."""
        if self in (self.COMPLETED, self.TOOL_CALLS):
            return "success"
        if self is self.INCOMPLETE:
            return "incomplete"
        if self is self.FILTERED:
            return "filtered"
        if self is self.CANCELLED:
            return "cancelled"
        if self is self.UNKNOWN:
            return "unknown"
        return "error"


def classify_generation(
    finish_reason: str | None,
    *,
    has_tool_calls: bool,
    legacy_complete: bool = False,
) -> GenerationStatus:
    """Classify a generation without assuming that EOF means completion.

    ``legacy_complete`` is only for a complete non-streaming CompletionResult
    from providers predating finish_reason. Stream consumers must leave it off.
    Callers validate tool completeness and schema before admitting execution.
    """
    if finish_reason is None and legacy_complete:
        finish_reason = "tool_calls" if has_tool_calls else "stop"
    if finish_reason in ("stop", "tool_calls"):
        if has_tool_calls:
            return GenerationStatus.TOOL_CALLS
        return (
            GenerationStatus.COMPLETED
            if finish_reason == "stop"
            else GenerationStatus.ERROR
        )
    return {
        "length": GenerationStatus.INCOMPLETE,
        "content_filter": GenerationStatus.FILTERED,
        "cancelled": GenerationStatus.CANCELLED,
        "error": GenerationStatus.ERROR,
    }.get(finish_reason or "unknown", GenerationStatus.UNKNOWN)
