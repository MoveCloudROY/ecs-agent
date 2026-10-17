"""Structured delegation outcomes and the legacy text/tuple boundary."""

import asyncio
import json
from collections.abc import Iterator
from dataclasses import dataclass, replace

from ecs_agent.core.run_result import RunResult, RunStatus
from ecs_agent.types import FileRefPart, ImageUrlPart, Message


def _message_text(message: Message | None) -> str:
    """Preserve non-text result references at the legacy text tool boundary."""
    if message is None:
        return ""
    if not message.parts:
        return message.content
    parts: list[dict[str, str | None]] = []
    for part in message.parts:
        if isinstance(part, FileRefPart):
            parts.append({"type": "file", "file_id": part.file_id, "filename": part.filename})
        elif isinstance(part, ImageUrlPart):
            parts.append({"type": "image_url", "url": part.url, "detail": part.detail})
    return json.dumps({"content": message.content, "parts": parts}, ensure_ascii=False)

LegacyDelegationResult = tuple[str, bool, str | None]


@dataclass(slots=True)
class DelegationResult:
    """Final output is separate from partial output and task failure."""

    status: RunStatus
    final_content: str = ""
    partial_content: str = ""
    error: str | None = None
    run_result: RunResult | None = None

    @property
    def success(self) -> bool:
        """Whether the child reached a verified successful terminal."""
        return self.status is RunStatus.COMPLETED

    @classmethod
    def from_run(cls, result: RunResult) -> "DelegationResult":
        """Derive all delegation fields from the runner's terminal snapshot."""
        return cls(
            status=result.status,
            final_content=_message_text(result.final_message),
            partial_content=_message_text(result.partial_message),
            error=None if result.success else (result.error or f"Subagent stopped: {result.reason}"),
            run_result=result,
        )

    def as_tuple(self) -> LegacyDelegationResult:
        """Adapt to the existing text tool result without relabeling partials."""
        if self.success:
            return self.final_content, True, None
        error = self.error or f"Subagent stopped: {self.status.value}"
        text = error if error.startswith("Error") else f"Error: {error}"
        if self.partial_content:
            text += f"\nPartial output (unfinished):\n{self.partial_content}"
        return text, False, error

    def __iter__(self) -> Iterator[str | bool | None]:
        """Keep tuple-unpacking callers working during migration."""
        return iter(self.as_tuple())


@dataclass(slots=True)
class DelegationOutcomeCapture:
    """Keep the child's final snapshot when cancellation unwinds its caller."""

    run_result: RunResult | None = None
    timeout_scope: asyncio.Timeout | None = None

    def cancellation_reason(self) -> str:
        """Distinguish the enclosing deadline from an external cancellation."""
        if self.timeout_scope is not None and self.timeout_scope.expired():
            return "timeout"
        return "external_cancellation"

    def failure(self, status: RunStatus, error: str) -> DelegationResult:
        """Apply the enclosing timeout/cancellation cause to the child snapshot."""
        if self.run_result is None:
            return DelegationResult(status=status, error=error)
        result = replace(
            self.run_result, status=status, reason=status.value, error=error,
            partial_message=self.run_result.partial_message or self.run_result.final_message,
            final_message=None,
        )
        return DelegationResult.from_run(result)


def normalize_delegation_result(
    value: DelegationResult | LegacyDelegationResult,
) -> DelegationResult:
    """Accept legacy callbacks that already supply an explicit success flag."""
    if isinstance(value, DelegationResult):
        return value
    text, success, error = value
    return DelegationResult(
        status=RunStatus.COMPLETED if success else RunStatus.ERROR,
        final_content=text if success else "",
        error=error or (text if not success else None),
    )
