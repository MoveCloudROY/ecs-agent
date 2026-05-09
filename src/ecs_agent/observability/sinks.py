"""Telemetry sink contracts and test-friendly implementations."""

from __future__ import annotations

from typing import Protocol

from ecs_agent.observability.schema import TelemetryRecord, TelemetryScore


class TelemetrySink(Protocol):
    """Async sink interface for internal telemetry records and scores."""

    async def emit(self, record: TelemetryRecord) -> None:
        """Emit a telemetry record."""

    async def score(self, score: TelemetryScore) -> None:
        """Emit a telemetry score."""

    async def flush(self) -> None:
        """Flush buffered telemetry."""

    async def shutdown(self) -> None:
        """Release sink resources."""


class NoOpTelemetrySink:
    """Telemetry sink that accepts all operations without side effects."""

    async def emit(self, record: TelemetryRecord) -> None:
        """Accept a telemetry record silently."""
        _ = record

    async def score(self, score: TelemetryScore) -> None:
        """Accept a telemetry score silently."""
        _ = score

    async def flush(self) -> None:
        """Flush silently."""

    async def shutdown(self) -> None:
        """Shutdown silently."""


class RecordingTelemetrySink:
    """Telemetry sink that records emitted data in insertion order for tests."""

    def __init__(self) -> None:
        self.records: list[TelemetryRecord] = []
        self.scores: list[TelemetryScore] = []
        self.operations: list[tuple[str, TelemetryRecord | TelemetryScore]] = []

    async def emit(self, record: TelemetryRecord) -> None:
        """Record a telemetry record."""
        self.records.append(record)
        self.operations.append(("emit", record))

    async def score(self, score: TelemetryScore) -> None:
        """Record a telemetry score."""
        self.scores.append(score)
        self.operations.append(("score", score))

    async def flush(self) -> None:
        """Flush recorded telemetry without changing recorded state."""

    async def shutdown(self) -> None:
        """Shutdown without changing recorded state."""


__all__ = [
    "NoOpTelemetrySink",
    "RecordingTelemetrySink",
    "TelemetrySink",
]
