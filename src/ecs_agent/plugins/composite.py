"""Fan-out telemetry sink with per-sink error isolation."""

from __future__ import annotations

from ecs_agent.logging import get_logger
from ecs_agent.observability.schema import TelemetryRecord, TelemetryScore
from ecs_agent.observability.sinks import TelemetrySink

logger = get_logger(__name__)


class CompositeTelemetrySink:
    """Deliver telemetry to mounted sinks sequentially, isolating failures.

    Fan-out is sequential in mount order so record ordering guarantees that
    single-sink installations rely on (e.g. trace rotation before children)
    hold for every mounted sink. A failing sink is logged and counted; it
    never interrupts other sinks or the agent run.
    """

    def __init__(self) -> None:
        self._sinks: dict[str, TelemetrySink] = {}
        self.failure_count = 0
        self.failures_by_sink: dict[str, int] = {}
        self.last_error: str | None = None

    def add(self, name: str, sink: TelemetrySink) -> None:
        """Mount a sink under a unique name."""
        if name in self._sinks:
            raise ValueError(f"telemetry sink {name!r} is already mounted")
        self._sinks[name] = sink

    def remove(self, name: str) -> TelemetrySink | None:
        """Unmount and return the sink registered under a name, if any."""
        return self._sinks.pop(name, None)

    def sinks(self) -> tuple[tuple[str, TelemetrySink], ...]:
        """Return the mounted (name, sink) pairs in mount order."""
        return tuple(self._sinks.items())

    async def emit(self, record: TelemetryRecord) -> None:
        """Fan a telemetry record out to every mounted sink."""
        for name, sink in list(self._sinks.items()):
            try:
                await sink.emit(record)
            except Exception as exc:
                self._record_failure(name, "emit", exc)

    async def score(self, score: TelemetryScore) -> None:
        """Fan a telemetry score out to every mounted sink."""
        for name, sink in list(self._sinks.items()):
            try:
                await sink.score(score)
            except Exception as exc:
                self._record_failure(name, "score", exc)

    async def flush(self) -> None:
        """Flush every mounted sink."""
        for name, sink in list(self._sinks.items()):
            try:
                await sink.flush()
            except Exception as exc:
                self._record_failure(name, "flush", exc)

    async def shutdown(self) -> None:
        """Shut every mounted sink down."""
        for name, sink in list(self._sinks.items()):
            try:
                await sink.shutdown()
            except Exception as exc:
                self._record_failure(name, "shutdown", exc)

    def _record_failure(self, name: str, operation: str, exc: Exception) -> None:
        self.failure_count += 1
        self.failures_by_sink[name] = self.failures_by_sink.get(name, 0) + 1
        self.last_error = str(exc)
        logger.error(
            "plugin_sink_operation_failed",
            plugin=name,
            operation=operation,
            exception=str(exc),
        )


__all__ = ["CompositeTelemetrySink"]
