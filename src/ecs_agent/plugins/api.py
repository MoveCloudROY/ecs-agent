"""Observability plugin contract.

A plugin bundles one observability integration (tracing backend, metrics
backend, exporter) behind a single mountable interface. Capabilities are
structural:

- ``telemetry_sink()`` — joins the neutral telemetry-record pipeline
  (``ObservabilitySubscriber`` → ``TelemetryRecord``/``TelemetryScore``).
- ``event_subscriptions()`` — taps raw EventBus events, for integrations
  that need low-cardinality raw data (e.g. Prometheus counters) or events
  that never reach the record pipeline.

A plugin may provide both capabilities, or only lifecycle behavior.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from ecs_agent.observability.install import EventSubscription
from ecs_agent.observability.sinks import TelemetrySink


@runtime_checkable
class ObservabilityPlugin(Protocol):
    """One observability integration mountable on a world.

    Lifecycle: ``start(world)`` at install (allocate clients/servers) →
    running → ``flush()`` on demand → ``shutdown()`` at uninstall.
    """

    name: str
    propagate_to_children: bool

    def telemetry_sink(self) -> TelemetrySink | None:
        """Return the record-pipeline sink capability, or None."""

    def event_subscriptions(self, world: Any) -> tuple[EventSubscription, ...]:
        """Return raw EventBus subscriptions to register on a world."""

    async def start(self, world: Any) -> None:
        """Allocate plugin resources when installed on a world."""

    async def flush(self) -> None:
        """Flush buffered telemetry."""

    async def shutdown(self) -> None:
        """Release plugin resources."""


__all__ = [
    "EventSubscription",
    "ObservabilityPlugin",
]
