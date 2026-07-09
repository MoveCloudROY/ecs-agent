"""Adapter that mounts a bare TelemetrySink as an observability plugin."""

from __future__ import annotations

from typing import Any

from ecs_agent.observability.install import EventSubscription
from ecs_agent.observability.sinks import TelemetrySink


class TelemetrySinkPlugin:
    """Mount any ``TelemetrySink`` as a record-pipeline plugin."""

    def __init__(
        self,
        name: str,
        sink: TelemetrySink,
        *,
        propagate_to_children: bool = False,
    ) -> None:
        self.name = name
        self.propagate_to_children = propagate_to_children
        self._sink = sink

    def telemetry_sink(self) -> TelemetrySink | None:
        """Return the wrapped sink."""
        return self._sink

    def event_subscriptions(self, world: Any) -> tuple[EventSubscription, ...]:
        """No raw-event capability."""
        _ = world
        return ()

    async def start(self, world: Any) -> None:
        """No resources to allocate."""
        _ = world

    async def flush(self) -> None:
        """Flush the wrapped sink."""
        await self._sink.flush()

    async def shutdown(self) -> None:
        """Shut the wrapped sink down."""
        await self._sink.shutdown()


__all__ = ["TelemetrySinkPlugin"]
