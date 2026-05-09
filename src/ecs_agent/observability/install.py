"""Observability installation helpers for ECS worlds."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from ecs_agent.observability.sinks import TelemetrySink
from ecs_agent.observability.subscriber import ObservabilitySubscriber


EventSubscription = tuple[type[Any], Callable[[Any], Awaitable[None]]]


@dataclass(slots=True)
class ObservabilityHandle:
    """Handle for an observability installation on one world."""

    world: Any
    sink: TelemetrySink
    config: Any = None

    async def flush(self) -> None:
        """Flush the installed telemetry sink."""
        await self.sink.flush()

    async def shutdown(self) -> None:
        """Shutdown the installed telemetry sink."""
        await self.sink.shutdown()

    def uninstall(self) -> ObservabilityHandle | None:
        """Uninstall observability from the associated world."""
        return uninstall_observability(self.world)


def install_observability(
    world: Any,
    sink: TelemetrySink,
    config: Any = None,
) -> ObservabilityHandle:
    """Install observability bookkeeping on a world idempotently."""
    installed_handle = getattr(world, "_ecs_agent_observability_handle", None)
    installed_sink = getattr(world, "_ecs_agent_observability_sink", None)
    if isinstance(installed_handle, ObservabilityHandle):
        if installed_sink is sink:
            return installed_handle
        raise ValueError("observability is already installed on this world with a different sink")

    subscriber = ObservabilitySubscriber(sink, world=world)
    subscriptions = list(subscriber.subscriptions())
    for event_type, handler in subscriptions:
        world.event_bus.subscribe(event_type, handler)

    handle = ObservabilityHandle(world=world, sink=sink, config=config)
    setattr(world, "_ecs_agent_observability_handle", handle)
    setattr(world, "_ecs_agent_observability_sink", sink)
    setattr(world, "_ecs_agent_observability_config", config)
    setattr(world, "_ecs_agent_observability_subscriber", subscriber)
    setattr(world, "_ecs_agent_observability_subscriptions", subscriptions)
    return handle


def uninstall_observability(world: Any) -> ObservabilityHandle | None:
    """Unsubscribe observability handlers from a world idempotently."""
    installed = getattr(world, "_ecs_agent_observability_handle", None)
    subscriptions = getattr(world, "_ecs_agent_observability_subscriptions", ())
    if not isinstance(installed, ObservabilityHandle):
        return None

    for event_type, handler in subscriptions:
        world.event_bus.unsubscribe(event_type, handler)

    for attribute in (
        "_ecs_agent_observability_handle",
        "_ecs_agent_observability_sink",
        "_ecs_agent_observability_config",
        "_ecs_agent_observability_subscriber",
        "_ecs_agent_observability_subscriptions",
    ):
        if hasattr(world, attribute):
            delattr(world, attribute)
    return installed


__all__ = [
    "EventSubscription",
    "ObservabilityHandle",
    "install_observability",
    "uninstall_observability",
]
