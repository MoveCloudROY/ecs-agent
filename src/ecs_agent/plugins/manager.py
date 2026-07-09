"""Install, uninstall, and propagate observability plugins on worlds."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ecs_agent.logging import get_logger
from ecs_agent.observability.install import (
    EventSubscription,
    install_observability,
    uninstall_observability,
)
from ecs_agent.plugins.api import ObservabilityPlugin
from ecs_agent.plugins.composite import CompositeTelemetrySink

logger = get_logger(__name__)

_PLUGINS_HANDLE_ATTR = "_ecs_agent_plugins_handle"
_OBSERVABILITY_SINK_ATTR = "_ecs_agent_observability_sink"
_OBSERVABILITY_CONFIG_ATTR = "_ecs_agent_observability_config"


@dataclass(slots=True)
class _PluginEntry:
    """Bookkeeping for one installed plugin."""

    plugin: ObservabilityPlugin
    subscriptions: tuple[EventSubscription, ...]


class PluginsHandle:
    """Handle for the observability plugins installed on one world."""

    def __init__(self, world: Any) -> None:
        self.world = world
        self.composite = CompositeTelemetrySink()
        self._entries: dict[str, _PluginEntry] = {}
        self._uninstalled = False

    @property
    def plugins(self) -> tuple[ObservabilityPlugin, ...]:
        """Return the installed plugins in installation order."""
        return tuple(entry.plugin for entry in self._entries.values())

    def plugin(self, name: str) -> ObservabilityPlugin | None:
        """Return the installed plugin registered under a name, if any."""
        entry = self._entries.get(name)
        return None if entry is None else entry.plugin

    async def add(self, plugin: ObservabilityPlugin) -> None:
        """Start a plugin on this world and mount its capabilities."""
        if self._uninstalled:
            raise ValueError("plugins have been uninstalled from this world")
        if plugin.name in self._entries:
            raise ValueError(
                f"observability plugin {plugin.name!r} is already installed"
            )

        await plugin.start(self.world)

        sink = plugin.telemetry_sink()
        if sink is not None:
            self._ensure_record_pipeline()
            self.composite.add(plugin.name, sink)

        subscriptions = tuple(plugin.event_subscriptions(self.world))
        for event_type, handler in subscriptions:
            self.world.event_bus.subscribe(event_type, handler)

        self._entries[plugin.name] = _PluginEntry(
            plugin=plugin,
            subscriptions=subscriptions,
        )

    async def remove(self, name: str) -> ObservabilityPlugin | None:
        """Unmount a plugin by name, shut it down, and return it."""
        entry = self._entries.pop(name, None)
        if entry is None:
            return None

        for event_type, handler in entry.subscriptions:
            self.world.event_bus.unsubscribe(event_type, handler)
        self.composite.remove(name)
        await self._shutdown_plugin(entry.plugin)
        return entry.plugin

    async def flush(self) -> None:
        """Flush every installed plugin, isolating failures."""
        for entry in list(self._entries.values()):
            try:
                await entry.plugin.flush()
            except Exception as exc:
                logger.error(
                    "plugin_operation_failed",
                    plugin=entry.plugin.name,
                    operation="flush",
                    exception=str(exc),
                )

    async def shutdown(self) -> None:
        """Shut every installed plugin down, isolating failures."""
        for entry in list(self._entries.values()):
            await self._shutdown_plugin(entry.plugin)

    async def uninstall(self) -> None:
        """Unsubscribe all capabilities, shut plugins down, clean the world."""
        if self._uninstalled:
            return
        self._uninstalled = True

        for entry in list(self._entries.values()):
            for event_type, handler in entry.subscriptions:
                self.world.event_bus.unsubscribe(event_type, handler)
            self.composite.remove(entry.plugin.name)

        if getattr(self.world, _OBSERVABILITY_SINK_ATTR, None) is self.composite:
            uninstall_observability(self.world)

        for entry in list(self._entries.values()):
            await self._shutdown_plugin(entry.plugin)
        self._entries.clear()

        if getattr(self.world, _PLUGINS_HANDLE_ATTR, None) is self:
            delattr(self.world, _PLUGINS_HANDLE_ATTR)

    def propagate_to(self, child_world: Any) -> None:
        """Wire this installation's capabilities onto a child world.

        The shared composite sink is installed on the child so child records
        reach every record plugin under the parent's backends. Raw-event
        plugins join only when they set ``propagate_to_children``. Plugins
        are not started again; child worlds reuse parent resources.
        """
        if getattr(self.world, _OBSERVABILITY_SINK_ATTR, None) is self.composite:
            install_observability(child_world, self.composite)

        for entry in self._entries.values():
            if not entry.plugin.propagate_to_children:
                continue
            for event_type, handler in entry.plugin.event_subscriptions(child_world):
                child_world.event_bus.subscribe(event_type, handler)

    def _ensure_record_pipeline(self) -> None:
        """Install the record pipeline (subscriber → composite) on demand."""
        if getattr(self.world, _OBSERVABILITY_SINK_ATTR, None) is self.composite:
            return
        install_observability(self.world, self.composite)

    async def _shutdown_plugin(self, plugin: ObservabilityPlugin) -> None:
        try:
            await plugin.shutdown()
        except Exception as exc:
            logger.error(
                "plugin_operation_failed",
                plugin=plugin.name,
                operation="shutdown",
                exception=str(exc),
            )


async def install_plugins(
    world: Any,
    plugins: Sequence[ObservabilityPlugin] = (),
) -> PluginsHandle:
    """Install observability plugins on a world and return their handle.

    Installs at most once per world; a partial failure rolls the
    installation back (started plugins are shut down) before re-raising.
    """
    existing = getattr(world, _PLUGINS_HANDLE_ATTR, None)
    if isinstance(existing, PluginsHandle):
        raise ValueError("observability plugins are already installed on this world")

    handle = PluginsHandle(world)
    setattr(world, _PLUGINS_HANDLE_ATTR, handle)
    try:
        for plugin in plugins:
            await handle.add(plugin)
    except Exception:
        await handle.uninstall()
        raise
    return handle


async def uninstall_plugins(world: Any) -> PluginsHandle | None:
    """Uninstall the world's observability plugins idempotently."""
    handle = getattr(world, _PLUGINS_HANDLE_ATTR, None)
    if not isinstance(handle, PluginsHandle):
        return None
    await handle.uninstall()
    return handle


def propagate_plugins(parent_world: Any, child_world: Any) -> None:
    """Install the parent world's observability pipeline on a child world.

    Prefers the parent's plugin installation; falls back to sharing a
    directly installed low-level sink so worlds without plugins keep their
    existing delegation behavior. Does nothing when the parent has no
    observability installed.
    """
    handle = getattr(parent_world, _PLUGINS_HANDLE_ATTR, None)
    if isinstance(handle, PluginsHandle):
        handle.propagate_to(child_world)
        return

    parent_sink = getattr(parent_world, _OBSERVABILITY_SINK_ATTR, None)
    if parent_sink is None:
        return
    parent_config = getattr(parent_world, _OBSERVABILITY_CONFIG_ATTR, None)
    install_observability(child_world, parent_sink, config=parent_config)


__all__ = [
    "PluginsHandle",
    "install_plugins",
    "propagate_plugins",
    "uninstall_plugins",
]
