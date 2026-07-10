"""Observability plugin system: one interface for tracing/metrics backends.

Mount any number of observability integrations on a world through a single
installer::

    from ecs_agent.plugins import install_plugins

    handle = await install_plugins(world, [plugin_a, plugin_b])
    ...
    await handle.flush()
    await handle.shutdown()
"""

from ecs_agent.plugins.api import EventSubscription, ObservabilityPlugin
from ecs_agent.plugins.composite import CompositeTelemetrySink
from ecs_agent.plugins.manager import (
    PluginsHandle,
    install_plugins,
    propagate_plugins,
    uninstall_plugins,
)
from ecs_agent.plugins.sink_adapter import TelemetrySinkPlugin

__all__ = [
    "CompositeTelemetrySink",
    "EventSubscription",
    "ObservabilityPlugin",
    "PluginsHandle",
    "TelemetrySinkPlugin",
    "install_plugins",
    "propagate_plugins",
    "uninstall_plugins",
]
