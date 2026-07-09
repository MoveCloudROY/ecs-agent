"""PrometheusPlugin lifecycle, configuration, and installation tests."""

from __future__ import annotations

import builtins
import sys
from typing import Any

import httpx
import pytest

from ecs_agent.components.definitions import TerminalComponent
from ecs_agent.core import Runner, World
from ecs_agent.observability.sinks import RecordingTelemetrySink
from ecs_agent.types import RunStartedEvent, SystemExecutionCompletedEvent


class TerminatingSystem:
    """System that terminates a run on its first tick."""

    async def process(self, world: World) -> None:
        """Attach a terminal component."""
        world.add_component(world.create_entity(), TerminalComponent(reason="done"))


def test_prometheus_plugin_satisfies_protocol() -> None:
    """PrometheusPlugin satisfies the runtime-checkable plugin protocol."""
    from ecs_agent.plugins import ObservabilityPlugin
    from ecs_agent.plugins.prometheus import PrometheusPlugin

    plugin = PrometheusPlugin()
    assert isinstance(plugin, ObservabilityPlugin)
    assert plugin.name == "prometheus"


def test_prometheus_module_import_does_not_import_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Importing the plugin module does not require prometheus-client."""
    for module_name in [
        name
        for name in sys.modules
        if name == "prometheus_client" or name.startswith("prometheus_client.")
    ]:
        monkeypatch.delitem(sys.modules, module_name)
    monkeypatch.delitem(sys.modules, "ecs_agent.plugins.prometheus", raising=False)

    original_import = builtins.__import__

    def guarded_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "prometheus_client" or name.startswith("prometheus_client."):
            raise AssertionError("prometheus_client import attempted too early")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    import ecs_agent.plugins.prometheus as prometheus_plugin

    assert prometheus_plugin.PrometheusConfig().start_server is False


@pytest.mark.asyncio
async def test_install_without_client_raises_actionable_import_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Metric creation raises a clear optional-extra ImportError only on install."""
    from ecs_agent.plugins import install_plugins
    from ecs_agent.plugins.prometheus import PrometheusPlugin

    # A None entry in sys.modules makes the import machinery raise
    # ModuleNotFoundError, simulating an uninstalled optional extra.
    monkeypatch.setitem(sys.modules, "prometheus_client", None)

    world = World()
    with pytest.raises(
        ImportError,
        match="Install ecs-agent\\[prometheus\\] to use Prometheus metrics",
    ):
        await install_plugins(world, [PrometheusPlugin()])

    assert getattr(world, "_ecs_agent_plugins_handle", None) is None


def test_prometheus_config_with_env_fills_absent_server_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """with_env resolves server fields from env without overriding explicit values."""
    from ecs_agent.plugins.prometheus import PrometheusConfig

    monkeypatch.setenv("ECS_AGENT_PROMETHEUS_PORT", "9188")
    monkeypatch.setenv("ECS_AGENT_PROMETHEUS_ADDR", "127.0.0.1")

    resolved = PrometheusConfig().with_env()
    assert resolved.port == 9188
    assert resolved.addr == "127.0.0.1"

    explicit = PrometheusConfig(port=9300, addr="0.0.0.0").with_env()
    assert explicit.port == 9300
    assert explicit.addr == "0.0.0.0"


@pytest.mark.asyncio
async def test_plugin_install_wires_event_subscriptions() -> None:
    """Installed plugin observes raw events through the world's bus."""
    from ecs_agent.plugins import install_plugins
    from ecs_agent.plugins.prometheus import PrometheusPlugin, render_metrics

    world = World()
    plugin = PrometheusPlugin()
    await install_plugins(world, [plugin])

    await world.event_bus.publish(
        SystemExecutionCompletedEvent(
            system="tests.DemoSystem", status="success", duration_seconds=0.01
        )
    )

    assert plugin.metrics is not None
    output = render_metrics(plugin.metrics)
    assert b'ecs_agent_system_executions_total{status="success",system="tests.DemoSystem"} 1.0' in output


@pytest.mark.asyncio
async def test_uninstall_plugins_stops_metric_collection() -> None:
    """Uninstalling stops the plugin from observing further events."""
    from ecs_agent.plugins import install_plugins, uninstall_plugins
    from ecs_agent.plugins.prometheus import PrometheusPlugin, render_metrics

    world = World()
    plugin = PrometheusPlugin()
    await install_plugins(world, [plugin])
    await uninstall_plugins(world)

    await world.event_bus.publish(
        RunStartedEvent(max_ticks=1, start_tick=0, active_entities=7)
    )

    assert plugin.metrics is not None
    output = render_metrics(plugin.metrics)
    assert b"ecs_agent_active_entities 7.0" not in output


@pytest.mark.asyncio
async def test_embedded_metrics_server_lifecycle() -> None:
    """start_server=True serves /metrics at install and closes on shutdown."""
    from ecs_agent.plugins import install_plugins
    from ecs_agent.plugins.prometheus import PrometheusConfig, PrometheusPlugin

    world = World()
    plugin = PrometheusPlugin(PrometheusConfig(start_server=True, port=0, addr="127.0.0.1"))
    handle = await install_plugins(world, [plugin])

    assert plugin.server_handle is not None
    port = plugin.server_handle.server.server_address[1]
    async with httpx.AsyncClient() as client:
        response = await client.get(f"http://127.0.0.1:{port}/metrics")
    assert response.status_code == 200
    assert "ecs_agent_runs_total" in response.text

    await handle.shutdown()
    assert plugin.server_handle is None
    with pytest.raises(httpx.ConnectError):
        async with httpx.AsyncClient() as client:
            await client.get(f"http://127.0.0.1:{port}/metrics", timeout=1.0)


@pytest.mark.asyncio
async def test_prometheus_and_langfuse_mount_together() -> None:
    """Both plugin capabilities observe one run on the same world."""
    from ecs_agent.plugins import install_plugins
    from ecs_agent.plugins.langfuse import LangfusePlugin
    from ecs_agent.plugins.prometheus import PrometheusPlugin, render_metrics

    world = World()
    world.register_system(TerminatingSystem(), priority=0)
    sink = RecordingTelemetrySink()
    langfuse_plugin = LangfusePlugin(sink=sink)
    prometheus_plugin = PrometheusPlugin()
    await install_plugins(world, [langfuse_plugin, prometheus_plugin])

    await Runner().run(world, max_ticks=5)

    assert any(record.name == "runner.run" for record in sink.records)
    assert prometheus_plugin.metrics is not None
    output = render_metrics(prometheus_plugin.metrics)
    assert b"ecs_agent_runs_total" in output
    assert b"ecs_agent_runner_ticks_total" in output
