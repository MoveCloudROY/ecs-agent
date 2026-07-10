"""LangfusePlugin lifecycle, configuration, and installation tests."""

from __future__ import annotations

import builtins
import sys
from typing import Any

import pytest

from ecs_agent.core import World
from ecs_agent.observability.sinks import NoOpTelemetrySink, RecordingTelemetrySink


class LifecycleRecordingSink(RecordingTelemetrySink):
    """Recording sink counting lifecycle calls."""

    def __init__(self) -> None:
        super().__init__()
        self.flush_count = 0
        self.shutdown_count = 0

    async def flush(self) -> None:
        """Count flush calls."""
        self.flush_count += 1

    async def shutdown(self) -> None:
        """Count shutdown calls."""
        self.shutdown_count += 1


def test_langfuse_plugin_module_import_does_not_import_sdk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Importing the plugin module does not require the Langfuse SDK."""
    original_import = builtins.__import__

    def guarded_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "langfuse" or name.startswith("langfuse."):
            raise AssertionError("langfuse import attempted too early")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    import ecs_agent.plugins.langfuse as langfuse_plugin

    assert langfuse_plugin.LangfuseConfig().enabled is True


def test_langfuse_plugin_satisfies_protocol() -> None:
    """LangfusePlugin satisfies the runtime-checkable plugin protocol."""
    from ecs_agent.plugins import ObservabilityPlugin
    from ecs_agent.plugins.langfuse import LangfusePlugin

    plugin = LangfusePlugin()
    assert isinstance(plugin, ObservabilityPlugin)
    assert plugin.name == "langfuse"


@pytest.mark.asyncio
async def test_install_without_sdk_raises_actionable_import_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Client creation raises a clear optional-extra ImportError only on install."""
    from ecs_agent.plugins import install_plugins
    from ecs_agent.plugins.langfuse import LangfusePlugin

    original_import = builtins.__import__

    def missing_langfuse(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "langfuse" or name.startswith("langfuse."):
            raise ImportError("not installed")
        return original_import(name, *args, **kwargs)

    for module_name in [
        name
        for name in sys.modules
        if name == "langfuse" or name.startswith("langfuse.")
    ]:
        monkeypatch.delitem(sys.modules, module_name)
    monkeypatch.setattr(builtins, "__import__", missing_langfuse)

    world = World()
    with pytest.raises(
        ImportError,
        match="Install ecs-agent\\[langfuse\\] to use Langfuse observability",
    ):
        await install_plugins(world, [LangfusePlugin()])

    assert getattr(world, "_ecs_agent_plugins_handle", None) is None
    assert getattr(world, "_ecs_agent_observability_sink", None) is None


@pytest.mark.asyncio
async def test_plugin_resolves_env_only_for_absent_config_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Install resolves Langfuse env aliases without overriding explicit config."""
    from ecs_agent.plugins import install_plugins
    from ecs_agent.plugins.langfuse import LangfuseConfig, LangfusePlugin

    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "env-public-value")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "env-private-value")
    monkeypatch.setenv("LANGFUSE_BASE_URL", "env-base-url-value")
    monkeypatch.setenv("LANGFUSE_HOST", "env-host-value")
    sink = RecordingTelemetrySink()

    world = World()
    plugin = LangfusePlugin(
        LangfuseConfig(public_key="explicit-public", host=None),
        sink=sink,
    )
    handle = await install_plugins(world, [plugin])

    assert handle.plugin("langfuse") is plugin
    assert plugin.config.public_key == "explicit-public"
    assert plugin.config.secret_key == "env-private-value"
    assert plugin.config.host == "env-host-value"
    assert plugin.config.enabled is True
    assert plugin.telemetry_sink() is sink


@pytest.mark.asyncio
async def test_disabled_config_installs_noop_sink() -> None:
    """A disabled config mounts a no-op sink instead of creating a client."""
    from ecs_agent.plugins import install_plugins
    from ecs_agent.plugins.langfuse import LangfuseConfig, LangfusePlugin

    world = World()
    plugin = LangfusePlugin(LangfuseConfig(enabled=False))
    await install_plugins(world, [plugin])

    assert isinstance(plugin.telemetry_sink(), NoOpTelemetrySink)


@pytest.mark.asyncio
async def test_injected_client_reaches_langfuse_sink() -> None:
    """An injected client is wrapped in a LangfuseTelemetrySink at start."""
    from ecs_agent.plugins import install_plugins
    from ecs_agent.plugins.langfuse import (
        LangfuseConfig,
        LangfusePlugin,
        LangfuseTelemetrySink,
    )

    class StubClient:
        """Minimal stand-in for a Langfuse client."""

    client = StubClient()
    world = World()
    plugin = LangfusePlugin(LangfuseConfig(public_key="pk", secret_key="sk"), client=client)
    await install_plugins(world, [plugin])

    sink = plugin.telemetry_sink()
    assert isinstance(sink, LangfuseTelemetrySink)
    assert sink.client is client


@pytest.mark.asyncio
async def test_flush_and_shutdown_delegate_to_sink() -> None:
    """Plugin flush/shutdown reach the mounted sink exactly once."""
    from ecs_agent.plugins import install_plugins
    from ecs_agent.plugins.langfuse import LangfusePlugin

    world = World()
    sink = LifecycleRecordingSink()
    handle = await install_plugins(world, [LangfusePlugin(sink=sink)])

    await handle.flush()
    await handle.shutdown()

    assert sink.flush_count == 1
    assert sink.shutdown_count == 1
