"""Plugin core contracts: protocol, fan-out composite sink, and manager."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from ecs_agent.components.definitions import TerminalComponent
from ecs_agent.core import Runner, World
from ecs_agent.observability import (
    RecordingTelemetrySink,
    TelemetryRecord,
    install_observability,
)
from ecs_agent.observability.install import EventSubscription
from ecs_agent.observability.sinks import TelemetrySink
from ecs_agent.types import RunStartedEvent


class OrderRecordingSink(RecordingTelemetrySink):
    """Recording sink that appends its label to a shared order log on emit."""

    def __init__(self, label: str, order_log: list[str]) -> None:
        super().__init__()
        self.label = label
        self.order_log = order_log
        self.flush_count = 0
        self.shutdown_count = 0

    async def emit(self, record: TelemetryRecord) -> None:
        """Record the record and note fan-out order."""
        self.order_log.append(self.label)
        await super().emit(record)

    async def flush(self) -> None:
        """Count flush calls."""
        self.flush_count += 1

    async def shutdown(self) -> None:
        """Count shutdown calls."""
        self.shutdown_count += 1


class FailingSink(RecordingTelemetrySink):
    """Sink whose operations raise after recording the attempt."""

    async def emit(self, record: TelemetryRecord) -> None:
        """Raise on emit."""
        _ = record
        raise RuntimeError("sink emit failed")

    async def flush(self) -> None:
        """Raise on flush."""
        raise RuntimeError("sink flush failed")


class RecordingPlugin:
    """Record-pipeline plugin capturing lifecycle calls for assertions."""

    def __init__(
        self,
        name: str,
        *,
        sink: TelemetrySink | None = None,
        propagate_to_children: bool = False,
    ) -> None:
        self.name = name
        self.propagate_to_children = propagate_to_children
        self.sink: TelemetrySink = RecordingTelemetrySink() if sink is None else sink
        self.started_worlds: list[Any] = []
        self.flush_count = 0
        self.shutdown_count = 0

    def telemetry_sink(self) -> TelemetrySink | None:
        """Expose the record-pipeline capability."""
        return self.sink

    def event_subscriptions(self, world: Any) -> tuple[EventSubscription, ...]:
        """No raw-event capability."""
        _ = world
        return ()

    async def start(self, world: Any) -> None:
        """Record the world this plugin was started on."""
        self.started_worlds.append(world)

    async def flush(self) -> None:
        """Count flush calls."""
        self.flush_count += 1

    async def shutdown(self) -> None:
        """Count shutdown calls."""
        self.shutdown_count += 1


class FlushFailingPlugin(RecordingPlugin):
    """Plugin whose flush and shutdown raise."""

    async def flush(self) -> None:
        """Raise on flush."""
        raise RuntimeError("plugin flush failed")

    async def shutdown(self) -> None:
        """Raise on shutdown."""
        raise RuntimeError("plugin shutdown failed")


class EventTapPlugin:
    """Raw-event plugin capturing RunStartedEvent instances."""

    def __init__(self, name: str = "tap", *, propagate_to_children: bool = False) -> None:
        self.name = name
        self.propagate_to_children = propagate_to_children
        self.events: list[RunStartedEvent] = []
        self.started_worlds: list[Any] = []
        self.shutdown_count = 0

    def telemetry_sink(self) -> TelemetrySink | None:
        """No record-pipeline capability."""
        return None

    def event_subscriptions(self, world: Any) -> tuple[EventSubscription, ...]:
        """Subscribe to run-start events on the given world."""
        _ = world

        async def on_run_started(event: RunStartedEvent) -> None:
            self.events.append(event)

        return ((RunStartedEvent, on_run_started),)

    async def start(self, world: Any) -> None:
        """Record the world this plugin was started on."""
        self.started_worlds.append(world)

    async def flush(self) -> None:
        """No-op flush."""

    async def shutdown(self) -> None:
        """Count shutdown calls."""
        self.shutdown_count += 1


class TerminatingSystem:
    """System that terminates a run on its first tick."""

    async def process(self, world: World) -> None:
        """Attach a terminal component."""
        world.add_component(world.create_entity(), TerminalComponent(reason="done"))


def _record(name: str = "unit.test") -> TelemetryRecord:
    return TelemetryRecord(
        trace_id="trace",
        run_id="run",
        observation_id="obs",
        name=name,
        kind="event",
    )


def test_removed_observability_modules_stay_unreferenced() -> None:
    """Nothing references the deleted pre-plugin module paths."""
    # Assemble the needles at runtime so this guard does not match itself.
    removed_modules = tuple(f"ecs_agent.{name}" for name in ("metrics", "integrations"))
    project_root = Path(__file__).parent.parent
    offenders: list[str] = []
    for directory in ("src", "tests", "examples", "docs"):
        for pattern in ("*.py", "*.md"):
            for path in (project_root / directory).rglob(pattern):
                if "plans" in path.parts or "__pycache__" in path.parts:
                    continue
                content = path.read_text(encoding="utf-8")
                for needle in removed_modules:
                    if needle in content:
                        offenders.append(f"{path.relative_to(project_root)}: {needle}")
    readme = (project_root / "README.md").read_text(encoding="utf-8")
    for needle in removed_modules:
        if needle in readme:
            offenders.append(f"README.md: {needle}")

    assert offenders == []


def test_plugins_public_surface_importable() -> None:
    """Plugin API is importable from the plugins package."""
    from ecs_agent.plugins import (
        CompositeTelemetrySink,
        ObservabilityPlugin,
        PluginsHandle,
        TelemetrySinkPlugin,
        install_plugins,
        propagate_plugins,
        uninstall_plugins,
    )

    assert callable(install_plugins)
    assert callable(uninstall_plugins)
    assert callable(propagate_plugins)
    assert ObservabilityPlugin.__name__ == "ObservabilityPlugin"
    assert PluginsHandle.__name__ == "PluginsHandle"
    assert TelemetrySinkPlugin.__name__ == "TelemetrySinkPlugin"
    assert CompositeTelemetrySink.__name__ == "CompositeTelemetrySink"


def test_plugin_implementations_satisfy_protocol() -> None:
    """Test plugins and the sink adapter satisfy the runtime-checkable protocol."""
    from ecs_agent.plugins import ObservabilityPlugin, TelemetrySinkPlugin

    adapter = TelemetrySinkPlugin("recording", RecordingTelemetrySink())
    assert isinstance(adapter, ObservabilityPlugin)
    assert isinstance(RecordingPlugin("record"), ObservabilityPlugin)
    assert isinstance(EventTapPlugin(), ObservabilityPlugin)


@pytest.mark.asyncio
async def test_install_plugins_starts_plugins_and_returns_handle() -> None:
    """Install starts each plugin on the world and exposes them on the handle."""
    from ecs_agent.plugins import install_plugins

    world = World()
    record_plugin = RecordingPlugin("record")
    tap_plugin = EventTapPlugin()

    handle = await install_plugins(world, [record_plugin, tap_plugin])

    assert record_plugin.started_worlds == [world]
    assert tap_plugin.started_worlds == [world]
    assert {plugin.name for plugin in handle.plugins} == {"record", "tap"}
    assert handle.plugin("record") is record_plugin
    assert handle.plugin("missing") is None


@pytest.mark.asyncio
async def test_install_plugins_rejects_duplicate_plugin_names() -> None:
    """Two plugins with the same name cannot be installed together."""
    from ecs_agent.plugins import install_plugins

    world = World()

    with pytest.raises(ValueError, match="already installed"):
        await install_plugins(world, [RecordingPlugin("dup"), EventTapPlugin(name="dup")])

    assert getattr(world, "_ecs_agent_plugins_handle", None) is None


@pytest.mark.asyncio
async def test_install_plugins_rejects_second_install() -> None:
    """A world accepts only one plugins installation at a time."""
    from ecs_agent.plugins import install_plugins

    world = World()
    await install_plugins(world, [RecordingPlugin("record")])

    with pytest.raises(ValueError, match="already installed"):
        await install_plugins(world, [EventTapPlugin()])


@pytest.mark.asyncio
async def test_record_fanout_preserves_registration_order() -> None:
    """Records fan out to every record plugin in registration order."""
    from ecs_agent.plugins import install_plugins

    world = World()
    order_log: list[str] = []
    first = RecordingPlugin("first", sink=OrderRecordingSink("first", order_log))
    second = RecordingPlugin("second", sink=OrderRecordingSink("second", order_log))
    await install_plugins(world, [first, second])

    composite = getattr(world, "_ecs_agent_observability_sink")
    await composite.emit(_record())

    assert order_log == ["first", "second"]


@pytest.mark.asyncio
async def test_record_fanout_isolates_plugin_sink_errors() -> None:
    """One failing sink neither raises nor starves the other plugins."""
    from ecs_agent.plugins import install_plugins

    world = World()
    failing = RecordingPlugin("failing", sink=FailingSink())
    healthy = RecordingPlugin("healthy")
    await install_plugins(world, [failing, healthy])

    composite = getattr(world, "_ecs_agent_observability_sink")
    await composite.emit(_record())

    healthy_sink = healthy.sink
    assert isinstance(healthy_sink, RecordingTelemetrySink)
    assert len(healthy_sink.records) == 1
    assert composite.failure_count == 1
    assert composite.failures_by_sink == {"failing": 1}


@pytest.mark.asyncio
async def test_handle_flush_and_shutdown_fan_out_with_isolation() -> None:
    """Handle flush/shutdown reach every plugin even when one raises."""
    from ecs_agent.plugins import install_plugins

    world = World()
    failing = FlushFailingPlugin("failing")
    healthy = RecordingPlugin("healthy")
    handle = await install_plugins(world, [failing, healthy])

    await handle.flush()
    await handle.shutdown()

    assert healthy.flush_count == 1
    assert healthy.shutdown_count == 1


@pytest.mark.asyncio
async def test_uninstall_plugins_unsubscribes_and_cleans_world() -> None:
    """Uninstall removes subscriptions, world attrs, and shuts plugins down."""
    from ecs_agent.plugins import install_plugins, uninstall_plugins

    world = World()
    record_plugin = RecordingPlugin("record")
    tap_plugin = EventTapPlugin()
    handle = await install_plugins(world, [record_plugin, tap_plugin])

    removed = await uninstall_plugins(world)
    assert removed is handle
    assert await uninstall_plugins(world) is None
    assert record_plugin.shutdown_count == 1
    assert tap_plugin.shutdown_count == 1
    assert getattr(world, "_ecs_agent_plugins_handle", None) is None
    assert getattr(world, "_ecs_agent_observability_handle", None) is None

    await world.event_bus.publish(
        RunStartedEvent(max_ticks=1, start_tick=0, active_entities=0)
    )
    assert tap_plugin.events == []


@pytest.mark.asyncio
async def test_event_subscription_plugin_receives_raw_events() -> None:
    """Raw-event capability plugins receive events published on the world bus."""
    from ecs_agent.plugins import install_plugins

    world = World()
    tap_plugin = EventTapPlugin()
    await install_plugins(world, [tap_plugin])

    await world.event_bus.publish(
        RunStartedEvent(max_ticks=3, start_tick=0, active_entities=1)
    )

    assert len(tap_plugin.events) == 1
    assert tap_plugin.events[0].max_ticks == 3


@pytest.mark.asyncio
async def test_event_only_install_skips_record_pipeline() -> None:
    """Installing only raw-event plugins does not build the record pipeline."""
    from ecs_agent.plugins import install_plugins

    world = World()
    await install_plugins(world, [EventTapPlugin()])

    assert getattr(world, "_ecs_agent_observability_handle", None) is None
    assert getattr(world, "_ecs_agent_observability_sink", None) is None


@pytest.mark.asyncio
async def test_dynamic_add_wires_new_plugin() -> None:
    """Adding a plugin later starts it and mounts its capabilities."""
    from ecs_agent.plugins import install_plugins

    world = World()
    handle = await install_plugins(world, [])
    record_plugin = RecordingPlugin("record")

    await handle.add(record_plugin)

    assert record_plugin.started_worlds == [world]
    composite = getattr(world, "_ecs_agent_observability_sink")
    await composite.emit(_record())
    record_sink = record_plugin.sink
    assert isinstance(record_sink, RecordingTelemetrySink)
    assert len(record_sink.records) == 1


@pytest.mark.asyncio
async def test_dynamic_remove_unsubscribes_and_shuts_down_plugin() -> None:
    """Removing a plugin unsubscribes it, shuts it down, and frees its name."""
    from ecs_agent.plugins import install_plugins

    world = World()
    tap_plugin = EventTapPlugin()
    handle = await install_plugins(world, [tap_plugin])

    removed = await handle.remove("tap")
    assert removed is tap_plugin
    assert tap_plugin.shutdown_count == 1
    assert await handle.remove("tap") is None

    await world.event_bus.publish(
        RunStartedEvent(max_ticks=1, start_tick=0, active_entities=0)
    )
    assert tap_plugin.events == []

    replacement = EventTapPlugin()
    await handle.add(replacement)
    await world.event_bus.publish(
        RunStartedEvent(max_ticks=1, start_tick=0, active_entities=0)
    )
    assert len(replacement.events) == 1


@pytest.mark.asyncio
async def test_propagate_plugins_shares_composite_sink_with_child() -> None:
    """Child worlds share the parent composite so records reach all plugins."""
    from ecs_agent.plugins import install_plugins, propagate_plugins

    parent = World()
    child = World()
    await install_plugins(parent, [RecordingPlugin("record")])

    propagate_plugins(parent, child)

    parent_sink = getattr(parent, "_ecs_agent_observability_sink")
    child_sink = getattr(child, "_ecs_agent_observability_sink")
    assert child_sink is parent_sink


@pytest.mark.asyncio
async def test_propagate_plugins_low_level_fallback() -> None:
    """Propagation still works when only a bare sink was installed."""
    from ecs_agent.plugins import propagate_plugins

    parent = World()
    child = World()
    sink = RecordingTelemetrySink()
    install_observability(parent, sink)

    propagate_plugins(parent, child)

    assert getattr(child, "_ecs_agent_observability_sink", None) is sink


@pytest.mark.asyncio
async def test_propagate_plugins_without_observability_is_noop() -> None:
    """Propagating from a world with no observability leaves the child untouched."""
    from ecs_agent.plugins import propagate_plugins

    parent = World()
    child = World()

    propagate_plugins(parent, child)

    assert getattr(child, "_ecs_agent_observability_sink", None) is None


@pytest.mark.asyncio
async def test_propagate_plugins_event_capability_opt_in() -> None:
    """Only plugins that opt in receive raw events from child worlds."""
    from ecs_agent.plugins import install_plugins, propagate_plugins

    parent = World()
    child = World()
    opted_in = EventTapPlugin(name="opted-in", propagate_to_children=True)
    opted_out = EventTapPlugin(name="opted-out")
    await install_plugins(parent, [opted_in, opted_out])

    propagate_plugins(parent, child)
    await child.event_bus.publish(
        RunStartedEvent(max_ticks=1, start_tick=0, active_entities=0)
    )

    assert len(opted_in.events) == 1
    assert opted_out.events == []


@pytest.mark.asyncio
async def test_delegation_child_observability_reaches_all_plugins() -> None:
    """Subagent child worlds report records into every mounted record plugin."""
    from ecs_agent.plugins import install_plugins
    from ecs_agent.systems.subagent.delegation import DelegationExecutor

    parent = World()
    child = World()
    first = RecordingPlugin("first")
    second = RecordingPlugin("second")
    await install_plugins(parent, [first, second])

    DelegationExecutor().install_child_observability(
        parent_world=parent,
        child_world=child,
        parent_observation_id="root-observation",
    )

    assert getattr(child, "_ecs_agent_observability_sink") is getattr(
        parent, "_ecs_agent_observability_sink"
    )
    assert getattr(child, "_ecs_agent_parent_observation_id") == "root-observation"

    child.register_system(TerminatingSystem(), priority=0)
    await Runner().run(child, max_ticks=5)

    for plugin in (first, second):
        plugin_sink = plugin.sink
        assert isinstance(plugin_sink, RecordingTelemetrySink)
        assert any(record.name == "runner.run" for record in plugin_sink.records)


@pytest.mark.asyncio
async def test_runner_run_records_reach_all_plugins() -> None:
    """A full runner lifecycle delivers trace records to every record plugin."""
    from ecs_agent.plugins import install_plugins

    world = World()
    world.register_system(TerminatingSystem(), priority=0)
    first = RecordingPlugin("first")
    second = RecordingPlugin("second")
    await install_plugins(world, [first, second])

    await Runner().run(world, max_ticks=5)

    for plugin in (first, second):
        plugin_sink = plugin.sink
        assert isinstance(plugin_sink, RecordingTelemetrySink)
        root_records = [
            record for record in plugin_sink.records if record.name == "runner.run"
        ]
        assert len(root_records) == 1
        assert root_records[0].kind == "trace"

    first_sink = first.sink
    second_sink = second.sink
    assert isinstance(first_sink, RecordingTelemetrySink)
    assert isinstance(second_sink, RecordingTelemetrySink)
    assert [record.name for record in first_sink.records] == [
        record.name for record in second_sink.records
    ]
