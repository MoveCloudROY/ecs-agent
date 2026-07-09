# Observability Plugins

`ecs_agent.plugins` is the single interface for mounting observability
integrations — tracing backends, metrics backends, exporters — on a `World`.
Any number of plugins mount side by side through one installer with one
lifecycle, replacing per-integration install functions.

```python
from ecs_agent.plugins import install_plugins
from ecs_agent.plugins.langfuse import LangfuseConfig, LangfusePlugin
from ecs_agent.plugins.prometheus import PrometheusConfig, PrometheusPlugin

handle = await install_plugins(
    world,
    [
        LangfusePlugin(LangfuseConfig(environment="production")),
        PrometheusPlugin(PrometheusConfig(start_server=True, port=9100)),
    ],
)

try:
    await Runner().run(world)
finally:
    await handle.flush()
    await handle.shutdown()
```

## Architecture

The framework publishes two complementary telemetry surfaces, and a plugin
declares which one(s) it consumes:

```
                        ┌────────────────────────────────────────────┐
EventBus ──(~35 evts)──▶│ ObservabilitySubscriber                    │
                        │   events → TelemetryRecord/TelemetryScore  │
                        └───────────────┬────────────────────────────┘
                                        ▼
                        CompositeTelemetrySink (per-sink isolation)
                         ├─▶ plugin A. telemetry_sink()   ← record consumers
                         └─▶ plugin B. telemetry_sink()
EventBus ──(raw evts)──────▶ plugin C. event_subscriptions()  ← raw-event consumers
```

- **Record pipeline** (`telemetry_sink()`): the `ObservabilitySubscriber`
  maps runner/system/LLM/tool/stream/subagent lifecycle events into neutral
  `TelemetryRecord` / `TelemetryScore` values with stable trace, run, and
  observation IDs. Tracing backends (Langfuse, future OTLP exporters)
  consume this pipeline. The pipeline is built lazily on the first plugin
  that provides a sink.
- **Raw events** (`event_subscriptions()`): plugins that need
  low-cardinality raw data — or events that never reach the record pipeline
  (tool approvals, message-bus, checkpoint, MCTS) — subscribe to the
  `World` event bus directly. Prometheus metrics consume this surface.

A plugin may implement both capabilities.

## The plugin protocol

`ObservabilityPlugin` is a runtime-checkable `Protocol`:

```python
class ObservabilityPlugin(Protocol):
    name: str                        # unique per world
    propagate_to_children: bool      # opt into child-world raw events

    def telemetry_sink(self) -> TelemetrySink | None: ...
    def event_subscriptions(self, world) -> tuple[EventSubscription, ...]: ...
    async def start(self, world) -> None: ...    # allocate clients/servers
    async def flush(self) -> None: ...
    async def shutdown(self) -> None: ...        # release resources
```

Lifecycle: `start(world)` runs at install (this is where SDK clients and
HTTP servers are created, so constructing a plugin never imports optional
dependencies), then the plugin receives records/events until `shutdown()`
at uninstall. Install-time errors raise loudly and roll the installation
back; runtime errors in one plugin are logged and isolated so they never
break other plugins or the agent run.

## Installer API

```python
from ecs_agent.plugins import (
    install_plugins,     # async: mount plugins, returns PluginsHandle
    uninstall_plugins,   # async: unsubscribe + shutdown everything
    propagate_plugins,   # sync: wire a child world into the parent pipeline
    TelemetrySinkPlugin, # adapter: mount any bare TelemetrySink
)

handle = await install_plugins(world, [plugin_a, plugin_b])
handle.plugins            # installed plugins, in order
handle.plugin("langfuse") # lookup by name
await handle.add(plugin)  # mount another plugin later
await handle.remove("prometheus")  # unmount + shutdown one plugin
await handle.flush()      # flush all plugins (failures isolated)
await handle.shutdown()   # shut all plugins down
await handle.uninstall()  # unsubscribe, shutdown, clean world state
```

- One installation per world: a second `install_plugins` call raises
  `ValueError` until the first is uninstalled.
- Plugin names must be unique per world.
- A custom `TelemetrySink` mounts without writing a plugin class:
  `TelemetrySinkPlugin("my-sink", sink)`.

## Subagent child worlds

Subagent delegation calls `propagate_plugins(parent_world, child_world)`
internally: the child world shares the parent's composite sink, so child
LLM/tool observations land in every mounted record plugin, nested under the
parent's `subagent.<name>` span. Raw-event plugins do not observe child
worlds unless they set `propagate_to_children = True`
(e.g. `PrometheusConfig(propagate_to_children=True)`).

## Standard protocols

- **W3C Trace Context** — cross-world trace propagation uses `traceparent`
  headers (`ecs_agent.observability.generate_traceparent` /
  `propagate_trace_context`), so external systems can join agent traces.
- **Neutral internal schema** — `TelemetryRecord` is the interchange format
  between the subscriber and plugins. It carries the same information as an
  OpenTelemetry span plus LLM-generation fields; the table below maps the
  vocabularies. The framework intentionally does not run the OTel SDK as its
  own pipeline (the Langfuse SDK v4 already embeds an OTel tracer in-process;
  a second tracer stack would fight over context), but the plugin interface
  is sized so an OTLP exporter can be added as a plugin without core changes.

| `TelemetryRecord` field | OpenTelemetry / GenAI semconv | Langfuse |
| --- | --- | --- |
| `trace_id` | trace ID | trace ID |
| `observation_id` / `parent_observation_id` | span ID / parent span ID | observation ID / parent |
| `kind` (`trace`/`span`/`generation`/`tool`/`event`) | span kind + `gen_ai.operation.name` | observation type |
| `name` | span name | observation name |
| `model` | `gen_ai.request.model` | `model` |
| `usage_details.prompt_tokens` | `gen_ai.usage.input_tokens` | `usage_details` |
| `usage_details.completion_tokens` | `gen_ai.usage.output_tokens` | `usage_details` |
| `status` / `error` | span status | level / status message |
| `start_time` / `end_time` / `latency_ms` | span timestamps | timestamps |

## Writing a plugin

A tracing-style plugin implements `telemetry_sink()`; a metrics-style plugin
implements `event_subscriptions()`. Everything else is lifecycle glue:

```python
import json

from ecs_agent.observability.schema import TelemetryRecord, TelemetryScore


class JsonlSink:
    """Append every telemetry record to a JSONL file."""

    def __init__(self, path: str) -> None:
        self._path = path

    async def emit(self, record: TelemetryRecord) -> None:
        with open(self._path, "a", encoding="utf-8") as fp:
            fp.write(json.dumps(record.to_payload()) + "\n")

    async def score(self, score: TelemetryScore) -> None: ...
    async def flush(self) -> None: ...
    async def shutdown(self) -> None: ...


handle = await install_plugins(
    world, [TelemetrySinkPlugin("jsonl", JsonlSink("trace.jsonl"))]
)
```

For a full plugin class (config, env resolution, lazy optional imports,
owned servers), `ecs_agent/plugins/langfuse.py` and
`ecs_agent/plugins/prometheus.py` are the reference implementations.

## Built-in plugins

- [Langfuse tracing](langfuse.md) — `ecs_agent.plugins.langfuse`, extra
  `ecs-agent[langfuse]`.
- [Prometheus metrics](metrics.md) — `ecs_agent.plugins.prometheus`, extra
  `ecs-agent[prometheus]`.
