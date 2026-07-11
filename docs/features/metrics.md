# Prometheus Metrics

`ecs_agent.plugins.prometheus` provides Prometheus metrics as an
[observability plugin](plugins.md). The plugin subscribes to the `World`
event bus, records framework-owned runtime events into a private
`CollectorRegistry`, and exposes Prometheus text format without using the
global default registry.

## Installation

Prometheus support is optional. Install it with the `prometheus` extra:

```bash
uv pip install "ecs-agent[prometheus]"
```

The `prometheus_client` package is imported lazily when the plugin starts,
so the core library installs and imports without it.

## Quick Start

```python
from ecs_agent.core import Runner, World
from ecs_agent.plugins import install_plugins
from ecs_agent.plugins.prometheus import PrometheusPlugin, render_metrics

world = World()
plugin = PrometheusPlugin()
handle = await install_plugins(world, [plugin])

# Register systems and add agent entities as usual.
await Runner().run(world, max_ticks=3)

body = render_metrics(plugin.metrics)
await handle.shutdown()
```

The plugin mounts alongside any other observability plugin (for example
`LangfusePlugin`) in the same `install_plugins` call. Use
`uninstall_plugins(world)` to remove the event-bus subscriptions; the
recorder on `plugin.metrics` remains renderable for a final scrape or test
assertions.

`PrometheusConfig` controls the recorder and the embedded endpoint:

- `registry` / `metric_contract`: inject a custom `CollectorRegistry` or
  metric contract (defaults: private registry, built-in contract).
- `start_server`, `port`, `addr`: own an embedded `/metrics` HTTP server
  for the lifetime of the installation. `port`/`addr` also resolve from
  `ECS_AGENT_PROMETHEUS_PORT` / `ECS_AGENT_PROMETHEUS_ADDR` when unset,
  falling back to `9100` on `0.0.0.0`.
- `propagate_to_children`: opt into recording events from subagent child
  worlds (off by default).

```python
from ecs_agent.plugins.prometheus import PrometheusConfig, PrometheusPlugin

plugin = PrometheusPlugin(PrometheusConfig(start_server=True, port=9100))
handle = await install_plugins(world, [plugin])
# ... /metrics is live until:
await handle.shutdown()
```

## Endpoint Modes

Besides the embedded server, standalone endpoint helpers use the registry
owned by the `PrometheusMetrics` instance you pass in.

```python
from ecs_agent.plugins.prometheus import (
    make_metrics_asgi_app,
    make_metrics_wsgi_app,
    render_metrics,
    start_metrics_server,
)

metrics = plugin.metrics
body = render_metrics(metrics)
asgi_app = make_metrics_asgi_app(metrics)
wsgi_app = make_metrics_wsgi_app(metrics)
handle = start_metrics_server(9100, addr="127.0.0.1", metrics=metrics)

try:
    ...
finally:
    handle.close(timeout=5)
```

- `render_metrics(metrics)` returns Prometheus text bytes for custom handlers, tests, or CLI output.
- `make_metrics_asgi_app(metrics)` returns a small framework-free ASGI callable suitable for mounting at `/metrics`.
- `make_metrics_wsgi_app(metrics)` returns a framework-free WSGI callable suitable for mounting at `/metrics`.
- `start_metrics_server(port, addr=..., metrics=...)` starts a standalone HTTP server. The returned handle supports `close()`, `shutdown()`, `server_close()`, `join()`, and tuple-unpacking for access to the underlying server/thread.

## Prometheus and Grafana UI Demo

For a runnable local Prometheus + Grafana setup, see [`examples/prometheus/`](../../examples/prometheus/). The demo starts an ecs-agent process with a standalone `:9100/metrics` endpoint, a Docker Compose Prometheus server that scrapes it, and a Grafana instance with an automatically provisioned Prometheus datasource plus `ecs-agent Overview` dashboard. Open `http://localhost:3000` with `admin` / `admin` to view charts, or use `http://localhost:9090` for raw PromQL queries such as `ecs_agent_runs_total`, `ecs_agent_llm_invocations_total`, or `rate(ecs_agent_llm_invocations_total[1m])`.

## Metric Contract

Prometheus names use the `ecs_agent_` prefix. Duration histograms are in seconds. Counters are listed without the Prometheus client's generated `_created` samples.

| Metric | Type | Labels | Meaning |
| --- | --- | --- | --- |
| `ecs_agent_runs_total` | Counter | `status` | Agent run outcomes. |
| `ecs_agent_runner_ticks_total` | Counter | `status` | Completed runner tick outcomes. |
| `ecs_agent_runner_tick_duration_seconds` | Histogram | `status` | Runner tick duration. |
| `ecs_agent_system_executions_total` | Counter | `system`, `status` | System execution outcomes. |
| `ecs_agent_system_execution_duration_seconds` | Histogram | `system`, `status` | Per-system execution duration. |
| `ecs_agent_active_entities` | Gauge | none | Latest observed active entity count. |
| `ecs_agent_llm_invocations_total` | Counter | `provider`, `model`, `operation`, `status`, `streaming` | Logical framework-owned LLM invocation outcomes. |
| `ecs_agent_llm_invocation_duration_seconds` | Histogram | `provider`, `model`, `operation`, `status`, `streaming` | Logical LLM invocation duration. |
| `ecs_agent_llm_tokens_total` | Counter | `provider`, `model`, `token_type` | Normalized prompt/completion/total/cache token usage when providers return usage. |
| `ecs_agent_llm_retries_total` | Counter | `provider`, `model`, `reason` | Retry attempts emitted by retry-capable model wrappers. |
| `ecs_agent_tool_calls_total` | Counter | `tool`, `status` | Tool call outcomes. |
| `ecs_agent_tool_call_duration_seconds` | Histogram | `tool`, `status` | Tool call runtime. |
| `ecs_agent_tool_denied_total` | Counter | `tool`, `reason` | Denied tool attempts. |
| `ecs_agent_tool_approved_total` | Counter | `tool`, `policy` | Approved tool attempts. |
| `ecs_agent_errors_total` | Counter | `system`, `error_type` | Captured system error outcomes without raw exception text. |
| `ecs_agent_terminals_total` | Counter | `reason` | Terminal run outcomes. |
| `ecs_agent_stream_events_total` | Counter | `event`, `status` | Stream start/delta/end/interruption lifecycle counts. |
| `ecs_agent_stream_first_delta_seconds` | Histogram | `provider`, `model`, `operation`, `status` | Time to first stream delta. |
| `ecs_agent_stream_duration_seconds` | Histogram | `provider`, `model`, `operation`, `status` | Total stream duration. |
| `ecs_agent_subagent_lifecycle_total` | Counter | `phase`, `status` | Subagent lifecycle events. |
| `ecs_agent_message_bus_events_total` | Counter | `event`, `operation` | Message-bus publish/deliver/timeout/response style operations. |
| `ecs_agent_checkpoint_operations_total` | Counter | `operation`, `status` | Checkpoint save/restore operations. |
| `ecs_agent_compaction_operations_total` | Counter | `operation`, `status` | Conversation compaction operations. |
| `ecs_agent_mcts_nodes_scored_total` | Counter | `phase`, `status` | MCTS node scoring outcomes without node IDs or scores as labels. |
| `ecs_agent_plan_steps_total` | Counter | `operation`, `status` | Planning and replanning step operations. |
| `ecs_agent_tool_result_cached_total` | Counter | `status` | Tool-result cache writes/reuse outcomes. |

Business and quality metrics are intentionally not inferred in this phase. They need explicit application events in a future extension so the framework does not guess business semantics from prompts, responses, tool payloads, or result text.

## Label Safety Policy

Allowed labels are the low-cardinality contract values: `system`, `status`, `reason`, `operation`, `provider`, `model`, `streaming`, `event`, `phase`, `policy`, `tool`, `error_type`, and `token_type`.

The metrics layer rejects labels outside that set. It never exports ID-class labels such as entity, tool-call, request, correlation, trace, session, checkpoint, node, branch, or message identifiers. It also does not export world names, raw topics, prompts, responses, tool arguments/results, raw exception strings, artifact paths, API keys, or tokens.

Label values are sanitized to bounded strings. Empty values, very long values, whitespace-bearing values, path-like values, and quoted values fall back to `unknown` rather than becoming high-cardinality labels.

## Event Sources

Metrics are recorded from explicit event-bus events rather than log scraping:

- Runner/system lifecycle events feed run, tick, system, terminal, and active-entity metrics.
- LLM invocation, retry, and stream events feed model, token, retry, and stream metrics.
- Tool execution and approval events feed tool counters and durations.
- Delegation, message-bus, checkpoint, compaction, MCTS, plan-step, and tool-result cache events feed runtime-control metrics.

## Live Smoke Test

The live smoke test is optional and environment-gated. It skips cleanly without `LLM_API_KEY` and uses only environment variables for live configuration:

```bash
LLM_API_KEY="$LLM_API_KEY" \
  LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1 \
  LLM_MODEL=qwen3.5-flash \
  LLM_API_FORMAT=openai_chat_completions \
  uv run pytest tests/live/test_prometheus_metrics_live.py -v
```

Supported `LLM_API_FORMAT` values are `openai_chat_completions`, `openai_responses`, and `anthropic_messages`; shorthand values `openai`, `chat`, `responses`, and `anthropic` are also accepted. All live tests resolve the variable through the shared helper in `tests/live/api_format.py`, so the same value drives every live file.

The smoke's per-request LLM timeout defaults to 30 seconds and can be raised for slower models with `LLM_LIVE_TIMEOUT` (seconds). Transient endpoint errors — timeouts, rate limits, 5xx, connection drops — skip the smoke with a diagnostic reason instead of failing; auth and request-mapping errors still fail.
