# Prometheus + Grafana Dashboard Demo

This example connects a running ecs-agent process to a real Prometheus scrape and
a Grafana dashboard.

The demo runs the agent on the host machine, listens on `0.0.0.0:9100`, and is
available locally at `http://127.0.0.1:9100/metrics`. Prometheus runs in Docker
and scrapes that endpoint through `host.docker.internal:9100`.

## 1. Start the ecs-agent metrics endpoint

From the repository root:

```bash
uv run python examples/prometheus/agent_metrics_demo.py
```

The script uses `FakeModel`, so no `LLM_API_KEY` is required. It records a demo
agent run every five seconds and keeps the `/metrics` server alive until you
press `Ctrl+C`.

For a single smoke run:

```bash
uv run python examples/prometheus/agent_metrics_demo.py --iterations 1
```

## 2. Start Prometheus and Grafana

In another terminal:

```bash
cd examples/prometheus
docker compose up
```

Open Grafana:

```text
http://localhost:3000
```

Use the default demo credentials:

```text
username: admin
password: admin
```

Grafana automatically provisions:

- a Prometheus datasource pointing at `http://prometheus:9090`
- an `ecs-agent Overview` dashboard in the `ecs-agent` folder

Prometheus is also available for raw PromQL exploration:

```text
http://localhost:9090
```

## 3. View ecs-agent charts

Open Grafana, then navigate to:

```text
Dashboards → ecs-agent → ecs-agent Overview
```

The dashboard includes charts for:

- agent run rate by status
- LLM invocation rate by provider/model/operation/status
- p95 LLM duration
- LLM token throughput by token type
- total captured errors
- runner tick rate
- tool call rate
- stream event rate and stream p95 duration

## 4. Query ecs-agent metrics manually

Try these PromQL queries in the Prometheus expression browser:

```promql
ecs_agent_runs_total
ecs_agent_runner_ticks_total
ecs_agent_llm_invocations_total
ecs_agent_llm_tokens_total
rate(ecs_agent_llm_invocations_total[1m])
histogram_quantile(0.95, rate(ecs_agent_llm_invocation_duration_seconds_bucket[5m]))
```

You should see the `job="ecs-agent-demo"` label on scraped samples. The same
queries are used in the Grafana dashboard panels. The ecs-agent metrics
themselves keep only low-cardinality labels such as `status`, `system`,
`provider`, `model`, `operation`, and `streaming`; they do not expose entity IDs,
tool-call IDs, raw prompts, responses, API keys, or tokens.

## Troubleshooting

- If Grafana has no data, open Prometheus at `http://localhost:9090` and check
  **Status → Target health** for the `ecs-agent-demo` target.
- If Prometheus shows the target as down, confirm the demo is still running and
  `curl http://127.0.0.1:9100/metrics` returns `ecs_agent_*` output.
- On Linux, this compose file maps `host.docker.internal` to Docker's host
  gateway. If your Docker runtime does not support `host-gateway`, replace the
  target in `prometheus.yml` with your host IP.
- If port `9100` is already in use, start the demo with `--metrics-port <port>`
  and update `prometheus.yml` to match.
