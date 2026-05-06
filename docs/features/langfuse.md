# Langfuse Observability

`ecs_agent.integrations.langfuse` provides a native integration with [Langfuse](https://langfuse.com/) for open-source LLM observability. It captures traces, spans, and observations from the `World` event bus and exports them to Langfuse.

## Installation

The Langfuse integration is optional. Install it with the `langfuse` extra:

```bash
uv pip install "ecs-agent[langfuse]"
```

## Quick Start

Install the observability handler on any `World` before running agents.

```python
import os
from ecs_agent.core import World, Runner
from ecs_agent.integrations.langfuse import install_langfuse_observability, LangfuseConfig

world = World()

# Install Langfuse observability
handle = install_langfuse_observability(
    world,
    LangfuseConfig(
        public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
        secret_key=os.environ["LANGFUSE_SECRET_KEY"],
        host=os.getenv("LANGFUSE_HOST") or os.getenv("LANGFUSE_BASE_URL"),
        environment="production",
    )
)

try:
    await Runner().run(world)
finally:
    # Ensure all traces are flushed before exit
    await handle.flush()
    await handle.shutdown()
```

## Configuration

The integration uses the following environment variables for configuration:

- `LANGFUSE_PUBLIC_KEY`: Your Langfuse project public key.
- `LANGFUSE_SECRET_KEY`: Your Langfuse project secret key.
- `LANGFUSE_HOST` or `LANGFUSE_BASE_URL`: The Langfuse API host.

> **Security Note**: Never hardcode your secret keys in source code. Use environment variables or a secret manager. If your credentials have been exposed outside a secure environment, we recommend a full credential rotation immediately.

## Langfuse Sessions

Langfuse Sessions require `session_id` as a trace-level session attribute, not only as `metadata.session_id`. The SDK v4 adapter creates observations with `start_as_current_observation(...)` and calls `propagate_attributes(session_id=...)` while the root observation is active, so Langfuse can group the complete trace chain in the Sessions UI.

If you provide `LangfuseConfig(session_id="...")`, the value is exported both in sanitized metadata for debugging and through `propagate_attributes(...)` for Langfuse session grouping. A `session_id` placed only inside custom metadata remains metadata-only and is not sufficient for Sessions UI grouping.

## Runner Context vs Langfuse Configuration

`Runner.run(...)` and `LangfuseConfig(...)` configure different layers and do not overwrite each other.

| Source | Parameter | Purpose | Overwrites the other layer? |
| --- | --- | --- | --- |
| `Runner.run(...)` | `trace_id` | ecs-agent trace context for the active or inherited execution chain; maps to Langfuse trace context during export. | Does not overwrite `LangfuseConfig`. |
| `Runner.run(...)` | `run_id` | Internal ecs-agent id for one runner execution chain. | Not a Langfuse `session_id`; does not overwrite it. |
| `Runner.run(...)` | `parent_observation_id` | Parent span id for nested child-world observations. | Does not affect Langfuse configuration. |
| `Runner.run(...)` | `emit_root_trace` | Controls whether this runner emits its own root trace record. | Does not affect Langfuse configuration. |
| `LangfuseConfig(...)` | `session_id` | Langfuse Sessions grouping attribute. | Does not overwrite `run_id` or `trace_id`. |
| `LangfuseConfig(...)` | `user_id` | Langfuse user attribution. | Does not affect runner context. |
| `LangfuseConfig(...)` | `environment`, `release`, `tags` | Langfuse trace/export attributes and metadata. | Does not affect runner context. |
| `LangfuseConfig(...)` | `public_key`, `secret_key`, `host` | Langfuse project connection settings. | Used only by the Langfuse client/export layer. |

`Runner.run(...)` owns ecs-agent execution context. Its observability parameters are internal hooks used mostly for nested runs such as subagents:

- `trace_id`: inherited by child worlds so their observations stay inside the active parent Langfuse trace.
- `run_id`: stable ecs-agent execution-scope id for one runner chain. It is not a Langfuse Session id.
- `parent_observation_id`: default parent span for child-world observations, typically the `subagent.<name>` span id.
- `emit_root_trace`: disables a child world's own root trace when it should be nested under a parent span.

`LangfuseConfig(...)` owns Langfuse project/session/export attributes:

- `session_id`: propagated as a Langfuse Sessions attribute and also included in sanitized metadata for debugging.
- `user_id`: exported as the Langfuse user attribute where supported.
- `environment`, `release`, and `tags`: exported as Langfuse trace/export attributes and metadata.
- `public_key`, `secret_key`, and `host`: used only to connect to the Langfuse project.

For normal top-level runs, prefer the default call:

```python
await Runner().run(world)
```

Do not pass a fixed top-level `trace_id` unless you intentionally want multiple runner calls to be associated with the same Langfuse trace. Subagent internals pass these runner parameters automatically so child LLM/tool observations are nested correctly; application code usually only needs `LangfuseConfig` for session, user, environment, release, and credential settings.

## Trace Structure

The integration treats a `Runner.run()` call as the execution scope, while Langfuse traces are scoped to user turns:

- **Interactive agents**: every `UserInputReceivedEvent` starts a new `user.turn` trace. The trace covers everything that happens from that user input until the next user input or run completion.
- **One-shot agents**: if a world starts with an initial `ConversationComponent` user message and does not receive interactive input, the existing single trace for that run is preserved.
- **Subagents**: each subagent delegation is a `subagent.<name>` span inside the active user-turn trace. LLM calls, tool calls, retrieval, streaming, and other child-world observations from that subagent are nested under the subagent span instead of creating a second top-level trace.

Observations captured include:

- **User Input**: The prompt or input text that started the turn.
- **LLM Generations**: Full prompt and completion details, model name, model parameters, and integer token counters such as `prompt_tokens`, `completion_tokens`, `total_tokens`, and cache token counts when the provider supplies them. `LLM_MODEL` is treated as operational configuration rather than a secret, so model identifiers such as `deepseek-v4-flash` remain visible in the Langfuse model field and dashboards.
- **Tool Calls**: Tool names, arguments, results, errors, and recorded execution duration.
- **Subagent Runs**: Parent spans for delegated child agents, with child LLM/tool observations nested beneath them.
- **Streaming**: Real-time token delivery events under the active generation.
- **Retries**: Automatic retry attempts and reasons.
- **Errors**: Captured system errors and stack traces.
- **Context Pressure**: Information about conversation compaction or windowing.
- **Scores**: Automated evaluation scores if provided.

For Langfuse SDK v4, completed ECS records are exported with the manual observation lifecycle. When the SDK exposes its OpenTelemetry-backed manual span hooks, the adapter creates the observation with the recorded operation `start_time` and then calls `end(end_time=...)` with the recorded operation end timestamp, both converted to Langfuse's nanosecond epoch format. This keeps LLM reasoning, tool execution, and subagent span durations aligned with the actual work rather than the telemetry export call duration, preventing the Langfuse UI from displaying zero-latency observations for operations that already completed before export. Older or test clients that do not expose historical OTel start hooks fall back to `start_observation(...)` plus `end(end_time=...)`. Active root observations still use `start_as_current_observation(...)` so `session_id` can be propagated while the trace context is current.

## Alerts and Monitoring

The integration exports alert-ready score and context data for downstream use within the Langfuse platform. The adapter focuses on exporting the necessary telemetry so these features can be configured within your Langfuse project.

## Data Privacy and Redaction

The integration follows a strict data privacy policy. While raw prompts, responses, tool arguments, and results can be captured for debugging, they are sanitized before export.

- **Mandatory Redaction**: Sensitive patterns (like API keys or tokens) are automatically redacted from payloads.
- **Redaction Reports**: Exported metadata includes counts and names of applied redaction rules, but never the redacted content itself.
- **Model Names**: `LLM_MODEL` is intentionally not redacted because model identifiers drive Langfuse generation grouping and dashboard filters. Do not encode credentials, tenant secrets, or private data in model names.

## Telemetry Resilience

Telemetry failures (e.g., network issues or Langfuse API downtime) do not fail agent runs. The integration captures events asynchronously and handles export errors gracefully to ensure your agent remains operational even if observability is interrupted.

## Live Smoke Tests

The project includes live integration tests that verify the Langfuse adapter with real LLM calls. These tests are optional and skip automatically if the required environment variables are missing.

To run the live tests, set the following environment variables in your shell:

- `RUN_LANGFUSE_LIVE_TESTS`: Set to `1`
- `LANGFUSE_PUBLIC_KEY`: Your public key
- `LANGFUSE_SECRET_KEY`: Your secret key
- `LANGFUSE_HOST`: Your host
- `LLM_API_KEY`: Your LLM API key
- `LLM_BASE_URL`: Your LLM base URL
- `LLM_MODEL`: Your LLM model

Then execute the full suite or specific node tests:

```bash
# Run all Langfuse live tests
uv run pytest tests/live/test_langfuse_observability_live.py -v

# Run specific provider tests
uv run pytest tests/live/test_langfuse_observability_live.py::test_live_langfuse_openai_chat_agent_run -v
uv run pytest tests/live/test_langfuse_observability_live.py::test_live_langfuse_openai_responses_agent_run -v
uv run pytest tests/live/test_langfuse_observability_live.py::test_live_langfuse_anthropic_messages_agent_run -v
```

For the plan-and-task example against an Anthropic-compatible endpoint, enable the example installer and flush on process exit:

```bash
export PLAN_TASK_LANGFUSE=1
export PLAN_TASK_LANGFUSE_ENVIRONMENT=dev
export PLAN_TASK_LANGFUSE_RELEASE=local-test
export PLAN_TASK_LANGFUSE_SESSION_ID="plan-task-dev-1"
export LLM_API_FORMAT=anthropic_messages
export LLM_MODEL=deepseek-v4-flash

# Set LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_BASE_URL,
# LLM_BASE_URL, and LLM_API_KEY in your shell or secret manager first.
uv run python examples/e2e/plan_and_task/main.py
```

Note: Do not commit actual secret values to version control.
