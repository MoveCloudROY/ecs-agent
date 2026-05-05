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

## Trace Structure

The integration produces one trace per `Runner.run()` call. Observations captured include:

- **User Input**: The initial prompt that triggered the run.
- **LLM Generations**: Full prompt and completion details, including token usage and model parameters.
- **Tool Calls**: Tool names, arguments, and results.
- **Streaming**: Real-time token delivery events.
- **Retries**: Automatic retry attempts and reasons.
- **Errors**: Captured system errors and stack traces.
- **Context Pressure**: Information about conversation compaction or windowing.
- **Scores**: Automated evaluation scores if provided.

## Alerts and Monitoring

The integration exports alert-ready score and context data for downstream use within the Langfuse platform. The adapter focuses on exporting the necessary telemetry so these features can be configured within your Langfuse project.

## Data Privacy and Redaction

The integration follows a strict data privacy policy. While raw prompts, responses, tool arguments, and results can be captured for debugging, they are sanitized before export.

- **Mandatory Redaction**: Sensitive patterns (like API keys or tokens) are automatically redacted from payloads.
- **Redaction Reports**: Exported metadata includes counts and names of applied redaction rules, but never the redacted content itself.

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

Note: Do not commit actual secret values to version control.
