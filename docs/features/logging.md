# Structured Logging

The framework uses `structlog` for structured logging, allowing for easy parsing and observability in both development and production environments.

## API Reference

The primary way to interact with the logging system is through `configure_logging()` and `get_logger()`.

- `configure_logging(json_output: bool = False, level: str = "INFO", module_levels: dict[str, str] | None = None, colors: bool = True)`: Initializes the global logging configuration with per-module filtering, caller info, and exception formatting.
- `get_logger(name: str)`: Returns a structured logger instance for the given name.

### JSON vs. Console Output
- **`json_output=False`**: Uses `ConsoleRenderer`, providing a colorized, human-readable output that is ideal for local development.
- **`json_output=True`**: Uses `JSONRenderer`, producing machine-readable JSON logs for production observability systems like ELK or Datadog.

### Enhanced Features

- **Caller Information**: Automatically captures file, function, and line number for each log entry
- **Exception Formatting**: Pretty-prints exceptions with full tracebacks
- **Per-Module Log Levels**: Fine-grained control over logging verbosity per module
- **Colored Output**: Syntax-highlighted console output for development
- **Stdlib Bridge**: Redirects standard library logging to structlog

## Logging Processors

The system applies several standard processors to ensure every log entry is consistent:
- `merge_contextvars`: Merges any thread-local or async-local context variables into the log.
- `_filter_by_level`: Filters logs according to the configured logging level.
- `add_log_level`: Adds the severity level (e.g., `info`, `error`) to each entry.
- `TimeStamper(fmt="iso")`: Adds an ISO-formatted timestamp to every log.

## Instrumented Modules

The following core modules and systems have structured logging instrumentation:

### Core Runtime (`core/`)
- **`runner.py`**: Run start/end, tick lifecycle, completion events with timing
- **`system.py`**: System execution start/end, priority group execution, exceptions
- **`world.py`**: Entity creation, component add/remove operations
- **`event_bus.py`**: Publish, deliver, timeout, and response events with topic/subscriber metadata

### Key Systems (`systems/`)
- **`reasoning.py`**: Reasoning start/complete events with model and entity metadata
- **`tool_execution.py`**: Tool invocation, completion (with duration), and failure events
- **`checkpoint.py`**: Checkpoint save events with success status and duration
- **`planning.py`**: Planning request, step completion (with step metadata and duration), and error events

## Logging Levels Policy

Events are logged at appropriate levels:
- **DEBUG**: High-frequency operations, internal state transitions
- **INFO**: Lifecycle milestones (system start, completion, checkpoint save)
- **WARNING**: Recoverable anomalies, degraded performance
- **ERROR**: Failures, exceptions, unrecoverable errors

## Sensitive Data Policy

The logging system enforces strict guardrails to prevent sensitive data leakage:

**FORBIDDEN FIELDS** (never logged):
- Raw conversation message `content` (user or assistant messages)
- Raw tool call `arguments` (may contain secrets, credentials, or PII)
- API keys, tokens, or authentication credentials
- Full HTTP request/response bodies
- Serialized world state payloads (checkpoint snapshots)

**ALLOWED METADATA** (safe to log):
- Entity IDs, system names, model names
- Event names, log levels, timestamps
- Duration metrics (`duration_ms`)
- Success/failure status (`success`, `reason`)
- Tool names (but not arguments)
- Message counts, step indices, step descriptions

All tests in `tests/test_logging.py::TestSensitiveDataPolicy` verify these guardrails are enforced.

## Usage Example

You can import logging utilities from `ecs_agent.logging` or directly from `ecs_agent`.

```python
from ecs_agent.logging import configure_logging, get_logger

# Configure with per-module levels
configure_logging(
    json_output=False,
    level="INFO",
    module_levels={
        "ecs_agent.providers.openai_provider": "DEBUG",
        "ecs_agent.systems.reasoning": "DEBUG",
        "httpx": "WARNING",  # Suppress noisy HTTP logs
    },
    colors=True,  # Enable colored output
)

# Get a logger instance
logger = get_logger(__name__)

# Logs include caller info automatically
logger.info("system_initialized", status="ready")
# Output: [INFO] system_initialized | status=ready | caller=main.py:15:setup()

# Exception formatting
try:
    raise ValueError("Example error")
except Exception as exc:
    logger.error("operation_failed", exception=str(exc))
    # Pretty-printed traceback in output
```

## Structured Event Examples

### Reasoning System
```json
{"event": "reasoning_start", "entity_id": 1, "model": "gpt-4", "system": "ReasoningSystem", "level": "info", "timestamp": "2026-03-05T00:00:00Z"}
{"event": "reasoning_complete", "entity_id": 1, "model": "gpt-4", "system": "ReasoningSystem", "level": "info", "timestamp": "2026-03-05T00:00:05Z"}
```

### Tool Execution System
```json
{"event": "tool_called", "tool_name": "search", "level": "info", "timestamp": "2026-03-05T00:00:00Z"}
{"event": "tool_result", "tool_name": "search", "success": true, "duration_ms": 123.45, "level": "info", "timestamp": "2026-03-05T00:00:00Z"}
{"event": "tool_failed", "tool_name": "missing_tool", "reason": "Error: unknown tool 'missing_tool'", "level": "error", "timestamp": "2026-03-05T00:00:00Z"}
```

### Checkpoint System
```json
{"event": "checkpoint_saved", "success": true, "duration_ms": 0.05, "checkpoint_id": 0, "level": "info", "timestamp": "2026-03-05T00:00:00Z"}
```

### Planning System
```json
{"event": "planning_request", "message_count": 2, "level": "info", "timestamp": "2026-03-05T00:00:00Z"}
{"event": "planning_step_completed", "step_index": 0, "step_description": "Do task A", "duration_ms": 234.56, "level": "info", "timestamp": "2026-03-05T00:00:05Z"}
{"event": "planning_error", "exception": "Provider failed!", "system_name": "PlanningSystem", "level": "error", "timestamp": "2026-03-05T00:00:05Z"}
```

## Internal Usage

Core systems (`Runner`, `World`, `ReasoningSystem`, `ToolExecutionSystem`, `CheckpointSystem`, `PlanningSystem`) use structured logging internally. This provides a clear, unified view of system operations without needing to manually add logs to every component.

For provider-level logging, see the individual provider implementations (`OpenAIProvider`, `ClaudeProvider`, `RetryProvider`) which emit HTTP request/response metadata at DEBUG level.

## Testing & Verification

All logging behavior is verified through comprehensive tests:
- `tests/test_logging.py`: 42 tests covering event contracts, sensitive data policy, and level enforcement
- `tests/test_enhanced_logging.py`: Module-level filtering and caller info tests
- `tests/test_real_llm_integration.py`: Real LLM logging verification (env-gated)

Run logging tests:
```bash
# All logging tests
uv run pytest tests/test_logging.py tests/test_enhanced_logging.py -v

# Sensitive data policy tests only
uv run pytest tests/test_logging.py::TestSensitiveDataPolicy -v

# Real LLM logging (requires LLM_API_KEY)
LLM_API_KEY=$YOUR_KEY uv run pytest tests/test_real_llm_integration.py -k "logging" -v
```
