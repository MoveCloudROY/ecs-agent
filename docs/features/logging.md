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

## Internal Usage

All core systems (e.g., `RetryProvider`, `WorldSerializer`) use this structured logging internally. This provides a clear, unified view of system operations without needing to manually add logs to every component.
