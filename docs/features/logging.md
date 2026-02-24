# Structured Logging

The framework uses `structlog` for structured logging, allowing for easy parsing and observability in both development and production environments.

## API Reference

The primary way to interact with the logging system is through `configure_logging()` and `get_logger()`.

- `configure_logging(json_output: bool = False, level: str = "INFO")`: Initializes the global logging configuration.
- `get_logger(name: str)`: Returns a structured logger instance for the given name.

### JSON vs. Console Output
- **`json_output=False`**: Uses `ConsoleRenderer`, providing a colorized, human-readable output that is ideal for local development.
- **`json_output=True`**: Uses `JSONRenderer`, producing machine-readable JSON logs for production observability systems like ELK or Datadog.

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

# 1. Initialize global logging
configure_logging(json_output=False, level="DEBUG")

# 2. Get a logger instance
logger = get_logger(__name__)

# 3. Use structured logging
logger.info("system_initialized", status="ready", components=["llm", "world"])
logger.debug("received_response", response_length=150, latency=0.45)
logger.warning("retry_attempt", attempt=2, error="Rate limit exceeded")
logger.error("connection_failed", error="Endpoint not reachable")
```

## Internal Usage

All core systems (e.g., `RetryProvider`, `WorldSerializer`) use this structured logging internally. This provides a clear, unified view of system operations without needing to manually add logs to every component.
