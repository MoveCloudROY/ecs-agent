# Retry Logic

Transient failures such as rate limits (HTTP 429) or temporary server issues (HTTP 5xx) are common when working with LLM APIs. `RetryModel` provides a transparent way to add exponential backoff and retry logic to any `LLMModel`.

## Overview

`RetryModel` is a wrapper that implements the same `LLMModel` interface. It uses the `tenacity` library to handle retries with exponential backoff.

- **Non-streaming calls**: Retried automatically based on the configuration.
- **Streaming calls**: Bypassed directly to the base provider (not retried).

## Configuration

You can customize the retry behavior using the `RetryConfig` dataclass.

```python
from ecs_agent.types import RetryConfig

config = RetryConfig(
    max_attempts=3,                 # Default: 3
    multiplier=1.0,                 # Default: 1.0
    min_wait=4.0,                   # Default: 4.0 seconds
    max_wait=60.0,                  # Default: 60.0 seconds
    retry_status_codes=(429, 500, 502, 503, 504) # Default
)
```

### Retry Criteria
`RetryModel` will attempt a retry if:
- It receives an `httpx.HTTPStatusError` with a status code included in `retry_status_codes`.
- It encounters an `httpx.RequestError` (like network timeouts or connection issues).

## Usage Example

Wrap any existing model (such as one returned by `Model(...)`) with `RetryModel`.

```python
import asyncio
from ecs_agent.providers import Model
from ecs_agent import RetryModel
from ecs_agent.providers.config import ApiFormat
from ecs_agent.types import Message, RetryConfig

async def main():
    base_model = Model(
        "qwen3.5-plus",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key="...",
        api_format=ApiFormat.OPENAI_CHAT_COMPLETIONS,
    )
    
    # Customize retry logic to be more aggressive
    retry_config = RetryConfig(max_attempts=5, multiplier=2.0)
    
    # Wrap the model
    model = RetryModel(base_model, retry_config=retry_config)
    
    messages = [Message(role="user", content="What is the capital of France?")]
    
    # This call will automatically retry up to 5 times on transient errors
    result = await model.complete(messages)
    print(result.message.content)

if __name__ == "__main__":
    asyncio.run(main())
```

## Logging

When a retry occurs, it is logged at the `WARNING` level. The log entries include structured fields to help you monitor API reliability:

- `attempt`: The current attempt number.
- `error`: The error message or status code that triggered the retry.
- `wait_seconds`: The amount of time the system will wait before the next attempt.

## Caveats

- **Streaming**: As mentioned, streaming calls are NOT retried. If a stream fails, you must handle the error in your application logic.
- **Permanent Errors**: Errors that are not considered transient (e.g., HTTP 400 Bad Request, 401 Unauthorized, 403 Forbidden) are NOT retried and will raise immediately.
