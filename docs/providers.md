# Providers

This document provides a reference for the LLM providers available in the ECS Agent framework.

## LLMProvider Protocol

The `LLMProvider` protocol defines the interface for all language model implementations. It's located in `ecs_agent.providers.protocol`.

```python
from typing import Any, Protocol, runtime_checkable
from collections.abc import AsyncIterator
from ecs_agent.types import Message, CompletionResult, StreamDelta, ToolSchema

@runtime_checkable
class LLMProvider(Protocol):
    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        stream: bool = False,
        response_format: dict[str, Any] | None = None,
    ) -> CompletionResult | AsyncIterator[StreamDelta]:
        ...
```

The `complete` method returns a `CompletionResult` when `stream=False` and an `AsyncIterator[StreamDelta]` when `stream=True`.

## OpenAIProvider

`OpenAIProvider` is an OpenAI-compatible HTTP provider using `httpx.AsyncClient`. It works with OpenAI's API as well as compatible alternatives like Dashscope, vLLM, or Ollama.

### Configuration

```python
from ecs_agent.providers import OpenAIProvider

provider = OpenAIProvider(
    api_key="your-api-key",
    base_url="https://api.openai.com/v1",
    model="gpt-4o-mini",
    connect_timeout=10.0,
    read_timeout=120.0,
    write_timeout=10.0,
    pool_timeout=10.0
)
```

### Behavior

- **Non-streaming**: Sends a POST request to `/chat/completions` and returns a `CompletionResult`.
- **Streaming**: Sends a POST request with `stream=True`. It iterates through server-sent events (SSE), yielding `StreamDelta` objects. It handles partial tool call arguments by accumulating them before yielding.
- **Error Handling**: `httpx.HTTPStatusError` and `httpx.RequestError` are logged and re-raised.

### Response Format Helper

The `pydantic_to_response_format` function helps convert a Pydantic `BaseModel` into an OpenAI-compatible `response_format` dictionary.

```python
from pydantic import BaseModel
from ecs_agent.providers.openai_provider import pydantic_to_response_format

class User(BaseModel):
    name: str
    age: int

response_format = pydantic_to_response_format(User)
# Result: {'type': 'json_schema', 'json_schema': {'name': 'User', 'schema': {...}, 'strict': True}}
```

## FakeProvider

`FakeProvider` is designed for deterministic testing. It returns a sequence of pre-configured responses.

### Usage

```python
from ecs_agent.providers import FakeProvider
from ecs_agent.types import CompletionResult, Message

responses = [
    CompletionResult(message=Message(role="assistant", content="Hello!")),
    CompletionResult(message=Message(role="assistant", content="How can I help?"))
]
provider = FakeProvider(responses=responses)

# First call returns "Hello!"
# Second call returns "How can I help?"
# Third call raises IndexError
```

### Behavior

- **Sequential**: Returns responses in the order they were provided. If the index exceeds the list length, it raises `IndexError`.
- **Streaming**: When `stream=True`, it yields character-by-character `StreamDelta` objects. The final delta contains the `finish_reason="stop"` and usage information.
- **Verification**: Stores the `last_response_format` for use in test assertions.

## RetryProvider

`RetryProvider` adds resilience to any `LLMProvider` by wrapping it and implementing retry logic using `tenacity`.

### Usage

```python
from ecs_agent.providers import OpenAIProvider
from ecs_agent import RetryProvider
from ecs_agent.types import RetryConfig

base_provider = OpenAIProvider(api_key="...")
retry_config = RetryConfig(
    max_attempts=3,
    multiplier=1.0,
    min_wait=4.0,
    max_wait=60.0,
    retry_status_codes=(429, 500, 502, 503, 504)
)

provider = RetryProvider(provider=base_provider, retry_config=retry_config)
```

### Behavior

- **Non-streaming**: Automatically retries on `httpx.HTTPStatusError` (for specific status codes) and `httpx.RequestError`. It logs retry attempts at the `WARNING` level.
- **Streaming**: Calls are passed through directly to the underlying provider. **Streaming calls are not retried.**
- **Default Config**: If `retry_config` is not provided, it uses standard defaults (3 attempts, exponential backoff starting at 4 seconds).

## ClaudeProvider

`ClaudeProvider` is a native Anthropic API provider with full SSE streaming support. It communicates directly with the Anthropic Messages API using `httpx.AsyncClient`.

### Configuration

```python
from ecs_agent.providers import ClaudeProvider

provider = ClaudeProvider(
    api_key="your-anthropic-api-key",
    base_url="https://api.anthropic.com",
    model="claude-3-5-haiku-latest",
    max_tokens=4096,
    connect_timeout=10.0,
    read_timeout=120.0,
    write_timeout=10.0,
    pool_timeout=10.0,
)
```

### Constructor Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `api_key` | `str` | (required) | Anthropic API key |
| `base_url` | `str` | `"https://api.anthropic.com"` | API base URL |
| `model` | `str` | `"claude-3-5-haiku-latest"` | Model identifier |
| `max_tokens` | `int` | `4096` | Maximum tokens in response |
| `connect_timeout` | `float` | `10.0` | Connection timeout in seconds |
| `read_timeout` | `float` | `120.0` | Read timeout in seconds |
| `write_timeout` | `float` | `10.0` | Write timeout in seconds |
| `pool_timeout` | `float` | `10.0` | Connection pool timeout in seconds |

### Behavior

- **Non-streaming**: Sends a POST request to `/v1/messages` with the Anthropic message format and returns a `CompletionResult`.
- **Streaming**: Uses SSE streaming with `content_block_delta` events. Accumulates text deltas and tool use inputs, yielding `StreamDelta` objects.
- **Tool Use**: Supports Anthropic's native tool use format, converting between the framework's `ToolSchema`/`ToolCall` format and Anthropic's `tool_use` blocks.
- **Error Handling**: `httpx.HTTPStatusError` and `httpx.RequestError` are logged and re-raised.
- **Headers**: Sends `x-api-key` and `anthropic-version: 2023-06-01` headers.

### Usage with RetryProvider

```python
from ecs_agent import RetryProvider, RetryConfig
from ecs_agent.providers import ClaudeProvider

provider = RetryProvider(
    provider=ClaudeProvider(api_key="...", model="claude-sonnet-4-20250514"),
    config=RetryConfig(max_retries=3),
)
```

## LiteLLMProvider

`LiteLLMProvider` enables access to 100+ LLM providers through a single unified interface via the `litellm` library. This is an optional dependency — install with `pip install litellm`.

### Configuration

```python
from ecs_agent.providers import LiteLLMProvider

# OpenAI
provider = LiteLLMProvider(model="gpt-4o", api_key="sk-...")

# Anthropic
provider = LiteLLMProvider(model="claude-sonnet-4-20250514", api_key="sk-ant-...")

# Any litellm-supported model
provider = LiteLLMProvider(model="ollama/llama3", base_url="http://localhost:11434")
```

### Constructor Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `model` | `str` | (required) | litellm model identifier (e.g. `gpt-4o`, `claude-sonnet-4-20250514`, `ollama/llama3`) |
| `api_key` | `str | None` | `None` | API key (can also be set via environment variables) |
| `base_url` | `str | None` | `None` | Custom base URL for self-hosted models |

### Behavior

- **Non-streaming**: Calls `litellm.acompletion()` and returns a `CompletionResult`.
- **Streaming**: Calls `litellm.acompletion(stream=True)` and yields `StreamDelta` objects.
- **Tool Use**: Converts between the framework's `ToolSchema` format and litellm's tool format.
- **Optional Dependency**: `litellm` is not a hard dependency. An `ImportError` with a helpful message is raised if litellm is not installed.

### Supported Providers (via litellm)

litellm supports 100+ providers including: OpenAI, Anthropic, Google Gemini, AWS Bedrock, Azure OpenAI, Ollama, vLLM, Together AI, Groq, Mistral, and many more.

---

## EmbeddingProvider Protocol

The `EmbeddingProvider` protocol defines the interface for converting text into numerical vectors. It is located in `ecs_agent.providers.embedding_protocol`.

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class EmbeddingProvider(Protocol):
    async def embed(self, texts: list[str]) -> list[list[float]]:
        ...
```

## OpenAIEmbeddingProvider

`OpenAIEmbeddingProvider` is an OpenAI-compatible provider for generating text embeddings.

### Configuration

```python
from ecs_agent.providers.embedding_provider import OpenAIEmbeddingProvider

provider = OpenAIEmbeddingProvider(
    api_key="your-api-key",
    model="text-embedding-3-small"
)
```

### Behavior
- **Batching**: Sends a POST request to `/embeddings` with a list of input texts.
- **Error Handling**: Retries are not built-in; use a wrapper if needed. `httpx` errors are logged and re-raised.

## FakeEmbeddingProvider

`FakeEmbeddingProvider` returns deterministic vectors based on the hash of the input text. Ideal for testing and development without API costs.

### Usage

```python
from ecs_agent.providers.fake_embedding_provider import FakeEmbeddingProvider

provider = FakeEmbeddingProvider(dimension=384)
vectors = await provider.embed(["hello", "world"])
```

## VectorStore Protocol

The `VectorStore` protocol defines the interface for storing and searching vectors. Located in `ecs_agent.providers.vector_store`.

```python
from typing import Any, Protocol, runtime_checkable

@runtime_checkable
class VectorStore(Protocol):
    async def add(self, id: str, vector: list[float], metadata: dict[str, Any] | None = None) -> None: ...
    async def search(self, query_vector: list[float], top_k: int = 5) -> list[tuple[str, float]]: ...
    async def delete(self, id: str) -> None: ...
```

## InMemoryVectorStore

`InMemoryVectorStore` provides a simple, dictionary-backed vector store with cosine similarity search. It optionally uses `numpy` for faster computations if available.

### Usage

```python
from ecs_agent.providers.vector_store import InMemoryVectorStore

store = InMemoryVectorStore(dimension=384)
await store.add("doc1", [0.1, 0.2, ...], metadata={"text": "content"})
results = await store.search([0.1, 0.2, ...], top_k=5)
```

## Choosing a Provider

- **Production**: Use `OpenAIProvider` for real API interaction. Wrap it in a `RetryProvider` to handle transient network issues or rate limits.
- **Testing**: Use `FakeProvider` for unit tests where you need predictable, deterministic results without making real network requests.
- **Resilience**: Always consider wrapping your primary provider in a `RetryProvider` for production environments.
- **Claude-native**: Use `ClaudeProvider` for direct Anthropic API access with native tool use support.
- **Multi-provider**: Use `LiteLLMProvider` when you need to switch between different providers without changing code.
