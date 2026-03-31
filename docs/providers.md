# Providers

This document is a reference for all LLM providers in the ECS Agent framework.

---

## Provider Architecture Overview

The provider stack is organized around three linked concepts: a canonical model ID (`provider/model`) that carries both routing provider and API model name, a `ProviderConfig` that defines endpoint/auth/protocol settings, and event-driven accounting that measures usage and cache behavior. The Quick Start below shows the full end-to-end flow.

### Quick Start

```python
import os

from ecs_agent.accounting.subscriber import AccountingSubscriber
from ecs_agent.core import World
from ecs_agent.providers import OpenAIProvider
from ecs_agent.providers.config import ApiFormat, ProviderConfig
from ecs_agent.providers.model_id import parse_model_id

# 1) Parse canonical provider/model ID
model_id = parse_model_id("aliyun/qwen3.5-flash")

# 2) Build provider config from the provider part
config = ProviderConfig(
    provider_id=model_id.provider,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=os.environ["LLM_API_KEY"],
    api_format=ApiFormat.OPENAI_CHAT_COMPLETIONS,
)

# 3) Build provider with ProviderConfig + model part
provider = OpenAIProvider(
    api_key="",  # ignored when provider_config is provided
    provider_config=config,
    model=model_id.model,
)

# 4) Attach accounting subscriber to world events
world = World()
subscriber = AccountingSubscriber()
subscriber.subscribe(world.event_bus)
```

### Canonical Model IDs

```python
from ecs_agent.providers.model_id import parse_model_id, format_model_id, ModelId

# Parse a canonical ID
model_id = parse_model_id("aliyun/qwen3.5-flash")
# ModelId(provider="aliyun", model="qwen3.5-flash")

# Format back to string
canonical = format_model_id(model_id)
# "aliyun/qwen3.5-flash"

# Invalid — colon-delimited IDs are rejected with ValueError:
# parse_model_id("aliyun:qwen3.5-flash")  # raises ValueError
# parse_model_id("gpt-4o")               # raises ValueError (missing provider prefix)
# parse_model_id("/gpt-4o")              # raises ValueError (empty provider)
```

Rules enforced by `parse_model_id`:
- Exactly one `/` separator — no colons, no bare model names
- Both provider and model components must be non-empty

### ProviderConfig and ApiFormat

`ProviderConfig` holds all connection parameters for a provider endpoint. `ApiFormat` selects the wire protocol.

```python
from ecs_agent.providers.config import ProviderConfig, ApiFormat

# OpenAI-compatible Chat Completions (most common)
chat_config = ProviderConfig(
    provider_id="aliyun",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key="your-api-key",
    api_format=ApiFormat.OPENAI_CHAT_COMPLETIONS,
)

# OpenAI-compatible Responses API
responses_config = ProviderConfig(
    provider_id="aliyun",
    base_url="https://dashscope.aliyuncs.com/api/v2/apps/protocols/compatible-mode/v1",
    api_key="your-api-key",
    api_format=ApiFormat.OPENAI_RESPONSES,
)

# Anthropic-compatible Messages API
anthropic_config = ProviderConfig(
    provider_id="moonshot",
    base_url="https://dashscope.aliyuncs.com/apps/anthropic",
    api_key="your-api-key",
    api_format=ApiFormat.ANTHROPIC_MESSAGES,
)
```

Available `ApiFormat` values:

| Value | Wire protocol |
|---|---|
| `OPENAI_CHAT_COMPLETIONS` | POST `/chat/completions` (standard OpenAI-compat) |
| `OPENAI_RESPONSES` | POST `/responses` (OpenAI Responses API) |
| `OPENAI_EMBEDDINGS` | POST `/embeddings` |
| `OPENAI_FILES` | POST `/files` (file upload) |
| `ANTHROPIC_MESSAGES` | POST `/v1/messages` (Anthropic Messages) |

`ProviderConfig` fields:

| Field | Type | Default | Description |
|---|---|---|---|
| `provider_id` | `str` | (required) | Logical provider name, e.g. `"aliyun"` |
| `base_url` | `str` | (required) | API base URL |
| `api_key` | `str` | (required) | Bearer token / API key |
| `api_format` | `ApiFormat` | (required) | Wire protocol selection |
| `extra_headers` | `dict[str, str]` | `{}` | Additional HTTP headers |
| `timeout` | `float \| None` | `None` | Global timeout override (seconds) |

---

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

---

## OpenAIProvider

`OpenAIProvider` is an OpenAI-compatible HTTP provider using `httpx.AsyncClient`. It works with OpenAI's API as well as compatible alternatives like DashScope, vLLM, or Ollama. Internally it dispatches to explicit **chat completions** or **responses** adapters based on the `api_format` / `use_responses_api` setting.

### Configuration

```python
from ecs_agent.providers import OpenAIProvider

# Chat Completions (default)
provider = OpenAIProvider(
    api_key="your-api-key",
    base_url="https://api.openai.com/v1",
    model="gpt-4o-mini",
    connect_timeout=10.0,
    read_timeout=120.0,
    write_timeout=10.0,
    pool_timeout=10.0,
)

# Responses API
provider = OpenAIProvider(
    api_key="your-api-key",
    base_url="https://dashscope.aliyuncs.com/api/v2/apps/protocols/compatible-mode/v1",
    model="qwen3.5-flash",
    use_responses_api=True,
)

# Advanced — pass an explicit ProviderConfig
from ecs_agent.providers.config import ProviderConfig, ApiFormat

provider = OpenAIProvider(
    api_key="",  # ignored when provider_config is given
    provider_config=ProviderConfig(
        provider_id="aliyun",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key="your-api-key",
        api_format=ApiFormat.OPENAI_CHAT_COMPLETIONS,
    ),
)
```

### Chat Completions Adapter

The default adapter (`ApiFormat.OPENAI_CHAT_COMPLETIONS`) sends POST requests to `/chat/completions`. It handles:

- Multimodal messages: `TextPart`, `ImageUrlPart`, and `FileRefPart` are serialized to the OpenAI vision format.
- Streaming: yields `StreamDelta` objects from SSE chunks; emits a single terminal `LLMInvocationEvent` with normalized `UsageRecord`.
- Usage normalization: reads `cached_tokens` from the OpenAI usage response to populate `UsageRecord.cached_input_tokens`.

### Responses Adapter

Set `use_responses_api=True` or `api_format=ApiFormat.OPENAI_RESPONSES` to activate the Responses API adapter. It sends POST requests to `/responses`.

Threading state (`previous_response_id`) is **not** stored on the provider instance — it lives on an ECS component:

```python
from ecs_agent.components.definitions import ResponsesAPIStateComponent

world.add_component(agent, ResponsesAPIStateComponent(previous_response_id=None))
```

`ReasoningSystem` reads and writes `ResponsesAPIStateComponent` automatically on each tick.

### Response Format Helper

```python
from pydantic import BaseModel
from ecs_agent.providers.openai_provider import pydantic_to_response_format

class User(BaseModel):
    name: str
    age: int

response_format = pydantic_to_response_format(User)
# Result: {'type': 'json_schema', 'json_schema': {'name': 'User', 'schema': {...}, 'strict': True}}
```

---

## ClaudeProvider

`ClaudeProvider` is an Anthropic-compatible provider with full SSE streaming and cache-aware usage normalization. It communicates with the Anthropic Messages API format using `httpx.AsyncClient`.

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

For Anthropic-compatible endpoints (e.g. Aliyun Kimi):

```python
provider = ClaudeProvider(
    api_key="your-api-key",
    base_url="https://dashscope.aliyuncs.com/apps/anthropic",
    model="kimi-k2.5",
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
- **Cache-aware usage**: Normalizes `cache_creation_input_tokens` and `cache_read_input_tokens` from Anthropic responses into the canonical `UsageRecord` fields.
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

---

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

---

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

---

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
| `api_key` | `str \| None` | `None` | API key (can also be set via environment variables) |
| `base_url` | `str \| None` | `None` | Custom base URL for self-hosted models |

### Behavior

- **Non-streaming**: Calls `litellm.acompletion()` and returns a `CompletionResult`.
- **Streaming**: Calls `litellm.acompletion(stream=True)` and yields `StreamDelta` objects.
- **Tool Use**: Converts between the framework's `ToolSchema` format and litellm's tool format.
- **Optional Dependency**: `litellm` is not a hard dependency. An `ImportError` with a helpful message is raised if litellm is not installed.

### Supported Providers (via litellm)

litellm supports 100+ providers including: OpenAI, Anthropic, Google Gemini, AWS Bedrock, Azure OpenAI, Ollama, vLLM, Together AI, Groq, Mistral, and many more.

---

## Usage Accounting

The framework emits one canonical `LLMInvocationEvent` per LLM call. Use `AccountingSubscriber` to capture cost and cache hit-rate metrics.

### AccountingSubscriber

```python
from ecs_agent.accounting.subscriber import AccountingSubscriber

subscriber = AccountingSubscriber()
subscriber.subscribe(world.event_bus)

# ... run your agent ...

# Query per-provider/model aggregate cache hit-rate
stats = subscriber.get_aggregate_stats("aliyun", "qwen3.5-flash")
if stats is not None:
    print(f"Cache hit rate: {stats.hit_rate}")  # float 0.0–1.0, or None
    print(f"Cache read tokens: {stats.cache_read_tokens}")
    print(f"Total prompt tokens: {stats.total_prompt_tokens}")
```

`AccountingSubscriber` computes token-weighted aggregate hit rate across all observed invocations for a given `(provider_id, model)` pair:

```
hit_rate = sum(cache_read_tokens) / sum(total_prompt_tokens)
```

where `total_prompt_tokens = uncached_input_tokens + cache_write_tokens + cache_read_tokens`. If the denominator is zero, `hit_rate` is `None`.

### Custom Pricing Catalog

Pass a custom `PricingCatalog` to override built-in pricing:

```python
from ecs_agent.accounting.catalog import PricingCatalog, ModelPricing
from ecs_agent.accounting.subscriber import AccountingSubscriber

catalog = PricingCatalog()
catalog.register("aliyun", "qwen3.5-flash", ModelPricing(
    input_per_million=0.5,
    output_per_million=1.5,
    cached_input_per_million=0.1,
    cache_write_per_million=None,
))

subscriber = AccountingSubscriber(pricing_catalog=catalog)
```

### UsageRecord

All provider adapters normalize usage into a canonical `UsageRecord`:

```python
from ecs_agent.accounting.models import UsageRecord, StreamCompleteness

usage = UsageRecord(
    prompt_tokens=1024,
    completion_tokens=256,
    total_tokens=1280,
    cached_input_tokens=512,        # OpenAI cached_tokens
    cache_creation_tokens=None,     # Anthropic cache_creation_input_tokens
    cache_read_tokens=512,          # Anthropic cache_read_input_tokens
    stream_completeness=StreamCompleteness.COMPLETE,
    provider_id="aliyun",
    model="qwen3.5-flash",
)
```

`StreamCompleteness` values:
- `COMPLETE` — full usage data available
- `PARTIAL` — stream was interrupted; usage may be incomplete
- `UNKNOWN` — usage chunk was not received (e.g. server dropped the final SSE event)

---

## Embeddings

### EmbeddingProvider Protocol

The `EmbeddingProvider` protocol defines the interface for converting text into numerical vectors. Located in `ecs_agent.providers.embedding_protocol`.

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class EmbeddingProvider(Protocol):
    async def embed(self, texts: list[str]) -> list[list[float]]:
        ...
```

### OpenAIEmbeddingProvider

`OpenAIEmbeddingProvider` is an OpenAI-compatible provider for generating text embeddings aligned with the new `ProviderConfig` model.

```python
from ecs_agent.providers.embedding_provider import OpenAIEmbeddingProvider

provider = OpenAIEmbeddingProvider(
    api_key="your-api-key",
    model="text-embedding-3-small"
)
```

### FakeEmbeddingProvider

`FakeEmbeddingProvider` returns deterministic vectors based on the hash of the input text. Ideal for testing and development without API costs.

```python
from ecs_agent.providers.fake_embedding_provider import FakeEmbeddingProvider

provider = FakeEmbeddingProvider(dimension=384)
vectors = await provider.embed(["hello", "world"])
```

---

## File Upload

`OpenAIFilesService` provides a typed file-upload service for OpenAI-compatible endpoints.

```python
from ecs_agent.providers.openai_files import OpenAIFilesService
from ecs_agent.providers.config import ProviderConfig, ApiFormat

config = ProviderConfig(
    provider_id="openai",
    base_url="https://api.openai.com/v1",
    api_key="your-api-key",
    api_format=ApiFormat.OPENAI_FILES,
)

service = OpenAIFilesService(provider_config=config)
file_ref = await service.upload(path="/path/to/document.pdf", purpose="assistants")
# file_ref.file_id — stable reference usable in FileRefPart messages
```

Uploaded file references can be embedded in multimodal messages via `FileRefPart`:

```python
from ecs_agent.types import Message, TextPart, FileRefPart

msg = Message(
    role="user",
    content="",
    parts=[
        TextPart(text="Summarize this document."),
        FileRefPart(file_id=file_ref.file_id),
    ],
)
```

---

## Multimodal Messages

The framework supports multimodal content via typed `Message.parts` entries.

### Core Types

Import path:

```python
from ecs_agent.types import Message, TextPart, ImageUrlPart, FileRefPart
```

Type definitions (from `ecs_agent.types`):

```python
@dataclass(slots=True)
class TextPart:
    text: str

@dataclass(slots=True)
class ImageUrlPart:
    url: str
    detail: str | None = None

@dataclass(slots=True)
class FileRefPart:
    file_id: str
    filename: str | None = None

MessagePart = TextPart | ImageUrlPart | FileRefPart

@dataclass(slots=True)
class Message:
    role: str
    content: str
    parts: list[MessagePart] | None = None
```

`Message.parts=None` means text-only mode (backward compatible).

### Usage Pattern

```python
from ecs_agent.types import Message, TextPart, ImageUrlPart, FileRefPart

msg = Message(
    role="user",
    content="Please review the image and file.",
    parts=[
        TextPart(text="Focus on entities and relationships."),
        ImageUrlPart(url="https://example.com/diagram.png", detail="high"),
        FileRefPart(file_id="file-abc123", filename="spec.pdf"),
    ],
)
```

If both `content` and `parts` are set, adapters preserve both.

### OpenAI Chat Completions Wire Format

- `TextPart` → `{"type": "text", "text": part.text}`
- `ImageUrlPart` → `{"type": "image_url", "image_url": {"url": part.url, "detail": part.detail}}` (`detail` omitted when `None`)
- `FileRefPart` → `{"type": "file", "file": {"file_id": part.file_id, "filename": part.filename}}` (`filename` omitted when `None`)
- If `message.content` is non-empty and `parts` is set, `content` is prepended as a text block.

```json
{
  "role": "user",
  "content": [
    {"type": "text", "text": "Please review the image and file."},
    {"type": "text", "text": "Focus on entities and relationships."},
    {"type": "image_url", "image_url": {"url": "https://example.com/diagram.png", "detail": "high"}},
    {"type": "file", "file": {"file_id": "file-abc123", "filename": "spec.pdf"}}
  ]
}
```

### OpenAI Responses API Wire Format

- Text type is role-aware: `"input_text"` (user) or `"output_text"` (assistant)
- `ImageUrlPart` → `{"type": "input_image", "image_url": part.url, "detail": part.detail}` (`detail` omitted when `None`)
- `FileRefPart` → `{"type": "input_file", "file_id": part.file_id, "filename": part.filename}` (`filename` omitted when `None`)

```json
{
  "type": "message",
  "role": "user",
  "content": [
    {"type": "input_text", "text": "Please review the image and file."},
    {"type": "input_text", "text": "Focus on entities and relationships."},
    {"type": "input_image", "image_url": "https://example.com/diagram.png", "detail": "high"},
    {"type": "input_file", "file_id": "file-abc123", "filename": "spec.pdf"}
  ]
}
```

### Anthropic Messages Wire Format

- `TextPart` → `{"type": "text", "text": part.text}`
- `ImageUrlPart` → `{"type": "image", "source": {"type": "url", "url": part.url}}`
- Vision requires adapter config `supports_vision=True`; otherwise image parts raise `ValueError`.
- `FileRefPart` is not supported by the Anthropic adapter and always raises `ValueError`.

### `ImageUrlPart.detail` (OpenAI Vision)

`detail` accepts `"low"`, `"high"`, or `"auto"`; set `None` to omit it from payloads.

### `FileRefPart` and File Uploads

Use `FileRefPart` with IDs returned by `OpenAIFilesService` (see **File Upload** above).

### Live Vision Test Note

For live vision tests, set env vars such as `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL`, and `IMAGE_URL` (never hardcode secrets).

---

## Vector Store

### VectorStore Protocol

The `VectorStore` protocol defines the interface for storing and searching vectors. Located in `ecs_agent.providers.vector_store`.

```python
from typing import Any, Protocol, runtime_checkable

@runtime_checkable
class VectorStore(Protocol):
    async def add(self, id: str, vector: list[float], metadata: dict[str, Any] | None = None) -> None: ...
    async def search(self, query_vector: list[float], top_k: int = 5) -> list[tuple[str, float]]: ...
    async def delete(self, id: str) -> None: ...
```

### InMemoryVectorStore

`InMemoryVectorStore` provides a simple, dictionary-backed vector store with cosine similarity search. It optionally uses `numpy` for faster computations if available.

```python
from ecs_agent.providers.vector_store import InMemoryVectorStore

store = InMemoryVectorStore(dimension=384)
await store.add("doc1", [0.1, 0.2, ...], metadata={"text": "content"})
results = await store.search([0.1, 0.2, ...], top_k=5)
```

---

## Choosing a Provider

- **Production**: Use `OpenAIProvider` for real API interaction. Wrap it in a `RetryProvider` to handle transient network issues or rate limits.
- **Testing**: Use `FakeProvider` for unit tests where you need predictable, deterministic results without making real network requests.
- **Resilience**: Always consider wrapping your primary provider in a `RetryProvider` for production environments.
- **Claude-native**: Use `ClaudeProvider` for direct Anthropic API access with native tool use support and cache-aware accounting.
- **Multi-provider**: Use `LiteLLMProvider` when you need to switch between different providers without changing code.
- **Accounting**: Attach `AccountingSubscriber` to the `EventBus` to track cost and cache hit-rate metrics across invocations.
