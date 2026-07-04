# Models and Provider Configuration

This guide covers how ECS Agent selects, constructs, and configures LLM models.

The high-level rule is simple:

- Use `Model(...)` when you already know the endpoint settings.
- Use `get_model("provider/model", registry=...)` when you want registry-based routing.
- Drop down to `ProviderConfig` and concrete model classes only when you need lower-level control.

---

## Unified `Model(...)` Constructor

The recommended way to create any LLM model is the `Model(...)` factory. It automatically selects the correct implementation class (`OpenAIModel`, `ClaudeModel`, or `LiteLLMModel`) based on the `api_format` or `model_type` argument.

```python
from ecs_agent.providers import Model
from ecs_agent.providers.config import ApiFormat

# OpenAI-compatible Chat Completions
model = Model(
    "qwen3.5-flash",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key="your-api-key",
    api_format=ApiFormat.OPENAI_CHAT_COMPLETIONS,
)

# OpenAI Responses API
model = Model(
    "qwen3.5-flash",
    base_url="https://dashscope.aliyuncs.com/api/v2/apps/protocols/compatible-mode/v1",
    api_key="your-api-key",
    api_format=ApiFormat.OPENAI_RESPONSES,
)

# Anthropic Messages API
model = Model(
    "claude-3-5-haiku-latest",
    base_url="https://api.anthropic.com",
    api_key="your-api-key",
    api_format=ApiFormat.ANTHROPIC_MESSAGES,
)

# Explicit model_type (skips api_format inference)
from ecs_agent.providers import ModelType
model = Model("gpt-4o", base_url="...", api_key="...", model_type=ModelType.OPENAI)

# api_format as string
model = Model("gpt-4o", base_url="...", api_key="...", api_format="openai_chat_completions")
```

### `Model(...)` Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model_id` | `str` | (required, positional) | Model name, e.g. `"gpt-4o"`, `"claude-3-5-haiku-latest"` |
| `base_url` | `str` | (required) | Provider endpoint base URL |
| `api_key` | `str` | (required) | Bearer token / API key |
| `api_format` | `ApiFormat \| str \| None` | `None` | Wire protocol. Overrides model_type default when both are given. Conflicts raise `ValueError`. |
| `model_type` | `str \| type \| None` | `None` | Implementation to use: `"openai"`, `"claude"`, `"litellm"`, or the class itself. Inferred from `api_format` when omitted. |
| `provider_id` | `str` | `""` | Label stored in `ProviderConfig` (not used at runtime) |
| `extra_headers` | `dict[str, str] \| None` | `None` | Additional HTTP headers |
| `timeout` | `float \| None` | `None` | Global request timeout (seconds) |
| `enable_store` | `bool` | `False` | Enable conversation storage (Responses API feature) |
| `enable_prompt_caching` | `bool` | `True` | Emit Anthropic `cache_control` breakpoints (Claude adapter only; ignored by other formats). Set `False` to revert to the pre-caching request shape. |
| `**kwargs` | | | Forwarded to the underlying model constructor (e.g. `connect_timeout`, `max_tokens`) |

### Anthropic Prompt Caching

When `enable_prompt_caching=True` (default), `ClaudeModel` places `cache_control: {"type": "ephemeral"}` breakpoints at three positions in each request — the last **tool** definition, the cache-stable **system** prefix, and the last **message** — for automatic incremental caching. This is GA and needs no `anthropic-beta` header.

To keep the cached prefix byte-stable, the rendered system prompt is split by `SystemPromptRenderSystem` into:

- **stable prefix** — base instructions + tool/skill/MCP inventory + scratchbook metadata (byte-stable across turns), emitted as the first `system` block with the cache breakpoint;
- **volatile tail** — compaction summary (`_chat_history_summary_xml`) and workflow state (`_workflow_state_prompt`), emitted as a second, un-cached `system` block.

When the entire prompt body is volatile (e.g. a workflow-driven prompt where the whole system prompt is `${_workflow_state_prompt}`), the stable prefix is empty and the volatile content is sent as the sole system message — nothing is cached, which is correct since such prompts change per state. Verify cache hits via `usage.cache_read_tokens` (populated from the response's `cache_read_input_tokens`).

### Actual token usage per entity

`ReasoningSystem` records the **provider-reported** token usage on the entity after every call as a `TokenUsageComponent` — the ground truth (more accurate than any local estimate). It carries the last call's counts (`last_prompt_tokens`, `last_completion_tokens`, `last_total_tokens`, `last_cache_read_tokens`, `last_cache_creation_tokens`) and running totals (`total_*`, `call_count`):

```python
from ecs_agent.components import TokenUsageComponent

usage = world.get_component(agent_id, TokenUsageComponent)
if usage is not None:
    print(usage.last_prompt_tokens, usage.total_tokens, usage.call_count)
```

The component is absent until the first call and is not created when the provider reports no usage.

**Compaction calibration.** Once this component exists, `CompactionSystem` calibrates its threshold check against ground truth: it uses the real `last_prompt_tokens` (the exact size of the last input — system, tools and history) plus a local estimate of only the messages appended since that call (`last_prompt_message_count` anchors the boundary). Before the first call — or after compaction shrinks the conversation below the anchor — it falls back to a pure local estimate (`ecs_agent.token_counting`).

### Context trimming → compaction pipeline

When an entity has a `ContextTrimConfig`, `CompactionSystem` runs a failover pipeline before summarizing: **estimate → trim → (still over) summarize**.

- **Budget** = `ContextTrimConfig.max_tokens`, or — when `None` — derived from the model's context window via `ecs_agent.context_windows.resolve_context_budget(model_id)` (window minus an output reserve).
- **Trim** permanently drops the oldest tool spans (atomic assistant-tool-call + results), then optionally strips replayed reasoning (`trim_reasoning`), until the estimate is under budget. It's cheap (no LLM) and rewrites history in place.
  - `protect_recent_turns: int` keeps the most recent N messages untouched (tool spans reaching into them are kept whole; their reasoning is not stripped).
  - Reasoning stripping never touches a tool-calling assistant message (its thinking + signature is load-bearing for extended-thinking tool-use replay) and always keeps the newest reasoning-bearing message.
  - `token_estimation_chars_per_token` controls the fallback estimate when tiktoken is unavailable (used consistently across the trim path).
- If trimming frees enough space → **no summary is produced** this turn. If essential content still exceeds budget → it **falls back to compaction summarization**. There is no "raise on overflow" — `overflow_behavior` defaults to a non-raising `"warn"`.

```python
from ecs_agent.components import ContextTrimConfig

# Trim oldest tool results to fit the model's window before compacting.
world.add_component(agent_id, ContextTrimConfig())            # budget from model window
world.add_component(agent_id, ContextTrimConfig(max_tokens=100_000))  # explicit budget
```

Without a `ContextTrimConfig`, compaction behaves exactly as before (summarize at `threshold_tokens`). Model windows are catalogued in `ecs_agent/context_windows.py` (extend `CONTEXT_WINDOWS` as needed).

### Selection Rules

- **`api_format` only**: infers `model_type` (`OPENAI_CHAT_COMPLETIONS` / `OPENAI_RESPONSES` → `openai`; `ANTHROPIC_MESSAGES` → `claude`).
- **`model_type` only**: infers default `api_format` (`openai` → `OPENAI_CHAT_COMPLETIONS`; `claude` → `ANTHROPIC_MESSAGES`).
- **Both given**: `api_format` overrides the model_type default. Incompatible combinations raise `ValueError` containing "conflict".
- **Neither given**: raises `ValueError`.

---

## Model Selection Overview

The model stack is organized around three linked concepts: a canonical model ID (`provider/model`) that carries both routing provider and API model name, a `ProviderConfig` that defines endpoint/auth/protocol settings, and event-driven accounting that measures usage and cache behavior. The quick start below shows the full end-to-end flow.

### Quick Start

```python
import os

from ecs_agent.accounting.subscriber import AccountingSubscriber
from ecs_agent.core import World
from ecs_agent.providers.registry import ProviderRegistry, get_model

# 1) Load provider configs from TOML file
registry = ProviderRegistry.from_toml("providers.toml")
# or inline:
# registry = ProviderRegistry.from_dict({
#     "aliyun": {
#         "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
#         "api_format": "openai_chat_completions",
#         "api_key_env": "LLM_API_KEY",
#     }
# })

# 2) One call: model ID → correct LLMModel instance (determined by api_format)
model = get_model("aliyun/qwen3.5-flash", registry=registry)

# 3) Attach accounting to the World's event bus
world = World()
subscriber = AccountingSubscriber()
subscriber.subscribe(world.event_bus)
```

---

## Registry-based Model Selection

`ProviderRegistry` maps provider IDs to endpoint/auth/protocol configs. `get_model` resolves a `provider/model` identifier and delegates final construction to the unified `Model(...)` entry point.

### TOML Configuration

```toml
[providers.aliyun]
base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
api_format = "openai_chat_completions"
api_key_env = "LLM_API_KEY"

[providers.aliyun-responses]
base_url = "https://dashscope.aliyuncs.com/api/v2/apps/protocols/compatible-mode/v1"
api_format = "openai_responses"
api_key_env = "LLM_API_KEY"

[providers.moonshot]
base_url = "https://dashscope.aliyuncs.com/apps/anthropic"
api_format = "anthropic_messages"
api_key_env = "LLM_API_KEY"
default_max_tokens = 8192
```

### Loading a Registry

```python
from ecs_agent.providers.registry import ProviderRegistry, get_model

# From TOML file
registry = ProviderRegistry.from_toml("providers.toml")

# From dict (useful in tests)
registry = ProviderRegistry.from_dict({
    "aliyun": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_format": "openai_chat_completions",
        "api_key_env": "LLM_API_KEY",
    }
})
```

### `get_model`

```python
from ecs_agent.providers.registry import get_model

# Resolves provider ID → ProviderEntry → Model(...) → correct LLMModel instance
model = get_model("aliyun/qwen3.5-flash", registry=registry)

# Override API key at call time
model = get_model("aliyun/qwen3.5-flash", registry=registry, api_key="sk-...")
```

API key resolution order: explicit `api_key` argument → `ProviderEntry.api_key` → env var named by `ProviderEntry.api_key_env`.

Dispatch by `api_format`:

| `api_format` | Model type returned |
|---|---|
| `openai_chat_completions` | `OpenAIModel` |
| `openai_responses` | `OpenAIModel` (Responses API) |
| `anthropic_messages` | `ClaudeModel` |
| `openai_embeddings` | raises `ValueError` — use `get_embedding_provider` |
| `openai_files` | raises `ValueError` — use `get_file_service` |

### `ProviderEntry` Fields

| Field | Type | Default | Description |
|---|---|---|---|
| `base_url` | `str` | (required) | Endpoint base URL (trailing slash stripped) |
| `api_format` | `ApiFormat` | (required) | Wire protocol; parsed eagerly |
| `api_key` | `str \| None` | `None` | Literal API key (not shown in repr) |
| `api_key_env` | `str \| None` | `None` | Env var name to read the key from |
| `extra_headers` | `dict[str, str]` | `{}` | Extra HTTP headers |
| `timeout` | `float \| None` | `None` | Global timeout override (seconds) |
| `default_max_tokens` | `int` | `4096` | Used as `max_tokens` for `ClaudeModel` |

---

## ProviderConfig and ApiFormat

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

## LLMModel Protocol

The `LLMModel` protocol defines the interface for all language model implementations. It's located in `ecs_agent.providers.protocol`.

```python
from typing import Any, Protocol, runtime_checkable
from collections.abc import AsyncIterator
from ecs_agent.types import Message, CompletionResult, StreamDelta, ToolSchema

@runtime_checkable
class LLMModel(Protocol):
    @property
    def model_id(self) -> str:
        """Canonical model identifier, e.g. 'qwen3.5-flash'."""
        ...

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

### Using LLMModel with LLMComponent

```python
from ecs_agent.components import LLMComponent
from ecs_agent.providers import Model
from ecs_agent.providers.config import ApiFormat

model = Model(
    "qwen3.5-flash",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key="your-api-key",
    api_format=ApiFormat.OPENAI_CHAT_COMPLETIONS,
)

# LLMComponent takes the model object directly (not a string)
llm = LLMComponent(model=model, system_prompt="You are a helpful assistant.")
```

### Switching models at runtime

Set `pending_model` on `LLMComponent` to a new `LLMModel` object before the tick runs. The system will swap it in, update `model`, and clear `pending_model` after the call:

```python
llm_component = world.get_component(entity, LLMComponent)
llm_component.pending_model = get_model("aliyun/qwen3.5-flash", registry=registry)
```

---

## OpenAIModel

`OpenAIModel` is an OpenAI-compatible HTTP model using `httpx.AsyncClient`. It works with OpenAI's API as well as compatible alternatives like DashScope, vLLM, or Ollama. Internally it dispatches to explicit **chat completions** or **responses** adapters based on the `api_format` in the `ProviderConfig`.

### Configuration

```python
from ecs_agent.providers import OpenAIModel
from ecs_agent.providers.config import ApiFormat, ProviderConfig

# Chat Completions
config = ProviderConfig(
    provider_id="openai",
    base_url="https://api.openai.com/v1",
    api_key="your-api-key",
    api_format=ApiFormat.OPENAI_CHAT_COMPLETIONS,
)
model = OpenAIModel(
    config=config,
    model="gpt-4o-mini",
    connect_timeout=10.0,
    read_timeout=120.0,
    write_timeout=10.0,
    pool_timeout=10.0,
)

# Responses API
responses_config = ProviderConfig(
    provider_id="aliyun",
    base_url="https://dashscope.aliyuncs.com/api/v2/apps/protocols/compatible-mode/v1",
    api_key="your-api-key",
    api_format=ApiFormat.OPENAI_RESPONSES,
)
model = OpenAIModel(config=responses_config, model="qwen3.5-flash")
```

### Chat Completions Adapter

The default adapter (`ApiFormat.OPENAI_CHAT_COMPLETIONS`) sends POST requests to `/chat/completions`. It handles:

- Multimodal messages: `ImageUrlPart` and `FileRefPart` are serialized to the OpenAI vision format. Text goes in `message.content`.
- Streaming: yields `StreamDelta` objects from SSE chunks; emits a single terminal `LLMInvocationEvent` with normalized `UsageRecord`.
- Usage normalization: reads `cached_tokens` from the OpenAI usage response to populate `UsageRecord.cached_input_tokens`.

### Responses Adapter

Set `api_format=ApiFormat.OPENAI_RESPONSES` in the `ProviderConfig` to activate the Responses API adapter. It sends POST requests to `/responses`.

For multimodal vision input, build user messages with `content=` for the text prompt and `Message.parts` containing `ImageUrlPart(url=...)` entries. The adapter converts `ImageUrlPart` into Responses API `input_image` items automatically (see `examples/vision_agent.py` for a full runnable example).

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

## ClaudeModel

`ClaudeModel` is an Anthropic-compatible model with full SSE streaming and cache-aware usage normalization. It communicates with the Anthropic Messages API format using `httpx.AsyncClient`.


### Configuration

```python
from ecs_agent.providers import ClaudeModel
from ecs_agent.providers.config import ApiFormat, ProviderConfig

# Direct Anthropic API
config = ProviderConfig(
    provider_id="anthropic",
    base_url="https://api.anthropic.com",
    api_key="your-anthropic-api-key",
    api_format=ApiFormat.ANTHROPIC_MESSAGES,
)
model = ClaudeModel(config=config, model="claude-3-5-haiku-latest", max_tokens=4096)
```

For Anthropic-compatible endpoints (e.g. Aliyun Kimi):

```python
from ecs_agent.providers import ClaudeModel
from ecs_agent.providers.config import ApiFormat, ProviderConfig

# Anthropic-compatible endpoint (e.g. Aliyun Kimi)
config = ProviderConfig(
    provider_id="moonshot",
    base_url="https://dashscope.aliyuncs.com/apps/anthropic",
    api_key="your-api-key",
    api_format=ApiFormat.ANTHROPIC_MESSAGES,
)
model = ClaudeModel(config=config, model="kimi-k2.5")
```

### Constructor Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `config` | `ProviderConfig` | (required) | Endpoint/auth/protocol config |
| `model` | `str` | `"claude-3-5-haiku-latest"` | Model identifier |
| `max_tokens` | `int` | `4096` | Maximum tokens in response |
| `connect_timeout` | `float` | `10.0` | Connection timeout in seconds |
| `read_timeout` | `float` | `120.0` | Read timeout in seconds |
| `write_timeout` | `float` | `10.0` | Write timeout in seconds |
| `pool_timeout` | `float` | `10.0` | Connection pool timeout in seconds |
| `supports_vision` | `bool` | `False` | Enable image input parts |

### Behavior

- **Non-streaming**: Sends a POST request to `/v1/messages` with the Anthropic message format and returns a `CompletionResult`.
- **Streaming**: Uses SSE streaming with `content_block_delta` events. Accumulates text deltas and tool use inputs, yielding `StreamDelta` objects.
- **Tool Use**: Supports Anthropic's native tool use format, converting between the framework's `ToolSchema`/`ToolCall` format and Anthropic's `tool_use` blocks.
- **Cache-aware usage**: Normalizes `cache_creation_input_tokens` and `cache_read_input_tokens` from Anthropic responses into the canonical `UsageRecord` fields.
- **Error Handling**: `httpx.HTTPStatusError` and `httpx.RequestError` are logged and re-raised.
- **Headers**: Sends `x-api-key` and `anthropic-version: 2023-06-01` headers.

### Usage with RetryModel

```python
from ecs_agent import RetryModel, RetryConfig
from ecs_agent.providers import ClaudeModel
from ecs_agent.providers.config import ApiFormat, ProviderConfig

config = ProviderConfig(
    provider_id="anthropic",
    base_url="https://api.anthropic.com",
    api_key="your-api-key",
    api_format=ApiFormat.ANTHROPIC_MESSAGES,
)
model = RetryModel(
    model=ClaudeModel(config=config, model="claude-sonnet-4-20250514"),
    retry_config=RetryConfig(max_retries=3),
)
```

---

## FakeModel

`FakeModel` is designed for deterministic testing. It returns a sequence of pre-configured responses.


### Usage

```python
from ecs_agent.providers import FakeModel
from ecs_agent.types import CompletionResult, Message

responses = [
    CompletionResult(message=Message(role="assistant", content="Hello!")),
    CompletionResult(message=Message(role="assistant", content="How can I help?"))
]
model = FakeModel(responses=responses, model_id="test-model")

# First call returns "Hello!"
# Second call returns "How can I help?"
# Third call raises IndexError
```

### Constructor Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `responses` | `list[CompletionResult]` | (required) | Sequence of responses to return |
| `model_id` | `str` | `"fake"` | Identifier returned by `model.model_id` |

### Behavior

- **Sequential**: Returns responses in the order they were provided. If the index exceeds the list length, it raises `IndexError`.
- **Streaming**: When `stream=True`, it yields character-by-character `StreamDelta` objects. The final delta contains the `finish_reason="stop"` and usage information.
- **Verification**: Stores the `last_response_format` for use in test assertions.

---

## RetryModel

`RetryModel` adds resilience to any `LLMModel` by wrapping it and implementing retry logic using `tenacity`.


### Usage

```python
from ecs_agent.providers import OpenAIModel
from ecs_agent import RetryModel
from ecs_agent.types import RetryConfig
from ecs_agent.providers.config import ApiFormat, ProviderConfig

config = ProviderConfig(
    provider_id="openai",
    base_url="https://api.openai.com/v1",
    api_key="your-api-key",
    api_format=ApiFormat.OPENAI_CHAT_COMPLETIONS,
)
base_model = OpenAIModel(config=config, model="gpt-4o-mini")
retry_config = RetryConfig(
    max_attempts=3,
    multiplier=1.0,
    min_wait=4.0,
    max_wait=60.0,
    retry_status_codes=(429, 500, 502, 503, 504)
)

model = RetryModel(model=base_model, retry_config=retry_config)
```

### Behavior

- **Non-streaming**: Automatically retries on `httpx.HTTPStatusError` (for specific status codes) and `httpx.RequestError`. It logs retry attempts at the `WARNING` level.
- **Streaming**: Calls are passed through directly to the underlying provider. **Streaming calls are not retried.**
- **Default Config**: If `retry_config` is not provided, it uses standard defaults (3 attempts, exponential backoff starting at 4 seconds).

---

## LiteLLMModel

`LiteLLMModel` enables access to 100+ LLM providers through a single unified interface via the `litellm` library. This is an optional dependency — install with `pip install litellm`.


### Configuration

```python
from ecs_agent.providers import LiteLLMModel

# OpenAI
model = LiteLLMModel(model="gpt-4o", api_key="sk-...")

# Anthropic
model = LiteLLMModel(model="claude-sonnet-4-20250514", api_key="sk-ant-...")

# Any litellm-supported model
model = LiteLLMModel(model="ollama/llama3", base_url="http://localhost:11434")
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
from ecs_agent.types import Message, FileRefPart

msg = Message(
    role="user",
    content="Summarize this document.",
    parts=[
        FileRefPart(file_id=file_ref.file_id),
    ],
)

---

## Multimodal Messages

The framework supports multimodal content via typed `Message.parts` entries.

### Core Types

Import path:

```python
from ecs_agent.types import Message, ImageUrlPart, FileRefPart
```

Type definitions (from `ecs_agent.types`):

```python
@dataclass(slots=True)
class ImageUrlPart:
    url: str
    detail: str | None = None

@dataclass(slots=True)
class FileRefPart:
    file_id: str
    filename: str | None = None

MessagePart = ImageUrlPart | FileRefPart

@dataclass(slots=True)
class Message:
    role: str
    content: str          # canonical text — always use this for text
    parts: list[MessagePart] | None = None  # non-text media only
```

`Message.content` is the canonical text field. `Message.parts` holds non-text media (`ImageUrlPart`, `FileRefPart`) only. Text always goes in `content`, never in `parts`.

### Usage Pattern

```python
from ecs_agent.types import Message, ImageUrlPart, FileRefPart

msg = Message(
    role="user",
    content="Please review the image and file. Focus on entities and relationships.",
    parts=[
        ImageUrlPart(url="https://example.com/diagram.png", detail="high"),
        FileRefPart(file_id="file-abc123", filename="spec.pdf"),
    ],
)
```

Text lives in `content`; media-only parts go in `parts`. Adapters prepend `content` as a text block when `parts` is also set.

### OpenAI Chat Completions Wire Format

- `message.content` → `{"type": "text", "text": message.content}` (prepended when non-empty)
- `ImageUrlPart` → `{"type": "image_url", "image_url": {"url": part.url, "detail": part.detail}}` (`detail` omitted when `None`)
- `FileRefPart` → `{"type": "file", "file": {"file_id": part.file_id, "filename": part.filename}}` (`filename` omitted when `None`)

```json
{
  "role": "user",
  "content": [
    {"type": "text", "text": "Please review the image and file. Focus on entities and relationships."},
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
    {"type": "input_text", "text": "Please review the image and file. Focus on entities and relationships."},
    {"type": "input_image", "image_url": "https://example.com/diagram.png", "detail": "high"},
    {"type": "input_file", "file_id": "file-abc123", "filename": "spec.pdf"}
  ]
}
```

### Anthropic Messages Wire Format

- `message.content` → `{"type": "text", "text": message.content}` (prepended when non-empty)
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

## Choosing a Model

- **Production**: Use `OpenAIModel` for real API interaction. Wrap it in a `RetryModel` to handle transient network issues or rate limits.
- **Testing**: Use `FakeModel` for unit tests where you need predictable, deterministic results without making real network requests.
- **Resilience**: Always consider wrapping your primary model in a `RetryModel` for production environments.
- **Claude-native**: Use `ClaudeModel` for direct Anthropic API access with native tool use support and cache-aware accounting.
- **Multi-provider**: Use `LiteLLMModel` when you need to switch between different providers without changing code.
- **Accounting**: Attach `AccountingSubscriber` to the `EventBus` to track cost and cache hit-rate metrics across invocations.
