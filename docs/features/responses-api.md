# OpenAI Responses API

`OpenAIModel` supports the OpenAI `/v1/responses` endpoint when constructed via `Model(..., api_format=ApiFormat.OPENAI_RESPONSES)`, which offers enhanced metadata tracking and streaming capabilities compared to the standard Chat Completions API.

Requests and SSE decoding use the official [OpenAI Python SDK](https://developers.openai.com/api/reference/python)
(`AsyncOpenAI.responses.create`). The SDK is installed as a project dependency
and loaded lazily on the first Responses request. The public `Model` and
`OpenAIModel` interfaces are unchanged.

The SDK uses the model's HTTP client, preserving `base_url`, API keys, extra
headers and timeout settings. Streaming keeps its separate per-chunk read
timeout (`stream_read_timeout`, unbounded by default). Call `await model.aclose()`
to release the shared connection pool.

SDK automatic retries are disabled; `RetryModel` remains responsible for retry
policy and accounting. HTTP status and connection errors are translated back to
the existing `httpx.HTTPStatusError` / `httpx.RequestError` contract, including
transport timeout subtypes. Explicit SDK stream errors and malformed SSE JSON
propagate rather than being treated as successful completion. Message, tool,
usage and terminal-event mapping still supports the documented gateway dialects.

## Overview

The Responses API provides:
- Incremental response IDs for tracking multi-turn conversations
- Enhanced usage metadata (prompt tokens, completion tokens, total tokens)
- Automatic fallback to Chat Completions if Responses API is unavailable
- Full streaming support with Server-Sent Events (SSE)

## Usage

### Basic Setup

Enable Responses API by selecting `ApiFormat.OPENAI_RESPONSES`:

```python
from ecs_agent.providers import Model
from ecs_agent.providers.config import ApiFormat

model = Model(
    "gpt-4o",
    base_url="https://api.openai.com/v1",
    api_key="your-api-key",
    api_format=ApiFormat.OPENAI_RESPONSES,
)
```

### With Components

```python
from ecs_agent import World
from ecs_agent.components import LLMComponent, ResponsesAPIStateComponent, ConversationComponent
from ecs_agent.types import Message

world = World()
entity = world.create_entity()

# Add LLM component with Responses API enabled
world.add_component(
    entity,
    LLMComponent(model=model, system_prompt="You are a helpful assistant."),
)

# Add ResponsesAPIStateComponent to track previous_response_id
world.add_component(
    entity,
    ResponsesAPIStateComponent(previous_response_id=None),
)

# Add conversation
world.add_component(
    entity,
    ConversationComponent(messages=[Message(role="user", content="Hello!")]),
)
```

### Streaming

The Responses API supports streaming just like Chat Completions:

```python
async for delta in model.complete(messages, stream=True):
    print(delta.content, end="", flush=True)
```

The stream parser understands both SSE dialects found in the wild:

- **Standard OpenAI events** — `response.output_text.delta` (string delta),
  `response.function_call_arguments.delta`, `response.completed`,
  `response.failed` (returns `FinishReason.ERROR` with the provider error in
  `StreamDelta.error`, retaining any usage).
- **Legacy object-delta events** — `response.output_item.delta` carrying
  `{"type": "content_delta"|"arguments_delta", ...}` payloads, terminated by
  `response.done`.

When a `response.output_item.done` item carries a complete `arguments` string,
it overrides whatever accumulated from delta events. Likewise, if a message
item arrives with no text deltas at all (some gateways send only
`output_item.added` + `output_item.done`), its text is recovered from the done
item's content — without duplicating text that already streamed.

Reasoning models stream their summary on a separate channel
(`response.reasoning_summary_text.delta`); the parser surfaces it as
`StreamDelta.reasoning_content` so callers can split private reasoning from the
user-facing answer, exactly as they would on Chat Completions.

Every stream ends with exactly one terminal delta, carrying the finish reason,
usage and response id — see [Terminal state](#terminal-state) below.

## Terminal state

How a turn ended is read from the `status` on the provider's `response` object
— `completed`, `incomplete`, `failed`, `cancelled` — not from the name of the
event that delivered it. Streaming and non-streaming share one parse, so both
report the same [`FinishReason`](./streaming.md#finish-reasons):

| Provider status | `FinishReason` | Notes |
|---|---|---|
| `completed` | `STOP`, or `TOOL_CALLS` when the turn produced function calls | Matches what Chat Completions reports for the same turn |
| `incomplete` | `LENGTH` / `CONTENT_FILTER` from `incomplete_details.reason` | The truncated turn's usage is still recorded |
| `cancelled` | `CANCELLED` | |
| `failed` | `ERROR` | Preserves provider code/message in `error` and any reported usage |

Two degradations are handled explicitly:

- **No `status` field.** The legacy `response.done` dialect (and some gateways
  on failures) omit it; the event name is then used as the fallback status.
- **No terminal event at all.** A stream that ends on the Chat-style `[DONE]`
  sentinel or a closed connection gets a synthesised terminal delta with
  `UNKNOWN` and the id seen on `response.created`, plus a
  `responses_stream_missing_terminal_event` warning. Usage is unknowable there
  and stays `None` — but the response id still reaches the caller, so token
  accounting and stored-response chaining do not silently stall. `UNKNOWN` is
  deliberately not reported as `STOP`: a completion the provider never stated
  must not look clean.

## Tool result contract

A `role="tool"` message must carry the `tool_call_id` of the call it answers —
without it the result cannot be rendered as a `function_call_output`, and
dropping it would strand the matching `function_call` in the request. Adapters
therefore reject such a history with `ValueError` naming the offending message
index, rather than silently skipping the message. The same check runs on the
Chat Completions, LiteLLM and Anthropic paths. Assembled outbound histories
are repaired for interrupted calls by `repair_dangling_tool_spans` before the
request is built. Validation also rejects orphan/duplicate results, repeated call
IDs, out-of-order results, and unclosed call spans. Explicit incremental Responses
input with `previous_response_id` may begin with results for remote calls; its
locally declared calls must still form complete spans. A declared full local
history (`covered_message_count > 0`) receives the full-history checks.

## Structured Output

Pass a `response_format` (e.g. from
[`pydantic_to_response_format`](./structured-output.md)) and the adapter
translates it into the Responses-native `text.format` block. The Responses API
does **not** accept the top-level `response_format` parameter, and it expects
the schema flattened rather than nested under a `json_schema` key, so the
adapter rewrites both automatically:

```python
# Chat-shaped input (what pydantic_to_response_format produces)
{"type": "json_schema", "json_schema": {"name": ..., "schema": ..., "strict": true}}

# Sent to /v1/responses as
{"text": {"format": {"type": "json_schema", "name": ..., "schema": ..., "strict": true}}}
```

The same `response_format` value therefore works unchanged across both
`OPENAI_CHAT_COMPLETIONS` and `OPENAI_RESPONSES`.

## Reasoning Content

For reasoning models, `type: "reasoning"` output items are parsed into
`CompletionResult.reasoning_content` (their `summary_text` blocks joined with
newlines). Non-reasoning responses leave `reasoning_content` as `None`.

## Storage and automatic chaining

`enable_store` maps directly to the SDK's `store` parameter:

| Configuration | Wire request |
|---|---|
| `True` | `store: true` |
| `False` | `store: false` |
| `None` / unset (default) | Parameter omitted; the server default applies |

Framework `None` means SDK omission, not JSON null. Storage is independent of
Prompt Cache and of `previous_response_id`. A request with `store=False` can
reference an older stored response; its own response ID does not become the next
automatic anchor. Explicit `thread_response_id` calls follow this same rule.

Automatic chaining is a separate opt-in: `enable_response_chaining=True`.
`ReasoningSystem` verifies the provider/endpoint/model/auth scope, conversation
identity/revision, and a fingerprint of the actual covered protocol items before
selecting an anchor. Counts alone and legacy state without provenance cannot
activate a chain. `RetryModel` forwards verified references. Storage must be
known from explicit `store=True` or the response's reported storage state; an ID
alone is not evidence of storage.

When chaining, only the uncovered protocol suffix is sent. Message boundaries
are recomputed from the verified item prefix, so one message containing multiple
reasoning/tool items is handled correctly. A request with no incremental input
rebuilds without a chain instead of duplicating all history on the remote chain.

Use `replace_conversation_history` from `ecs_agent.history` for reset, editing or
compaction. It changes the history revision and invalidates the anchor. Workflow
reset also starts a new history identity. Restored checkpoints are checked against
the current provider and actual outbound prefix; an expired remote reference is
handled by the same error path as any other invalid anchor. Direct list edits
that change the protocol prefix are detected before the next request.

An attributable 400/404 can trigger one retry without the reference, before any
stream output, only when full local history is available and known opaque
reasoning is replayable. An explicit incremental-only caller cannot recover
missing remote history this way. Unsupported chaining disables automatic chaining
for that model instance; an expired ID alone does not disable future valid chains.
Explicit caller-provided references are never silently discarded by that cache.

```toml
[providers.openai]
base_url = "https://api.openai.com/v1"
api_format = "openai_responses"
api_key_env = "LLM_API_KEY"
enable_store = true
enable_response_chaining = true
```

### Replayable output

Assistant messages retain a JSON-only `ProviderOutput` with all original output
items, source identity, completeness and storage metadata. `CompletionResult`
exposes the same object. Streaming aggregates item completion and final snapshots
without duplicating function calls; world serialization/checkpoints preserve the
state. Tool execution/display use the normalized message while same-origin replay
uses the original item order, IDs, opaque reasoning and argument strings.

The adapter requests `reasoning.encrypted_content`, including for stored responses,
so an expired remote chain can have a local backup. This follows the
[reasoning replay guidance](https://developers.openai.com/api/docs/guides/reasoning).
If the provider does not supply replayable opaque state, a reference failure is
surfaced instead of claiming a lossless recovery. Incomplete output is retained
for diagnosis and is not replayed as a completed tool turn.

Editing a message projection or changing provider invalidates its native replay
and uses its portable visible projection, with a `responses_output_projection`
log entry. Private provider state is never sent to a different source. Such a
projection is intentionally lossy; the framework does not claim reasoning-state
continuity across incompatible providers. Compaction keeps pending tool turns
intact and only removes closed tool spans as units.

## Automatic Fallback

If the Responses API endpoint is not available, `OpenAIModel` automatically falls back to Chat Completions:

```python
# This works even if base_url doesn't support /v1/responses
model = Model(
    "qwen3.5-plus",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # May not support Responses API
    api_key="your-api-key",
    api_format=ApiFormat.OPENAI_RESPONSES,
)
```

Endpoint fallback covers ordinary and streaming calls before any output. A 404
about a previous response is not an unsupported endpoint. Explicit storage
settings disable Chat fallback because dropping `store=False` or `store=True`
would change the requested contract; unsupported storage is reported as an error.

## Response Metadata

When using Responses API, additional metadata is available:

- **response_id**: Unique identifier for each API response
- **previous_response_id**: Links responses in multi-turn conversations
- **usage**: Detailed token usage (`prompt_tokens`, `completion_tokens`,
  `total_tokens`, plus cache fields when the provider reports them)

Access this via the response object:

```python
result = await model.complete(messages)
print(f"Response ID: {result.response_id}")
print(f"Usage: {result.usage}")
```

## Component: ResponsesAPIStateComponent

Track Responses API state across multiple turns:

```python
from ecs_agent.components import ResponsesAPIStateComponent

# Initialize with no previous response
world.add_component(
    entity,
    ResponsesAPIStateComponent(previous_response_id=None),
)

# After first response, update with response_id for context
state = world.get_component(entity, ResponsesAPIStateComponent)
if state:
    # The system will automatically update previous_response_id after each turn
    print(f"Previous Response ID: {state.previous_response_id}")
```

## When to Use

Use Responses API when:
- You need detailed response tracking across multi-turn conversations
- You want enhanced usage metadata for billing or monitoring
- You're using OpenAI's latest models with full Responses API support

Use Chat Completions when:
- You need maximum provider compatibility
- Your provider doesn't support `/v1/responses`
- You don't need response ID tracking

## Implementation Details

`OpenAIModel` detects Responses API support by:
1. Attempting a request to `/v1/responses`
2. Falling back to `/v1/chat/completions` on endpoint HTTP 404 only when storage is unspecified
3. Caching the decision for subsequent requests

Reference errors are handled on `/v1/responses`; see
[Storage and automatic chaining](#storage-and-automatic-chaining) for replay and
compatibility boundaries.

## See Also

- [OpenAIModel](../models.md#openaimodel) — Full OpenAI-compatible model documentation
- [Streaming](./streaming.md) — SSE streaming details
- [Components](../components.md#responsesapistatecomponent) — Component reference
