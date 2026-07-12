# OpenAI Responses API

`OpenAIModel` supports the OpenAI `/v1/responses` endpoint when constructed via `Model(..., api_format=ApiFormat.OPENAI_RESPONSES)`, which offers enhanced metadata tracking and streaming capabilities compared to the standard Chat Completions API.

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
  `response.failed` (raises `ValueError` with the provider's error code and
  message).
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

## Stored-Response Chaining (`previous_response_id`)

`ReasoningSystem` records each `response_id` in `ResponsesAPIStateComponent`
and passes it to the next request as `previous_response_id`. Two gates decide
whether the chain is actually sent:

1. **`enable_store`** — the chain is only included when the model was built
   with `Model(..., enable_store=True)`. With `store=false` the referenced
   response was never persisted server-side, and providers reject the id.
2. **Provider support** — some gateways reject `previous_response_id` on the
   plain HTTP endpoint (e.g. `400: previous_response_id is only supported on
   Responses WebSocket v2`). When a 400 blames the chain, the adapter retries
   the request once without it, logs a
   `responses_previous_response_id_rejected` warning, and stops sending the
   chain for the lifetime of that model instance. The 400 is attributed to the
   parameter via the structured error body (`error.param` / `error.message` /
   `error.code`), so an unrelated 400 that merely echoes the id elsewhere in
   the payload does not trigger a spurious retry; a non-JSON body falls back to
   a raw substring scan.

Both degradations are lossless: the full message history is always sent in
`input`, so dropping the chain never loses context. `store: true/false` itself
is still sent according to `enable_store`.

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

The fallback is automatic and transparent — your code doesn't need to change.
It covers both non-streaming and streaming calls: on a streaming 404 the
adapter's `raise_for_status` fires before any delta is yielded, so the request
is replayed against Chat Completions with no duplicated output.

## Response Metadata

When using Responses API, additional metadata is available:

- **response_id**: Unique identifier for each API response
- **previous_response_id**: Links responses in multi-turn conversations
- **usage_tokens**: Detailed token usage (prompt, completion, total)

Access this via the response object:

```python
result = await model.complete(messages)
print(f"Response ID: {result.response_id}")
print(f"Usage: {result.usage_tokens}")
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
2. Falling back to `/v1/chat/completions` on HTTP 404
3. Caching the decision for subsequent requests

A 400 that blames `previous_response_id` stays on the Responses endpoint: the
adapter drops the chain, retries once, and keeps using `/v1/responses` (see
[Stored-Response Chaining](#stored-response-chaining-previous_response_id)).

This ensures zero-configuration compatibility across providers.

## See Also

- [OpenAIModel](../models.md#openaimodel) — Full OpenAI-compatible model documentation
- [Streaming](./streaming.md) — SSE streaming details
- [Components](../components.md#responsesapistatecomponent) — Component reference
