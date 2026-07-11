# Streaming

The ECS-based LLM Agent framework supports real-time response streaming for models that implement it (such as `OpenAIModel`). Streaming allows you to process and display the LLM's response as it is being generated, rather than waiting for the entire completion to finish.

## API Reference

To enable streaming, set `stream=True` when calling `model.complete()`. This changes the return type from `CompletionResult` to an `AsyncIterator[StreamDelta]`.

```python
async def complete(
    self,
    messages: list[Message],
    stream: bool = False,
    **kwargs: Any
) -> CompletionResult | AsyncIterator[StreamDelta]:
    ...
```

### StreamDelta

Each chunk emitted by the iterator is a `StreamDelta` object with the following fields:

- `content: str | None`: The partial text content of the response.
- `reasoning_content: str | None`: The streamed reasoning/thinking text when provider exposes it.
- `tool_calls: list[ToolCall] | None`: Partial tool calls (accumulated by `OpenAIModel`).
- `finish_reason: str | None`: The reason why the generation stopped (e.g., `"stop"`, `"tool_calls"`).
- `usage: Usage | None`: Usage statistics, typically only provided in the final delta.

For Chat Completions streams, `OpenAIModel` requests the usage chunk via
`stream_options: {"include_usage": true}` and tolerates its `"choices": []`
shape (some gateways send it unconditionally). The usage — including
`cached_input_tokens` from `prompt_tokens_details` — arrives as a trailing
usage-only delta *after* the delta carrying `finish_reason`, so consumers must
keep iterating to the end of the stream rather than stopping at
`finish_reason`.

## Usage Example

The following pattern demonstrates how to consume a streamed response in real-time.

```python
import sys
from ecs_agent.providers import Model
from ecs_agent.providers.config import ApiFormat
from ecs_agent.types import Message

model = Model(
    "qwen3.5-plus",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key="...",
    api_format=ApiFormat.OPENAI_CHAT_COMPLETIONS,
)
messages = [Message(role="user", content="Tell me a short story.")]

# Call model with streaming enabled
delta_iterator = await model.complete(messages, stream=True)

async for delta in delta_iterator:
    if delta.content:
        # Write chunks to stdout without newlines
        sys.stdout.write(delta.content)
        sys.stdout.flush()
    
    if delta.finish_reason:
        print(f"\nFinished: {delta.finish_reason}")
        if delta.usage:
            print(f"Total tokens: {delta.usage.total_tokens}")
```

## Provider Implementation Details

### OpenAIModel
`OpenAIModel` uses real Server-Sent Events (SSE) streaming. It automatically accumulates partial tool call arguments from the stream, allowing you to see the full tool call in the final delta when the `finish_reason` is `"tool_calls"`.

### FakeModel
`FakeModel` simulates streaming by emitting the full response character-by-character (or chunk-by-chunk) with small delays, which is useful for testing UI/UX without consuming API credits.

## Caveats

- **Structured Output**: Streaming is NOT compatible with `response_format` (JSON mode). If you need structured output, you must use non-streaming calls.
- **RetryModel**: `RetryModel` does NOT retry streaming calls. If a streaming connection fails halfway, the error is passed through to the consumer.
- **Tool Calls**: While tool calls are streamed, they are usually only useful once the full arguments have been accumulated.
See [`examples/streaming_system_agent.py`](../../examples/streaming_system_agent.py) for a complete demo.

## System-Level Streaming

In addition to direct model-level streaming, the framework supports system-level streaming through the `ReasoningSystem` and `StreamingComponent`.

### Setup

```python
from ecs_agent.components import StreamingComponent

world.add_component(agent, StreamingComponent(enabled=True))
```

### How It Works

When an entity has `StreamingComponent(enabled=True)`, the `ReasoningSystem` automatically:

1. Calls `model.complete(stream=True)` instead of the standard call.
2. Publishes `StreamStartEvent(entity_id)`.
3. Publishes `StreamReasoningDeltaEvent(entity_id, reasoning_delta)` for reasoning chunks (if present).
4. Publishes `StreamReasoningEndEvent(entity_id)` when reasoning phase ends.
5. Publishes `StreamContentStartEvent(entity_id)` when assistant content phase starts.
6. For each content chunk, publishes `StreamContentDeltaEvent(entity_id, delta)`.
7. On completion, publishes `StreamEndEvent(entity_id, timestamp)`.
8. Accumulates all chunks into a final `CompletionResult` as normal.

### Subscribing to Stream Events

```python
from ecs_agent.types import (
    StreamContentDeltaEvent,
    StreamContentStartEvent,
    StreamEndEvent,
    StreamReasoningDeltaEvent,
    StreamReasoningEndEvent,
    StreamStartEvent,
)

async def on_start(event: StreamStartEvent):
    print("Streaming started...")

async def on_reasoning_delta(event: StreamReasoningDeltaEvent):
    print(f"[think:{event.reasoning_delta}]", end="", flush=True)

async def on_reasoning_end(event: StreamReasoningEndEvent):
    print("\n[reasoning ended]")

async def on_content_start(event: StreamContentStartEvent):
    print("[content started]")

async def on_delta(event: StreamContentDeltaEvent):
    if event.delta:
        print(event.delta, end="", flush=True)

async def on_end(event: StreamEndEvent):
    print("\nStreaming complete.")

world.event_bus.subscribe(StreamStartEvent, on_start)
world.event_bus.subscribe(StreamReasoningDeltaEvent, on_reasoning_delta)
world.event_bus.subscribe(StreamReasoningEndEvent, on_reasoning_end)
world.event_bus.subscribe(StreamContentStartEvent, on_content_start)
world.event_bus.subscribe(StreamContentDeltaEvent, on_delta)
world.event_bus.subscribe(StreamEndEvent, on_end)
```

This approach decouples streaming consumers from the provider, allowing multiple subscribers to react to streaming events independently.

### Example

See `examples/streaming_system_agent.py` for a runnable end-to-end system-level streaming demo.

## Subagent Streaming Telemetry

When a subagent is launched with `background=True` and `stream=True`, the `SubagentSystem` bridges streaming events from the child world to the parent world's `EventBus`. This provides real-time visibility into the subagent's progress without requiring the parent agent to poll for results.

### Event Types

Three event types are used for subagent telemetry:

1. **`SubagentStreamStartEvent`**: Emitted when the subagent begins its first reasoning or content generation.
2. **`SubagentStreamDeltaEvent`**: Emitted for each chunk of text or reasoning generated by the subagent.
3. **`SubagentStreamEndEvent`**: Emitted when the subagent finishes its generation.

### Event Fields

All subagent stream events share these common fields:

- `session_id: str`: The unique session ID of the subagent delegation.
- `parent_entity_id: EntityId`: The ID of the parent entity that launched the subagent.
- `category: str`: The subagent category (e.g., `"researcher"`).
- `child_world_name: str`: The name of the child world where the subagent is running.
- `seq: int`: A monotonically increasing sequence number for events within the session.
- `timestamp: str`: ISO 8601 timestamp of the event.

**`SubagentStreamDeltaEvent`** also includes:

- `delta: str`: The partial text content of the assistant's response.
- `reasoning_delta: str | None`: The partial reasoning/thinking text (if supported by the provider).

**`SubagentStreamEndEvent`** also includes:

- `total_tokens: int | None`: Total tokens consumed by the subagent (if available).

### Usage Notes

- **Telemetry-First**: This is a telemetry-only feature. The final, durable result of the subagent must still be retrieved via `subagent_result`.
- **No Durable Replay**: Subagent stream events are transient and are not stored in the `World` or `Scratchbook`. They are only available to active `EventBus` subscribers.
- **EventBus-Only**: These events are published directly to the parent world's `EventBus`. They do not appear in the parent agent's conversation history.
