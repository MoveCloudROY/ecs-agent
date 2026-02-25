# Streaming

The ECS-based LLM Agent framework supports real-time response streaming for providers that implement it (like `OpenAIProvider`). Streaming allows you to process and display the LLM's response as it is being generated, rather than waiting for the entire completion to finish.

## API Reference

To enable streaming, set `stream=True` when calling `provider.complete()`. This changes the return type from `CompletionResult` to an `AsyncIterator[StreamDelta]`.

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
- `tool_calls: list[ToolCall] | None`: Partial tool calls (accumulated by `OpenAIProvider`).
- `finish_reason: str | None`: The reason why the generation stopped (e.g., `"stop"`, `"tool_calls"`).
- `usage: Usage | None`: Usage statistics, typically only provided in the final delta.

## Usage Example

The following pattern demonstrates how to consume a streamed response in real-time.

```python
import sys
from ecs_agent.providers import OpenAIProvider
from ecs_agent.types import Message

provider = OpenAIProvider(api_key="...", model="qwen3.5-plus")
messages = [Message(role="user", content="Tell me a short story.")]

# Call provider with streaming enabled
delta_iterator = await provider.complete(messages, stream=True)

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

### OpenAIProvider
The `OpenAIProvider` uses real Server-Sent Events (SSE) streaming. It automatically accumulates partial tool call arguments from the stream, allowing you to see the full tool call in the final delta when the `finish_reason` is `"tool_calls"`.

### FakeProvider
The `FakeProvider` simulates streaming by emitting the full response character-by-character (or chunk-by-chunk) with small delays, which is useful for testing UI/UX without consuming API credits.

## Caveats

- **Structured Output**: Streaming is NOT compatible with `response_format` (JSON mode). If you need structured output, you must use non-streaming calls.
- **RetryProvider**: The `RetryProvider` does NOT retry streaming calls. If a streaming connection fails halfway, the error is passed through to the consumer.
- **Tool Calls**: While tool calls are streamed, they are usually only useful once the full arguments have been accumulated.
See [`examples/streaming_system_agent.py`](../../examples/streaming_system_agent.py) for a complete demo.

## System-Level Streaming

In addition to direct provider-level streaming, the framework supports system-level streaming through the `ReasoningSystem` and `StreamingComponent`.

### Setup

```python
from ecs_agent.components import StreamingComponent

world.add_component(agent, StreamingComponent(enabled=True))
```

### How It Works

When an entity has `StreamingComponent(enabled=True)`, the `ReasoningSystem` automatically:

1. Calls `provider.complete(stream=True)` instead of the standard call.
2. Publishes `StreamStartEvent(entity_id)`.
3. For each content chunk, publishes `StreamDeltaEvent(entity_id, delta)`.
4. On completion, publishes `StreamEndEvent(entity_id, result)`.
5. Accumulates all chunks into a final `CompletionResult` as normal.

### Subscribing to Stream Events

```python
from ecs_agent.types import StreamStartEvent, StreamDeltaEvent, StreamEndEvent

async def on_start(event: StreamStartEvent):
    print("Streaming started...")

async def on_delta(event: StreamDeltaEvent):
    if event.delta.content:
        print(event.delta.content, end="", flush=True)

async def on_end(event: StreamEndEvent):
    print("\nStreaming complete.")

world.event_bus.subscribe(StreamStartEvent, on_start)
world.event_bus.subscribe(StreamDeltaEvent, on_delta)
world.event_bus.subscribe(StreamEndEvent, on_end)
```

This approach decouples streaming consumers from the provider, allowing multiple subscribers to react to streaming events independently.

### Example
,
