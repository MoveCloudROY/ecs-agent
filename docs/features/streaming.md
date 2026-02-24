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
