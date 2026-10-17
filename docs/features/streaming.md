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
- `reasoning_signature: str | None`: The provider's signature over the thinking
  block (Anthropic extended thinking), emitted once when the thinking block
  closes. Kept on the final assistant `Message` so signed thinking can be
  replayed on later turns.
- `tool_calls: list[ToolCall] | None`: Completed tool calls with fully parsed
  arguments, emitted exactly once per stream — when the argument streams
  finish (at the `finish_reason` chunk, or at stream end for gateways that
  omit it). Arguments that never became valid JSON are preserved under a
  `{"_partial": "<raw text>"}` placeholder.
- `finish_reason: FinishReason | None`: Why generation stopped, normalized
  across providers — see [Finish reasons](#finish-reasons).
- `usage: Usage | None`: Usage statistics, typically only provided in the final delta.

For Chat Completions streams, `OpenAIModel` requests the usage chunk via
`stream_options: {"include_usage": true}` and tolerates its `"choices": []`
shape (some gateways send it unconditionally). The usage — including
`cached_input_tokens` from `prompt_tokens_details` — arrives as a trailing
usage-only delta *after* the delta carrying `finish_reason`, so consumers must
keep iterating to the end of the stream rather than stopping at
`finish_reason`.

## Finish reasons

`finish_reason` is a `FinishReason` (a `StrEnum`, so it compares equal to its
string value). Providers report this in their own dialect — OpenAI says
`length` where Anthropic says `max_tokens` — and every adapter maps into one
shared vocabulary, so a caller can detect truncation without knowing which
provider answered:

| Value | Meaning | Provider dialects mapped |
|---|---|---|
| `STOP` | Model ended the turn on its own | `stop`, `end_turn`, `stop_sequence` |
| `LENGTH` | Output hit a token limit and was truncated | `length`, `max_tokens`, `max_output_tokens` |
| `TOOL_CALLS` | Turn ended by calling tools | `tool_calls`, `tool_use`, Responses `completed` + function-call output |
| `CONTENT_FILTER` | Provider filtered or refused the output | `content_filter`, `refusal` |
| `CANCELLED` | Provider reports the response was cancelled | `cancelled` |
| `UNKNOWN` | The response ended without the provider stating a reason | unrecognised values; a stream closing on a bare `[DONE]` or EOF |
| `ERROR` | Provider reports generation failure | Responses `failed` |

`UNKNOWN` is a fact about the response, not a parse failure. It is never
reported as `STOP`: claiming a clean completion the provider never stated would
hide truncation. Unrecognised dialect values are logged
(`unmapped_finish_reason`) so the vocabulary can be extended.

The same value is on `CompletionResult.finish_reason` for non-streaming calls,
so both transports answer "why did this end?" identically.

### Agent generation outcomes

`ReasoningSystem` preserves `finish_reason` and `generation_status` on assistant
messages, invocation / observation events, and `StreamEndEvent`. The outcome is
one of `completed`, `tool_calls`, `incomplete`, `filtered`, `cancelled`, `unknown`,
or `error`. Only a completed final answer produces `ReasoningCompleteEvent` and
`TerminalComponent(reason="reasoning_complete")`; complete tool calls continue
the tool loop. Other outcomes stop the turn with their corresponding
`reasoning_<status>` terminal reason. Partial text remains available with its
outcome, and incomplete or malformed tool arguments are not executed.

Responses generation failures returned in an HTTP-success response produce
`FinishReason.ERROR` and `CompletionResult.error` / `StreamDelta.error`, retaining
any reported usage. HTTP / transport errors still raise exceptions. A plain EOF
without a finish reason is `UNKNOWN`, even if some output was received. Legacy
non-streaming providers returning a complete `CompletionResult` without a finish
reason retain their completed-result contract; this compatibility rule does not
apply to stream iterators.

Usage completeness is independent: an answer truncated at its token limit can
still have a complete token report. Missing usage is `UNKNOWN`; usage received
before interruption or failure is retained with its reported completeness. The
shared observation helper emits streamed invocation accounting after consumption,
when the usage and terminal outcome are available.

Streams returned by `complete_with_llm_invocation_event` own their underlying
iterator. When stopping consumption early, call `await stream.aclose()` (or use
`contextlib.aclosing`) to release the HTTP connection and publish one terminal
event. Closing before a finish reason arrives reports cancellation; it does not
infer success from partial text. Repeated close calls do not duplicate accounting.
Tool fragments use the same aggregation as Reasoning, followed by tool identity
and JSON Schema validation before a successful tool-call outcome is reported.

Interactive plan/task hosts clear recoverable reasoning terminal markers after
reporting the outcome and wait for new user input. Child worlds keep their
terminal markers for task outcome inspection. Truncation does not automatically
trigger a continuation or model retry.

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
`OpenAIModel` uses real Server-Sent Events (SSE) streaming. It accumulates partial tool call arguments internally and emits the completed tool calls exactly once; partially accumulated arguments are never re-emitted per chunk. On Chat Completions that happens at the `finish_reason` chunk (or at stream end for gateways that omit it); on the Responses API it happens when the call's `output_item.done` arrives, ahead of the terminal delta.

### ClaudeModel
`ClaudeModel` parses the Anthropic SSE event stream: `text_delta` → `content`, `thinking_delta` → `reasoning_content`, `signature_delta` → `reasoning_signature` (emitted at the thinking block's `content_block_stop`), and tool calls once per `content_block_stop`. Usage from `message_start` (input/cache tokens) and `message_delta` (output tokens) is merged and emitted on the final delta together with the stop reason.

### FakeModel
`FakeModel` simulates streaming by emitting the full response character-by-character, with the final chunk carrying tool calls (if any), `finish_reason`, and usage — mirroring the real adapters' contract so streaming + tool-call paths are testable without API credits.

## Caveats

- **Structured Output**: Streaming is NOT compatible with `response_format` (JSON mode). If you need structured output, you must use non-streaming calls.
- **RetryModel**: `RetryModel` does NOT retry streaming calls. If a streaming connection fails halfway, the error is passed through to the consumer.
- **Tool Calls**: Tool calls arrive on a single delta with fully parsed arguments once their argument streams complete; there is no per-chunk partial emission to accumulate on the consumer side.
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

### Delivery Modes

`StreamingComponent(non_blocking_delta_publish=...)` selects how stream
events reach subscribers:

- **Blocking (default, `False`)**: every event is awaited inline — each delta
  waits for all subscribers before the next chunk is consumed. Deterministic
  and backpressured, but token throughput is gated on the slowest subscriber.
- **Non-blocking (`True`)**: events are enqueued to a per-stream FIFO
  publisher and delivered by a single background consumer, strictly in
  stream order — `StreamEndEvent` (and reasoning-end/content-start markers)
  can never overtake an earlier delta. The turn finishes without waiting for
  subscribers; the queued tail flushes in the background. On cancellation or
  stream error the queued tail is dropped and `StreamEndEvent` is delivered
  inline so consumers unblock.

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
