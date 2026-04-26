# OpenAI Responses API

The `OpenAIProvider` supports the OpenAI `/v1/responses` endpoint, which offers enhanced metadata tracking and streaming capabilities compared to the standard Chat Completions API.

## Overview

The Responses API provides:
- Incremental response IDs for tracking multi-turn conversations
- Enhanced usage metadata (prompt tokens, completion tokens, total tokens)
- Automatic fallback to Chat Completions if Responses API is unavailable
- Full streaming support with Server-Sent Events (SSE)

## Usage

### Basic Setup

Enable Responses API with the `use_responses_api=True` parameter:

```python
from ecs_agent.providers import OpenAIProvider

provider = OpenAIProvider(
    api_key="your-api-key",
    base_url="https://api.openai.com/v1",
    model="gpt-4o",
    use_responses_api=True,  # Enable Responses API
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
    LLMComponent(provider=provider, model="gpt-4o", system_prompt="You are a helpful assistant."),
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
async for delta in provider.stream(messages):
    print(delta.content, end="", flush=True)
```

## Automatic Fallback

If the Responses API endpoint is not available (e.g., provider doesn't support it), the `OpenAIProvider` automatically falls back to Chat Completions:

```python
# This works even if base_url doesn't support /v1/responses
provider = OpenAIProvider(
    api_key="your-api-key",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # May not support Responses API
    model="qwen3.5-plus",
    use_responses_api=True,  # Will fall back to Chat Completions if needed
)
```

The fallback is automatic and transparent — your code doesn't need to change.

## Response Metadata

When using Responses API, additional metadata is available:

- **response_id**: Unique identifier for each API response
- **previous_response_id**: Links responses in multi-turn conversations
- **usage_tokens**: Detailed token usage (prompt, completion, total)

Access this via the response object:

```python
result = await provider.complete(messages)
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

The `OpenAIProvider` detects Responses API support by:
1. Attempting a request to `/v1/responses`
2. Falling back to `/v1/chat/completions` on HTTP 404
3. Caching the decision for subsequent requests

This ensures zero-configuration compatibility across providers.

## See Also

- [OpenAIModel](../models.md#openaimodel) — Full OpenAI-compatible model documentation
- [Streaming](./streaming.md) — SSE streaming details
- [Components](../components.md#responsesapistatecomponent) — Component reference
