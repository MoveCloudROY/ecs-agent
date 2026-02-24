<div align="center">
  <h1>ecs-agent</code></h1>

  <p>
    <strong>Entity-Component-System architecture for composable AI agents.</strong>
  </p>
</div>


****

Build modular, testable LLM agents by composing behavior from dataclass components, async systems, and pluggable providers — no inheritance hierarchies, just clean composition.

## Features

- **ECS Architecture** — Entities hold identity, components hold data, systems hold logic. Mix and match freely.
- **Async-Native** — Built on `asyncio` with structured concurrency. Systems at the same priority run concurrently via `TaskGroup`.
- **Provider-Agnostic** — Swap between OpenAI-compatible APIs without touching agent logic. Ships with `OpenAIProvider`, `FakeProvider` (testing), and `RetryProvider` (tenacity).
- **Streaming** — First-class SSE streaming with `AsyncIterator[StreamDelta]` for real-time token delivery.
- **Tool Use** — Register tool schemas and async handlers. The framework manages the LLM ↔ tool call loop automatically.
- **Planning & ReAct** — Built-in `PlanningSystem` and `ReplanningSystem` for multi-step reasoning with dynamic plan adjustment.
- **Multi-Agent** — Multiple agent entities in one `World`, collaborating through an `EventBus` and inbox-based messaging.
- **Structured Output** — JSON mode with Pydantic schema support for type-safe LLM responses.
- **Serialization** — Save and restore full `World` state (entities, components, conversation history) via `WorldSerializer`.
- **Type-Safe** — Full type annotations, `dataclass(slots=True)` components, mypy strict mode. Errors surface at write-time, not runtime.
- **Tool Auto-Discovery & Approval** — Secure your agent with manual approval policies for sensitive tool calls.
- **MCTS Plan Optimization** — Find optimal execution paths using Monte Carlo Tree Search (MCTS) for complex goals.
- **RAG (Vector Search)** — Retrieval-Augmented Generation with pluggable embedding providers and vector stores.

## Installation

```bash
# Clone and install with uv
git clone https://github.com/your-org/ecs-agent.git
cd ecs-agent
uv sync --group dev
# Install with embeddings support (optional)
uv pip install -e ".[embeddings]"
```

> **Requires Python ≥ 3.11**

## Quick Start

```python
import asyncio

from ecs_agent.components import ConversationComponent, LLMComponent
from ecs_agent.core import Runner, World
from ecs_agent.providers import FakeProvider
from ecs_agent.systems.reasoning import ReasoningSystem
from ecs_agent.systems.memory import MemorySystem
from ecs_agent.systems.error_handling import ErrorHandlingSystem
from ecs_agent.types import CompletionResult, Message


async def main() -> None:
    world = World()

    # Create a provider (FakeProvider for demo; swap to OpenAIProvider for real LLMs)
    provider = FakeProvider(
        responses=[
            CompletionResult(
                message=Message(role="assistant", content="Hello! How can I help you?")
            )
        ]
    )

    # Create an agent entity and attach components
    agent = world.create_entity()
    world.add_component(agent, LLMComponent(provider=provider, model="fake", system_prompt="You are a helpful assistant."))
    world.add_component(agent, ConversationComponent(messages=[Message(role="user", content="Hi there!")]))

    # Register systems (priority controls execution order)
    world.register_system(ReasoningSystem(priority=0), priority=0)
    world.register_system(MemorySystem(), priority=10)
    world.register_system(ErrorHandlingSystem(priority=99), priority=99)

    # Run the agent loop
    runner = Runner()
    await runner.run(world, max_ticks=3)

    # Read results
    conv = world.get_component(agent, ConversationComponent)
    if conv:
        for msg in conv.messages:
            print(f"{msg.role}: {msg.content}")


asyncio.run(main())
```

## Using a Real LLM

Copy `.env.example` to `.env` and add your API credentials:

```bash
cp .env.example .env
```

```ini
LLM_API_KEY=your-api-key-here
LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_MODEL=qwen3.5-plus
```

Then use `OpenAIProvider` (works with any OpenAI-compatible API):

```python
from ecs_agent.providers import OpenAIProvider

provider = OpenAIProvider(
    api_key="your-api-key",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="qwen3.5-plus",
)
```

Wrap with `RetryProvider` for automatic retries on transient failures:

```python
from ecs_agent import RetryProvider, RetryConfig

provider = RetryProvider(
    provider=OpenAIProvider(api_key="...", base_url="...", model="..."),
    config=RetryConfig(max_retries=3, initial_wait=1.0, max_wait=30.0),
)
```

## Architecture

```
src/ecs_agent/
├── core/
│   ├── world.py             # World — entity/component/system registry
│   ├── runner.py             # Runner — tick loop until TerminalComponent
│   ├── system.py             # System Protocol + SystemExecutor
│   ├── component.py          # ComponentStore
│   ├── entity.py             # EntityIdGenerator
│   ├── query.py              # Query engine for entity filtering
│   └── event_bus.py          # Pub/sub EventBus
├── components/
│   └── definitions.py        # 12 component dataclasses
├── providers/
│   ├── protocol.py           # LLMProvider Protocol
│   ├── openai_provider.py    # OpenAI-compatible HTTP provider (httpx)
│   ├── fake_provider.py      # Deterministic test provider
│   └── retry_provider.py     # Retry wrapper (tenacity)
├── systems/                  # 10 built-in systems
│   ├── reasoning.py          # LLM inference
│   ├── planning.py           # Multi-step plan execution
│   ├── replanning.py         # Dynamic plan adjustment
│   ├── tool_execution.py     # Tool call dispatch
│   ├── memory.py             # Conversation memory management
│   ├── collaboration.py      # Multi-agent messaging
│   └── error_handling.py     # Error capture and recovery
│   ├── tree_search.py        # MCTS plan optimization
│   ├── tool_approval.py      # Human-in-the-loop approval
│   └── rag.py                # Retrieval-Augmented Generation
├── types.py                  # Core types (EntityId, Message, ToolCall, etc.)
├── serialization.py          # WorldSerializer for save/load
└── logging.py                # structlog configuration
```

### How It Works

The **Runner** repeatedly ticks the **World** until a `TerminalComponent` is attached to an entity (or `max_ticks` is reached). Each tick:

1. Systems execute in priority order (lower = earlier)
2. Systems at the same priority run **concurrently**
3. Systems read/write components on entities — that's the entire data flow

```
World
 ├── Entity 0 ── [LLMComponent, ConversationComponent, PlanComponent, ...]
 ├── Entity 1 ── [LLMComponent, ConversationComponent, CollaborationComponent, ...]
 └── Systems ─── [ReasoningSystem(0), PlanningSystem(0), ToolExecutionSystem(5), MemorySystem(10), ...]
                          │
                    Runner.run()
                          │
              Tick 1 → Tick 2 → ... → TerminalComponent found → Done
```

### Components

| Component | Purpose |
|-----------|---------|
| `LLMComponent` | Provider, model, system prompt |
| `ConversationComponent` | Message history with optional size limit |
| `PlanComponent` | Multi-step plan with progress tracking |
| `ToolRegistryComponent` | Tool schemas and async handler functions |
| `PendingToolCallsComponent` | Tool calls awaiting execution |
| `ToolResultsComponent` | Results from completed tool calls |
| `CollaborationComponent` | Inbox for multi-agent messaging |
| `OwnerComponent` | Parent entity reference |
| `SystemPromptComponent` | Dedicated system prompt storage |
| `KVStoreComponent` | Generic key-value scratch space |
| `ErrorComponent` | Error details for failed operations |
| `TerminalComponent` | Signals agent completion |
| `ToolApprovalComponent` | Policy-based tool call filtering |
| `SandboxConfigComponent` | Execution limits for tools |
| `PlanSearchComponent` | MCTS search configuration |
| `RAGTriggerComponent` | Vector search retrieval state |
| `EmbeddingComponent` | Embedding provider reference |
| `VectorStoreComponent` | Vector store reference |

## Examples

The `examples/` directory contains 9 runnable demos:

| Example | Description |
|---------|-------------|
| [`chat_agent.py`](examples/chat_agent.py) | Minimal agent with FakeProvider — good starting point |
| [`tool_agent.py`](examples/tool_agent.py) | Tool use with automatic call/result cycling |
| [`react_agent.py`](examples/react_agent.py) | ReAct pattern — Thought → Action → Observation loop |
| [`plan_and_execute_agent.py`](examples/plan_and_execute_agent.py) | Dynamic replanning with RetryProvider and configurable timeouts |
| [`streaming_agent.py`](examples/streaming_agent.py) | Real-time token streaming via SSE |
| [`retry_agent.py`](examples/retry_agent.py) | RetryProvider with custom retry configuration |
| [`multi_agent.py`](examples/multi_agent.py) | Two agents collaborating through inbox messaging |
| [`structured_output_agent.py`](examples/structured_output_agent.py) | Pydantic schema → JSON mode for type-safe responses |
| [`serialization_demo.py`](examples/serialization_demo.py) | Save and restore World state to/from JSON |
| [`tool_approval_agent.py`](examples/tool_approval_agent.py) | Manual approval flow for sensitive tools |
| [`tree_search_agent.py`](examples/tree_search_agent.py) | MCTS-based planning for complex goals |
| [`rag_agent.py`](examples/rag_agent.py) | Retrieval-Augmented Generation demo |

Run any example:

```bash
# FakeProvider examples (no API key needed)
uv run python examples/chat_agent.py
uv run python examples/tool_agent.py

# Real LLM examples (requires .env with API credentials)
uv run python examples/react_agent.py
uv run python examples/streaming_agent.py
```

## Development

### Tests

```bash
# Run all tests
uv run pytest

# Run a single test file
uv run pytest tests/test_world.py

# Run tests matching a keyword
uv run pytest -k "streaming"

# Verbose output
uv run pytest -v
```

### Type Checking

```bash
# Full strict type check
uv run mypy src/ecs_agent/

# Single file
uv run mypy src/ecs_agent/core/world.py
```

### Project Configuration

- **Build**: hatchling
- **Package manager**: uv (lockfile: `uv.lock`)
- **pytest**: `asyncio_mode = "auto"` — async tests run without explicit event loop setup
- **mypy**: `strict = true`, `python_version = "3.11"`

## Documentation

See [`docs/`](docs/) for detailed guides:

- [Getting Started](docs/getting-started.md) — Installation, first agent, key concepts
- [Architecture](docs/architecture.md) — ECS pattern, data flow, system lifecycle
- [Core Concepts](docs/core-concepts.md) — World, Entity, Component, System, Runner
- [Components](docs/components.md) — All 12 components with usage examples
- [Systems](docs/systems.md) — All 7 systems with configuration details
- [Providers](docs/providers.md) — LLM provider protocol, built-in providers
- [API Reference](docs/api-reference.md) — Complete API surface
- [Examples](docs/examples.md) — Walkthrough of all 9 examples
- [Streaming](docs/features/streaming.md) — SSE streaming setup and usage
- [Retry](docs/features/retry.md) — RetryProvider configuration
- [Serialization](docs/features/serialization.md) — World state persistence
- [Logging](docs/features/logging.md) — structlog integration
- [Structured Output](docs/features/structured-output.md) — Pydantic schema → JSON mode

## License

This project is currently unlicensed. See the repository for updates.
