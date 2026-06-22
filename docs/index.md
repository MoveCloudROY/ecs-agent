# ecs_agent

ECS-based LLM Agent framework

## Why ecs_agent?

The Entity-Component-System (ECS) pattern makes LLM agents composable and testable. By separating data (Components) from logic (Systems), you can build complex agent behaviors without the typical spaghetti code of stateful objects. Each agent is an Entity in a World, making it easy to inspect, serialize, and debug every step of the reasoning process.

## Key Features

*   **ECS Architecture** — Composition-first design. Mix 30+ components to build custom agents without inheritance bloat.
*   **Multi-Agent Orchestration** — Spawn subagents for subtasks. Parent-child messaging via pub/sub. Unified delegation API.
*   **Tree-Structured Conversations** — Branch reasoning paths, navigate multiple strategies, linearize for LLM compatibility.
*   **Advanced Planning** — Built-in Planning & ReAct systems with dynamic replanning. MCTS optimization for complex goals.
*   **Production Infrastructure** — Multiple LLM model integrations, streaming tokens (SSE), checkpoint/undo, conversation compaction, tool approval.
*   **Type-Safe & Testable** — Full type annotations, strict mypy, FakeModel for deterministic tests, World serialization.

## Documentation

*   [Getting Started](getting-started.md)
*   [Architecture](architecture.md)
*   [Core Concepts](core-concepts.md)
*   [Components](components.md)
*   [Systems](systems.md)
*   [Models](models.md)
*   [Features]
    *   [Streaming](features/streaming.md)
    *   [Retry](features/retry.md)
    *   [Serialization](features/serialization.md)
    *   [Structured Logging](features/logging.md)
    *   [Prometheus Metrics](features/metrics.md)
    *   [Structured Output](features/structured-output.md)
*   [Context Management](features/context-management.md)
*   [Tool Discovery & Approval](features/tool-discovery.md)
*   [User Input](features/user-input.md)
*   [Agent Scratchbook](features/scratchbook.md) — Persistent artifact storage.
*   [Agent DSL](features/agent-dsl.md) — JSON and Markdown configuration format.

*   [Examples](examples.md)
*   [API Reference](api-reference.md)

## Quick Start

```python
import asyncio
from ecs_agent.core import World, Runner
from ecs_agent.components import LLMComponent, ConversationComponent
from ecs_agent.providers import FakeModel
from ecs_agent.systems.reasoning import ReasoningSystem
from ecs_agent.systems.error_handling import ErrorHandlingSystem
from ecs_agent.types import CompletionResult, Message

async def main():
    world = World()
    model = FakeModel([CompletionResult(message=Message(role="assistant", content="Hello!"))])
    agent = world.create_entity()
    world.add_component(agent, LLMComponent(model=model, system_prompt="You are helpful."))
    world.add_component(agent, ConversationComponent(messages=[Message(role="user", content="Hi!")]))
    world.register_system(ReasoningSystem(priority=0), priority=0)
    world.register_system(ErrorHandlingSystem(priority=99), priority=99)
    runner = Runner()
    await runner.run(world, max_ticks=3)

asyncio.run(main())
```

## Requirements

*   Python >= 3.11
*   Version: 0.1.0
*   Runtime dependencies: `httpx`, `tenacity`, `structlog`, `pydantic`
