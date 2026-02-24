# ecs_agent

ECS-based LLM Agent framework

## Why ecs_agent?

The Entity-Component-System (ECS) pattern makes LLM agents composable and testable. By separating data (Components) from logic (Systems), you can build complex agent behaviors without the typical spaghetti code of stateful objects. Each agent is an Entity in a World, making it easy to inspect, serialize, and debug every step of the reasoning process.

## Key Features

*   ECS core for modular agent design.
*   7 specialized systems for reasoning, planning, memory, and execution.
*   3 provider types for LLM integration and testing.
*   Async streaming support for real-time responses.
*   Tenacity-based retry logic for reliable API calls.
*   Full World serialization for state persistence.
*   Pydantic-powered structured output and tool schemas.
*   Structured logging for observability.

## Documentation

*   [Getting Started](getting-started.md)
*   [Architecture](architecture.md)
*   [Core Concepts](core-concepts.md)
*   [Components](components.md)
*   [Systems](systems.md)
*   [Providers](providers.md)
*   [Features]
    *   [Streaming](features/streaming.md)
    *   [Retry](features/retry.md)
    *   [Serialization](features/serialization.md)
    *   [Structured Logging](features/logging.md)
    *   [Structured Output](features/structured-output.md)
*   [Examples](examples.md)
*   [API Reference](api-reference.md)

## Quick Start

```python
import asyncio
from ecs_agent.core import World, Runner
from ecs_agent.components import LLMComponent, ConversationComponent
from ecs_agent.providers import FakeProvider
from ecs_agent.systems.reasoning import ReasoningSystem
from ecs_agent.systems.memory import MemorySystem
from ecs_agent.systems.error_handling import ErrorHandlingSystem
from ecs_agent.types import CompletionResult, Message

async def main():
    world = World()
    provider = FakeProvider([CompletionResult(message=Message(role="assistant", content="Hello!"))])
    agent = world.create_entity()
    world.add_component(agent, LLMComponent(provider=provider, model="fake", system_prompt="You are helpful."))
    world.add_component(agent, ConversationComponent(messages=[Message(role="user", content="Hi!")]))
    world.register_system(ReasoningSystem(priority=0), priority=0)
    world.register_system(MemorySystem(), priority=10)
    world.register_system(ErrorHandlingSystem(priority=99), priority=99)
    runner = Runner()
    await runner.run(world, max_ticks=3)

asyncio.run(main())
```

## Requirements

*   Python >= 3.11
*   Version: 0.1.0
*   Runtime dependencies: `httpx`, `tenacity`, `structlog`, `pydantic`
