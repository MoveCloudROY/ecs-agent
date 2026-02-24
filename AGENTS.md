# AGENTS.md — ecs-agent

## Project Overview

ECS-based LLM Agent framework in Python. Uses Entity-Component-System architecture
to compose AI agents from dataclass components, async systems, and LLM providers.

- **Python**: ≥3.11 (uses `X | None`, `list[str]`, `match` syntax)
- **Package manager**: `uv` (lockfile: `uv.lock`)
- **Build backend**: `hatchling`
- **Source layout**: `src/ecs_agent/` (installed as `ecs_agent`)
- **Tests**: `tests/`

## Build & Test Commands

```bash
# Install dependencies (dev group)
uv sync --group dev

# Run all tests
uv run pytest

# Run a single test file
uv run pytest tests/test_world.py

# Run a single test by name
uv run pytest tests/test_world.py::test_world_create_entity_returns_incrementing_ids

# Run tests matching a keyword
uv run pytest -k "streaming"

# Run with verbose output
uv run pytest -v

# Type-check (strict mode)
uv run mypy src/ecs_agent/

# Type-check a single file
uv run mypy src/ecs_agent/core/world.py
```

### Key pytest config (`pyproject.toml`)

- `testpaths = ["tests"]`
- `asyncio_mode = "auto"` — async tests run automatically, no explicit event loop setup

### mypy config (`pyproject.toml`)

- `strict = true`
- `python_version = "3.11"`
- `disallow_untyped_defs = true`
- `disallow_incomplete_defs = true`

No ruff, black, isort, or flake8 configured. No Makefile, CI pipeline, or pre-commit hooks.

## Code Style

### Imports

Order: stdlib → typing/collections.abc → third-party → internal package.

```python
from __future__ import annotations          # when forward refs needed

import json                                  # stdlib
from typing import Any, TypeVar              # typing
from collections.abc import AsyncIterator    # collections.abc

import httpx                                 # third-party
from tenacity import AsyncRetrying           # third-party

from ecs_agent.logging import get_logger     # internal
from ecs_agent.types import Message          # internal
```

- Use `from __future__ import annotations` only when forward references are needed
- Multi-symbol imports use parenthesized form with trailing comma
- Module-level `logger = get_logger(__name__)` immediately after imports

### Naming

| Element | Convention | Example |
|---------|-----------|---------|
| Classes | PascalCase | `OpenAIProvider`, `ErrorHandlingSystem` |
| Functions/methods | snake_case | `create_entity`, `add_component` |
| Constants | UPPER_SNAKE | (none currently; use when needed) |
| Private attrs | `_` prefix | `self._client`, `self._components` |
| Type aliases | PascalCase via `NewType` | `EntityId = NewType("EntityId", int)` |
| Test functions | `test_<unit>_<behavior>` | `test_world_create_entity_returns_incrementing_ids` |

### Type Annotations

- **All** public functions and methods must have return type annotations
- Use PEP 604 unions: `str | None`, not `Optional[str]`
- Use builtin generics: `list[str]`, `dict[str, Any]`, `tuple[EntityId, Message]`
- Use `collections.abc.AsyncIterator` for async generators, not `typing.AsyncIterator`
- Protocols for interfaces: `class LLMProvider(Protocol)` in `providers/protocol.py`
- `@runtime_checkable` on protocols that need `isinstance` checks
- **Never** use `as any`, `# type: ignore` unless unavoidable (document why)

### Dataclasses

All domain types and components use `@dataclass(slots=True)`:

```python
@dataclass(slots=True)
class LLMComponent:
    provider: LLMProvider
    model: str
    system_prompt: str = ""
```

- Pydantic `BaseModel` is used **only** in examples for structured output demos
- Library code must not hard-depend on pydantic at import time

### Docstrings

```python
"""One-line summary for simple items."""

"""Multi-line summary for complex items.

Longer description if needed.

Args:
    world: World instance to process
    max_ticks: Maximum iterations (default 100)

Returns:
    CompletionResult or async iterator of StreamDelta.
"""
```

- Module-level docstring on every `.py` file
- Class-level docstring (short, one line is fine)
- Args/Returns sections on public methods with non-obvious parameters

### Error Handling

- **No custom exception classes** — use stdlib and library exceptions
- Systems catch `Exception`, attach `ErrorComponent` + `TerminalComponent` to the entity
- Providers catch specific `httpx.HTTPStatusError` / `httpx.RequestError`
- Always log before re-raising: `logger.error("event_name", key=value, exception=str(exc))`
- Never use bare `except:` or empty `except Exception: pass`

### Logging

Uses `structlog` via `ecs_agent.logging`:

```python
from ecs_agent.logging import get_logger
logger = get_logger(__name__)

logger.info("event_name", entity_id=eid, detail=value)
logger.error("llm_http_error", status_code=code, exception=str(exc))
```

- Event names are snake_case strings (first positional arg)
- Structured key-value pairs as kwargs

### Async Patterns

- All systems implement `async def process(self, world: World) -> None`
- Systems at the same priority run concurrently via `asyncio.TaskGroup`
- Streaming uses `AsyncIterator[StreamDelta]` (async generator with `yield`)
- `httpx.AsyncClient` for HTTP; use `client.stream()` context manager for SSE
- Tests use `@pytest.mark.asyncio` (auto mode means just `async def test_...` works)

## Architecture Quick Reference

```
src/ecs_agent/
├── __init__.py              # Public API re-exports (__all__)
├── types.py                 # Core types: EntityId, Message, ToolCall, etc.
├── logging.py               # structlog config (get_logger, configure_logging)
├── serialization.py         # WorldSerializer for save/load
├── core/
│   ├── world.py             # World: entity/component/system manager
│   ├── runner.py            # Runner: tick loop until TerminalComponent
│   ├── system.py            # System Protocol + SystemExecutor
│   ├── component.py         # ComponentStore
│   ├── entity.py            # EntityIdGenerator
│   ├── query.py             # Query engine
│   └── event_bus.py         # Pub/sub EventBus
├── components/
│   └── definitions.py       # All 12 component dataclasses
├── providers/
│   ├── protocol.py          # LLMProvider Protocol
│   ├── openai_provider.py   # OpenAI-compatible HTTP provider
│   ├── fake_provider.py     # Deterministic test provider
│   └── retry_provider.py    # Retry wrapper (tenacity)
└── systems/                 # reasoning, planning, tool_execution, etc.
```

### Key API Patterns

```python
world = World()
entity = world.create_entity()             # returns EntityId
world.add_component(entity, SomeComponent(...))
comp = world.get_component(entity, SomeComponent)  # returns T | None
world.register_system(MySystem(), priority=0)
runner = Runner()
await runner.run(world, max_ticks=10)
```

### Exports

- `from ecs_agent import RetryProvider` — NOT from `ecs_agent.providers`
- `from ecs_agent.core import World` — core classes
- `from ecs_agent.providers.openai_provider import OpenAIProvider` — direct module import
- Check `__init__.py` files for canonical import paths before adding new imports