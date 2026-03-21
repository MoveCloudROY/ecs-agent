# User Input System

The `UserInputSystem` enables agents to request and wait for human input during execution, supporting both timed and infinite waits.

For multi-turn interactive agents, `UserInputSystem` is often paired with the opt-in `TerminalCleanupSystem`. This keeps `UserInputSystem` focused on input futures while allowing interactive runtimes to clear `TerminalComponent(reason="reasoning_complete")` after reasoning completes.

## Components

### UserInputComponent

| Field | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `prompt` | `str` | `""` | Prompt text to display to the user |
| `future` | `asyncio.Future[str] \| None` | `None` | Future that resolves with user input |
| `timeout` | `float \| None` | `None` | Seconds to wait; `None` for infinite wait |
| `result` | `str \| None` | `None` | The received user input |

## System Behavior

The `UserInputSystem` runs at priority `-10` (before reasoning) and processes entities with a `UserInputComponent`:

1. Creates an `asyncio.Future` if none exists on the component.
2. Publishes `UserInputRequestedEvent(entity_id, prompt)`.
3. Awaits the future with `asyncio.wait_for(asyncio.shield(future), timeout=component.timeout)`.
4. When `timeout=None`, the system waits **indefinitely** for input.
5. On resolve: stores the result in `UserInputComponent.result` and appends it as a user message to `ConversationComponent`.
6. On timeout: adds `ErrorComponent` and `TerminalComponent` to the entity.

## Setup

```python
from ecs_agent.components import UserInputComponent
from ecs_agent.systems.user_input import UserInputSystem

# Add input component to agent
world.add_component(agent, UserInputComponent(
    prompt="What would you like to do next?",
    timeout=None,  # Wait indefinitely
))

# Register system
world.register_system(UserInputSystem(priority=-10), priority=-10)
```

## Pairing with TerminalCleanupSystem

`TerminalCleanupSystem` is the recommended helper for interactive continuations that must continue after a successful reasoning turn. Register it after reasoning, typically with `priority=1`, so it can clear `reasoning_complete` before the next tick's Runner stop check.

```python
from ecs_agent.systems import TerminalCleanupSystem

world.register_system(TerminalCleanupSystem(priority=1), priority=1)
world.register_system(UserInputSystem(priority=-10), priority=-10)
```

This behavior is **opt-in**. Runner still stops on top-level `TerminalComponent` unless a cleanup system removes a selected reason first.

## Providing Input

External code provides input by resolving the future:

```python
from ecs_agent.types import UserInputRequestedEvent

async def handle_input(event: UserInputRequestedEvent):
    # Get input from user (e.g., stdin, UI, API)
    user_response = input(event.prompt)
    event.future.set_result(user_response)

world.event_bus.subscribe(UserInputRequestedEvent, handle_input)
```

## Infinite Wait

Both the `UserInputSystem` and the `Runner` support infinite waiting:

- **UserInputSystem**: Set `timeout=None` on `UserInputComponent` for indefinite wait.
- **Runner**: Set `max_ticks=None` on `runner.run()` for indefinite execution.

Together, these allow building interactive agents that wait for human input without artificial time limits:

```python
runner = Runner()
await runner.run(world, max_ticks=None)  # Run until TerminalComponent
```

## Events

- `UserInputRequestedEvent(entity_id, prompt)` — Published when input is needed.

## Imports

```python
from ecs_agent import UserInputComponent, UserInputRequestedEvent, UserInputSystem
```

All types are available from the top-level `ecs_agent` package.
