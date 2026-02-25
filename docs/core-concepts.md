# Core Concepts

This guide provides a detailed reference for the core primitives in the ECS library. It covers class definitions, method signatures, and basic usage patterns.

## EntityId
`ecs_agent.types`

An `EntityId` is a unique identifier for an entity. It's defined as a `NewType` of an integer.

```python
EntityId = NewType("EntityId", int)
```

## EntityIdGenerator
`ecs_agent.core.entity`

The generator manages the creation of unique IDs.

- `__init__(self) -> None`: Sets the internal counter to 0.
- `next(self) -> EntityId`: Increments and returns the next available ID.

## ComponentStore
`ecs_agent.core.component`

This internal class manages how components are mapped to entities. It uses an internal dictionary structured as `dict[type[Any], dict[EntityId, Any]]`.

- `add(self, entity_id: EntityId, component: Any) -> None`: Stores a component for a specific entity.
- `get(self, entity_id: EntityId, component_type: type[T]) -> T | None`: Retrieves a component by its type.
- `remove(self, entity_id: EntityId, component_type: type[Any]) -> None`: Deletes a component from an entity.
- `has(self, entity_id: EntityId, component_type: type[Any]) -> bool`: Checks if an entity has a specific component.
- `delete_entity(self, entity_id: EntityId) -> None`: Removes all components associated with an entity.
- `get_all(self, component_type: type[T]) -> dict[EntityId, T]`: Returns all entities and their instances of a specific component.

## World
`ecs_agent.core.world`

The `World` is the primary entry point for interacting with the ECS. It integrates the storage, event bus, and execution logic.

- `__init__(self) -> None`: Creates the `EntityIdGenerator`, `ComponentStore`, `SystemExecutor`, `EventBus`, and `Query` instances.
- `event_bus`: Property that provides access to the `EventBus`.
- `create_entity(self) -> EntityId`: Creates a new entity and returns its ID.
- `add_component(self, entity_id: EntityId, component: Any) -> None`: Attaches a component to an entity.
- `get_component(self, entity_id: EntityId, component_type: type[T]) -> T | None`: Finds a component for an entity.
- `remove_component(self, entity_id: EntityId, component_type: type[Any]) -> None`: Removes a component.
- `has_component(self, entity_id: EntityId, component_type: type[Any]) -> bool`: Verifies if an entity has a component.
- `delete_entity(self, entity_id: EntityId) -> None`: Fully removes an entity and its data.
- `register_system(self, system: System, priority: int) -> None`: Adds a system to the executor with a set priority.
- `async process(self) -> None`: Triggers the system execution cycle.
- `query(self, *component_types: type[Any]) -> list[tuple[EntityId, tuple[Any, ...]]]`: Finds entities matching a set of components.

### Usage Example
```python
from ecs_agent.core import World

world = World()
player = world.create_entity()
world.add_component(player, Position(x=0, y=0))

# Querying for entities
results = world.query(Position)
for entity_id, (pos,) in results:
    print(f"Entity {entity_id} is at {pos.x}, {pos.y}")
```

## System Protocol
`ecs_agent.core.system`

A `System` defines logic that operates on entities. It's a `typing.Protocol`, meaning any class with the correct `process` method qualifies. You don't need to inherit from a specific base class.

- `async def process(self, world: World) -> None`: The main logic loop for the system.

## SystemExecutor
`ecs_agent.core.system`

The executor manages when and how systems run.

- `register(self, system: System, priority: int) -> None`: Registers a system.
- `async execute(self, world: World) -> None`: Runs all registered systems. It groups them by priority and runs systems within the same priority level in parallel using an `asyncio.TaskGroup`.

## Query
`ecs_agent.core.query`

Queries allow you to filter entities based on their components.

- `__init__(self, component_store: ComponentStore) -> None`: Links the query engine to a component store.
- `get(self, *component_types: type[Any]) -> list[tuple[EntityId, tuple[Any, ...]]]`: Performs an intersection search. It returns only the entities that have every requested component type.

## EventBus
`ecs_agent.core.event_bus`

The `EventBus` facilitates decoupled communication between different parts of the system.

- `subscribe(self, event_type: type[T], handler: Callable[[T], Awaitable[None]]) -> None`: Registers a listener for an event.
- `unsubscribe(self, event_type: type[T], handler: Callable[[T], Awaitable[None]]) -> None`: Removes a listener.
- `async publish(self, event: T) -> None`: Sends an event to all subscribers. It uses `asyncio.gather` with `return_exceptions=True` to ensure one failing handler doesn't stop others.
- `clear(self) -> None`: Removes all subscribers.

### Usage Example
```python
from ecs_agent.core import EventBus

async def on_death(event: DeathEvent):
    print(f"Entity {event.entity_id} died")

bus = EventBus()
bus.subscribe(DeathEvent, on_death)
await bus.publish(DeathEvent(entity_id=123))
```

## Runner
`ecs_agent.core.runner`

The `Runner` automates the execution of the `World`'s processing cycle.

- `async run(self, world: World, max_ticks: int | None = 100, start_tick: int = 0) -> None`: Runs a loop that calls `world.process()`. Pass `max_ticks=None` for infinite execution. The loop runs until a `TerminalComponent` is found.
- `save_checkpoint(self, world: World, entity_id: EntityId) -> None`: Creates a state snapshot for the given entity.
- `load_checkpoint(self, world: World, entity_id: EntityId) -> None`: Restores the entity state from the last snapshot.

The loop stops if an entity with a `TerminalComponent` is found or if `max_ticks` is reached. If the loop hits the tick limit, it creates a new entity with a `TerminalComponent(reason="max_ticks")`.

### Usage Example
```python
from ecs_agent.core import World, Runner

world = World()
# ... setup entities and systems ...

runner = Runner()
await runner.run(world, max_ticks=None)  # Run until TerminalComponent found
```
