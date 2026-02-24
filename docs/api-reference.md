# API Reference

Detailed API documentation for the `ecs_agent` package.

## ecs_agent (top-level exports)

```python
__version__: str = "0.1.0"
```

The following types and classes are re-exported for convenience:

- `Message`, `CompletionResult`, `ToolSchema`, `EntityId`, `StreamDelta`, `RetryConfig` from `ecs_agent.types`
- `RetryProvider` from `ecs_agent.providers.retry_provider`
- `WorldSerializer` from `ecs_agent.serialization`
- `configure_logging`, `get_logger` from `ecs_agent.logging`

---

## ecs_agent.types

### Base Types

```python
EntityId = NewType("EntityId", int)
```

### Data Classes

```python
@dataclass(slots=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]

@dataclass(slots=True)
class Message:
    role: str
    content: str
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None

@dataclass(slots=True)
class ToolSchema:
    name: str
    description: str
    parameters: dict[str, Any]

@dataclass(slots=True)
class Usage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

@dataclass(slots=True)
class CompletionResult:
    message: Message
    usage: Usage | None = None

@dataclass(slots=True)
class StreamDelta:
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    finish_reason: str | None = None
    usage: Usage | None = None

@dataclass(slots=True)
class RetryConfig:
    max_attempts: int = 3
    multiplier: float = 1.0
    min_wait: float = 4.0
    max_wait: float = 60.0
    retry_status_codes: tuple[int, ...] = (429, 500, 502, 503, 504)
```

### Event Classes

```python
@dataclass(slots=True)
class ConversationTruncatedEvent:
    entity_id: EntityId
    removed_count: int

@dataclass(slots=True)
class ErrorOccurredEvent:
    entity_id: EntityId
    error: str
    system_name: str

@dataclass(slots=True)
class MessageDeliveredEvent:
    from_entity: EntityId
    to_entity: EntityId
    message: Message

@dataclass(slots=True)
class PlanStepCompletedEvent:
    entity_id: EntityId
    step_index: int
    step_description: str

@dataclass(slots=True)
class PlanRevisedEvent:
    entity_id: EntityId
    old_steps: list[str]
    new_steps: list[str]
```

---

## ecs_agent.core

### World

```python
class World:
    @property
    def event_bus(self) -> EventBus: ...
    def create_entity(self) -> EntityId: ...
    def add_component(self, entity_id: EntityId, component: Any) -> None: ...
    def get_component(self, entity_id: EntityId, component_type: type[T]) -> T: ...
    def remove_component(self, entity_id: EntityId, component_type: type) -> None: ...
    def has_component(self, entity_id: EntityId, component_type: type) -> bool: ...
    def delete_entity(self, entity_id: EntityId) -> None: ...
    def register_system(self, system: System, priority: int = 0) -> None: ...
    async def process(self) -> None: ...
    def query(self, *component_types: type) -> Query: ...
```

### Runner

```python
class Runner:
    async def run(self, world: World, max_ticks: int = 100) -> None: ...
```

### EventBus

```python
class EventBus:
    def subscribe(self, event_type: type[T], callback: Callable[[T], None]) -> None: ...
    def unsubscribe(self, event_type: type[T], callback: Callable[[T], None]) -> None: ...
    def publish(self, event: Any) -> None: ...
    def clear(self) -> None: ...
```

### EntityIdGenerator

```python
class EntityIdGenerator:
    def next(self) -> EntityId: ...
```

### Query

```python
class Query:
    def get(self, *component_types: type) -> list[tuple[EntityId, tuple[Any, ...]]]: ...
```

---

## ecs_agent.components

All components are implemented as `@dataclass(slots=True)`.

 `LLMComponent(provider: LLMProvider, model: str, system_prompt: str = "")`
 `ConversationComponent(messages: list[Message], max_messages: int = 100)`
 `KVStoreComponent(store: dict[str, Any])`
 `ToolRegistryComponent(tools: dict[str, ToolSchema], handlers: dict[str, Callable[..., Awaitable[str]]])`
 `PendingToolCallsComponent(tool_calls: list[ToolCall])`
 `ToolResultsComponent(results: dict[str, str])`
 `PlanComponent(steps: list[str], current_step: int = 0, completed: bool = False)`
 `CollaborationComponent(peers: list[EntityId], inbox: list[tuple[EntityId, Message]])`
 `OwnerComponent(owner_id: EntityId)`
 `ErrorComponent(error: str, system_name: str, timestamp: float)`
 `TerminalComponent(reason: str)`
 `SystemPromptComponent(content: str)`

---

## ecs_agent.systems

### ReasoningSystem

```python
class ReasoningSystem(priority: int = 0):
    async def process(self, world: World) -> None: ...
```

### MemorySystem

```python
class MemorySystem:
    async def process(self, world: World) -> None: ...
```

### PlanningSystem

```python
class PlanningSystem(priority: int = 0):
    async def process(self, world: World) -> None: ...
```

### ToolExecutionSystem

```python
class ToolExecutionSystem(priority: int = 0):
    async def process(self, world: World) -> None: ...
```

### CollaborationSystem

```python
class CollaborationSystem(priority: int = 0):
    async def process(self, world: World) -> None: ...
```

### ErrorHandlingSystem

```python
class ErrorHandlingSystem(priority: int = 99):
    async def process(self, world: World) -> None: ...
```

### ReplanningSystem

```python
class ReplanningSystem(priority: int = 7):
    async def process(self, world: World) -> None: ...
```

---

## ecs_agent.providers

### LLMProvider (Protocol)

```python
class LLMProvider(Protocol):
    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        stream: bool = False,
        response_format: dict[str, Any] | None = None,
    ) -> CompletionResult | AsyncIterator[StreamDelta]: ...
```

### OpenAIProvider

```python
class OpenAIProvider:
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o-mini",
        connect_timeout: float = 10.0,
        read_timeout: float = 120.0,
        write_timeout: float = 10.0,
        pool_timeout: float = 10.0,
    ): ...
```

### FakeProvider

```python
class FakeProvider:
    def __init__(self, responses: list[CompletionResult]): ...
```

### RetryProvider

```python
class RetryProvider:
    def __init__(
        self,
        provider: LLMProvider,
        retry_config: RetryConfig | None = None,
    ): ...
```

---

## ecs_agent.providers.openai_provider

```python
def pydantic_to_response_format(model: type) -> dict[str, Any]: ...
```

---

## ecs_agent.serialization

### WorldSerializer

```python
class WorldSerializer:
    @staticmethod
    def to_dict(world: World) -> dict[str, Any]: ...
    
    @staticmethod
    def from_dict(
        data: dict[str, Any],
        providers: dict[str, LLMProvider],
        tool_handlers: dict[str, Callable],
    ) -> World: ...
    
    @staticmethod
    def save(world: World, path: str | Path) -> None: ...
    
    @staticmethod
    def load(
        path: str | Path,
        providers: dict[str, LLMProvider],
        tool_handlers: dict[str, Callable],
    ) -> World: ...
```

---

## ecs_agent.logging

```python
def configure_logging(json_output: bool = False, level: str = "INFO") -> None: ...
def get_logger(name: str) -> BoundLogger: ...
```
