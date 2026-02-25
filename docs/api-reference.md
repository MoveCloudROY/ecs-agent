# API Reference

Detailed API documentation for the `ecs_agent` package.

## ecs_agent (top-level exports)

```python
__version__: str = "0.1.0"
```

The following types and classes are re-exported for convenience:

- `Message`, `CompletionResult`, `ToolSchema`, `EntityId`, `StreamDelta`, `RetryConfig`, `ApprovalPolicy`, `ToolTimeoutError` from `ecs_agent.types`
- `RetryProvider` from `ecs_agent.providers.retry_provider`
- `WorldSerializer` from `ecs_agent.serialization`
- `configure_logging`, `get_logger` from `ecs_agent.logging`
- `StreamingComponent`, `CheckpointComponent`, `CompactionConfigComponent`, `ConversationArchiveComponent`, `RunnerStateComponent`, `UserInputComponent` from `ecs_agent.components`
- `ClaudeProvider` from `ecs_agent.providers.claude_provider`
- `LiteLLMProvider` from `ecs_agent.providers.litellm_provider`
- `OpenAIEmbeddingProvider`, `FakeEmbeddingProvider` from `ecs_agent.providers`
- `RAGSystem`, `TreeSearchSystem`, `ToolApprovalSystem`, `CheckpointSystem`, `CompactionSystem`, `UserInputSystem` from `ecs_agent.systems`
- `StreamStartEvent`, `StreamDeltaEvent`, `StreamEndEvent`, `CheckpointCreatedEvent`, `CheckpointRestoredEvent`, `CompactionCompleteEvent`, `ToolApprovalRequestedEvent`, `ToolApprovedEvent`, `ToolDeniedEvent`, `RAGRetrievalCompletedEvent`, `UserInputRequestedEvent`, `MCTSNodeScoredEvent` from `ecs_agent.types`
- `scan_module`, `sandboxed_execute`, `tool` from `ecs_agent.tools`

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
```python
class ApprovalPolicy(Enum):
    ALWAYS_APPROVE = "always_approve"
    ALWAYS_DENY = "always_deny"
    REQUIRE_APPROVAL = "require_approval"

class ToolTimeoutError(Exception): ...
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

@dataclass(slots=True)
class ToolApprovalRequestedEvent:
    entity_id: EntityId
    tool_call: ToolCall
    future: asyncio.Future[bool]

@dataclass(slots=True)
class ToolApprovedEvent:
    entity_id: EntityId
    tool_call_id: str

@dataclass(slots=True)
class ToolDeniedEvent:
    entity_id: EntityId
    tool_call_id: str

@dataclass(slots=True)
class MCTSNodeScoredEvent:
    entity_id: EntityId
    node_path: list[str]
    score: float

@dataclass(slots=True)
class StreamStartEvent:
    entity_id: EntityId

@dataclass(slots=True)
class StreamDeltaEvent:
    entity_id: EntityId
    delta: StreamDelta

@dataclass(slots=True)
class StreamEndEvent:
    entity_id: EntityId
    result: CompletionResult

@dataclass(slots=True)
class CheckpointCreatedEvent:
    entity_id: EntityId
    snapshot_index: int

@dataclass(slots=True)
class CheckpointRestoredEvent:
    entity_id: EntityId
    snapshot_index: int

@dataclass(slots=True)
class CompactionCompleteEvent:
    entity_id: EntityId
    removed_count: int
    summary_length: int

@dataclass(slots=True)
class RAGRetrievalCompletedEvent:
    entity_id: EntityId
    query: str
    num_docs: int

@dataclass(slots=True)
class UserInputRequestedEvent:
    entity_id: EntityId
    prompt: str

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
    async def run(self, world: World, max_ticks: int | None = 100, start_tick: int = 0) -> None: ...
    def save_checkpoint(self, world: World, path: str | Path) -> None: ...
    @classmethod
    def load_checkpoint(cls, path: str | Path, providers: dict[str, LLMProvider], tool_handlers: dict[str, Callable]) -> tuple[World, int]: ...
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
 `StreamingComponent(enabled: bool = False)`
 `CheckpointComponent(snapshots: list[dict[str, Any]] = [], max_snapshots: int = 10)`
 `CompactionConfigComponent(threshold_tokens: int, summary_model: str)`
 `ConversationArchiveComponent(archived_summaries: list[str] = [])`
 `RunnerStateComponent(current_tick: int, is_paused: bool = False, checkpoint_path: str | None = None)`
 `UserInputComponent(prompt: str = "", future: asyncio.Future[str] | None = None, timeout: float | None = None, result: str | None = None)`
 `ToolApprovalComponent(policy: ApprovalPolicy, timeout: float | None = 30.0, approved_calls: list[str] = [], denied_calls: list[str] = [])`
 `SandboxConfigComponent(timeout: float = 30.0, max_output_size: int = 10000)`
 `PlanSearchComponent(max_depth: int = 5, max_branching: int = 3, exploration_weight: float = 1.414, best_plan: list[str] = [], search_active: bool = False)`
 `RAGTriggerComponent(query: str = "", top_k: int = 5, retrieved_docs: list[str] = [])`
 `EmbeddingComponent(provider: EmbeddingProvider, dimension: int = 0)`
 `VectorStoreComponent(store: VectorStore)`

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
### ToolApprovalSystem

```python
class ToolApprovalSystem(priority: int = -5):
    async def process(self, world: World) -> None: ...
```

### TreeSearchSystem

```python
class TreeSearchSystem(priority: int = 0):
    async def process(self, world: World) -> None: ...
```

### RAGSystem

```python
class RAGSystem(priority: int = -10):
    async def process(self, world: World) -> None: ...
```

### CheckpointSystem
```python
class CheckpointSystem:
    async def process(self, world: World) -> None: ...
    @staticmethod
    def undo(world: World, providers: dict[str, LLMProvider], tool_handlers: dict[str, Callable]) -> None: ...
```

### CompactionSystem

```python
class CompactionSystem(bisect_ratio: float = 0.5):
    async def process(self, world: World) -> None: ...
```

### UserInputSystem

```python
class UserInputSystem(priority: int = -10):
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
### ClaudeProvider

```python
class ClaudeProvider:
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.anthropic.com",
        model: str = "claude-3-5-haiku-latest",
        max_tokens: int = 4096,
        connect_timeout: float = 10.0,
        read_timeout: float = 120.0,
        write_timeout: float = 10.0,
        pool_timeout: float = 10.0,
    ): ...
```

### LiteLLMProvider

```python
class LiteLLMProvider:
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
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
---

## ecs_agent.tools

```python
def scan_module(module: ModuleType) -> tuple[dict[str, ToolSchema], dict[str, Callable[..., Awaitable[str]]]]: ...
async def sandboxed_execute(func: Callable[..., Awaitable[str]], args: dict[str, Any], timeout: float = 30.0, max_output_size: int = 10000) -> str: ...
def tool(name: str, description: str, parameters: dict[str, Any]) -> Callable: ...
```
