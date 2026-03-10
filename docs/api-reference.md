# API Reference

Detailed API documentation for the `ecs_agent` package.

## ecs_agent (top-level exports)

```python
__version__: str = "0.1.0"
```

The following types and classes are re-exported for convenience:

- `Message`, `CompletionResult`, `ToolSchema`, `EntityId`, `StreamDelta`, `RetryConfig`, `ApprovalPolicy`, `ToolTimeoutError`, `MessageBusEnvelope` from `ecs_agent.types`
- `RetryProvider` from `ecs_agent.providers.retry_provider`
- `WorldSerializer` from `ecs_agent.serialization`
- `configure_logging`, `get_logger` from `ecs_agent.logging`
- `StreamingComponent`, `CheckpointComponent`, `CompactionConfigComponent`, `ConversationArchiveComponent`, `RunnerStateComponent`, `UserInputComponent` from `ecs_agent.components`
- `ClaudeProvider` from `ecs_agent.providers.claude_provider`
- `LiteLLMProvider` from `ecs_agent.providers.litellm_provider`
- `OpenAIEmbeddingProvider`, `FakeEmbeddingProvider` from `ecs_agent.providers`
- `MessageBusSystem`, `RAGSystem`, `TreeSearchSystem`, `ToolApprovalSystem`, `CheckpointSystem`, `CompactionSystem`, `UserInputSystem`, `SubagentSystem` from `ecs_agent.systems`
- `StreamStartEvent`, `StreamDeltaEvent`, `StreamEndEvent`, `CheckpointCreatedEvent`, `CheckpointRestoredEvent`, `CompactionCompleteEvent`, `ToolApprovalRequestedEvent`, `ToolApprovedEvent`, `ToolDeniedEvent`, `RAGRetrievalCompletedEvent`, `UserInputRequestedEvent`, `MCTSNodeScoredEvent`, `MessageBusPublishedEvent`, `MessageBusDeliveredEvent`, `MessageBusResponseEvent`, `MessageBusTimeoutEvent` from `ecs_agent.types`
- `scan_module`, `sandboxed_execute`, `tool` from `ecs_agent.tools`
- `TaskStatus`, `TaskComponent`, `ScratchbookRef`, `ScratchbookRefComponent`, `ScratchbookIndexComponent` from `ecs_agent.types` and `ecs_agent.components`
- `SubagentConfig`, `SubagentLifecycleStatus`, `SubagentSessionRecord`, `InheritancePolicy` from `ecs_agent.types`
- `ScratchbookService`, `ScratchbookIndexer`, `ToolResultsSink` from `ecs_agent.scratchbook`
- `TaskExecutor`, `StateMachine`, `TaskPersistenceService`, `ContextResolver`, `DependencyAnalyzer`, `WavePlanner`, `TaskFetchingUnit` from `ecs_agent.task`
- `AgentSpec`, `validate_agent_spec`, `discover_agent_sources`, `load_json_agents`, `load_markdown_agent`, `resolve_agent_specs`, `compile_agent_specs`, `resolve_prompt_file` from `ecs_agent.dsl`
- `TaskCreatedEvent`, `TaskStatusChangedEvent`, `TaskBlockedEvent`, `TaskCompletedEvent`, `TaskFailedEvent` from `ecs_agent.types`


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
class MessageBusEnvelope:
    id: str
    source: EntityId
    type: str
    data: Message
    specversion: str = "1.0"
    topic: str | None = None
    reply_to: str | None = None
    correlation_id: str | None = None
    traceparent: str | None = None
```
```python
@dataclass(slots=True)
class InheritancePolicy:
    enabled: bool = True
    inherit_system_prompt: bool = True
    inherit_tools: list[str] = field(default_factory=list)
    inherit_permissions: bool = False
    allow_delegate_tool: bool = True
    tool_conflict_policy: str = "skip"
    missing_skill_policy: str = "warn"

@dataclass(slots=True)
class SubagentConfig:
    name: str
    provider: Any
    model: str
    system_prompt: str = ""
    skills: list[str] = field(default_factory=list)
    max_ticks: int = 10
    inheritance_policy: InheritancePolicy = field(default_factory=InheritancePolicy)

@dataclass(slots=True)
class SubagentSessionRecord:
    session_id: str
    category: str
    prompt: str
    parent_entity_id: EntityId
    created_at: str
    updated_at: str
    status: SubagentLifecycleStatus = "Idle"
    timeout_seconds: float | None = None
    result_excerpt: str | None = None
    error: str | None = None
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
class MessageBusPublishedEvent:
    entity_id: EntityId
    envelope: MessageBusEnvelope

@dataclass(slots=True)
class MessageBusDeliveredEvent:
    entity_id: EntityId
    envelope: MessageBusEnvelope

@dataclass(slots=True)
class MessageBusResponseEvent:
    entity_id: EntityId
    envelope: MessageBusEnvelope

@dataclass(slots=True)
class MessageBusTimeoutEvent:
    entity_id: EntityId
    correlation_id: str
```
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

### `register_entity(entity_id: EntityId, name: str, tags: set[str] | None = None) -> None`

Register entity with unique name and optional tags.

**Parameters:**
- `entity_id` — Entity to register
- `name` — Unique name for entity lookup
- `tags` — Optional set of tags for entity grouping

**Raises:**
- `ValueError` — If name already registered

**Example:**
```python
agent = world.create_entity()
world.register_entity(agent, "coordinator", tags={"manager", "primary"})
```

### `resolve_entity(name: str) -> EntityId | None`

Look up entity by registered name.

**Parameters:**
- `name` — Registered entity name

**Returns:**
EntityId if found, None otherwise

**Example:**
```python
coordinator_id = world.resolve_entity("coordinator")
if coordinator_id:
    print(f"Found entity: {coordinator_id}")
```

### `list_entities_by_tag(tag: str) -> list[EntityId]`

Find all entities with given tag.

**Parameters:**
- `tag` — Tag string to search

**Returns:**
List of entity IDs with tag (empty list if tag not found)

**Example:**
```python
workers = world.list_entities_by_tag("worker")
for worker_id in workers:
    print(f"Worker: {worker_id}")
```

### `unregister_entity(entity_id: EntityId) -> None`

Remove entity from registry and tag indexes (no-op if not found).

**Parameters:**
- `entity_id` — Entity to unregister

**Example:**
```python
world.unregister_entity(agent)  # Remove from registry
```

### `remove_system(handle: SystemHandle) -> None`

Queue system for removal at next tick boundary.

**Parameters:**
- `handle` — System handle from register_system

**Example:**
```python
handle = world.register_system(MySystem(), priority=0)
world.remove_system(handle)  # Queued, applied at pre-tick
```

### `replace_system(handle: SystemHandle, system: System, priority: int | None = None) -> None`

Queue system replacement at next tick boundary.

**Parameters:**
- `handle` — System handle to replace
- `system` — New system instance
- `priority` — Optional new priority (defaults to original)

**Example:**
```python
world.replace_system(handle, NewReasoningSystem(), priority=5)
```

### `apply_pending_system_operations() -> None`

Apply queued system operations (called automatically by Runner at pre-tick).

**Example:**
```python
# Manual application (normally not needed)
world.apply_pending_system_operations()
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
 `MessageBusConfigComponent(max_queue_size: int = 1000, publish_timeout: float = 2.0, request_timeout: float = 30.0, cleanup_interval: float = 10.0)`
 `MessageBusSubscriptionComponent(subscriptions: set[str], inbox: deque[MessageBusEnvelope])`
 `MessageBusConversationComponent(active_requests: dict[str, float])`
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

### MessageBusSystem

```python
class MessageBusSystem(priority: int = 5):
    async def process(self, world: World) -> None: ...
    def subscribe(self, world: World, entity_id: EntityId, topic: str) -> None: ...
    def unsubscribe(self, world: World, entity_id: EntityId, topic: str) -> None: ...
    async def publish(self, world: World, entity_id: EntityId, topic: str, message: Message) -> None: ...
    async def request(self, world: World, entity_id: EntityId, topic: str, message: Message, timeout: float | None = None) -> MessageBusEnvelope: ...
    async def respond(self, world: World, entity_id: EntityId, request_envelope: MessageBusEnvelope, message: Message) -> None: ...
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

### SubagentSystem

```python
class SubagentSystem(priority: int = -1, default_timeout: float | None = None):
    async def process(self, world: World) -> None: ...
    def install_subagent_tool(self, world: World, entity_id: EntityId, tool_name: str = "subagent", override: bool = False) -> None: ...
    def install_delegate_tool(self, world: World, entity_id: EntityId, tool_name: str = "delegate", override: bool = False) -> None: ...
    def install_subagent_control_tools(self, world: World, entity_id: EntityId) -> None: ...
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

---

## ecs_agent.conversation_tree

### `revert_to_message(tree: ConversationTreeComponent, target_message_id: str) -> str`

Move active branch pointer to target message (non-destructive).

**Parameters:**
- `tree` — Conversation tree component
- `target_message_id` — Message ID to revert to

**Returns:**
Target message ID (for verification)

**Raises:**
- `ValueError` — If no active branch (current_branch_id is None)
- `KeyError` — If target message not found in tree.messages

**Example:**
```python
from ecs_agent.conversation_tree import revert_to_message

# Revert to earlier message
revert_to_message(tree, msg2.id)  # Move active branch leaf to msg2
```
---

## ecs_agent.scratchbook

### ScratchbookService
```python
class ScratchbookService:
    def __init__(self, root: Path | str): ...
    def write_artifact(self, artifact_id: str, category: str, data: dict[str, Any]) -> None: ...
    def read_artifact(self, artifact_id: str, category: str) -> dict[str, Any] | None: ...
    def append_log(self, log_name: str, category: str, line: str) -> None: ...
    def write_index(self, index_name: str, category: str, data: dict[str, Any]) -> None: ...
    def read_index(self, index_name: str, category: str) -> dict[str, Any] | None: ...
    def list_artifacts(self, category: str) -> list[str]: ...
    def delete_artifact(self, artifact_id: str, category: str) -> None: ...
```

### ScratchbookIndexer
```python
class ScratchbookIndexer:
    def __init__(self, root: Path | str): ...
    def add_entry(self, stable_id: str, artifact_id: str, artifact_type: str, category: str, content_hash: str) -> None: ...
    def lookup_by_task_id(self, task_id: str) -> list[dict[str, Any]]: ...
    def lookup_by_artifact_type(self, artifact_type: str) -> list[dict[str, Any]]: ...
    def lookup_by_category(self, category: str) -> list[dict[str, Any]]: ...
```

---

## ecs_agent.task

### TaskExecutor
```python
class TaskExecutor(priority: int = 0):
    async def execute_dispatch_request(self, world: World, entity_id: EntityId, request: DispatchRequest) -> ExecutionResult: ...
```

### TaskPersistenceService
```python
class TaskPersistenceService:
    def __init__(self, scratchbook: ScratchbookService): ...
    def persist_task_snapshot(self, task_id: str, task_component: TaskComponent) -> ScratchbookRef: ...
    def read_task_snapshot(self, task_id: str) -> dict[str, Any] | None: ...
    def append_task_event(self, task_id: str, event: TaskEvent) -> None: ...
    def read_task_events(self, task_id: str) -> list[dict[str, Any]]: ...
```

### ContextResolver
```python
class ContextResolver:
    def __init__(self, service: ScratchbookService): ...
    def resolve_context(self, task: TaskComponent, running_task_ids: set[str] | None = None) -> ResolvedContext | ContextResolutionError: ...
```

---

## ecs_agent.dsl

Agent DSL configuration and compilation utilities.

### AgentSpec

```python
@dataclass(slots=True)
class AgentSpec:
    name: str
    mode: str
    model: str
    prompt: str | dict[str, str]
    tools: dict[str, bool] | None = None
    metadata: dict[str, Any] | None = None
```

Normalized agent specification. All loaders produce this canonical representation.

**Fields:**
- `name` — Agent identifier (derived from JSON key or Markdown filename)
- `mode` — Execution mode (`"primary"` creates runnable entity, `"library"` for config-only templates)
- `model` — LLM model identifier
- `prompt` — System prompt (string or `{"file": "path/to/prompt.txt"}`)
- `tools` — Optional tool permission mapping (`{"tool_name": true/false}`)
- `metadata` — Optional arbitrary metadata dictionary

### validate_agent_spec

```python
def validate_agent_spec(data: dict[str, Any], *, source_name: str = "") -> AgentSpec:
    ...
```

Validate raw dictionary and convert to AgentSpec. Fail-fast on invalid/unknown fields.

**Parameters:**
- `data` — Raw agent configuration dict
- `source_name` — Optional source identifier for error messages (e.g., filename)

**Returns:**
Validated AgentSpec instance

**Raises:**
- `ValueError` — On validation failure (missing required fields, invalid types, unknown fields)

**Example:**
```python
from ecs_agent.dsl import validate_agent_spec

spec = validate_agent_spec({
    "name": "assistant",
    "mode": "primary",
    "model": "gpt-4",
    "prompt": "You are a helpful assistant."
})
```

### discover_agent_sources

```python
def discover_agent_sources(directory: Path | str) -> list[Path]:
    ...
```

Find all `*.json` and `*.md` agent configuration files in a directory.

**Parameters:**
- `directory` — Directory to search (non-recursive)

**Returns:**
List of Path objects sorted deterministically (JSON first alphabetically, then Markdown alphabetically)

**Example:**
```python
from ecs_agent.dsl import discover_agent_sources

sources = discover_agent_sources("./agents")
# [PosixPath('agents/primary.json'), PosixPath('agents/assistant.md')]
```

### load_json_agents

```python
def load_json_agents(path: Path | str) -> list[AgentSpec]:
    ...
```

Load agent specifications from a JSON file containing `{"agents": {...}}` structure.

**Parameters:**
- `path` — Path to JSON file

**Returns:**
List of AgentSpec (one per agent in the JSON)

**Raises:**
- `ValueError` — Invalid JSON structure or missing `"agents"` key
- File I/O errors if path doesn't exist

**Example:**
```python
from ecs_agent.dsl import load_json_agents

specs = load_json_agents("agents.json")
for spec in specs:
    print(f"{spec.name}: {spec.model}")
```

### load_markdown_agent

```python
def load_markdown_agent(path: Path | str) -> AgentSpec:
    ...
```

Load single agent from Markdown file. Filename becomes agent name (without `.md`). YAML frontmatter provides configuration, body text becomes prompt.

**Parameters:**
- `path` — Path to `.md` file

**Returns:**
Single AgentSpec instance

**Raises:**
- `ValueError` — Invalid YAML frontmatter, missing required fields, or validation errors
- File I/O errors if path doesn't exist

**Example:**
```python
from ecs_agent.dsl import load_markdown_agent

spec = load_markdown_agent("assistant.md")
# spec.name == "assistant" (derived from filename)
```

### resolve_agent_specs

```python
def resolve_agent_specs(specs: list[AgentSpec]) -> list[AgentSpec]:
    ...
```

Resolve name conflicts using **last-one-wins** policy. Later specs with duplicate names override earlier ones.

**Parameters:**
- `specs` — List of AgentSpec (possibly with duplicate names)

**Returns:**
Deduplicated list (preserving last occurrence of each name)

**Example:**
```python
from ecs_agent.dsl import resolve_agent_specs

specs = [
    AgentSpec(name="bot", mode="primary", model="gpt-3.5", prompt="v1"),
    AgentSpec(name="bot", mode="primary", model="gpt-4", prompt="v2"),
]
resolved = resolve_agent_specs(specs)
# resolved[0].model == "gpt-4" (last definition wins)
```

### compile_agent_specs

```python
def compile_agent_specs(
    specs: list[AgentSpec],
    factory: Callable[[AgentSpec], tuple[EntityId, LLMProvider]]
) -> World:
    ...
```

Compile agent specifications into an ECS World. Creates entities for agents with `mode="primary"` using the provided factory function.

**Parameters:**
- `specs` — List of AgentSpec to compile
- `factory` — Callback `(spec) -> (entity_id, provider)` that creates entity and instantiates LLM provider

**Returns:**
World instance with compiled entities and components

**Raises:**
- `ValueError` — No primary mode agent found (at least one `mode="primary"` required)

**Example:**
```python
from ecs_agent.dsl import compile_agent_specs
from ecs_agent.core import World
from ecs_agent.providers import OpenAIProvider

def my_factory(spec: AgentSpec):
    world_temp = World()
    entity = world_temp.create_entity()
    provider = OpenAIProvider(api_key="sk-xxx", model=spec.model)
    return entity, provider

world = compile_agent_specs(specs, my_factory)
```

### resolve_prompt_file

```python
def resolve_prompt_file(prompt_ref: dict[str, str], agent_source_path: Path) -> str:
    ...
```

Resolve `{"file": "path/to/prompt.txt"}` reference to actual file content. Security: rejects absolute paths and parent directory traversal.

**Parameters:**
- `prompt_ref` — Dictionary with `{"file": "relative/path.txt"}`
- `agent_source_path` — Path to agent config file (used as base for relative resolution)

**Returns:**
File content as string

**Raises:**
- `ValueError` — Absolute path, parent traversal (`..`), or file not found

**Example:**
```python
from pathlib import Path
from ecs_agent.dsl import resolve_prompt_file

content = resolve_prompt_file(
    {"file": "prompts/assistant.txt"},
    Path("agents/config.json")
)
# Reads from agents/prompts/assistant.txt
```
