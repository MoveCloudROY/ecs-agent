# Component Reference

This page documents all 26 core component dataclasses used in the ECS agent architecture. All components use `@dataclass(slots=True)` for memory efficiency and can be imported from `ecs_agent.components`.

## Core Agent Components

### LLMComponent
Links an agent to an LLM model implementation for reasoning and planning.

| Name | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `model` | `LLMModel` | (none) | The live LLM model instance |
| `system_prompt` | `str` | `""` | Optional system prompt override |
| `pending_model` | `LLMModel | None` | `None` | Queued model switch (applied at next request start) |

**Used by:** `ReasoningSystem`, `PlanningSystem`, `ReplanningSystem`

**Usage:**
```python
from ecs_agent.components import LLMComponent
from ecs_agent.providers import Model
from ecs_agent.providers.config import ApiFormat

llm = LLMComponent(
    model=Model(
        "gpt-4o",
        base_url="https://api.openai.com/v1",
        api_key="...",
        api_format=ApiFormat.OPENAI_CHAT_COMPLETIONS,
    ),
    system_prompt="You are a helpful assistant."
)
world.add_component(agent, llm)

# Queue model switch (takes effect at next LLM request)
llm.pending_model = Model(
    "gpt-3.5-turbo",
    base_url="https://api.openai.com/v1",
    api_key="...",
    api_format=ApiFormat.OPENAI_CHAT_COMPLETIONS,
)
```

### ConversationComponent
Maintains the full history of messages for an agent's conversation.

| Name | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `messages` | `list[Message]` | (none) | History of conversation messages |
| `max_messages` | `int` | `100` | Maximum number of messages to retain |

**Used by:** `ReasoningSystem`, `PlanningSystem`, `MessageBusSystem`, `ToolExecutionSystem`, `ReplanningSystem`

**Usage:**
```python
from ecs_agent.components import ConversationComponent
world.add_component(agent, ConversationComponent(messages=[]))
```

### SystemPromptComponent
Legacy component that stores the system prompt template and rendered content. For new agents, prefer `SystemPromptConfigSpec` with `SystemPromptRenderSystem`.

| Name | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `template` | `str` | `""` | Base system prompt template with placeholders |
| `sections` | `list[PromptSectionSpec]` | `[]` | List of prompt sections to inject into placeholders |
| `content` | `str` | `""` | Rendered system prompt output |

**Used by:** `ReasoningSystem`, `PlanningSystem`, `ReplanningSystem` (legacy path)

---

### SystemPromptConfigSpec
New-style prompt spec using `${name}` placeholder templates. Processed by `SystemPromptRenderSystem`.

| Name | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `template_source` | `PromptTemplateSource` | required | Inline string or file path for the template |
| `placeholder_specs` | `dict[str, PlaceholderSpec]` | `{}` | Per-placeholder resolver specs (static, callable, or file) |

**Produces:** `RenderedSystemPromptComponent` (via `SystemPromptRenderSystem`)

**Usage:**
```python
from ecs_agent.prompts.contracts import SystemPromptConfigSpec, PromptTemplateSource

world.add_component(entity, SystemPromptConfigSpec(
    template_source=PromptTemplateSource(
        inline="You are a helpful assistant. Tools: ${_installed_tools}"
    )
))
```

---

### RenderedSystemPromptComponent
cached/frozen rendered system prompt produced by `SystemPromptRenderSystem` on the first successful render-system pass. Reused on subsequent ticks; consumed by `ReasoningSystem`, `PlanningSystem`, `ReplanningSystem`.

| Name | Type | Description |
| :--- | :--- | :--- |
| `text` | `str` | Fully rendered system prompt with all `${name}` placeholders resolved |

---

### RenderedUserPromptComponent
Output component written by `UserPromptNormalizationSystem` containing the normalized user message for the current tick. Stored conversation history is never mutated.

| Name | Type | Description |
| :--- | :--- | :--- |
| `text` | `str` | Normalized user message with trigger injections prepended |
### `EntityRegistryComponent`

Internal registry for named entity resolution and tagging. Usually managed by World via `register_entity()`, not directly attached to entities.

**Fields:**
- `entity_id: EntityId` — Entity being registered
- `name: str` — Unique name for lookup
- `tags: set[str]` — Tag set (default: empty set)
- `metadata: dict[str, Any]` — Additional metadata (default: empty dict)

**Example:**
```python
from ecs_agent.components import EntityRegistryComponent

# Usually created via world.register_entity(), not directly
agent = world.create_entity()
world.register_entity(agent, "coordinator", tags={"manager", "primary"})
```

## Tool Components

### ToolRegistryComponent
Stores the definitions and execution handlers for all tools available to the agent.

| Name | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `tools` | `dict[str, ToolSchema]` | (none) | Mapping of tool names to their schemas |
| `handlers` | `dict[str, Callable[..., Awaitable[str]]]` | (none) | Async handlers for tool execution |

**Used by:** `ReasoningSystem`, `PlanningSystem`, `ToolExecutionSystem`

**Usage:**
```python
from ecs_agent.components import ToolRegistryComponent
world.add_component(agent, ToolRegistryComponent(tools=tools, handlers=handlers))
```

### PendingToolCallsComponent
Captures tool calls generated by reasoning or planning that are waiting for execution.

| Name | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `tool_calls` | `list[ToolCall]` | (none) | List of requested tool calls |

**Added by:** `ReasoningSystem`, `PlanningSystem`  
**Consumed by:** `ToolExecutionSystem` (removes the component after processing)

**Usage:**
```python
from ecs_agent.components import PendingToolCallsComponent
world.add_component(agent, PendingToolCallsComponent(tool_calls=requested_calls))
```

### ToolResultsComponent
Holds the output results of executed tool calls before they are returned to the conversation.

| Name | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `results` | `dict[str, str]` | (none) | Mapping of call ID to result string |

**Added by:** `ToolExecutionSystem`

**Usage:**
```python
from ecs_agent.components import ToolResultsComponent
world.add_component(agent, ToolResultsComponent(results={"call_1": "Success output"}))
```

## Planning Components

### PlanComponent
Maintains a ReAct plan, tracking individual steps and completion status.

| Name | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `steps` | `list[str]` | (none) | Sequential list of plan steps |
| `current_step` | `int` | `0` | Current active step index |
| `completed` | `bool` | `False` | Completion flag for the plan |

**Used by:** `PlanningSystem`, `ReplanningSystem`

**Usage:**
```python
from ecs_agent.components import PlanComponent
world.add_component(agent, PlanComponent(steps=["Analyze", "Execute", "Verify"]))
```

## Collaboration Components

### MessageBusConfigComponent
Configures the behavior and limits for the message bus.

| Name | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `max_queue_size` | `int` | `1000` | Maximum number of buffered messages per subscriber |
| `publish_timeout` | `float` | `2.0` | Timeout in seconds for blocking publishes (backpressure) |
| `request_timeout` | `float` | `30.0` | Default timeout for request-response conversations |
| `cleanup_interval` | `float` | `10.0` | Seconds between cleanup of stale request states |

**Used by:** `MessageBusSystem`

**Usage:**
```python
from ecs_agent.components import MessageBusConfigComponent
world.add_component(agent, MessageBusConfigComponent(request_timeout=60.0))
```

### MessageBusSubscriptionComponent
Tracks an entity's topic subscriptions and message queues.

| Name | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `subscriptions` | `set[str]` | `set()` | Set of topics the entity is subscribed to |
| `inbox` | `deque[MessageBusEnvelope]` | `deque()` | Incoming message queue for the subscriber |

**Used by:** `MessageBusSystem`

**Usage:**
```python
from ecs_agent.components import MessageBusSubscriptionComponent
world.add_component(agent, MessageBusSubscriptionComponent(subscriptions={"agent.updates", "system.alerts"}))
```

### MessageBusConversationComponent
Manages active request-response conversations for an entity.

| Name | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `active_requests` | `dict[str, float]` | `{}` | Mapping of correlation IDs to request expiration timestamps |

**Used by:** `MessageBusSystem`

**Usage:**
```python
from ecs_agent.components import MessageBusConversationComponent
world.add_component(agent, MessageBusConversationComponent())
```

## State & Lifecycle Components

### ErrorComponent
Captures diagnostic information when a system encounters an exception.

| Name | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `error` | `str` | (none) | The error message or traceback |
| `system_name` | `str` | (none) | System that triggered the error |
| `timestamp` | `float` | (none) | Unix timestamp of occurrence |

**Added by:** `ReasoningSystem`, `PlanningSystem` on failure  
**Consumed by:** `ErrorHandlingSystem` (processed and then removed)

**Usage:**
```python
import time
from ecs_agent.components import ErrorComponent
world.add_component(agent, ErrorComponent(
    error="API key expired",
    system_name="ReasoningSystem",
    timestamp=time.time()
))
```

### InterruptionComponent

Signals agent should stop gracefully with partial content preservation.

| Name | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `reason` | `InterruptionReason` | (none) | Enum: USER_REQUESTED, SYSTEM_PAUSE, ERROR, COMPLETION |
| `message` | `str` | `""` | Human-readable reason string |
| `metadata` | `dict[str, Any]` | `{}` | Structured context (default: empty dict) |
| `timestamp` | `float` | (auto) | Auto-generated via time.time() |

**Added by:** External code (user interruption), systems (error conditions)
**Consumed by:** `Runner` (detects and halts execution)

**Usage:**
```python
from ecs_agent.components import InterruptionComponent
from ecs_agent.types import InterruptionReason

world.add_component(agent, InterruptionComponent(
    reason=InterruptionReason.USER_REQUESTED,
    message="User clicked stop",
    metadata={"source": "web_ui"}
))
```

### TerminalComponent
A marker component that signifies the agent has finished its task.

| Name | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `reason` | `str` | (none) | Final completion status or reason |

**Checked by:** Runner loop to determine when to stop execution.

In interactive flows, a `TerminalComponent` can also be handled by the opt-in `TerminalCleanupSystem`, which clears selected reasons after terminal-producing systems run. By default, that helper only clears `reasoning_complete` and leaves other terminal reasons intact.

**Usage:**
```python
from ecs_agent.components import TerminalComponent
world.add_component(agent, TerminalComponent(reason="Task goal reached."))
```

### OwnerComponent
Establishes a hierarchical relationship between entities.

| Name | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `owner_id` | `EntityId` | (none) | ID of the parent/owner entity |

**Usage:**
```python
from ecs_agent.components import OwnerComponent
world.add_component(child, OwnerComponent(owner_id=root_agent_id))
```

### ChildStubComponent
Marker component placed on parent-world stub entities that track a delegated child subagent. Entities with this component are skipped by `ReasoningSystem` so the parent world never attempts LLM inference on delegation stubs.

No fields — this is a pure marker component.

**Added by:** `SubagentSystem` (on the parent-world stub entity at delegation time)
**Checked by:** `ReasoningSystem` (skips entities that have this component)

**Usage:**
```python
from ecs_agent.components import ChildStubComponent

# Normally added automatically by SubagentSystem, not manually
world.add_component(stub_entity, ChildStubComponent())
```
### KVStoreComponent
Provides a generic key-value store for custom system memory.

| Name | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `store` | `dict[str, Any]` | (none) | Backing dictionary for storage |

**Used by:** Available for custom implementations and extensions.

**Usage:**
```python
from ecs_agent.components import KVStoreComponent
world.add_component(agent, KVStoreComponent(store={"count": 5}))
```

## Advanced & Feature-Specific Components

### ToolApprovalComponent
Configures how tool calls are approved or denied before execution.

| Name | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `policy` | `ApprovalPolicy` | (none) | `ALWAYS_APPROVE`, `ALWAYS_DENY`, or `REQUIRE_APPROVAL` |
| `timeout` | `float | None` | `30.0` | Seconds to wait for human approval in `REQUIRE_APPROVAL` mode |
| `approved_calls` | `list[str]` | `[]` | History of approved tool call IDs |
| `denied_calls` | `list[str]` | `[]` | History of denied tool call IDs |
,
**Used by:** `ToolApprovalSystem`

**Usage:**
```python
from ecs_agent.components import ToolApprovalComponent
from ecs_agent.types import ApprovalPolicy

world.add_component(agent, ToolApprovalComponent(policy=ApprovalPolicy.REQUIRE_APPROVAL, timeout=60.0))
```

### SandboxConfigComponent
Defines execution limits for code-running tools or sandboxed environments.

| Name | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `timeout` | `float` | `30.0` | Maximum execution time in seconds |
| `max_output_size` | `int` | `10000` | Maximum allowed size of execution output in bytes |
**Used by:** Tools using `Sandbox` logic.

**Usage:**
```python
from ecs_agent.components import SandboxConfigComponent
world.add_component(agent, SandboxConfigComponent(timeout=10.0, max_output_size=5000))
```

### PlanSearchComponent
Configures Monte Carlo Tree Search (MCTS) for finding the optimal plan path.

| Name | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `max_depth` | `int` | `5` | Maximum search depth in the MCTS tree |
| `max_branching` | `int` | `3` | Maximum number of child nodes to expand per parent |
| `exploration_weight` | `float` | `1.414` | UCB1 exploration constant |
| `best_plan` | `list[str]` | `[]` | The best plan path found after search completes |
| `search_active` | `bool` | `False` | Internal flag indicating search is in progress |
**Used by:** `TreeSearchSystem` (mutually exclusive with `PlanComponent`)

**Usage:**
```python
from ecs_agent.components import PlanSearchComponent
world.add_component(agent, PlanSearchComponent(max_depth=10, max_branching=5))
```

### RAGTriggerComponent
Triggers a vector search and stores the retrieved document snippets.

| Name | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `query` | `str` | `""` | The search query string; cleared after retrieval |
| `top_k` | `int` | `5` | Number of documents to retrieve |
| `retrieved_docs` | `list[str]` | `[]` | Snippets of retrieved text |
,
**Used by:** `RAGSystem`

**Usage:**
```python
from ecs_agent.components import RAGTriggerComponent
world.add_component(agent, RAGTriggerComponent(query="How to use ECS?", top_k=3))
```

### EmbeddingComponent
Provides an entity with access to an embedding provider.
,
| Name | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `provider` | `EmbeddingProvider` | (none) | The embedding provider instance |
| `dimension` | `int` | `0` | Expected vector dimension |
,
**Used by:** `RAGSystem`

**Usage:**
```python
from ecs_agent.components import EmbeddingComponent
from ecs_agent.providers.fake_embedding_provider import FakeEmbeddingProvider

world.add_component(agent, EmbeddingComponent(provider=FakeEmbeddingProvider(), dimension=384))
```

### VectorStoreComponent
Links an entity to a vector store for storage and retrieval.
,
| Name | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `store` | `VectorStore` | (none) | The vector store instance |
,
**Used by:** `RAGSystem`
,
**Usage:**
```python
from ecs_agent.components import VectorStoreComponent
from ecs_agent.providers.vector_store import InMemoryVectorStore

world.add_component(agent, VectorStoreComponent(store=InMemoryVectorStore(dimension=384)))
```

### StreamingComponent
Enables system-level streaming output for an entity's LLM responses.

| Name | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `enabled` | `bool` | `False` | Whether streaming is active for this entity |

**Used by:** `ReasoningSystem` (checks this to decide streaming vs batch mode)

**Usage:**
```python
from ecs_agent.components import StreamingComponent
world.add_component(agent, StreamingComponent(enabled=True))
```

### CheckpointComponent
Stores snapshots of world state for undo/restore operations.

| Name | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `snapshots` | `list[dict[str, Any]]` | `[]` | Stack of serialized world state snapshots |
| `max_snapshots` | `int` | `10` | Maximum number of snapshots to retain |

**Used by:** `CheckpointSystem`

**Usage:**
```python
from ecs_agent.components import CheckpointComponent
world.add_component(agent, CheckpointComponent(max_snapshots=5))
```

### CompactionConfigComponent
Configures when and how conversation compaction occurs.

| Name | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `threshold_tokens` | `int` | (none) | Token count threshold triggering compaction |
| `summary_model` | `str` | (none) | Model to use for generating summaries |

**Used by:** `CompactionSystem`

**Usage:**
```python
from ecs_agent.components import CompactionConfigComponent
world.add_component(agent, CompactionConfigComponent(threshold_tokens=4000, summary_model="qwen-plus"))
```

### ConversationArchiveComponent
Stores archived conversation summaries created by compaction.

| Name | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `archived_summaries` | `list[str]` | `[]` | List of past conversation summaries |

**Used by:** `CompactionSystem`

**Usage:**
```python
from ecs_agent.components import ConversationArchiveComponent
world.add_component(agent, ConversationArchiveComponent())
```

### RunnerStateComponent
Tracks the current state of the runner execution on an entity.

| Name | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `current_tick` | `int` | (none) | Current tick number in the runner loop |
| `is_paused` | `bool` | `False` | Whether the runner is paused |
| `checkpoint_path` | `str | None` | `None` | Path to the last saved checkpoint file |

**Used by:** `Runner.save_checkpoint()`, `Runner.load_checkpoint()`

**Usage:**
```python
from ecs_agent.components import RunnerStateComponent
# Usually added automatically by Runner, but can be pre-configured
world.add_component(agent, RunnerStateComponent(current_tick=0))
```

### UserInputComponent
Enables an entity to request and receive async user input.

| Name | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `prompt` | `str` | `""` | The prompt to display to the user |
| `future` | `asyncio.Future[str] | None` | `None` | Future that resolves with user input |
| `timeout` | `float | None` | `None` | Seconds to wait; `None` for infinite wait |
| `result` | `str | None` | `None` | The received user input |

**Used by:** `UserInputSystem`

**Usage:**
```python
from ecs_agent.components import UserInputComponent
world.add_component(agent, UserInputComponent(prompt="Enter your name: ", timeout=None))
```
### ConversationTreeComponent
Tree-structured conversation with branching and linearization.

| Name | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `messages` | `dict[str, ConversationMessage]` | `{}` | All messages indexed by ID |
| `current_branch_id` | `str | None` | `None` | Active branch ID |
| `branches` | `dict[str, ConversationBranch]` | `{}` | All branches indexed by ID |

**Used by:** `conversation_tree` module, `ReasoningSystem` (auto-linearizes current branch)

**Usage:**
```python
from ecs_agent.components import ConversationTreeComponent
from ecs_agent.types import ConversationMessage, ConversationBranch

root_msg = ConversationMessage(
    id="msg_0",
    parent_message_id=None,
    role="user",
    content="Hello",
)
branch = ConversationBranch(branch_id="main", leaf_message_id="msg_0")

world.add_component(
    agent,
    ConversationTreeComponent(
        messages={"msg_0": root_msg},
        current_branch_id="main",
        branches={"main": branch},
    ),
)
```

### ResponsesAPIStateComponent
Tracks OpenAI Responses API state for multi-turn conversations.

| Name | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `previous_response_id` | `str | None` | `None` | Response ID from last API call |

**Used by:** `OpenAIModel` when configured for the Responses API

**Usage:**
```python
from ecs_agent.components import ResponsesAPIStateComponent

world.add_component(
    agent,
    ResponsesAPIStateComponent(previous_response_id=None),
)
# System automatically updates previous_response_id after each LLM call
```

### SubagentRegistryComponent
Registry of named subagent configurations and opt-in free-form delegation defaults.

| Name | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `subagents` | `dict[str, SubagentConfig]` | `{}` | Subagent configurations by name |
| `free_subagent_config` | `FreeSubagentConfig` | disabled | Defaults for dynamically created subagents when free-form mode is enabled |

**Used by:** `SubagentSystem`, `subagent` tool

**Usage:**
```python
from ecs_agent.components import SubagentRegistryComponent
from ecs_agent.types import FreeSubagentConfig, SubagentConfig

researcher = SubagentConfig(
    name="researcher",
    provider=your_provider,
    model="gpt-4o",
    system_prompt="You are a research assistant.",
    max_ticks=10,
    skills=[],
)

world.add_component(
    agent,
    SubagentRegistryComponent(
        subagents={"researcher": researcher},
        # Optional: let agents call unregistered names such as "security-reviewer".
        free_subagent_config=FreeSubagentConfig(enabled=True),
    ),
)
```

### ScratchbookRefComponent
Stores a reference to a specific scratchbook artifact.

| Name | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `artifact_id` | `str` | (none) | Unique identifier for the artifact file |
| `category` | `str` | (none) | Category subfolder path |
| `content_hash` | `str` | (none) | SHA256 hash of the artifact content |
| `timestamp` | `str` | (none) | Creation timestamp |

**Used by:** `TaskPersistenceService`, `ContextResolver`, `ScratchbookIndexer`

**Usage:**
```python
from ecs_agent.components import ScratchbookRefComponent

ref = ScratchbookRefComponent(
    artifact_id="report_001",
    category="tool_results",
    content_hash="abc123...",
    timestamp="1709827200.0"
)
world.add_component(agent, ref)
```

### ScratchbookIndexComponent
Maintains an index of available artifacts for an agent.

| Name | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `artifacts` | `dict[str, ScratchbookRef]` | `{}` | Mapping of artifact IDs to their metadata references |

**Used by:** `ScratchbookIndexer`

**Usage:**
```python
from ecs_agent.components import ScratchbookIndexComponent

world.add_component(agent, ScratchbookIndexComponent(artifacts={}))
```
### UserPromptConfigComponent
Configures opt-in prompt normalization and keyword injection.

| Name | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `triggers` | `dict[str, str]` | `{}` | Mapping of `@keyword` or `event:<name>` to template content |
| `enable_context_pool` | `bool` | `False` | Enable one-shot context collection |
| `context_pool_max_chars` | `int` | `8192` | Maximum characters for context block |

### PromptContextQueueComponent
Queue-backed storage for context items (tool results, subagent outputs) to be injected into outbound user messages.

| Name | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `entries` | `list[ContextEntry]` | `[]` | Collected context entries |

### PromptContextReservationComponent
Tracks an active reservation snapshot for deterministic prompt context injection.

| Name | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `reservation_id` | `str` | (required) | Unique reservation identifier |
| `created_at_tick` | `int` | (required) | Tick when reservation was created |
| `reserved_entries` | `list[ContextEntry]` | (required) | Snapshot of reserved context entries |


### WorkspaceBindingComponent
Binds a filesystem workspace root to an agent entity. Used by `BuiltinToolsSkill` and other
workspace-aware skills to scope file operations. The workspace is owned by the **entity**,
not the skill object, so the same skill can be installed on different agents with different
workspace roots without any shared-state conflicts.

| Name | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `workspace_root` | `Path \| str` | (required) | Absolute path to the agent's workspace root |

**Used by:** `BuiltinToolsSkill`, `SkillRuntime` (workspace-bound skill materialization), `SubagentSystem` (workspace inheritance)

**Usage:**
```python
from ecs_agent.components.definitions import WorkspaceBindingComponent
from pathlib import Path

world.add_component(agent, WorkspaceBindingComponent(workspace_root=Path("/workspace")))
# When a subagent is spawned, it inherits this binding by default (InheritancePolicy).
```

## Workflow Components

### WorkflowDefinitionComponent
Holds the compiled workflow definition for an entity. This component is NOT serialized by design.

| Name | Type | Description |
| :--- | :--- | :--- |
| `compiled` | `CompiledWorkflow` | The frozen, validated workflow model |

**Used by:** `WorkflowStateSystem`, `WorkflowPromptPlaceholderProvider`

### WorkflowRuntimeComponent
Holds the mutable runtime workflow state for an entity. This component IS serialized.

| Name | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `current_state_id` | `str` | (required) | The ID of the active workflow state |
| `transition_history` | `list[str]` | `[]` | History of committed transition IDs |

**Used by:** `WorkflowStateSystem`, `WorkflowPromptPlaceholderProvider`

### WorkflowBindingComponent
Binds an agent key to the workflow for this entity. This component IS serialized.

| Name | Type | Description |
| :--- | :--- | :--- |
| `agent_key` | `str` | The key used to look up prompt profiles |

**Used by:** `WorkflowPromptPlaceholderProvider`

### WorkflowGateSnapshotComponent
Records the last evaluated gate snapshot for debugging and logging. This component IS serialized.

| Name | Type | Description |
| :--- | :--- | :--- |
| `state_id` | `str` | The state ID being evaluated |
| `evaluated_at_tick` | `int` | The tick number of evaluation |
| `matched_transition_id` | `str | None` | The ID of the matched transition, if any |

**Added by:** `WorkflowStateSystem`

### WorkflowLastTransitionComponent
Records the most recent committed transition for history and exact-once semantics. This component IS serialized.

| Name | Type | Description |
| :--- | :--- | :--- |
| `from_state_id` | `str` | The source state ID |
| `to_state_id` | `str` | The target state ID |
| `transition_id` | `str` | The ID of the committed transition |
| `tick` | `int` | The tick number when the transition occurred |

**Added by:** `WorkflowStateSystem`
