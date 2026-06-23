# Built-in Systems Reference

This document provides a comprehensive guide to the fourteen built-in systems available in the ECS Agent framework. These systems handle the core logic of agent behavior, from reasoning and planning to tool execution and error management.

## Recommended System Priority Order

The table below summarizes the recommended priorities for each system. Priority values determine the execution order within each world tick, where lower numbers run first. Systems at the same priority level run concurrently.

### System Lifecycle and Queue Semantics

The `World` provides methods to dynamically manage systems: `register_system`, `remove_system`, and `replace_system`. 

- **Queued Operations**: Removal and replacement operations are queued.
- **Tick Boundaries**: Operations are applied only when `apply_pending_system_operations()` is called. The `Runner` automatically calls this at the start of each tick.
- **Handles**: Each system is assigned a unique string handle upon registration, which must be used for removal or replacement.


The table below summarizes the recommended priorities for each system. Priority values determine the execution order within each world tick, where lower numbers run first.

| System | Recommended Priority | Purpose |
| :--- | :--- | :--- |
| UserInputSystem | -10 | Captures async user input before reasoning. |
| RAGSystem | -10 | Retrieves context via vector search before reasoning. |
| SystemPromptRenderSystem | -20 | Resolves `${name}` placeholders from `SystemPromptConfigSpec` and produces a cached `RenderedSystemPromptComponent` on first render. |
| UserPromptNormalizationSystem | -10 | Injects trigger templates into user messages and produces `RenderedUserPromptComponent`. ContextPool injection happens later at call-time. |
| WorkflowStateSystem | -25 | Evaluates workflow gates and commits transitions. |
| SubagentWaitSystem | -5 | Handles `subagent_wait` tool and background session notifications. |
| PromptContextCollectorSystem | 0 | Collects tool/subagent results into the context pool. |
| ToolApprovalSystem | -5 | Filters pending tool calls before execution. |
| ReasoningSystem | 0 | Generates responses using an LLM. |
| PlanningSystem | 0 | Manages step-by-step execution of a plan. |
| TreeSearchSystem | 0 | Uses MCTS to find the best plan path. |
| MessageBusSystem | 5 | Handles pub/sub and request-response messaging. |
| ToolExecutionSystem | 5 | Executes pending tool calls and returns results. |
| SubagentSystem | -1 | Manages subagent delegation and execution. |
| ReplanningSystem | 7 | Periodically revises the current plan based on progress. |
| CheckpointSystem | (configurable) | Creates world state snapshots for undo. |
| CompactionSystem | (configurable) | Compresses conversation history via LLM summarization. |
| ErrorHandlingSystem | 99 | Processes and logs errors found on entities. |

---


## System Lifecycle Management

Systems can be dynamically registered, removed, and replaced at runtime using queue-based operations.

### Registration

Systems are registered with `world.register_system(system, priority)` which returns a `SystemHandle` for later reference:

```python
from ecs_agent.systems.reasoning import ReasoningSystem

handle = world.register_system(ReasoningSystem(priority=0), priority=0)
```

### Dynamic Removal and Replacement

Systems can be removed or replaced at runtime using queue-based operations:

- `world.remove_system(handle)` — Queue system for removal
- `world.replace_system(handle, new_system, priority)` — Queue system replacement
- `world.apply_pending_system_operations()` — Apply queued operations (called automatically by Runner)

**Queue Semantics:**

- All lifecycle operations are **queued**, not applied immediately
- Operations execute in **FIFO order** at the **pre-tick boundary** (before `world.process()`)
- Runner automatically calls `apply_pending_system_operations()` before each tick
- Mid-tick replacement requests wait until the next tick starts

```python
# Queue removal
world.remove_system(handle)
# Removal takes effect at next tick boundary

# Queue replacement
world.replace_system(handle, NewSystem(), priority=5)
# Replacement takes effect at next tick boundary
```

This ensures deterministic system execution and prevents mid-tick mutations.

---
## 1. SystemPromptRenderSystem

The `SystemPromptRenderSystem` resolves all `${name}` placeholders from a `SystemPromptConfigSpec` component and writes a cached `RenderedSystemPromptComponent` for LLM callers on the first successful render-system pass. Subsequent ticks reuse this frozen component and skip re-rendering. It replaces the legacy `SystemPromptAssemblySystem`.

- **Constructor**: `__init__(self)`
- **Queries**: `SystemPromptConfigSpec`
- **Produces**: `RenderedSystemPromptComponent`
- **Recommended Priority**: -20 (must run before reasoning)

### Behavior
The system reads the `SystemPromptConfigSpec.template_source` (inline string or file path) and substitutes all `${name}` occurrences. Built-in placeholders `${_installed_tools}`, `${_installed_skills}`, `${_installed_mcps}`, and `${_installed_subagents}` are populated from entity metadata automatically. Callable placeholder resolvers are called once during the first successful render-system pass. The resulting prompt is frozen (freeze semantics) and reused on subsequent ticks. Missing or failing placeholders raise `ValueError` immediately — no silent fallback.

### Usage Example
```python
from ecs_agent.prompts.contracts import SystemPromptConfigSpec, PromptTemplateSource
from ecs_agent.systems.system_prompt_render_system import SystemPromptRenderSystem

world.add_component(entity, SystemPromptConfigSpec(
    template_source=PromptTemplateSource(
        inline="You are a helpful assistant. Tools: ${_installed_tools}"
    )
))
world.register_system(SystemPromptRenderSystem(), priority=-20)
```

---

## 1b. UserPromptNormalizationSystem

The `UserPromptNormalizationSystem` processes the latest user message in `ConversationComponent`, injects any matching `@keyword` or `event:<name>` trigger templates, and writes a `RenderedUserPromptComponent`. ContextPool injection is handled separately at call-time by `prepare_outbound_messages()`. Stored conversation history is never mutated.

- **Constructor**: `__init__(self)`
- **Queries**: `UserPromptConfigComponent`, `ConversationComponent`
- **Produces**: `RenderedUserPromptComponent`
- **Recommended Priority**: -10 (must run before reasoning)

### Behavior
Scans the last user message for registered trigger patterns. Matched triggers prepend their template content to the rendered user prompt. The original user text comes last. The `RenderedUserPromptComponent.text` is what LLM callers send to the provider — the stored message is unchanged. Note that ContextPool injection is NOT performed by this system; it is applied at call-time by `prepare_outbound_messages()`.

### Usage Example
```python
from ecs_agent.components import UserPromptConfigComponent
from ecs_agent.systems.user_prompt_normalization_system import UserPromptNormalizationSystem

world.add_component(entity, UserPromptConfigComponent(
    triggers={"@test": "Use testing best practices."}
))
world.register_system(UserPromptNormalizationSystem(), priority=-10)
```

---

## 1c. WorkflowStateSystem

The `WorkflowStateSystem` evaluates declarative gate expressions against an entity's components and commits state transitions in the workflow graph.

- **Constructor**: `__init__(self, priority: int = -25)`
- **Queries**: `WorkflowDefinitionComponent`, `WorkflowRuntimeComponent`
- **Produces**: `WorkflowLastTransitionComponent`, `WorkflowGateSnapshotComponent`
- **Recommended Priority**: -25

### Behavior
The system iterates through all transitions defined for the entity's current state in the `WorkflowDefinitionComponent`. It evaluates each transition's gate expression (e.g., `has(C)`, `field(C, "attr") == value`). 

- If exactly one transition matches, the system updates `WorkflowRuntimeComponent.current_state_id`, appends the transition ID to `transition_history`, and attaches a `WorkflowLastTransitionComponent`.
- If zero transitions match, it is a no-op (state remains unchanged).
- If more than one transition matches simultaneously, the system attaches an `ErrorComponent` and a `TerminalComponent` with the reason `workflow_ambiguous_transition`.

### Usage Example
```python
from ecs_agent.systems.workflow_state import WorkflowStateSystem

world.register_system(WorkflowStateSystem(priority=-25), priority=-25)
```

---

## 2. ReasoningSystem

The ReasoningSystem serves as the primary cognitive engine for an entity. It coordinates with an LLM provider to generate text responses and identify necessary tool interactions.

- **Constructor**: `__init__(self, priority: int = 0)`
- **Queries**: `LLMComponent`, `ConversationComponent`
- **Optional Components**: `SystemPromptComponent`, `ToolRegistryComponent`, `StreamingComponent`
- **Modifies**: `ConversationComponent.messages` (appends the LLM response), potentially adds `PendingToolCallsComponent`.
- **Events Published**: `StreamStartEvent`, `StreamReasoningDeltaEvent`, `StreamReasoningEndEvent`, `StreamContentStartEvent`, `StreamContentDeltaEvent`, `StreamEndEvent` (when streaming is enabled)
- **Recommended Priority**: 0

### Behavior
The system gathers the system prompt and conversation history to build a complete message list. It then calls `provider.complete` using the entity's LLM configuration and any registered tools. The resulting message is appended to the conversation. If the LLM requests specific tools, the system attaches a `PendingToolCallsComponent` to the entity.

### Streaming Mode
When entity has `StreamingComponent(enabled=True)`, the system calls `provider.complete(stream=True)`, publishes `StreamStartEvent`, emits reasoning-phase events (`StreamReasoningDeltaEvent` -> `StreamReasoningEndEvent`) when reasoning chunks exist, emits `StreamContentStartEvent` before the first assistant content chunk, publishes `StreamContentDeltaEvent` for each content chunk, and finally publishes `StreamEndEvent`. Content chunks and tool calls are accumulated, and the final `CompletionResult` is returned as normal.

### Error Handling
If the LLM provider throws an `IndexError` or `StopIteration`, the system assumes the provider is exhausted and adds a `TerminalComponent(reason="provider_exhausted")`. Any other exceptions result in an `ErrorComponent` being attached to the entity.

### Usage Example
```python
from ecs_agent.systems.reasoning import ReasoningSystem
world.register_system(ReasoningSystem(priority=0), priority=0)
```

---

## 2. PlanningSystem

The PlanningSystem enables an entity to follow a structured sequence of actions to achieve a goal. It breaks down complex tasks into manageable steps.

- **Constructor**: `__init__(self, priority: int = 0)`
- **Queries**: `PlanComponent`, `LLMComponent`, `ConversationComponent`
- **Optional Components**: `SystemPromptComponent`, `ToolRegistryComponent`
- **Modifies**: `ConversationComponent.messages`, `PlanComponent.current_step`, `PlanComponent.completed`, potentially adds `PendingToolCallsComponent`.
- **Events Published**: `PlanStepCompletedEvent(entity_id, step_index, step_description)`
- **Recommended Priority**: 0

### Behavior
This system skips processing if the plan is already marked as completed. For active plans, it creates a context message indicating the current step (e.g., "Step 1/5: description") and sends it to the LLM. After the LLM provides a response, the system increments the step index and publishes a completion event. It marks the plan as finished once the final step is reached.

### Error Handling
Provider exhaustion leads to a `TerminalComponent`. Other exceptions trigger both an `ErrorComponent` and a `TerminalComponent(reason="planning_error")`.

### Usage Example
```python
from ecs_agent.systems.planning import PlanningSystem
world.register_system(PlanningSystem(priority=0), priority=0)
```

---

## 3. ToolExecutionSystem

The ToolExecutionSystem bridges the gap between LLM requests and actual code execution. It processes requests generated by the ReasoningSystem or PlanningSystem.

- **Constructor**: `__init__(self, priority: int = 0)`
- **Queries**: `PendingToolCallsComponent`, `ToolRegistryComponent`, `ConversationComponent`
- **Modifies**: Removes `PendingToolCallsComponent`, adds `ToolResultsComponent`, appends tool result messages to `ConversationComponent`.
- **Events Published**: None.
- **Recommended Priority**: 5

### Behavior
The system iterates through all tool calls in the `PendingToolCallsComponent`. It looks up the appropriate handler in the registry and executes it with the provided arguments. The results are formatted as messages with the "tool" role and added to the conversation.

### Error Handling
This system does not throw exceptions. If it encounters an unknown tool or a handler fails, it records the error as a string within the tool result message so the LLM can respond to the failure.

### Usage Example
```python
from ecs_agent.systems.tool_execution import ToolExecutionSystem
world.register_system(ToolExecutionSystem(priority=5), priority=5)
```

---

## 5. MessageBusSystem

The MessageBusSystem provides a robust messaging infrastructure for multi-agent communication. It supports asynchronous pub/sub messaging and synchronous request-response patterns with CloudEvents-aligned envelopes.

- **Constructor**: `__init__(self, priority: int = 5)`
- **Queries**: `MessageBusConfigComponent`, `MessageBusSubscriptionComponent`, `MessageBusConversationComponent`, `ConversationComponent`
- **Modifies**: `ConversationComponent` (appends delivered messages), `MessageBusSubscriptionComponent` (manages queues), `MessageBusConversationComponent` (tracks requests).
- **Events Published**: `MessageBusPublishedEvent`, `MessageBusDeliveredEvent`, `MessageBusResponseEvent`, `MessageBusTimeoutEvent`.
- **Recommended Priority**: 5

### Behavior
The system manages per-subscriber message queues with bounded buffering (default `max_queue_size=1000`). It processes outgoing messages from entities and delivers them to subscribers based on topic filters.

#### Pub/Sub
Entities can publish messages to any topic. Subscribers receive copies of these messages in their conversation history, formatted as "From {sender_id} on {topic}: {content}".

#### Request-Response
The system implements a request-response pattern using temporary inbox topics and correlation IDs.
- **Request**: An entity publishes a message with a `reply_to` topic. The system tracks this request and awaits a response.
- **Response**: The recipient responds to the `reply_to` topic. The system routes the response back to the requester and clears the conversation state.
- **Timeout**: If no response is received within the `request_timeout` (default 30s), a `MessageBusTimeoutEvent` is published and the conversation is cleaned up.

### Message Schema
All messages are wrapped in a `MessageBusEnvelope` following the CloudEvents spec:
- `id`: Unique message identifier
- `source`: Sending entity ID
- `type`: Message type (e.g., `ecs.message.pub`)
- `specversion`: "1.0"
- `correlationid`: ID used to link requests and responses
- `traceparent`: W3C TraceContext for distributed tracing

### Usage Example
```python
from ecs_agent.systems.message_bus import MessageBusSystem
world.register_system(MessageBusSystem(priority=5), priority=5)
```

---

## 6. ErrorHandlingSystem

The ErrorHandlingSystem acts as a centralized observer for failures across the world. It typically runs last to ensure it catches errors from all other systems.

- **Constructor**: `__init__(self, priority: int = 99)`
- **Queries**: `ErrorComponent`
- **Modifies**: Removes `ErrorComponent`.
- **Events Published**: `ErrorOccurredEvent(entity_id, error, system_name)`
- **Recommended Priority**: 99

### Behavior
This system identifies any entity with an `ErrorComponent`. It logs the error details, publishes an `ErrorOccurredEvent`, and then removes the component to prevent redundant processing in the next tick.

### Usage Example
```python
from ecs_agent.systems.error_handling import ErrorHandlingSystem
world.register_system(ErrorHandlingSystem(priority=99), priority=99)
```

---

## 7. ReplanningSystem

The ReplanningSystem allows an agent to adjust its course of action based on the results of previous steps. It ensures the plan remains relevant as the environment changes.

- **Constructor**: `__init__(self, priority: int = 7)`
- **Queries**: `PlanComponent`, `LLMComponent`, `ConversationComponent`
- **Optional Components**: `SystemPromptComponent`
- **Modifies**: `PlanComponent.steps` (replaces future steps).
- **Events Published**: `PlanRevisedEvent(entity_id, old_steps, new_steps)`
- **Recommended Priority**: 7

### Behavior
Replanning occurs when the plan's `current_step` moves past a internal checkpoint. The system sends a specialized prompt to the LLM asking for a revised step list in JSON format. If the LLM provides new steps, the system replaces the remaining portion of the plan and publishes a revision event.

### Error Handling
If the provider is exhausted or the LLM output fails to parse as valid JSON, the system silently advances its internal checkpoint. This prevents the agent from stalling or entering an infinite loop of replanning attempts.

### Usage Example
```python
from ecs_agent.systems.replanning import ReplanningSystem
world.register_system(ReplanningSystem(priority=7), priority=7)
```

---

## 8. ToolApprovalSystem

The ToolApprovalSystem provides a mechanism to filter or approve tool calls generated by the LLM before they are executed. This is essential for security and human-in-the-loop workflows.

- **Constructor**: `__init__(self, priority: int = -5)`
- **Queries**: `PendingToolCallsComponent`, `ToolApprovalComponent`, `ConversationComponent`
- **Modifies**: `PendingToolCallsComponent.tool_calls` (filters denied calls), `ConversationComponent.messages` (appends denial notifications).
- **Events Published**: `ToolApprovalRequestedEvent`, `ToolApprovedEvent`, `ToolDeniedEvent`.
- **Recommended Priority**: -5 (runs before `ToolExecutionSystem`)

### Behavior
The system checks the `ApprovalPolicy` on the entity. In `ALWAYS_APPROVE` mode, all calls pass through. In `ALWAYS_DENY`, all calls are removed and a system message is added. In `REQUIRE_APPROVAL`, the system publishes a `ToolApprovalRequestedEvent` and waits (up to a timeout, or indefinitely if `timeout` is `None`) for a response on the provided future. If approved, the call remains; if denied or timed out, it's removed.

### Usage Example
```python
from ecs_agent.systems.tool_approval import ToolApprovalSystem
world.register_system(ToolApprovalSystem(priority=-5), priority=-5)
```

---

## 9. TreeSearchSystem

The TreeSearchSystem implements Monte Carlo Tree Search (MCTS) to explore potential planning paths and select the most promising sequence of actions.

- **Constructor**: `__init__(self, priority: int = 0)`
- **Queries**: `PlanSearchComponent`, `LLMComponent`, `ConversationComponent`
- **Modifies**: `PlanSearchComponent.best_plan`, `PlanSearchComponent.search_active`.
- **Events Published**: `MCTSNodeScoredEvent`.
- **Recommended Priority**: 0 (runs alongside `ReasoningSystem`)

### Behavior
This system is mutually exclusive with `PlanComponent`. If a `PlanComponent` exists, the system skips the entity. For active searches, it performs selection (via UCB1), expansion, simulation (LLM scoring), and backpropagation. Once the search concludes (depth reached or no more expandable nodes), it populates `best_plan` with the optimal path.

### Usage Example
```python
from ecs_agent.systems.tree_search import TreeSearchSystem
world.register_system(TreeSearchSystem(priority=0), priority=0)
```

---

## 10. RAGSystem

The RAGSystem implements Retrieval-Augmented Generation by fetching relevant documents from a vector store and injecting them into the agent's conversation history.

- **Constructor**: `__init__(self, priority: int = -10)`
- **Queries**: `RAGTriggerComponent`, `EmbeddingComponent`, `VectorStoreComponent`, `ConversationComponent`
- **Modifies**: `ConversationComponent.messages` (inserts context messages), `RAGTriggerComponent.retrieved_docs`, `RAGTriggerComponent.query` (cleared).
- **Events Published**: `RAGRetrievalCompletedEvent`.
- **Recommended Priority**: -10 (runs before `ReasoningSystem`)

### Behavior
When a `RAGTriggerComponent` has a non-empty query, the system uses the `EmbeddingProvider` to embed the query and searches the `VectorStore`. The retrieved document snippets are inserted as system messages just before the last user message in the conversation.

### Usage Example
```python
from ecs_agent.systems.rag import RAGSystem
world.register_system(RAGSystem(priority=-10), priority=-10)
```

---

## 11. CheckpointSystem

Creates snapshots of the entire world state for undo/restore operations.

- **Constructor**: Default constructor.
- **Queries**: `CheckpointComponent`
- **Optional Components**: Uses `WorldSerializer.to_dict()` internally.
- **Modifies**: `CheckpointComponent.snapshots` (appends new snapshot).
- **Events Published**: `CheckpointCreatedEvent`.

### Behavior
On each tick, the system serializes the entire world state via `WorldSerializer.to_dict()` and pushes it onto the `CheckpointComponent.snapshots` stack. If `max_snapshots` is exceeded, the oldest snapshot is removed.

### Static Methods
- `undo(world, providers, tool_handlers)`: Pops the last snapshot, restores the world state via `WorldSerializer.from_dict()`, preserves the snapshot history, and publishes `CheckpointRestoredEvent`.

### Usage Example
```python
from ecs_agent.systems.checkpoint import CheckpointSystem
world.register_system(CheckpointSystem(), priority=15)
```

---

## 12. CompactionSystem

Compresses conversation history by summarizing older messages using the entity's LLM provider.

- **Constructor**: `__init__(self)` (no parameters)
- **Queries**: `CompactionConfigComponent`, `LLMComponent`, `ConversationComponent`
- **Optional Components**: `ConversationArchiveComponent`
- **Modifies**: `ConversationComponent.messages` (replaces summarized history with `[system_msg?] + [last_user_anchor?]` after compaction), `ConversationArchiveComponent.archived_summaries`.
- **Events Published**: `CompactionCompleteEvent`.

### Behavior
The system estimates token count using `word_count * 1.3`. When the estimate exceeds `CompactionConfigComponent.threshold_tokens`, the configured `compaction_method` selects which messages to summarize. `full_history` summarizes all non-system messages; `predrop_then_compact` first applies outbound-budget pruning to the summary input. After summarization, live conversation history is rebuilt from the original system message (if present) plus a minimal last-user continuation anchor (if present). If a matching `RenderedUserPromptComponent` exists for that last raw user message, its rendered text is used as the anchor so script-trigger prompts do not re-run after compaction. The summary is archived in `ConversationArchiveComponent` and stored in `CurrentCompactionSummaryComponent` for injection into the system prompt via `SystemPromptRenderSystem`.

### Usage Example
```python
from ecs_agent.systems.compaction import CompactionSystem
world.register_system(CompactionSystem(), priority=20)
```

---

## 13. UserInputSystem

Captures async user input, supporting infinite wait when timeout is None.

- **Constructor**: `__init__(self, priority: int = -10)`
- **Queries**: `UserInputComponent`, `ConversationComponent`
- **Modifies**: `UserInputComponent.result`, `ConversationComponent.messages` (appends user message).
- **Events Published**: `UserInputRequestedEvent`.

### Behavior
For each entity with a `UserInputComponent`, the system creates an `asyncio.Future` if none exists, publishes `UserInputRequestedEvent`, and awaits the future with `asyncio.wait_for(asyncio.shield(future), timeout=component.timeout)`. When `timeout=None`, it waits indefinitely. On resolve, the result is stored and appended as a user message to the conversation. On timeout, an `ErrorComponent` and `TerminalComponent` are added.

### Usage Example
```python
from ecs_agent.systems.user_input import UserInputSystem
world.register_system(UserInputSystem(priority=-10), priority=-10)
```

---

## 14. SubagentSystem

The SubagentSystem manages subagent delegation, allowing parent agents to spawn child agents for subtask execution with isolated contexts and automatic result aggregation.

- **Constructor**: `__init__(self, priority: int = -1, default_timeout: float | None = None, registry: ArtifactRegistry | None = None, max_background_concurrency: int = 5, allow_unregistered_subagents: bool = False, free_subagent_config: FreeSubagentConfig | None = None)`
- **Queries**: `SubagentRegistryComponent`, `ToolRegistryComponent` (`ToolRegistryComponent` alone when free-form mode is enabled)
- **Modifies**: `ToolRegistryComponent.tools` (registers `subagent` tool), `ToolRegistryComponent.handlers` (registers subagent handler).
- **Events Published**: `DelegationStartedEvent(parent_entity, child_entity, subagent_name, task)`, `DelegationCompletedEvent(parent_entity, child_entity, subagent_name, result)`
- **Recommended Priority**: -1 (runs before `ReasoningSystem`)

### Behavior
The system automatically registers a `subagent` tool for entities that have both `SubagentRegistryComponent` and `ToolRegistryComponent`. When `allow_unregistered_subagents=True`, it can also create a `SubagentRegistryComponent` for entities that only have `ToolRegistryComponent`, enabling dynamic subagent names. In that global free-form mode, register `SubagentSystem` before `SystemPromptRenderSystem` (for example priority `-30`) if `${_installed_subagents}` should advertise free-form delegation in the first rendered prompt.

When the subagent tool is called by an LLM, the system:
1. Looks up the subagent configuration by category name in the registry, or creates a dynamic config from the parent model if free-form mode is enabled and the category is unregistered
2. Creates a new child entity with the subagent's provider, model, and system prompt
3. Runs the child entity to completion (or until `max_ticks` is reached)
4. Returns the child's final assistant message as the tool result
5. Publishes delegation lifecycle events to the event bus

Each subagent runs in complete isolation with its own conversation history and state. The parent agent receives only the final result.

### Scheduler & Concurrency
The system manages a process-global FIFO queue for background sessions.
- **Concurrency Limit**: Configured via `max_background_concurrency`. Default is 5.
- **FIFO Queue**: Sessions are processed in the order they were launched.
- **Cap Conflict**: If multiple `SubagentSystem` instances are registered with different concurrency caps, a `ValueError` is raised.

### Tool Schema
The `subagent` tool accepts parameters:
- `category` (required): Name of the subagent to invoke. By default it must exist in the registry; with free-form mode enabled it may be any descriptive unregistered name.
- `prompt` (required): Task description for the subagent
- `load_skills` (optional): Additional skill names to load on the subagent
- `background` (optional): If true, executes asynchronously and returns session ID
- `stream` (optional): If true, enables streaming telemetry events to the parent EventBus
- `timeout` (optional): Maximum seconds to wait for completion

### Lifecycle States
Background sessions transition through these states:
- `queued`: Waiting in the FIFO queue.
- `running`: Currently executing.
- `succeeded`: Completed successfully.
- `failed`: Terminated with an error.
- `timed_out`: Terminated after exceeding timeout limit.
- `cancelled`: Terminated by explicit cancel request.

### Error Handling
If the specified subagent name is not found in the registry and free-form mode is disabled, the tool returns an error message. If free-form mode is enabled, the system uses the parent entity's `LLMComponent.model` to create a dynamic subagent config. If the subagent execution fails or times out, the error details are returned as the tool result.

### Usage Example
```python
from ecs_agent.systems.subagent import SubagentSystem
from ecs_agent.components import SubagentRegistryComponent
from ecs_agent.types import SubagentConfig

# Configure subagents
researcher = SubagentConfig(
    name="researcher",
    model=Model("gpt-4o", base_url="...", api_key="...", api_format=ApiFormat.OPENAI_CHAT_COMPLETIONS),
    system_prompt="You are a research specialist.",
    max_ticks=10,
)

# Register with parent entity
world.add_component(
    parent_entity,
    SubagentRegistryComponent(subagents={"researcher": researcher}),
)

# Register system
world.register_system(SubagentSystem(priority=-1), priority=-1)
```

### See Also
- [Subagent Feature Documentation](features/subagent.md) — Detailed guide with delegation patterns
- [SubagentRegistryComponent](components.md#subagentregistrycomponent) — Component reference
---

## 14b. SubagentWaitSystem

The `SubagentWaitSystem` handles the `subagent_wait` tool and manages the delivery of background session completion notifications to the parent agent.

- **Constructor**: `__init__(self, priority: int = -5, resume_callback: ResumeCallback | None = None)`
- **Queries**: `SubagentWaitComponent`, `SubagentNotificationQueueComponent`, `SubagentSessionTableComponent`, `ConversationComponent`
- **Modifies**: `SubagentWaitComponent` (resolves future, updates timeout/scope), `ConversationComponent` (appends notification messages), `SubagentNotificationQueueComponent` (marks notifications as delivered).
- **Recommended Priority**: -5 (REQUIRED to run before `ReasoningSystem`)

### Behavior
The system monitors entities for `SubagentWaitComponent`. It uses **wait-all semantics**: the future is only resolved when every session in the wait scope reaches a terminal state.
1. On first processing, snapshots all currently active sessions as the wait scope (when `session_ids` is `None`).
2. When all scoped sessions are terminal, resolves the wait future to wake the parent agent.
3. Batches all unread terminal notifications into a single `role="user"` message.
4. Appends the notification message to the parent's conversation history.
5. Marks the notifications as delivered.
6. Removes the `SubagentWaitComponent`.

On timeout, the system evaluates session statuses:
- If any sessions are still running, the deadline is extended (wait continues).
- If any sessions have failed, a `role="user"` failure notification is injected containing `subagent_resume` instructions so the LLM can restart failed sessions.
- If `auto_restart_budget > 0` and a `resume_callback` is configured, the system automatically restarts failed sessions up to the budget before surfacing failures.

### Usage Example
```python
from ecs_agent.systems.subagent_wait import SubagentWaitSystem

# Register system (priority -5 REQUIRED)
world.register_system(SubagentWaitSystem(priority=-5), priority=-5)

# With auto-restart enabled
subagent_system = SubagentSystem()
world.register_system(
    SubagentWaitSystem(
        priority=-5,
        resume_callback=subagent_system.make_resume_callback(),
    ),
    priority=-5,
)
```

---

## Complete Integration Example

The following code demonstrates how to register all built-in systems with their recommended execution order.

```python
from ecs_agent.core import World
from ecs_agent.systems.user_input import UserInputSystem
from ecs_agent.systems.rag import RAGSystem
from ecs_agent.systems.prompt_context_collector import PromptContextCollectorSystem
from ecs_agent.systems.tool_approval import ToolApprovalSystem
from ecs_agent.systems.reasoning import ReasoningSystem
from ecs_agent.systems.planning import PlanningSystem
from ecs_agent.systems.tree_search import TreeSearchSystem
from ecs_agent.systems.message_bus import MessageBusSystem
world.register_system(MessageBusSystem(priority=5), priority=5)
from ecs_agent.systems.tool_execution import ToolExecutionSystem
from ecs_agent.systems.replanning import ReplanningSystem
from ecs_agent.systems.checkpoint import CheckpointSystem
from ecs_agent.systems.compaction import CompactionSystem
from ecs_agent.systems.error_handling import ErrorHandlingSystem
from ecs_agent.systems.subagent import SubagentSystem

world = World()

# Input and Context
world.register_system(UserInputSystem(priority=-10), priority=-10)
world.register_system(RAGSystem(priority=-10), priority=-10)
#BH|world.register_system(PromptContextCollectorSystem(priority=0), priority=0)

# Safety and Filtering
world.register_system(ToolApprovalSystem(priority=-5), priority=-5)

# Cognitive and planning tasks
world.register_system(ReasoningSystem(priority=0), priority=0)
world.register_system(PlanningSystem(priority=0), priority=0)
world.register_system(TreeSearchSystem(priority=0), priority=0)

# Interaction and communication
world.register_system(MessageBusSystem(priority=5), priority=5)
world.register_system(ToolExecutionSystem(priority=5), priority=5)
world.register_system(SubagentSystem(priority=5), priority=5)

# Dynamic adjustment
world.register_system(ReplanningSystem(priority=7), priority=7)

# Maintenance
world.register_system(CheckpointSystem(), priority=15)
world.register_system(CompactionSystem(), priority=20)

# Global error handling (always run last)
world.register_system(ErrorHandlingSystem(priority=99), priority=99)
```
## 15. PromptContextCollectorSystem

The PromptContextCollectorSystem automatically gathers results from tool executions and subagent completions, storing them in the `PromptContextQueueComponent` for transient injection into the next LLM turn.

- **Constructor**: `__init__(self, priority: int = 0)`
- **Queries**: `PromptContextQueueComponent`, `ToolResultsComponent`, `SubagentSessionTableComponent`
- **Modifies**: `PromptContextQueueComponent.entries`.
- **Recommended Priority**: 0

### Behavior
The system monitors for new tool results or completed subagent sessions. It extracts the content, assigns a priority based on the source, and adds it to the context pool. Items are marked with a registration order to ensure deterministic rendering.

### Usage Example
```python
from ecs_agent.systems.prompt_context_collector import PromptContextCollectorSystem
world.register_system(PromptContextCollectorSystem(priority=0), priority=0)
```
