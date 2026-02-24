# Built-in Systems Reference

This document provides a comprehensive guide to the seven built-in systems available in the ECS Agent framework. These systems handle the core logic of agent behavior, from reasoning and planning to tool execution and error management.

## Recommended System Priority Order

The table below summarizes the recommended priorities for each system. Priority values determine the execution order within each world tick, where lower numbers run first.

| System | Recommended Priority | Purpose |
| :--- | :--- | :--- |
| RAGSystem | -10 | Retrieves context via vector search before reasoning. |
| ToolApprovalSystem | -5 | Filters pending tool calls before execution. |
| ReasoningSystem | 0 | Generates responses using an LLM. |
| PlanningSystem | 0 | Manages step-by-step execution of a plan. |
| TreeSearchSystem | 0 | Uses MCTS to find the best plan path. |
| CollaborationSystem | 5 | Ingests messages from other entities. |
| ToolExecutionSystem | 5 | Executes pending tool calls and returns results. |
| ReplanningSystem | 7 | Periodically revises the current plan based on progress. |
| MemorySystem | 10 | Truncates conversation history to stay within context limits. |
| ErrorHandlingSystem | 99 | Processes and logs errors found on entities. |

---

## 1. ReasoningSystem

The ReasoningSystem serves as the primary cognitive engine for an entity. It coordinates with an LLM provider to generate text responses and identify necessary tool interactions.

- **Constructor**: `__init__(self, priority: int = 0)`
- **Queries**: `LLMComponent`, `ConversationComponent`
- **Optional Components**: `SystemPromptComponent`, `ToolRegistryComponent`
- **Modifies**: `ConversationComponent.messages` (appends the LLM response), potentially adds `PendingToolCallsComponent`.
- **Events Published**: None directly.
- **Recommended Priority**: 0

### Behavior
The system gathers the system prompt and conversation history to build a complete message list. It then calls `provider.complete` using the entity's LLM configuration and any registered tools. The resulting message is appended to the conversation. If the LLM requests specific tools, the system attaches a `PendingToolCallsComponent` to the entity.

### Error Handling
If the LLM provider throws an `IndexError` or `StopIteration`, the system assumes the provider is exhausted and adds a `TerminalComponent(reason="provider_exhausted")`. Any other exceptions result in an `ErrorComponent` being attached to the entity.

### Usage Example
```python
from ecs_agent.systems.reasoning import ReasoningSystem
world.register_system(ReasoningSystem(priority=0), priority=0)
```

---

## 2. MemorySystem

The MemorySystem maintains the conversation history by pruning old messages once they exceed a defined limit. This ensures that LLM requests remain within context window constraints.

- **Constructor**: Uses default constructor.
- **Queries**: `ConversationComponent`
- **Modifies**: `ConversationComponent.messages` (truncates the list).
- **Events Published**: `ConversationTruncatedEvent(entity_id, removed_count)`
- **Recommended Priority**: 10

### Behavior
When the number of messages in a conversation exceeds the `max_messages` threshold, the system trims the list. It always preserves the system message at index 0 and keeps the most recent N messages. A `ConversationTruncatedEvent` is only published if the system actually removes one or more messages.

### Usage Example
```python
from ecs_agent.systems.memory import MemorySystem
world.register_system(MemorySystem(), priority=10)
```

---

## 3. PlanningSystem

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

## 4. ToolExecutionSystem

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

## 5. CollaborationSystem

The CollaborationSystem allows entities to communicate with one another by processing an incoming message inbox.

- **Constructor**: `__init__(self, priority: int = 0)`
- **Queries**: `CollaborationComponent`, `ConversationComponent`
- **Modifies**: Appends messages to `ConversationComponent`, clears the `inbox` in `CollaborationComponent`.
- **Events Published**: `MessageDeliveredEvent(from_entity, to_entity, message)`
- **Recommended Priority**: 5

### Behavior
The system drains all messages from the entity's inbox. Each message is converted into a user-role message formatted as "From {sender_id}: {content}" and added to the conversation history. It publishes a `MessageDeliveredEvent` for every message processed.

### Usage Example
```python
from ecs_agent.systems.collaboration import CollaborationSystem
world.register_system(CollaborationSystem(priority=5), priority=5)
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

## Complete Integration Example

The following code demonstrates how to register all built-in systems with their recommended execution order.

```python
from ecs_agent.core import World
from ecs_agent.systems.reasoning import ReasoningSystem
from ecs_agent.systems.memory import MemorySystem
from ecs_agent.systems.planning import PlanningSystem
from ecs_agent.systems.tool_execution import ToolExecutionSystem
from ecs_agent.systems.collaboration import CollaborationSystem
from ecs_agent.systems.error_handling import ErrorHandlingSystem
from ecs_agent.systems.replanning import ReplanningSystem

world = World()

# Cognitive and planning tasks
world.register_system(ReasoningSystem(priority=0), priority=0)
world.register_system(PlanningSystem(priority=0), priority=0)
# Interaction and communication
world.register_system(CollaborationSystem(priority=5), priority=5)
world.register_system(ToolExecutionSystem(priority=5), priority=5)
# Dynamic adjustment
world.register_system(ReplanningSystem(priority=7), priority=7)
# Maintenance
world.register_system(MemorySystem(), priority=10)
# Global error handling (always run last)
world.register_system(ErrorHandlingSystem(priority=99), priority=99)
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
The system checks the `ApprovalPolicy` on the entity. In `ALWAYS_APPROVE` mode, all calls pass through. In `ALWAYS_DENY`, all calls are removed and a system message is added. In `REQUIRE_APPROVAL`, the system publishes a `ToolApprovalRequestedEvent` and waits (up to a timeout) for a response on the provided future. If approved, the call remains; if denied or timed out, it's removed.

### Usage Example
```python
from ecs_agent.systems.tool_approval import ToolApprovalSystem
world.register_system(ToolApprovalSystem(priority=-5), priority=-5)
```

---

## 9. TreeSearchSystem

The TreeSearchSystem implements Monte Carlo Tree Search (MCTS) to explore potential planning paths and select the most promising sequence of actions.
,
-8
- **Constructor**: `__init__(self, priority: int = 0)`
- **Queries**: `PlanSearchComponent`, `LLMComponent`, `ConversationComponent`
- **Modifies**: `PlanSearchComponent.best_plan`, `PlanSearchComponent.search_active`.
- **Events Published**: `MCTSNodeScoredEvent`.
- **Recommended Priority**: 0 (runs alongside `ReasoningSystem`)

### Behavior
This system is mutually exclusive with `PlanComponent`. If a `PlanComponent` exists, the system skips the entity. For active searches, it performs selection (via UCB1), expansion, simulation (LLM scoring), and backpropagation. Once the search concludes (depth reached or no more expandable nodes), it populates `best_plan` with the optimal path.
,
### Usage Example
```python
from ecs_agent.systems.tree_search import TreeSearchSystem
world.register_system(TreeSearchSystem(priority=0), priority=0)
```
,
---
,
## 10. RAGSystem
,
The RAGSystem implements Retrieval-Augmented Generation by fetching relevant documents from a vector store and injecting them into the agent's conversation history.
,
- **Constructor**: `__init__(self, priority: int = -10)`
- **Queries**: `RAGTriggerComponent`, `EmbeddingComponent`, `VectorStoreComponent`, `ConversationComponent`
- **Modifies**: `ConversationComponent.messages` (inserts context messages), `RAGTriggerComponent.retrieved_docs`, `RAGTriggerComponent.query` (cleared).
- **Events Published**: `RAGRetrievalCompletedEvent`.
- **Recommended Priority**: -10 (runs before `ReasoningSystem`)
,
### Behavior
When a `RAGTriggerComponent` has a non-empty query, the system uses the `EmbeddingProvider` to embed the query and searches the `VectorStore`. The retrieved document snippets are inserted as system messages just before the last user message in the conversation.
,
### Usage Example
```python
from ecs_agent.systems.rag import RAGSystem
world.register_system(RAGSystem(priority=-10), priority=-10)
```