# Runtime Dynamic Control

The ECS Agent framework provides a set of runtime control capabilities that allow for dynamic management of entities, systems, and agent behavior during execution. These features enable high-level orchestration, graceful interruption, and flexible conversation management.

## Introduction

Runtime Dynamic Control encompasses five core capabilities:
1. **Entity Registry**: Named access and tagging for entity management.
2. **System Lifecycle**: Queued registration, removal, and replacement of systems.
3. **Model Switching**: Per-entity model and provider updates with isolated scope.
4. **Graceful Interruption**: Controlled stoppage of agent execution with state preservation.
5. **Conversation Revert**: Non-destructive navigation to historical conversation states.

## Capabilities

### 1. Entity Registry

The Entity Registry provides a central mechanism for resolving entities by name or tag. This is useful for multi-agent coordination where entities need to find each other.

- **Named Resolution**: Assign a unique name to an entity for global lookup.
- **Tagging**: Group entities with semantic tags (e.g., "worker", "researcher").
- **Automatic Lifecycle**: `delete_entity()` automatically unregisters the entity from the registry.

#### Usage

```python
# Register an agent with a name and tags
world.register_entity(agent_id, name="primary-agent", tags={"worker", "active"})

# Resolve entity ID by name
eid = world.resolve_entity("primary-agent")

# List all entities with a specific tag
workers = world.list_entities_by_tag("worker")

# Unregister an entity
world.unregister_entity(agent_id)
```

### 2. System Lifecycle

Systems can be dynamically registered, removed, or replaced at runtime. Operations are queued and applied at tick boundaries to ensure deterministic execution.

- **Queued Operations**: `remove_system()` and `replace_system()` are deferred until `apply_pending_system_operations()` is called.
- **System Handles**: `register_system()` returns a unique handle used for subsequent lifecycle operations.
- **Priority Preservation**: `replace_system()` preserves the original priority of the system being replaced.

#### Usage

```python
# Register a system and get its handle
handle = world.register_system(ReasoningSystem(priority=0), priority=0)

# Replace the system with a different implementation
new_system = AdvancedReasoningSystem()
world.replace_system(handle, new_system)

# Remove a system by handle
world.remove_system(handle)

# Apply pending operations (typically called by Runner before tick starts)
world.apply_pending_system_operations()
```

### 3. Model Switching

The `LLMComponent` supports updating the model or provider dynamically. These changes are isolated to the specific entity and do not affect others.

- **Pending Fields**: Use `pending_model` and `pending_provider` to stage updates.
- **Request-Start Sampling**: Systems sample the active model at the start of a request to ensure stability for in-flight completions.
- **Cross-Entity Isolation**: Updates to Entity A's configuration never impact Entity B.

#### Usage

```python
llm = world.get_component(agent_id, LLMComponent)

# Switch model for the next request
llm.pending_model = "gpt-4-turbo"

# Switch provider (e.g., fallback to local model)
llm.pending_provider = local_provider
```

### 4. Graceful Interruption

The `InterruptionComponent` allows for signaling that an entity's execution should stop. Unlike a hard cancellation, this mechanism allows systems to preserve partial state.

- **Interruption Reasons**: Supported reasons include `USER_REQUEST`, `ERROR`, `TIMEOUT`, and `POLICY_VIOLATION`.
- **State Preservation**: Systems check for interruption during long-running tasks (like streaming) and ensure partial content is saved to the conversation history before stopping.
- **Runner Detection**: The `Runner` detects `InterruptionComponent` and raises a `CancelledError` after ensuring tick consistency.

#### Usage

```python
from ecs_agent.types import InterruptionReason

# Signal interruption
world.add_component(agent_id, InterruptionComponent(
    reason=InterruptionReason.USER_REQUEST,
    metadata={"partial_allowed": True}
))
```

### 5. Conversation-Tree Revert

For entities using `ConversationTreeComponent`, the `revert_to_message()` utility provides a way to backtrack to a previous point in the conversation history without deleting subsequent nodes.

- **Non-Destructive**: Historical nodes are preserved; a new branch is created from the target message.
- **Branch Navigation**: Automatically switches the active branch to the newly created revert path.
- **Failure Semantics**: Raises `KeyError` if the target message ID is not found in the tree.

#### Usage

```python
from ecs_agent.conversation_tree import revert_to_message

# Revert conversation to a specific historical message
revert_to_message(tree_component, "msg_historical_id")

# The next ReasoningSystem tick will linearize from this point
```

## Constraints

- **Tick Boundaries**: System lifecycle operations (`remove`, `replace`) only take effect after `apply_pending_system_operations()` is called. Calling these mid-tick will not affect the currently executing tick.
- **Registry Names**: `register_entity()` will overwrite existing entries if the same name is used for a different entity ID.
- **Tree Revert**: Only works with `ConversationTreeComponent`. Does not support flat `ConversationComponent` history.
- **Interruption Scope**: Interruption is per-entity. One entity being interrupted does not stop other entities in the world.

## See Also
- [World API Reference](../api-reference.md#world)
- [Conversation Tree Guide](./tree-conversation.md)
- [Runner API Reference](../api-reference.md#runner)
