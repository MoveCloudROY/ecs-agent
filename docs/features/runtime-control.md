# Runtime Dynamic Control

The ECS Agent framework provides four runtime control capabilities for dynamic agent reconfiguration without restarting: **Entity Registry** (named entity resolution), **System Lifecycle** (dynamic system removal/replacement), **Model Switching** (per-entity provider updates), and **Graceful Interruption** (component-driven stopping).

## Entity Registry

Named entity resolution and tagging for managing multiple agents in a world.

### Methods

- `world.register_entity(entity_id, name, tags=None)` — Register entity with unique name and optional tags
- `world.resolve_entity(name)` — Look up entity by registered name (returns `EntityId | None`)
- `world.list_entities_by_tag(tag)` — Find all entities with given tag (returns `list[EntityId]`)
- `world.unregister_entity(entity_id)` — Remove from registry (called automatically by `delete_entity`)

### Example

```python
from ecs_agent.core import World

world = World()
agent1 = world.create_entity()
agent2 = world.create_entity()

# Register with names and tags
world.register_entity(agent1, "coordinator", tags={"manager", "primary"})
world.register_entity(agent2, "worker", tags={"worker", "secondary"})

# Resolve by name
coordinator_id = world.resolve_entity("coordinator")  # Returns agent1

# Find by tag
workers = world.list_entities_by_tag("worker")  # Returns [agent2]
managers = world.list_entities_by_tag("manager")  # Returns [agent1]
```

### Constraints

- Entity names must be unique within a world
- `register_entity` raises `ValueError` if name already registered
- `resolve_entity` returns `None` for missing names
- `list_entities_by_tag` returns empty list `[]` for missing tags
- `unregister_entity` is a no-op for missing entity IDs

## System Lifecycle Management

Dynamic system removal and replacement with queue-based tick-boundary semantics.

### Methods

- `handle = world.register_system(system, priority)` — Register system, returns `SystemHandle`
- `world.remove_system(handle)` — Queue system for removal at next tick boundary
- `world.replace_system(handle, new_system, priority=None)` — Queue system replacement at next tick boundary
- `world.apply_pending_system_operations()` — Apply queued operations (called automatically by `Runner`)

### Queue Semantics

- All lifecycle operations (remove/replace) are **queued**, not applied immediately
- Operations execute in FIFO order at the **pre-tick boundary** (before `world.process()`)
- Runner automatically calls `apply_pending_system_operations()` before each tick
- Mid-tick replacement requests wait until the next tick starts

### Example

```python
from ecs_agent.core import World, Runner
from ecs_agent.systems.reasoning import ReasoningSystem
from ecs_agent.systems.planning import PlanningSystem

world = World()
runner = Runner()

# Register initial systems
reasoning_handle = world.register_system(ReasoningSystem(priority=0), priority=0)
planning_handle = world.register_system(PlanningSystem(), priority=0)

# Run first tick with both systems
await runner.run(world, max_ticks=1)

# Queue planning system for removal
world.remove_system(planning_handle)

# Removal takes effect at pre-tick boundary of next tick
await runner.run(world, max_ticks=1, start_tick=1)

# Replace reasoning system
new_reasoning = ReasoningSystem(priority=0)
world.replace_system(reasoning_handle, new_reasoning, priority=5)

# Replacement takes effect at next tick
await runner.run(world, max_ticks=1, start_tick=2)
```

## Per-Entity Model Switching

Dynamic model updates for individual entities with in-flight request stability.

### Fields

- `LLMComponent.pending_model: LLMModel | None` — Queued model switch (applied at next request start)

### Behavior

- Pending fields are **sampled at request start** and used for the entire request
- In-flight requests use the sampled values, ignoring further updates
- Cross-entity isolation: Entity A's switch does not affect Entity B

### Example

```python
from ecs_agent.components import LLMComponent
from ecs_agent.providers import Model
from ecs_agent.providers.config import ApiFormat

world = World()
agent = world.create_entity()

llm = LLMComponent(model=Model(
    "gpt-4",
    base_url="https://api.openai.com/v1",
    api_key="...",
    api_format=ApiFormat.OPENAI_CHAT_COMPLETIONS,
))
world.add_component(agent, llm)

# ... agent generates with gpt-4 ...

# Queue model switch
llm.pending_model = Model(
    "gpt-3.5-turbo",
    base_url="https://api.openai.com/v1",
    api_key="...",
    api_format=ApiFormat.OPENAI_CHAT_COMPLETIONS,
)

# Next generation uses gpt-3.5-turbo
# (sampled at reasoning start, stable for entire request)
await world.process()
```

## Graceful Interruption

Component-driven agent stopping with partial content preservation.

### Component

**`InterruptionComponent`**: Signals agent should stop gracefully.

- `reason: InterruptionReason` — Enum: `USER_REQUESTED`, `SYSTEM_PAUSE`, `ERROR`, `COMPLETION`
- `message: str` — Human-readable reason
- `metadata: dict[str, Any]` — Structured context (e.g., `{"partial_chunks": 5}`)
- `timestamp: float` — Auto-generated via `time.time()`

### Behavior

Two distinct paths:

- **In-band interruption** (the component appears mid-stream, e.g. a stop
  button handler): `ReasoningSystem` detects it at its in-loop checks
  (before/after each delta), ends the turn **normally** — partial content is
  preserved in the conversation, partial tool calls are dropped (they would
  never execute and would leave a dangling span), `StreamEnd(status=
  "cancelled")` and the cancelled observability events are published, and the
  entity stays parked behind its `InterruptionComponent`. No exception is
  raised: a self-raised `CancelledError` would be silently ignored by the
  runner's system TaskGroup, swallowing the interruption. Other entities in
  the same world keep running.
- **External cancellation** (`task.cancel()` on the run): the
  `CancelledError` delivered at the await point is caught, partial content is
  preserved, an `InterruptionComponent` is attached if absent, and the
  exception is **re-raised** so cancellation propagates. When the `Runner`
  swallows it as an intentional stop (interruption present before the tick),
  it calls `task.uncancel()` to rebalance the cancelling count before
  returning gracefully.

Metadata is enriched with partial stream telemetry in both paths:
`partial_content`, `partial_chunks`, `partial_content_length`.

### Example

```python
from ecs_agent.components import LLMComponent, ConversationComponent
from ecs_agent.components.definitions import InterruptionComponent
from ecs_agent.types import InterruptionReason, Message

world = World()
agent = world.create_entity()

# ... add LLMComponent, ConversationComponent, start reasoning ...

# Interrupt during generation
world.add_component(agent, InterruptionComponent(
    reason=InterruptionReason.USER_REQUESTED,
    message="User clicked stop button",
    metadata={"source": "web_ui"}
))

# Next tick: ReasoningSystem ends the turn early and saves the partial response
await runner.run(world, max_ticks=1)

# Check partial content
conv = world.get_component(agent, ConversationComponent)
if conv and conv.messages:
    partial_response = conv.messages[-1].content  # Preserved even though interrupted
```

### Constraints

- External `CancelledError` **must be re-raised** after cleanup so cancellation propagates; in-band interruption must NOT raise
- Partial content preserved in both paths; partial tool calls are dropped (dangling spans are additionally repaired at send time by `repair_dangling_tool_spans`)
- Interruption state not overwritten if already present (metadata enriched instead)

## Constraints

- **Entity Registry**: Names must be unique (ValueError on duplicate), tags and metadata are optional
- **System Lifecycle**: Operations queued until tick boundary, applied in FIFO order
- **Model Switching**: Takes effect at next request start, sampled values stable for entire request
- **Graceful Interruption**: in-band interruption ends the turn without raising; external CancelledError is re-raised after partial content preservation
## Prompt Normalization & Injection

Dynamic prompt enhancement via trigger template injection and structured context queuing.

### Components

- **`UserPromptConfigComponent`**: Opt-in configuration for user-prompt normalization.
  - `enable_context_pool: bool` — Enable automatic context collection.
  - `triggers: dict[str, TriggerSpec]` — Mapping of `@keyword` or `event:<name>` to `TriggerSpec` objects.
  - `context_pool_max_chars: int` — Maximum size of the context block.
- **`PromptContextQueueComponent`**: Queue of `ContextEntry` items awaiting injection into the next outbound user message.

### Behavior

1. **Opt-in Only**: Behavior is only active if `UserPromptConfigComponent` is attached to the entity.
2. **Injection Order**: When a user message is processed:
   - `[PROMPT_INJECT:...]` marker is added if a keyword or event is detected.
   - The corresponding trigger template block is injected.
   - The context pool block (tool results, etc.) is injected.
   - The original user text follows.
3. **Deterministic Selection**: Triggers are resolved by `priority DESC`, then `registration_order ASC`, first-match.
4. **Reservation Lifecycle**:
   - **Reserve**: Context entries are snapshotted into a `PromptContextReservationComponent` before the LLM call.
   - **Retry**: If a request fails and retries, the same reserved payload is reused.
   - **Commit**: The queue is cleared only after a successful LLM response is received.
5. **Transient Injection**: Injected content is sent to the provider but **does not mutate stored conversation history**, keeping the long-term context clean.

### Example

```python
from ecs_agent.components import UserPromptConfigComponent, PromptContextQueueComponent
from ecs_agent.prompts import TriggerSpec

world.add_component(agent, UserPromptConfigComponent(
    triggers={
        "@code": TriggerSpec(
            pattern="@code",
            match_mode="keyword",
            action="replace",
            content="Use PEP8 style and include docstrings.",
        ),
        "event:tool_success": TriggerSpec(
            pattern="event:tool_success",
            match_mode="keyword",
            action="replace",
            content="Great job on the tool execution!",
        ),
    },
    enable_context_pool=True,
))
world.add_component(agent, PromptContextQueueComponent())

# User message: "@code Refactor this function"
# Sent to LLM: 1) [PROMPT_INJECT:@code] 2) Template 3) Context Pool 4) User Text
```

## See Also

- [Context Management](context-management.md) — Checkpoint, undo, compaction, conversation revert
- [Tree-Structured Conversations](tree-conversation.md) — Tree structure, branching, linearization
- [Systems](../systems.md) — System execution order and lifecycle
- [API Reference](../api-reference.md) — Complete method signatures
