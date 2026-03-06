# Context Management

The ECS Agent framework provides four mechanisms for managing conversation context: **Checkpoint** (undo/restore world state), **Compaction** (summarization), **Resume** (continue from saved state), and **Conversation Tree Revert** (navigate to historical conversation states).

## Checkpoint System

The `CheckpointSystem` creates snapshots of the entire world state on each tick, enabling undo operations.

### Components

- **`CheckpointComponent`**: Stores a stack of serialized world snapshots.
  - `snapshots: list[dict[str, Any]]` — Stack of world state snapshots (default: `[]`)
  - `max_snapshots: int` — Maximum snapshots to retain (default: `10`)

### Events

- `CheckpointCreatedEvent(entity_id, snapshot_index)` — Published when a new snapshot is created.
- `CheckpointRestoredEvent(entity_id, snapshot_index)` — Published when a snapshot is restored.

### Setup

```python
from ecs_agent.components import CheckpointComponent
from ecs_agent.systems.checkpoint import CheckpointSystem

world.add_component(agent, CheckpointComponent(max_snapshots=5))
world.register_system(CheckpointSystem(), priority=15)
```

### Undo Operation

```python
# Restore to the previous state
CheckpointSystem.undo(world, providers={"model": provider}, tool_handlers={"tool": handler})
```

The `undo` method pops the last snapshot, restores the world state via `WorldSerializer.from_dict()`, and preserves the remaining snapshot history.

## Compaction System

The `CompactionSystem` reduces conversation length by summarizing older messages using the entity's LLM provider.

### Components

- **`CompactionConfigComponent`**: Configures compaction thresholds and models.
  - `threshold_tokens: int` — Token count threshold triggering compaction
  - `summary_model: str` — Model identifier for summary generation

- **`ConversationArchiveComponent`**: Stores archived summaries.
  - `archived_summaries: list[str]` — Past conversation summaries (default: `[]`)

### Events

- `CompactionCompleteEvent(entity_id, removed_count, summary_length)` — Published after compaction.

### How It Works

1. The system estimates token count using `word_count * 1.3`.
2. When the estimate exceeds `threshold_tokens`, it splits messages at `bisect_ratio` (default: `0.5`).
3. The older half is summarized via the entity's LLM provider.
4. The summary is archived in `ConversationArchiveComponent`.
5. Older messages are replaced with a single summary message.

### Setup

```python
from ecs_agent.components import CompactionConfigComponent, ConversationArchiveComponent
from ecs_agent.systems.compaction import CompactionSystem

world.add_component(agent, CompactionConfigComponent(threshold_tokens=4000, summary_model="qwen-plus"))
world.add_component(agent, ConversationArchiveComponent())
world.register_system(CompactionSystem(bisect_ratio=0.5), priority=20)
```

## Resume from Checkpoint

The `Runner` supports saving and loading checkpoints for resuming execution.

### Save Checkpoint

```python
runner = Runner()
await runner.run(world, max_ticks=10)

# Save current state
runner.save_checkpoint(world, "checkpoint.json")
```

### Load and Resume

```python
# Load saved state
world, start_tick = Runner.load_checkpoint(
    "checkpoint.json",
    providers={"model": provider},
    tool_handlers={"tool": handler},
)

# Resume from where we left off
runner = Runner()
await runner.run(world, max_ticks=100, start_tick=start_tick)
```

The checkpoint includes the full world state plus `RunnerStateComponent` for tracking the tick position. When loading, `TerminalComponent` is excluded so execution can continue.

## Conversation Tree Revert

Non-destructive navigation to historical conversation states in tree-structured dialogues.

### Function

**`revert_to_message(tree: ConversationTreeComponent, target_message_id: str) -> str`**

Moves the active branch pointer to a target message without deleting historical nodes.

### Behavior

- Updates `current_branch.leaf_message_id` to `target_message_id`
- Returns target message ID for verification
- Next `ReasoningSystem.process()` call uses linearized history from reverted leaf
- All historical siblings and descendants remain in tree (non-destructive)

### Errors

- Raises `ValueError("No active branch to revert")` if `tree.current_branch_id is None`
- Raises `KeyError(f"Target message not found: {target_message_id}")` if target not in `tree.messages`

### Example

```python
from ecs_agent.conversation_tree import (
    ConversationTreeComponent,
    add_message,
    create_branch,
    switch_branch,
    revert_to_message,
    get_active_leaf,
)

tree = ConversationTreeComponent()

# Build conversation tree
msg1 = add_message(tree, role="user", content="What is 2+2?")
msg2 = add_message(tree, role="assistant", content="4", parent_id=msg1.id)
msg3 = add_message(tree, role="user", content="What is 3+3?", parent_id=msg2.id)

# Create and activate branch
create_branch(tree, "main", msg3.id)
switch_branch(tree, "main")

# ... agent generates response to "What is 3+3?" ...

# Revert to msg2 (before "What is 3+3?" question)
revert_to_message(tree, msg2.id)

# Next reasoning uses linearized history: [msg1, msg2] only
# (msg3 and subsequent responses still exist but not active)
```

### Integration with Reasoning

`ReasoningSystem` automatically checks for `ConversationTreeComponent` and uses the active branch:

1. `get_active_leaf(tree)` → current leaf message ID
2. `linearize(tree, leaf_id)` → chronological message list from root to leaf
3. Revert changes leaf pointer → next linearize() uses new path

### Use Cases

**Undo User Input**: Navigate back before a user message and try a different question:

```python
# User asked something, got response, wants to ask differently
conv = world.get_component(agent, ConversationTreeComponent)
if conv:
    # Find the message before user's last question
    target_msg_id = conv.messages["msg_before_question"].id
    revert_to_message(conv, target_msg_id)
    # Now add a different user question
```

**Compare Alternate Paths**: Save checkpoint, explore one branch, revert, explore another:

```python
# Checkpoint at decision point
decision_point = get_active_leaf(tree)

# Explore approach A
# ... generate responses ...
results_a = linearize(tree, get_active_leaf(tree))

# Revert and explore approach B
revert_to_message(tree, decision_point)
# ... generate different responses ...
results_b = linearize(tree, get_active_leaf(tree))
```

## See Also

- [Tree-Structured Conversations](tree-conversation.md) — Tree structure, branching, linearization
- [Runtime Control](runtime-control.md) — Entity registry, system lifecycle, model switching, interruption
- [Systems](../systems.md) — System execution order and lifecycle

## Complete Example

See [`examples/context_management_agent.py`](../../examples/context_management_agent.py) for a full working demo combining checkpoint, undo, and compaction.
