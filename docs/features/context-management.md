# Context Management

The ECS Agent framework provides three mechanisms for managing conversation context: **Checkpoint** (undo/restore), **Compaction** (summarization), and **Resume** (continue from saved state).

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

## Complete Example

See [`examples/context_management_agent.py`](../../examples/context_management_agent.py) for a full working demo combining checkpoint, undo, and compaction.
