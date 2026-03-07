# Agent Scratchbook

The Agent Scratchbook is a persistent filesystem-backed storage system for agent artifacts. It provides a structured way to store, categorize, and retrieve data produced during agent execution, such as tool results, planning snapshots, and task metadata.

## Overview

Agents often need to persist data that outlives a single tick or even a single session. The Scratchbook system provides:

- **Categorized Storage**: Artifacts are organized into subfolders (e.g., `tool_results/`, `planning/`, `tasks/`).
- **Atomic Operations**: Index updates use a temp-file + rename pattern to ensure consistency even during crashes.
- **Stable References**: Metadata-rich references (content hash, timestamp) for deterministic lookup.
- **Persistence Guarantees**: Snapshots and event logs are stored as plain JSON/JSONL for easy inspection and recovery.

## Architecture

The scratchbook system consists of three main parts.

1.  **ScratchbookService**: Low-level filesystem operations (read, write, append).
2.  **ScratchbookIndexer**: Maintains a metadata index for fast, stable artifact lookup.
3.  **ToolResultsSink**: Automatically persists tool execution results to the scratchbook.

## Components

### `ScratchbookRefComponent`
Stores a reference to a specific scratchbook artifact.

| Field | Type | Description |
|-------|------|-------------|
| `artifact_id` | `str` | Unique identifier for the artifact file |
| `category` | `str` | Category subfolder path |
| `content_hash` | `str` | SHA256 hash of the artifact content |
| `timestamp` | `str` | Creation timestamp |

### `ScratchbookIndexComponent`
Maintains an in-memory index of available artifacts.

| Field | Type | Description |
|-------|------|-------------|
| `artifacts` | `dict[str, ScratchbookRef]` | Mapping of artifact IDs to their metadata references |

## Usage Examples

### Basic Artifact Operations

```python
from pathlib import Path
from ecs_agent.scratchbook import ScratchbookService

service = ScratchbookService(root=".scratchbook")

# Write a categorized artifact
data = {"plan": ["step 1", "step 2"], "goal": "complete task"}
service.write_artifact(artifact_id="plan_001", category="planning", data=data)

# Read it back
loaded = service.read_artifact(artifact_id="plan_001", category="planning")
print(loaded["goal"])  # complete task

# List all planning artifacts
ids = service.list_artifacts(category="planning")
```

### Atomic Index Updates

The `ScratchbookService` provides `write_index` which uses an atomic replacement pattern.

```python
index_data = {
    "entries": [
        {"id": "task_1", "status": "completed"},
        {"id": "task_2", "status": "running"}
    ]
}
# This write is crash-safe
service.write_index("task_index.json", category="tasks", data=index_data)
```

## Best Practices

- **Use Categories**: Always group related artifacts into meaningful categories to keep the filesystem organized.
- **Content Hashes**: Use the `content_hash` in `ScratchbookRef` to verify data integrity when loading artifacts.
- **Atomic Writes**: Prefer `write_index` for metadata or index files that are frequently updated.
- **Human Readable**: Artifacts are stored as indented JSON, making them easy to debug via standard CLI tools.

## See Also

- [Task Orchestration System](task-system.md) — Uses Scratchbook for state persistence.
- [Context Management](context-management.md) — Checkpoint and undo capabilities.
- [API Reference](../api-reference.md) — Detailed signatures for `ScratchbookService`.
