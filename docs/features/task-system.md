# Task Orchestration System

The Task Orchestration System manages multi-step execution using a state-machine driven approach, dependency resolution, and wave-based parallel dispatch. It allows complex goals to be broken down into discrete, manageable tasks with clear success criteria and dependencies.

## Core Concepts

- **Task Status**: Six explicit states: `PENDING`, `READY`, `RUNNING`, `COMPLETED`, `FAILED`, and `BLOCKED`.
- **Dependency Resolution**: Automatic detection of unresolved dependencies and cycle prevention.
- **Wave Planning**: Grouping of ready tasks into waves for deterministic, parallel execution.
- **Mixed Backend**: Execution can be routed to either local tools or specialized subagents.
- **Dual-Path Persistence**: State is stored in both mutable snapshots and immutable event logs.

## The Task State Machine

Task transitions follow strict guardrails. Illegal transitions raise a `TaskStateTransitionError`.

| From | To | Guard |
|------|----|-------|
| `PENDING` | `READY` | - |
| `PENDING` | `BLOCKED` | Reason is required |
| `READY` | `RUNNING` | - |
| `READY` | `BLOCKED` | Reason is required |
| `RUNNING` | `COMPLETED` | - |
| `RUNNING` | `FAILED` | Reason is required |
| `RUNNING` | `BLOCKED` | Reason is required |
| `BLOCKED` | `READY` | Manual action required |
| `FAILED` | `READY` | Manual action + retries available |

## Components

### `TaskComponent`
The primary component for task definition and tracking.

| Field | Type | Description |
|-------|------|-------------|
| `task_id` | `str` | Stable identifier for the task |
| `description` | `str` | What the task does (supports placeholder rendering) |
| `expected_output` | `str` | What success looks like |
| `assigned_agent` | `EntityId \| str \| None` | EntityId (local), str (subagent), or None (default local) |
| `tools` | `list[str]` | List of tool names available for the task |
| `context_dependencies` | `list[str]` | List of scratchbook refs needed for execution |
| `status` | `TaskStatus` | Current state machine status |
| `priority` | `int` | Execution priority (higher = earlier) |
| `max_retries` | `int` | Maximum number of automatic retry attempts |

## Usage Examples

### Task Definition

Create a task that depends on a previous tool result.

```python
from ecs_agent.components import TaskComponent
from ecs_agent.types import TaskStatus

task = TaskComponent(
    task_id="analyze_report",
    description="Analyze the report at {{tool_results/report_id}}",
    expected_output="Summary of the report",
    assigned_agent="analyst_agent",
    tools=["read_file", "write_file"],
    context_dependencies=["tool_results/report_id"],
    status=TaskStatus.PENDING,
    priority=10
)
world.add_component(entity_id, task)
```

### Context Resolution

The `ContextResolver` automatically resolves `context_dependencies` from the scratchbook.

```python
from ecs_agent.task.context_resolver import ContextResolver
from ecs_agent.scratchbook import ScratchbookService

service = ScratchbookService(root=".scratchbook")
resolver = ContextResolver(service)

# Resolves all dependencies into a data dictionary
resolved = resolver.resolve_context(task)
if not isinstance(resolved, ContextResolutionError):
    print(resolved.resolved_data)
```

## Advanced Features

### Wave Dispatch
The `WavePlanner` and `TaskFetchingUnit` work together to generate parallel dispatch requests. Tasks are grouped into waves based on their dependency graph, ensuring that no task runs before its required context is ready.

### Manual Unblocking
When an upstream dependency fails, downstream tasks are moved to `BLOCKED`. These tasks require explicit manual action (`manual_unblock_task`) once the dependency is resolved.

### Dual-Path Persistence
Every task has a snapshot (its current state) and an event log (its full history). The `TaskPersistenceService` manages this to ensure high reliability and auditability.

## See Also

- [Agent Scratchbook](scratchbook.md) — The storage backend for tasks.
- [Subagent Delegation](subagent.md) — How tasks are delegated to subagents.
- [API Reference](../api-reference.md) — Signatures for `TaskExecutor`, `StateMachine`, and more.
