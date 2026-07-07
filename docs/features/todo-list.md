# Todo List

The `TodoSkill` gives an agent a session-scoped todo list it maintains itself
through a single `todo_write` tool. Multi-step tasks become locally driven:
the agent plans a checklist up front, keeps exactly one item `in_progress`,
marks items `completed` as it finishes them, and folds newly discovered work
into the list as it goes. Host code observes progress through a component and
an event — no new systems are involved.

## When to Use

- Multi-step tasks (3+ steps) where you want the agent to plan before acting
  and drive itself item by item.
- Host-side progress reporting (progress bars, dashboards, stall detection).
- Long sessions with compaction, where the current plan must survive history
  summarization.

Not for single-step or trivial tasks — the tool description tells the model to
skip the list in those cases.

## Installation

```python
from ecs_agent import SkillManager, TodoSkill

SkillManager().install(world, agent, TodoSkill())
```

`TodoSkill` is a tool bundle (`is_tool_bundle = True`), like
`BuiltinToolsSkill`: it registers `todo_write` on `ToolRegistryComponent`, is
not listed in `SkillComponent`, and injects no system-prompt fragment. It
composes with any other tool set. Manual registration works too:

```python
from ecs_agent.components import ToolRegistryComponent
from ecs_agent.skills.todo import TODO_WRITE_SCHEMA, todo_write

world.add_component(
    agent,
    ToolRegistryComponent(
        tools={"todo_write": TODO_WRITE_SCHEMA},
        handlers={"todo_write": todo_write},
    ),
)
```

## The `todo_write` Tool

`todo_write` replaces the **entire** list on every call. Full replacement
forces the model to re-read and re-assess its whole plan each time, which is
how mid-task discoveries get folded in — there are no item IDs and no
incremental add/complete/update tools.

Parameters:

```json
{
  "todos": [
    {"content": "Split utils into io/format modules", "status": "completed"},
    {"content": "Add unit tests for both new modules", "status": "in_progress"},
    {"content": "Update docs and README", "status": "pending"}
  ]
}
```

Result (echoed to the model as the tool result):

```
Todo list updated (1 in progress, 1/3 completed):
1. [x] Split utils into io/format modules
2. [→] Add unit tests for both new modules
3. [ ] Update docs and README
```

An empty `todos` array clears the list (`Todo list cleared.`).

### Validation

Validation is structural only — evolution constraints would block legitimate
replanning:

- `todos` must be a list of `{content, status}` objects;
- `content` must be a non-empty string;
- `status` must be `pending`, `in_progress`, or `completed`;
- at most **one** item may be `in_progress`.

Violations return an `Error: ...` string (mapped to `success=False` by
`ToolExecutionSystem`); the write is skipped and no event is published. Any
structurally valid rewrite is accepted: items may be added, split, removed, or
pushed back to `pending` at any time, so totals can grow and fractional
progress can legitimately regress (4/5 → 4/6).

### Soft warning on completed-item loss

When a previously-`completed` item disappears or is downgraded in the new list
(a known full-replacement retyping mistake), the write still succeeds and one
note line is appended so the model can self-check next turn:

```
note: 1 previously-completed item(s) removed or downgraded — confirm this was intentional.
```

## Reading State from Host Code

The list lives in `TodoListComponent` on the entity, lazily created by the
first `todo_write` call (hosts may also pre-seed it):

```python
from ecs_agent.components import TodoListComponent

todo = world.get_component(agent, TodoListComponent)
if todo is not None:
    done = sum(1 for item in todo.items if item.status == "completed")
    print(f"progress {done}/{len(todo.items)}")
```

Lists are per-entity: subagents installing `TodoSkill` get independent lists
with no extra configuration.

## Observing Updates

Every successful write publishes a `TodoListUpdatedEvent`:

```python
from ecs_agent.types import TodoListUpdatedEvent

async def on_todo_updated(event: TodoListUpdatedEvent) -> None:
    print(f"[entity {event.entity_id}] {event.completed_count}/{event.total_count}")

world.event_bus.subscribe(TodoListUpdatedEvent, on_todo_updated)
```

`event.items` is a snapshot copy, not the live list. Treat the stream as full
snapshots: counts may go up or down across events, so re-render the whole
table rather than assuming monotonic append.

## Compaction Behavior

`TodoCompactionContextProvider` is part of
`DEFAULT_COMPACTION_CONTEXT_PROVIDERS`: when `CompactionSystem` summarizes the
conversation, the current list snapshot is contributed to the summarization
request so the plan survives compaction verbatim. No configuration is needed.

## Design Rule: Never in the System Prompt

Todo state must **not** be rendered into the system prompt — neither the
stable prefix nor the volatile suffix. The request layout places both system
blocks before the conversation messages, and prompt caching is strict prefix
matching: any change to a system block invalidates the cached conversation
history behind it. The todo list is the highest-frequency mutable state in the
framework, so a system-prompt placeholder would defeat conversation caching
entirely. Visibility rides two cache-neutral channels instead: the
`todo_write` tool result (appended at the conversation tail) and the
compaction context provider (compaction is already a full cache reset).

## Permissions

`todo_write` is a pure in-memory state write with no side effects. When using
approval flows (`ToolApprovalComponent`) or permission filtering, add
`todo_write` to the allowlist so checklist maintenance never stalls on
approval prompts.

## Serialization

`TodoListComponent` round-trips through `WorldSerializer.save`/`load`; items
are restored as `TodoItem` instances. A resumed agent continues from the
persisted checklist.

## Example

See [`examples/todo_agent.py`](../../examples/todo_agent.py) — a dual-mode
demo (FakeModel or real provider) of plan → mid-task discovery → completion,
with host-side event observation.
