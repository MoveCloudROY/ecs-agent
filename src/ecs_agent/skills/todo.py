"""Session todo list tool: ``todo_write`` handler and the ``TodoSkill`` bundle.

The list lives in :class:`~ecs_agent.components.TodoListComponent` on the
calling entity and is replaced wholesale on every ``todo_write`` call.
Visibility rides the tool result (cache-neutral conversation tail) and the
compaction context provider — todo state is never rendered into the system
prompt, stable or volatile, because volatile-suffix mutations invalidate the
prompt cache for the entire conversation history.
"""

from typing import Any

from ecs_agent.components import TodoItem, TodoListComponent
from ecs_agent.core.world import World
from ecs_agent.logging import get_logger
from ecs_agent.skills.script_skill import ScriptSkill, ToolHandler
from ecs_agent.tools.context import current_tool_context
from ecs_agent.types import TODO_STATUSES, EntityId, TodoListUpdatedEvent, ToolSchema

logger = get_logger(__name__)

_STATUS_MARKERS: dict[str, str] = {
    "completed": "[x]",
    "in_progress": "[→]",
    "pending": "[ ]",
}

TODO_WRITE_SCHEMA = ToolSchema(
    name="todo_write",
    description=(
        "Maintain your task checklist for the current session. Replaces the "
        "ENTIRE list on every call. Use for multi-step tasks (3+ steps): create "
        "the list before starting, keep exactly ONE item in_progress at a time, "
        "and mark items completed IMMEDIATELY after finishing them. The list is "
        "living: when you discover new work mid-task, add items; when an item "
        "turns out larger than expected, split it; remove items that are no "
        "longer relevant. For discoveries outside the current task's scope, "
        "surface them to the user instead of silently expanding the list. Do "
        "not use for single-step or trivial tasks."
    ),
    parameters={
        "type": "object",
        "properties": {
            "todos": {
                "type": "array",
                "description": "The complete todo list (full replacement).",
                "items": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "Imperative description of the step.",
                        },
                        "status": {
                            "type": "string",
                            "enum": ["pending", "in_progress", "completed"],
                        },
                    },
                    "required": ["content", "status"],
                },
            }
        },
        "required": ["todos"],
    },
)


def _validate_todos(todos: Any) -> str | None:
    """Structural validation only — evolution constraints would block replanning."""
    if not isinstance(todos, list):
        return "Error: todos must be a list of {content, status} objects."

    in_progress_count = 0
    for index, item in enumerate(todos):
        if not isinstance(item, dict):
            return f"Error: todos[{index}] must be a {{content, status}} object."
        content = item.get("content")
        if not isinstance(content, str) or not content.strip():
            return f"Error: todos[{index}].content must be a non-empty string."
        status = item.get("status")
        if status not in TODO_STATUSES:
            return (
                f"Error: todos[{index}].status must be one of "
                f"pending|in_progress|completed (got {status!r})."
            )
        if status == "in_progress":
            in_progress_count += 1

    if in_progress_count > 1:
        return (
            f"Error: at most one todo may be 'in_progress' (got {in_progress_count}). "
            "Resend the full list with exactly one in_progress item."
        )
    return None


def _render_result(items: list[TodoItem], lost_completed: int) -> str:
    if not items:
        lines = ["Todo list cleared."]
    else:
        completed = sum(1 for item in items if item.status == "completed")
        in_progress = sum(1 for item in items if item.status == "in_progress")
        lines = [
            f"Todo list updated ({in_progress} in progress, "
            f"{completed}/{len(items)} completed):"
        ]
        for number, item in enumerate(items, start=1):
            lines.append(f"{number}. {_STATUS_MARKERS[item.status]} {item.content}")

    if lost_completed:
        lines.append(
            f"note: {lost_completed} previously-completed item(s) removed or "
            "downgraded — confirm this was intentional."
        )
    return "\n".join(lines)


async def todo_write(todos: Any = None) -> str:
    """Replace the calling entity's todo list; returns the rendered checklist."""
    error = _validate_todos(todos)
    if error is not None:
        return error

    context = current_tool_context()
    world = context.world
    entity_id = context.entity_id

    component = world.get_component(entity_id, TodoListComponent)
    if component is None:
        component = TodoListComponent()
        world.add_component(entity_id, component)

    previously_completed = {
        item.content for item in component.items if item.status == "completed"
    }
    new_items = [
        TodoItem(content=item["content"], status=item["status"]) for item in todos
    ]
    new_status_by_content = {item.content: item.status for item in new_items}
    lost_completed = sum(
        1
        for content in previously_completed
        if new_status_by_content.get(content) != "completed"
    )

    component.items = new_items
    completed_count = sum(1 for item in new_items if item.status == "completed")
    await world.event_bus.publish(
        TodoListUpdatedEvent(
            entity_id=entity_id,
            items=[TodoItem(content=item.content, status=item.status) for item in new_items],
            completed_count=completed_count,
            total_count=len(new_items),
        )
    )
    logger.info(
        "todo_write",
        entity_id=entity_id,
        total=len(new_items),
        completed=completed_count,
        lost_completed=lost_completed,
    )
    return _render_result(component.items, lost_completed)


class TodoSkill(ScriptSkill):
    """Tool bundle exposing ``todo_write`` for session task tracking."""

    name = "todo"
    description = "Session todo list maintenance via the todo_write tool."
    is_tool_bundle = True

    def tools(self) -> dict[str, tuple[ToolSchema, ToolHandler]]:
        return {"todo_write": (TODO_WRITE_SCHEMA, todo_write)}

    def system_prompt(self) -> str:
        return ""

    def install(self, world: World, entity_id: EntityId) -> None:
        return None

    def uninstall(self, world: World, entity_id: EntityId) -> None:
        return None
