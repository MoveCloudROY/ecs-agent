"""Tests for the TodoList tool: component, todo_write handler, TodoSkill bundle."""

import pytest

from ecs_agent.components import (
    CompactionConfigComponent,
    ConversationComponent,
    LLMComponent,
    SkillComponent,
    TodoItem,
    TodoListComponent,
    ToolRegistryComponent,
)
from ecs_agent.core import Runner, World
from ecs_agent.prompts.compaction_context import (
    DEFAULT_COMPACTION_CONTEXT_PROVIDERS,
    TodoCompactionContextProvider,
)
from ecs_agent.providers import FakeModel
from ecs_agent.skills import SkillManager
from ecs_agent.skills.todo import TODO_WRITE_SCHEMA, TodoSkill, todo_write
from ecs_agent.systems.compaction import CompactionSystem
from ecs_agent.systems.reasoning import ReasoningSystem
from ecs_agent.systems.tool_execution import ToolExecutionSystem
from ecs_agent.tools.context import ToolExecutionContext, use_tool_context
from ecs_agent.types import (
    CompletionResult,
    EntityId,
    Message,
    TodoListUpdatedEvent,
    ToolCall,
    ToolSchema,
)

# ---------------------------------------------------------------------------
# Task 1 — component
# ---------------------------------------------------------------------------


def test_todo_item_defaults_to_pending() -> None:
    item = TodoItem(content="Split utils into io/format modules")

    assert item.content == "Split utils into io/format modules"
    assert item.status == "pending"


def test_todo_list_component_defaults_to_empty_list() -> None:
    component = TodoListComponent()

    assert component.items == []


def test_todo_list_component_instances_do_not_share_items() -> None:
    first = TodoListComponent()
    second = TodoListComponent()
    first.items.append(TodoItem(content="only in first"))

    assert second.items == []


# ---------------------------------------------------------------------------
# Task 2 — todo_write handler + schema + event
# ---------------------------------------------------------------------------

THREE_TODOS = [
    {"content": "Split utils into io/format modules", "status": "completed"},
    {"content": "Add unit tests for both new modules", "status": "in_progress"},
    {"content": "Update docs and README", "status": "pending"},
]

THREE_TODOS_RENDER = (
    "Todo list updated (1 in progress, 1/3 completed):\n"
    "1. [x] Split utils into io/format modules\n"
    "2. [→] Add unit tests for both new modules\n"
    "3. [ ] Update docs and README"
)


def _tool_context(world: World, entity_id: EntityId) -> ToolExecutionContext:
    return ToolExecutionContext(
        world=world,
        entity_id=entity_id,
        tool_name="todo_write",
        tool_call_id="call_1",
    )


def test_todo_write_schema_shape() -> None:
    assert TODO_WRITE_SCHEMA.name == "todo_write"
    assert TODO_WRITE_SCHEMA.sandbox_compatible is False
    assert TODO_WRITE_SCHEMA.parameters["required"] == ["todos"]
    items_schema = TODO_WRITE_SCHEMA.parameters["properties"]["todos"]["items"]
    assert items_schema["required"] == ["content", "status"]
    assert items_schema["properties"]["status"]["enum"] == [
        "pending",
        "in_progress",
        "completed",
    ]


async def test_todo_write_creates_component_and_renders_checklist() -> None:
    world = World()
    entity = world.create_entity()

    with use_tool_context(_tool_context(world, entity)):
        result = await todo_write(todos=THREE_TODOS)

    assert result == THREE_TODOS_RENDER
    component = world.get_component(entity, TodoListComponent)
    assert component is not None
    assert [(item.content, item.status) for item in component.items] == [
        ("Split utils into io/format modules", "completed"),
        ("Add unit tests for both new modules", "in_progress"),
        ("Update docs and README", "pending"),
    ]


async def test_todo_write_replaces_entire_list() -> None:
    world = World()
    entity = world.create_entity()

    with use_tool_context(_tool_context(world, entity)):
        await todo_write(todos=THREE_TODOS)
        await todo_write(
            todos=[{"content": "A brand new plan", "status": "in_progress"}]
        )

    component = world.get_component(entity, TodoListComponent)
    assert component is not None
    assert [(item.content, item.status) for item in component.items] == [
        ("A brand new plan", "in_progress")
    ]


async def test_todo_write_rejects_non_list_without_touching_state() -> None:
    world = World()
    entity = world.create_entity()

    with use_tool_context(_tool_context(world, entity)):
        result = await todo_write(todos="not a list")

    assert result == "Error: todos must be a list of {content, status} objects."
    assert world.get_component(entity, TodoListComponent) is None


async def test_todo_write_rejects_non_object_item() -> None:
    world = World()
    entity = world.create_entity()

    with use_tool_context(_tool_context(world, entity)):
        result = await todo_write(todos=["just a string"])

    assert result == "Error: todos[0] must be a {content, status} object."


async def test_todo_write_rejects_empty_content() -> None:
    world = World()
    entity = world.create_entity()

    with use_tool_context(_tool_context(world, entity)):
        result = await todo_write(todos=[{"content": "   ", "status": "pending"}])

    assert result == "Error: todos[0].content must be a non-empty string."


async def test_todo_write_rejects_unknown_status() -> None:
    world = World()
    entity = world.create_entity()

    with use_tool_context(_tool_context(world, entity)):
        result = await todo_write(
            todos=[
                {"content": "ok item", "status": "pending"},
                {"content": "bad item", "status": "done"},
            ]
        )

    assert result == (
        "Error: todos[1].status must be one of pending|in_progress|completed "
        "(got 'done')."
    )


async def test_todo_write_rejects_multiple_in_progress() -> None:
    world = World()
    entity = world.create_entity()

    with use_tool_context(_tool_context(world, entity)):
        result = await todo_write(
            todos=[
                {"content": "first", "status": "in_progress"},
                {"content": "second", "status": "in_progress"},
            ]
        )

    assert result == (
        "Error: at most one todo may be 'in_progress' (got 2). "
        "Resend the full list with exactly one in_progress item."
    )


async def test_todo_write_missing_todos_argument_is_validation_error() -> None:
    world = World()
    entity = world.create_entity()

    with use_tool_context(_tool_context(world, entity)):
        result = await todo_write()

    assert result == "Error: todos must be a list of {content, status} objects."


async def test_todo_write_empty_list_clears() -> None:
    world = World()
    entity = world.create_entity()

    with use_tool_context(_tool_context(world, entity)):
        await todo_write(todos=[{"content": "pending item", "status": "pending"}])
        result = await todo_write(todos=[])

    assert result == "Todo list cleared."
    component = world.get_component(entity, TodoListComponent)
    assert component is not None
    assert component.items == []


async def test_todo_write_notes_completed_item_removal() -> None:
    world = World()
    entity = world.create_entity()

    with use_tool_context(_tool_context(world, entity)):
        await todo_write(todos=THREE_TODOS)
        result = await todo_write(
            todos=[{"content": "Update docs and README", "status": "in_progress"}]
        )

    assert result.endswith(
        "note: 1 previously-completed item(s) removed or downgraded "
        "— confirm this was intentional."
    )


async def test_todo_write_notes_completed_item_downgrade() -> None:
    world = World()
    entity = world.create_entity()

    with use_tool_context(_tool_context(world, entity)):
        await todo_write(todos=THREE_TODOS)
        result = await todo_write(
            todos=[
                {
                    "content": "Split utils into io/format modules",
                    "status": "pending",
                }
            ]
        )

    assert "note: 1 previously-completed item(s) removed or downgraded" in result


async def test_todo_write_no_note_when_completed_items_preserved() -> None:
    world = World()
    entity = world.create_entity()

    with use_tool_context(_tool_context(world, entity)):
        await todo_write(todos=THREE_TODOS)
        result = await todo_write(
            todos=THREE_TODOS
            + [{"content": "Newly discovered work", "status": "pending"}]
        )

    assert "note:" not in result


async def test_todo_write_publishes_event_with_snapshot_copy() -> None:
    world = World()
    entity = world.create_entity()
    events: list[TodoListUpdatedEvent] = []

    async def collect(event: TodoListUpdatedEvent) -> None:
        events.append(event)

    world.event_bus.subscribe(TodoListUpdatedEvent, collect)

    with use_tool_context(_tool_context(world, entity)):
        await todo_write(todos=THREE_TODOS)

    assert len(events) == 1
    event = events[0]
    assert event.entity_id == entity
    assert event.completed_count == 1
    assert event.total_count == 3

    component = world.get_component(entity, TodoListComponent)
    assert component is not None
    component.items[0].status = "pending"
    assert event.items[0].status == "completed"


async def test_todo_write_no_event_on_validation_failure() -> None:
    world = World()
    entity = world.create_entity()
    events: list[TodoListUpdatedEvent] = []

    async def collect(event: TodoListUpdatedEvent) -> None:
        events.append(event)

    world.event_bus.subscribe(TodoListUpdatedEvent, collect)

    with use_tool_context(_tool_context(world, entity)):
        await todo_write(todos="not a list")

    assert events == []


async def test_todo_write_outside_tool_context_raises() -> None:
    with pytest.raises(RuntimeError):
        await todo_write(todos=THREE_TODOS)


# ---------------------------------------------------------------------------
# Task 3 — TodoSkill bundle + exports
# ---------------------------------------------------------------------------


def test_todo_skill_install_registers_tool_without_skill_component() -> None:
    world = World()
    entity = world.create_entity()

    SkillManager().install(world, entity, TodoSkill())

    registry = world.get_component(entity, ToolRegistryComponent)
    assert registry is not None
    assert "todo_write" in registry.tools
    assert "todo_write" in registry.handlers

    skill_component = world.get_component(entity, SkillComponent)
    assert skill_component is None or "todo" not in skill_component.skills


def test_todo_skill_uninstall_removes_tool() -> None:
    world = World()
    entity = world.create_entity()
    manager = SkillManager()
    manager.install(world, entity, TodoSkill())

    manager.uninstall(world, entity, "todo")

    registry = world.get_component(entity, ToolRegistryComponent)
    assert registry is not None
    assert "todo_write" not in registry.tools
    assert "todo_write" not in registry.handlers


def test_todo_skill_double_install_raises_collision() -> None:
    world = World()
    entity = world.create_entity()
    manager = SkillManager()
    manager.install(world, entity, TodoSkill())

    with pytest.raises(ValueError, match="todo_write"):
        manager.install(world, entity, TodoSkill())


async def test_todo_lists_are_isolated_per_entity() -> None:
    world = World()
    first = world.create_entity()
    second = world.create_entity()
    manager = SkillManager()
    manager.install(world, first, TodoSkill())
    manager.install(world, second, TodoSkill())

    with use_tool_context(_tool_context(world, first)):
        await todo_write(todos=[{"content": "first's task", "status": "pending"}])
    with use_tool_context(_tool_context(world, second)):
        await todo_write(todos=[{"content": "second's task", "status": "completed"}])

    first_component = world.get_component(first, TodoListComponent)
    second_component = world.get_component(second, TodoListComponent)
    assert first_component is not None and second_component is not None
    assert [item.content for item in first_component.items] == ["first's task"]
    assert [item.content for item in second_component.items] == ["second's task"]


def test_top_level_exports() -> None:
    import ecs_agent

    assert ecs_agent.TodoSkill is TodoSkill
    assert ecs_agent.TodoItem is TodoItem
    assert ecs_agent.TodoListComponent is TodoListComponent
    assert ecs_agent.TodoListUpdatedEvent is TodoListUpdatedEvent


# ---------------------------------------------------------------------------
# Task 4 — TodoCompactionContextProvider
# ---------------------------------------------------------------------------


def test_todo_compaction_provider_returns_none_without_component() -> None:
    world = World()
    entity = world.create_entity()

    provider = TodoCompactionContextProvider()

    assert provider.render_compaction_context(world, entity) is None


def test_todo_compaction_provider_returns_none_when_list_is_empty() -> None:
    world = World()
    entity = world.create_entity()
    world.add_component(entity, TodoListComponent())

    provider = TodoCompactionContextProvider()

    assert provider.render_compaction_context(world, entity) is None


def test_todo_compaction_provider_renders_snapshot_block() -> None:
    world = World()
    entity = world.create_entity()
    world.add_component(
        entity,
        TodoListComponent(
            items=[
                TodoItem(
                    content="Split utils into io/format modules", status="completed"
                ),
                TodoItem(
                    content="Add unit tests for both new modules",
                    status="in_progress",
                ),
            ]
        ),
    )

    block = TodoCompactionContextProvider().render_compaction_context(world, entity)

    assert block == (
        "Current todo list (carry forward verbatim):\n"
        "- [completed] Split utils into io/format modules\n"
        "- [in_progress] Add unit tests for both new modules"
    )


def test_todo_compaction_provider_is_in_default_set() -> None:
    provider_ids = [
        provider.provider_id for provider in DEFAULT_COMPACTION_CONTEXT_PROVIDERS
    ]

    assert "todo_list" in provider_ids


class _RecordingModel(FakeModel):
    def __init__(self, responses: list[CompletionResult]) -> None:
        super().__init__(responses=responses)
        self.calls: list[list[Message]] = []

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        stream: bool = False,
        response_format: dict[str, object] | None = None,
    ) -> CompletionResult:
        _ = response_format
        self.calls.append(list(messages))
        result = await super().complete(messages, tools=tools, stream=stream)
        assert isinstance(result, CompletionResult)
        return result


async def test_compaction_summary_input_includes_todo_snapshot() -> None:
    world = World()
    model = _RecordingModel(
        responses=[
            CompletionResult(message=Message(role="assistant", content="summary"))
        ]
    )
    entity = world.create_entity()
    world.add_component(entity, LLMComponent(model=model))
    world.add_component(
        entity,
        ConversationComponent(
            messages=[
                Message(role="user", content="earlier conversation to compact"),
                Message(role="user", content="recent user turn"),
            ]
        ),
    )
    world.add_component(
        entity,
        CompactionConfigComponent(
            threshold_tokens=1, compaction_method="full_history"
        ),
    )
    world.add_component(
        entity,
        TodoListComponent(
            items=[
                TodoItem(
                    content="Split utils into io/format modules", status="completed"
                ),
                TodoItem(
                    content="Add unit tests for both new modules",
                    status="in_progress",
                ),
            ]
        ),
    )

    await CompactionSystem().process(world)

    summary_input = model.calls[0][1].content
    assert isinstance(summary_input, str)
    assert "Current todo list (carry forward verbatim):" in summary_input
    assert "- [completed] Split utils into io/format modules" in summary_input
    assert "- [in_progress] Add unit tests for both new modules" in summary_input


# ---------------------------------------------------------------------------
# Task 5 — end-to-end agent loop (FakeModel)
# ---------------------------------------------------------------------------


async def test_todo_e2e_agent_loop_with_midtask_replan() -> None:
    world = World()
    agent = world.create_entity()

    responses = [
        CompletionResult(
            message=Message(
                role="assistant",
                content="Planning the work.",
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        name="todo_write",
                        arguments={
                            "todos": [
                                {"content": "Read the config", "status": "in_progress"},
                                {"content": "Apply the fix", "status": "pending"},
                                {"content": "Verify with tests", "status": "pending"},
                            ]
                        },
                    )
                ],
            )
        ),
        CompletionResult(
            message=Message(
                role="assistant",
                content="Discovered extra work while reading.",
                tool_calls=[
                    ToolCall(
                        id="call_2",
                        name="todo_write",
                        arguments={
                            "todos": [
                                {"content": "Read the config", "status": "completed"},
                                {"content": "Apply the fix", "status": "in_progress"},
                                {"content": "Verify with tests", "status": "pending"},
                                {
                                    "content": "Fix discovered typo in docs",
                                    "status": "pending",
                                },
                            ]
                        },
                    )
                ],
            )
        ),
        CompletionResult(
            message=Message(
                role="assistant",
                content="Everything is finished.",
                tool_calls=[
                    ToolCall(
                        id="call_3",
                        name="todo_write",
                        arguments={
                            "todos": [
                                {"content": "Read the config", "status": "completed"},
                                {"content": "Apply the fix", "status": "completed"},
                                {"content": "Verify with tests", "status": "completed"},
                                {
                                    "content": "Fix discovered typo in docs",
                                    "status": "completed",
                                },
                            ]
                        },
                    )
                ],
            )
        ),
        CompletionResult(message=Message(role="assistant", content="All done.")),
    ]

    world.add_component(agent, LLMComponent(model=FakeModel(responses=responses)))
    world.add_component(
        agent,
        ConversationComponent(
            messages=[Message(role="user", content="Fix the config bug end to end.")]
        ),
    )
    SkillManager().install(world, agent, TodoSkill())

    events: list[TodoListUpdatedEvent] = []

    async def collect(event: TodoListUpdatedEvent) -> None:
        events.append(event)

    world.event_bus.subscribe(TodoListUpdatedEvent, collect)

    world.register_system(ReasoningSystem(priority=0), priority=0)
    world.register_system(ToolExecutionSystem(priority=5), priority=5)
    await Runner().run(world, max_ticks=12)

    component = world.get_component(agent, TodoListComponent)
    assert component is not None
    assert [item.status for item in component.items] == ["completed"] * 4

    # Full-snapshot event stream: totals grow on mid-task discovery (3 → 4),
    # so fractional progress may legitimately regress.
    assert [(event.completed_count, event.total_count) for event in events] == [
        (0, 3),
        (1, 4),
        (4, 4),
    ]

    conversation = world.get_component(agent, ConversationComponent)
    assert conversation is not None
    tool_messages = [
        message for message in conversation.messages if message.role == "tool"
    ]
    assert len(tool_messages) == 3
    assert isinstance(tool_messages[0].content, str)
    assert tool_messages[0].content.startswith(
        "Todo list updated (1 in progress, 0/3 completed):"
    )
