from __future__ import annotations

import pytest

from ecs_agent.components import TaskComponent
from ecs_agent.task.dependency_analyzer import (
    DependencyCycleError,
    MissingDependencyError,
    analyze_task_dependencies,
)
from ecs_agent.types import TaskStatus


def _task(
    task_id: str,
    *,
    priority: int = 0,
    context_dependencies: list[str] | None = None,
) -> TaskComponent:
    return TaskComponent(
        description=f"description for {task_id}",
        expected_output=f"output for {task_id}",
        assigned_agent=None,
        tools=[],
        context_dependencies=context_dependencies or [],
        task_id=task_id,
        status=TaskStatus.PENDING,
        priority=priority,
    )


def test_dependency_dag_classifies_ready_and_blocked_deterministically() -> None:
    tasks = [
        _task("task-a", priority=1),
        _task("task-b", priority=5),
        _task("task-c", priority=10, context_dependencies=["task-a"]),
        _task("task-d", priority=10, context_dependencies=["task-b", "task-c"]),
        _task("task-e", priority=0, context_dependencies=["task-a"]),
    ]

    result = analyze_task_dependencies(tasks)

    assert result.ready_task_ids == ("task-b", "task-a")
    assert result.blocked_task_ids == ("task-c", "task-d", "task-e")
    assert [status.task_id for status in result.ordered_statuses] == [
        "task-c",
        "task-d",
        "task-b",
        "task-a",
        "task-e",
    ]
    assert result.blocked_reasons["task-c"] == ("waiting for dependency: task-a",)
    assert result.blocked_reasons["task-d"] == (
        "waiting for dependency: task-b",
        "waiting for dependency: task-c",
    )

    order_index = {
        task_id: index for index, task_id in enumerate(result.topological_order)
    }
    assert order_index["task-a"] < order_index["task-c"]
    assert order_index["task-b"] < order_index["task-d"]
    assert order_index["task-c"] < order_index["task-d"]


def test_dependency_dag_empty_input_returns_empty_analysis() -> None:
    result = analyze_task_dependencies([])

    assert result.ready_task_ids == ()
    assert result.blocked_task_ids == ()
    assert result.topological_order == ()
    assert result.ordered_statuses == ()


def test_dependency_dag_single_task_no_dependencies_is_ready() -> None:
    result = analyze_task_dependencies([_task("task-solo", priority=3)])

    assert result.ready_task_ids == ("task-solo",)
    assert result.blocked_task_ids == ()
    assert result.topological_order == ("task-solo",)


def test_dependency_dag_raises_actionable_error_for_missing_dependency_id() -> None:
    tasks = [
        _task("task-a"),
        _task("task-b", context_dependencies=["task-missing"]),
    ]

    with pytest.raises(MissingDependencyError) as exc_info:
        analyze_task_dependencies(tasks)

    message = str(exc_info.value)
    assert "task-b" in message
    assert "task-missing" in message


def test_dependency_cycle_raises_explicit_error_with_cycle_trace() -> None:
    tasks = [
        _task("task-a", context_dependencies=["task-c"]),
        _task("task-b", context_dependencies=["task-a"]),
        _task("task-c", context_dependencies=["task-b"]),
    ]

    with pytest.raises(DependencyCycleError) as exc_info:
        analyze_task_dependencies(tasks)

    error = exc_info.value
    assert len(error.cycle_path) >= 2
    assert error.cycle_path[0] == error.cycle_path[-1]
    assert "dependency cycle detected" in str(error)


def test_dependency_cycle_message_contains_task_trace_details() -> None:
    tasks = [
        _task("task-x", context_dependencies=["task-y"]),
        _task("task-y", context_dependencies=["task-x"]),
    ]

    with pytest.raises(DependencyCycleError, match="task-x -> task-y -> task-x"):
        analyze_task_dependencies(tasks)
