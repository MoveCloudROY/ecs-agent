from __future__ import annotations

from dataclasses import dataclass
from graphlib import CycleError, TopologicalSorter

from ecs_agent.components import TaskComponent
from ecs_agent.logging import get_logger
from ecs_agent.types import TaskStatus

logger = get_logger(__name__)


class TaskDependencyAnalysisError(ValueError):
    pass


class MissingDependencyError(TaskDependencyAnalysisError):
    pass


class DependencyCycleError(TaskDependencyAnalysisError):
    def __init__(self, message: str, *, cycle_path: tuple[str, ...]) -> None:
        super().__init__(message)
        self.cycle_path = cycle_path


@dataclass(slots=True, frozen=True)
class TaskDependencyStatus:
    task_id: str
    priority: int
    dependencies: tuple[str, ...]
    is_ready: bool
    blocking_reasons: tuple[str, ...]


@dataclass(slots=True, frozen=True)
class DependencyAnalysisResult:
    ordered_statuses: tuple[TaskDependencyStatus, ...]
    topological_order: tuple[str, ...]
    ready_task_ids: tuple[str, ...]
    blocked_task_ids: tuple[str, ...]
    blocked_reasons: dict[str, tuple[str, ...]]


def analyze_task_dependencies(tasks: list[TaskComponent]) -> DependencyAnalysisResult:
    if not tasks:
        return DependencyAnalysisResult(
            ordered_statuses=(),
            topological_order=(),
            ready_task_ids=(),
            blocked_task_ids=(),
            blocked_reasons={},
        )

    tasks_by_id = _build_task_index(tasks)
    graph = _build_graph(tasks_by_id)
    _validate_dependencies_exist(tasks_by_id=tasks_by_id)
    topological_order = _compute_topological_order(graph)

    ordered_tasks = sorted(
        tasks_by_id.values(),
        key=lambda task: (-task.priority, task.task_id),
    )

    statuses: list[TaskDependencyStatus] = []
    ready_task_ids: list[str] = []
    blocked_task_ids: list[str] = []
    blocked_reasons: dict[str, tuple[str, ...]] = {}

    for task in ordered_tasks:
        unresolved_dependencies = tuple(
            dependency_id
            for dependency_id in sorted(set(task.context_dependencies))
            if tasks_by_id[dependency_id].status is not TaskStatus.COMPLETED
        )
        reasons = tuple(
            f"waiting for dependency: {dependency_id}"
            for dependency_id in unresolved_dependencies
        )
        is_ready = len(reasons) == 0

        if is_ready:
            ready_task_ids.append(task.task_id)
        else:
            blocked_task_ids.append(task.task_id)
            blocked_reasons[task.task_id] = reasons

        statuses.append(
            TaskDependencyStatus(
                task_id=task.task_id,
                priority=task.priority,
                dependencies=tuple(sorted(set(task.context_dependencies))),
                is_ready=is_ready,
                blocking_reasons=reasons,
            )
        )

    return DependencyAnalysisResult(
        ordered_statuses=tuple(statuses),
        topological_order=topological_order,
        ready_task_ids=tuple(ready_task_ids),
        blocked_task_ids=tuple(blocked_task_ids),
        blocked_reasons=blocked_reasons,
    )


def _build_task_index(tasks: list[TaskComponent]) -> dict[str, TaskComponent]:
    tasks_by_id: dict[str, TaskComponent] = {}

    for task in tasks:
        existing = tasks_by_id.get(task.task_id)
        if existing is not None:
            message = f"duplicate task id detected: {task.task_id}"
            logger.error(
                "task_dependency_duplicate_task_id",
                task_id=task.task_id,
            )
            raise TaskDependencyAnalysisError(message)
        tasks_by_id[task.task_id] = task

    return tasks_by_id


def _build_graph(tasks_by_id: dict[str, TaskComponent]) -> dict[str, tuple[str, ...]]:
    return {
        task_id: tuple(sorted(set(tasks_by_id[task_id].context_dependencies)))
        for task_id in sorted(tasks_by_id)
    }


def _validate_dependencies_exist(*, tasks_by_id: dict[str, TaskComponent]) -> None:
    known_ids = set(tasks_by_id)

    for task in sorted(tasks_by_id.values(), key=lambda item: item.task_id):
        for dependency_id in sorted(set(task.context_dependencies)):
            if dependency_id in known_ids:
                continue
            message = (
                "missing dependency id "
                f"'{dependency_id}' referenced by task '{task.task_id}'"
            )
            logger.error(
                "task_dependency_missing_dependency_id",
                task_id=task.task_id,
                dependency_id=dependency_id,
            )
            raise MissingDependencyError(message)


def _compute_topological_order(graph: dict[str, tuple[str, ...]]) -> tuple[str, ...]:
    try:
        sorter = TopologicalSorter(graph)
        return tuple(sorter.static_order())
    except CycleError as exc:
        cycle_nodes = (
            tuple(str(node) for node in exc.args[1]) if len(exc.args) > 1 else ()
        )
        cycle_trace = " -> ".join(cycle_nodes) if cycle_nodes else "unknown cycle"
        message = f"dependency cycle detected: {cycle_trace}"
        logger.error(
            "task_dependency_cycle_detected",
            cycle_path=cycle_nodes,
            exception=str(exc),
        )
        raise DependencyCycleError(message, cycle_path=cycle_nodes) from exc


__all__ = [
    "DependencyAnalysisResult",
    "DependencyCycleError",
    "MissingDependencyError",
    "TaskDependencyAnalysisError",
    "TaskDependencyStatus",
    "analyze_task_dependencies",
]
