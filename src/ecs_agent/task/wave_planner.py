"""Deterministic wave planner for task dispatch."""

from __future__ import annotations

from dataclasses import dataclass

from ecs_agent.logging import get_logger
from ecs_agent.task.dependency_analyzer import (
    DependencyAnalysisResult,
    TaskDependencyStatus,
)

logger = get_logger(__name__)


@dataclass(slots=True, frozen=True)
class Wave:
    """Single wave of ready tasks that can execute in parallel."""

    wave_number: int
    task_ids: tuple[str, ...]


@dataclass(slots=True, frozen=True)
class BlockedTask:
    """Task blocked by unresolved dependencies."""

    task_id: str
    priority: int
    blocking_reasons: tuple[str, ...]


@dataclass(slots=True, frozen=True)
class WavePlanResult:
    """Result of wave planning computation."""

    waves: tuple[Wave, ...]
    blocked_tasks: tuple[BlockedTask, ...]


class WavePlanner:
    """Compute ready waves from dependency analysis with stable ordering."""

    def compute_waves(self, analysis: DependencyAnalysisResult) -> WavePlanResult:
        """Compute waves of ready tasks ordered deterministically.

        Args:
            analysis: Dependency analysis result from analyze_task_dependencies

        Returns:
            WavePlanResult with waves and blocked tasks, both sorted by
            (priority DESC, task_id ASC) for deterministic dispatch.
        """
        if not analysis.ordered_statuses:
            logger.info("wave_planner_no_tasks")
            return WavePlanResult(waves=(), blocked_tasks=())

        # Build lookup for status by task_id
        status_by_id = {status.task_id: status for status in analysis.ordered_statuses}

        # Separate ready and blocked tasks
        ready_statuses: list[TaskDependencyStatus] = []
        blocked_statuses: list[TaskDependencyStatus] = []

        for task_id in analysis.ready_task_ids:
            status = status_by_id[task_id]
            ready_statuses.append(status)

        for task_id in analysis.blocked_task_ids:
            status = status_by_id[task_id]
            blocked_statuses.append(status)

        # Sort both lists by (priority DESC, task_id ASC) for determinism
        ready_statuses_sorted = sorted(
            ready_statuses, key=lambda s: (-s.priority, s.task_id)
        )
        blocked_statuses_sorted = sorted(
            blocked_statuses, key=lambda s: (-s.priority, s.task_id)
        )

        # Build waves (for now, single wave with all ready tasks)
        waves: list[Wave] = []
        if ready_statuses_sorted:
            wave = Wave(
                wave_number=0,
                task_ids=tuple(s.task_id for s in ready_statuses_sorted),
            )
            waves.append(wave)
            logger.info(
                "wave_planner_computed_wave",
                wave_number=0,
                task_count=len(ready_statuses_sorted),
            )

        # Build blocked tasks
        blocked_tasks = tuple(
            BlockedTask(
                task_id=s.task_id,
                priority=s.priority,
                blocking_reasons=s.blocking_reasons,
            )
            for s in blocked_statuses_sorted
        )

        if blocked_tasks:
            logger.info(
                "wave_planner_blocked_tasks",
                blocked_count=len(blocked_tasks),
                blocked_task_ids=tuple(b.task_id for b in blocked_tasks),
            )

        return WavePlanResult(waves=tuple(waves), blocked_tasks=blocked_tasks)


__all__ = [
    "BlockedTask",
    "Wave",
    "WavePlanResult",
    "WavePlanner",
]
