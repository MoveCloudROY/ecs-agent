"""Wave planner unit tests."""

from __future__ import annotations

import pytest

from ecs_agent.components import TaskComponent
from ecs_agent.task.dependency_analyzer import analyze_task_dependencies
from ecs_agent.task.wave_planner import WavePlanner
from ecs_agent.types import TaskStatus


class TestWavePlannerHappyPath:
    """Tests for stable wave ordering in happy-path scenarios."""

    def test_wave_order_deterministic_across_runs(self) -> None:
        """Same task graph yields same wave order across repeated runs."""
        tasks = [
            TaskComponent(
                task_id="task-3",
                description="Task 3",
                expected_output="Output 3",
                assigned_agent=None,
                tools=[],
                context_dependencies=[],
                status=TaskStatus.PENDING,
                priority=5,
            ),
            TaskComponent(
                task_id="task-1",
                description="Task 1",
                expected_output="Output 1",
                assigned_agent=None,
                tools=[],
                context_dependencies=[],
                status=TaskStatus.PENDING,
                priority=10,
            ),
            TaskComponent(
                task_id="task-2",
                description="Task 2",
                expected_output="Output 2",
                assigned_agent=None,
                tools=[],
                context_dependencies=[],
                status=TaskStatus.PENDING,
                priority=10,
            ),
        ]

        # Run wave computation multiple times
        planner = WavePlanner()
        analysis = analyze_task_dependencies(tasks)

        result1 = planner.compute_waves(analysis)
        result2 = planner.compute_waves(analysis)
        result3 = planner.compute_waves(analysis)

        # All runs should produce identical results
        assert result1.waves == result2.waves
        assert result2.waves == result3.waves
        assert result1.blocked_tasks == result2.blocked_tasks

        # Verify deterministic ordering: priority DESC, task_id ASC
        # Wave 0 should have task-1, task-2 (priority 10, sorted by id), then task-3 (priority 5)
        assert len(result1.waves) == 1  # All tasks ready, single wave
        wave0 = result1.waves[0]
        assert wave0.wave_number == 0
        assert wave0.task_ids == ("task-1", "task-2", "task-3")
        assert len(result1.blocked_tasks) == 0

    def test_wave_single_task(self) -> None:
        """Single task yields single wave with one task."""
        tasks = [
            TaskComponent(
                task_id="task-only",
                description="Only task",
                expected_output="Output",
                assigned_agent=None,
                tools=[],
                context_dependencies=[],
                status=TaskStatus.PENDING,
                priority=0,
            )
        ]

        planner = WavePlanner()
        analysis = analyze_task_dependencies(tasks)
        result = planner.compute_waves(analysis)

        assert len(result.waves) == 1
        assert result.waves[0].wave_number == 0
        assert result.waves[0].task_ids == ("task-only",)
        assert len(result.blocked_tasks) == 0

    def test_wave_empty_task_list(self) -> None:
        """Empty task list yields no waves."""
        tasks: list[TaskComponent] = []

        planner = WavePlanner()
        analysis = analyze_task_dependencies(tasks)
        result = planner.compute_waves(analysis)

        assert len(result.waves) == 0
        assert len(result.blocked_tasks) == 0

    def test_wave_priority_ordering_within_wave(self) -> None:
        """Tasks within wave are ordered by (priority DESC, task_id ASC)."""
        tasks = [
            TaskComponent(
                task_id="task-z",
                description="Task Z",
                expected_output="Output Z",
                assigned_agent=None,
                tools=[],
                context_dependencies=[],
                status=TaskStatus.PENDING,
                priority=1,
            ),
            TaskComponent(
                task_id="task-a",
                description="Task A",
                expected_output="Output A",
                assigned_agent=None,
                tools=[],
                context_dependencies=[],
                status=TaskStatus.PENDING,
                priority=10,
            ),
            TaskComponent(
                task_id="task-m",
                description="Task M",
                expected_output="Output M",
                assigned_agent=None,
                tools=[],
                context_dependencies=[],
                status=TaskStatus.PENDING,
                priority=5,
            ),
            TaskComponent(
                task_id="task-b",
                description="Task B",
                expected_output="Output B",
                assigned_agent=None,
                tools=[],
                context_dependencies=[],
                status=TaskStatus.PENDING,
                priority=10,
            ),
        ]

        planner = WavePlanner()
        analysis = analyze_task_dependencies(tasks)
        result = planner.compute_waves(analysis)

        assert len(result.waves) == 1
        wave0 = result.waves[0]
        # Expected order: priority 10 (task-a, task-b), priority 5 (task-m), priority 1 (task-z)
        assert wave0.task_ids == ("task-a", "task-b", "task-m", "task-z")


class TestWavePlannerUnresolvedDependencies:
    """Tests for blocked task handling."""

    def test_unresolved_dependency_blocks_task(self) -> None:
        """Tasks with unresolved dependencies are blocked with explicit reasons."""
        tasks = [
            TaskComponent(
                task_id="task-1",
                description="Task 1",
                expected_output="Output 1",
                assigned_agent=None,
                tools=[],
                context_dependencies=[],
                status=TaskStatus.PENDING,
                priority=10,
            ),
            TaskComponent(
                task_id="task-2",
                description="Task 2",
                expected_output="Output 2",
                assigned_agent=None,
                tools=[],
                context_dependencies=["task-1"],
                status=TaskStatus.PENDING,
                priority=5,
            ),
        ]

        planner = WavePlanner()
        analysis = analyze_task_dependencies(tasks)
        result = planner.compute_waves(analysis)

        # Wave 0 has only task-1 (ready)
        assert len(result.waves) == 1
        wave0 = result.waves[0]
        assert wave0.wave_number == 0
        assert wave0.task_ids == ("task-1",)

        # task-2 is blocked
        assert len(result.blocked_tasks) == 1
        blocked = result.blocked_tasks[0]
        assert blocked.task_id == "task-2"
        assert blocked.blocking_reasons == ("waiting for dependency: task-1",)

    def test_all_blocked_tasks_yields_no_waves(self) -> None:
        """When all tasks are blocked, no waves are produced."""
        tasks = [
            TaskComponent(
                task_id="task-1",
                description="Task 1",
                expected_output="Output 1",
                assigned_agent=None,
                tools=[],
                context_dependencies=["task-2"],
                status=TaskStatus.PENDING,
                priority=10,
            ),
            TaskComponent(
                task_id="task-2",
                description="Task 2",
                expected_output="Output 2",
                assigned_agent=None,
                tools=[],
                context_dependencies=["task-1"],
                status=TaskStatus.PENDING,
                priority=5,
            ),
        ]

        # This will raise DependencyCycleError during analysis
        with pytest.raises(Exception):  # DependencyCycleError
            analyze_task_dependencies(tasks)

    def test_blocked_tasks_sorted_deterministically(self) -> None:
        """Blocked tasks are sorted by (priority DESC, task_id ASC)."""
        tasks = [
            TaskComponent(
                task_id="task-1",
                description="Task 1",
                expected_output="Output 1",
                assigned_agent=None,
                tools=[],
                context_dependencies=[],
                status=TaskStatus.COMPLETED,  # Completed dependency
                priority=10,
            ),
            TaskComponent(
                task_id="task-z",
                description="Task Z",
                expected_output="Output Z",
                assigned_agent=None,
                tools=[],
                context_dependencies=["task-1"],
                status=TaskStatus.PENDING,
                priority=1,
            ),
            TaskComponent(
                task_id="task-a",
                description="Task A",
                expected_output="Output A",
                assigned_agent=None,
                tools=[],
                context_dependencies=["task-1"],
                status=TaskStatus.PENDING,
                priority=10,
            ),
            TaskComponent(
                task_id="task-b",
                description="Task B",
                expected_output="Output B",
                assigned_agent=None,
                tools=[],
                context_dependencies=["task-1"],
                status=TaskStatus.PENDING,
                priority=10,
            ),
        ]

        # Change task-1 status to PENDING so dependents are blocked
        tasks[0] = TaskComponent(
            task_id="task-1",
            description="Task 1",
            expected_output="Output 1",
            assigned_agent=None,
            tools=[],
            context_dependencies=[],
            status=TaskStatus.PENDING,
            priority=10,
        )

        planner = WavePlanner()
        analysis = analyze_task_dependencies(tasks)
        result = planner.compute_waves(analysis)

        # Wave 0 has only task-1
        assert len(result.waves) == 1
        assert result.waves[0].task_ids == ("task-1",)

        # Blocked tasks: task-a, task-b (priority 10), task-z (priority 1)
        assert len(result.blocked_tasks) == 3
        assert result.blocked_tasks[0].task_id == "task-a"
        assert result.blocked_tasks[1].task_id == "task-b"
        assert result.blocked_tasks[2].task_id == "task-z"
