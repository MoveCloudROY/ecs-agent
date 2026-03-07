"""Wave planner unit tests."""

from __future__ import annotations

from string import Template

import pytest

from ecs_agent.components import TaskComponent
from ecs_agent.task.dependency_analyzer import analyze_task_dependencies
from ecs_agent.task.fetching_unit import TaskFetchingUnit
from ecs_agent.task.wave_planner import Wave, WavePlanResult, WavePlanner
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


class _MutatingRenderer:
    def __init__(self, source_snapshot: dict[str, str]) -> None:
        self._source_snapshot = source_snapshot
        self._calls = 0

    def substitute(self, template: str, snapshot: dict[str, object]) -> str:
        self._calls += 1
        if self._calls == 1:
            self._source_snapshot["name"] = "MUTATED"
        return Template(template).substitute(snapshot)


class TestTaskFetchingUnit:
    def test_generate_dispatch_requests_expands_placeholders_from_ready_waves(
        self,
    ) -> None:
        tasks = [
            TaskComponent(
                task_id="task-1",
                description="Summarize $topic",
                expected_output="Return brief for $topic",
                assigned_agent="researcher",
                tools=["web_search"],
                context_dependencies=[],
                status=TaskStatus.READY,
                priority=10,
            ),
            TaskComponent(
                task_id="task-2",
                description="Draft review for $topic",
                expected_output="Checklist for $topic",
                assigned_agent=None,
                tools=["lint"],
                context_dependencies=[],
                status=TaskStatus.READY,
                priority=5,
            ),
        ]
        unit = TaskFetchingUnit()

        custom_plan = WavePlanResult(
            waves=(Wave(wave_number=0, task_ids=("task-1", "task-2")),),
            blocked_tasks=(),
        )
        requests = unit.generate_dispatch_requests(
            wave_plan=custom_plan,
            tasks=tasks,
            snapshot={"topic": "release-notes"},
            writer_id="task_fetching_unit",
        )

        assert tuple(request.task_id for request in requests) == ("task-1", "task-2")
        assert requests[0].description == "Summarize release-notes"
        assert requests[0].expected_output == "Return brief for release-notes"
        assert requests[0].wave_number == 0
        assert requests[0].sequence_number == 0
        assert requests[1].description == "Draft review for release-notes"
        assert requests[1].expected_output == "Checklist for release-notes"
        assert requests[1].wave_number == 0
        assert requests[1].sequence_number == 1

    def test_generate_dispatch_requests_preserves_dependency_safe_wave_order(
        self,
    ) -> None:
        tasks = [
            TaskComponent(
                task_id="task-a",
                description="A",
                expected_output="A",
                assigned_agent=None,
                tools=[],
                context_dependencies=[],
                status=TaskStatus.READY,
                priority=1,
            ),
            TaskComponent(
                task_id="task-b",
                description="B",
                expected_output="B",
                assigned_agent=None,
                tools=[],
                context_dependencies=[],
                status=TaskStatus.READY,
                priority=1,
            ),
            TaskComponent(
                task_id="task-c",
                description="C",
                expected_output="C",
                assigned_agent=None,
                tools=[],
                context_dependencies=[],
                status=TaskStatus.READY,
                priority=1,
            ),
        ]
        unit = TaskFetchingUnit()

        custom_plan = WavePlanResult(
            waves=(
                Wave(wave_number=0, task_ids=("task-b", "task-a")),
                Wave(wave_number=1, task_ids=("task-c",)),
            ),
            blocked_tasks=(),
        )
        requests = unit.generate_dispatch_requests(
            wave_plan=custom_plan,
            tasks=tasks,
            snapshot={},
            writer_id="task_fetching_unit",
        )

        assert tuple(request.task_id for request in requests) == (
            "task-b",
            "task-a",
            "task-c",
        )
        assert tuple(request.sequence_number for request in requests) == (0, 1, 2)
        assert tuple(request.wave_number for request in requests) == (0, 0, 1)

    def test_single_writer_rejects_non_authoritative_state_mutation(self) -> None:
        task = TaskComponent(
            task_id="task-1",
            description="Describe",
            expected_output="Output",
            assigned_agent=None,
            tools=[],
            context_dependencies=[],
            status=TaskStatus.READY,
        )
        plan = WavePlanner().compute_waves(analyze_task_dependencies([task]))
        unit = TaskFetchingUnit(state_owner="authoritative_fetcher")

        with pytest.raises(ValueError, match="single writer"):
            unit.generate_dispatch_requests(
                wave_plan=plan,
                tasks=[task],
                snapshot={},
                writer_id="other_system",
            )

    def test_frozen_snapshot_is_reused_for_all_requests_in_wave(self) -> None:
        source_snapshot = {"name": "Alice"}
        unit = TaskFetchingUnit(renderer=_MutatingRenderer(source_snapshot))
        tasks = [
            TaskComponent(
                task_id="task-1",
                description="$name-first",
                expected_output="$name-first-output",
                assigned_agent=None,
                tools=[],
                context_dependencies=[],
                status=TaskStatus.READY,
            ),
            TaskComponent(
                task_id="task-2",
                description="$name-second",
                expected_output="$name-second-output",
                assigned_agent=None,
                tools=[],
                context_dependencies=[],
                status=TaskStatus.READY,
            ),
        ]
        plan = WavePlanner().compute_waves(analyze_task_dependencies(tasks))

        requests = unit.generate_dispatch_requests(
            wave_plan=plan,
            tasks=tasks,
            snapshot=source_snapshot,
            writer_id="task_fetching_unit",
        )

        assert source_snapshot["name"] == "MUTATED"
        assert requests[0].description == "Alice-first"
        assert requests[1].description == "Alice-second"
