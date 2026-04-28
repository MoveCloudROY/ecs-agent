"""Task execution helpers for the plan-and-task E2E example."""

from __future__ import annotations

import datetime
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ecs_agent.logging import get_logger

from examples.e2e.plan_and_task.scratchbook_adapter import (
    PlanTaskScratchbookAdapter as ArtifactAdapter,
)
from examples.e2e.plan_and_task.plan_schema import (
    PlanTask,
    WorkflowPlan,
    parse_plan,
    validate_plan,
)
from examples.e2e.plan_and_task.state_models import (
    RuntimeState,
    SubagentRecord,
    TaskRecord,
)
from examples.e2e.plan_and_task.state_machine import (
    WorkflowStateMachine,
    _COMPILED_WORKFLOW,
)

logger = get_logger(__name__)

_PLAN_FILE_NAME = "workflow_plan.md"
_TASK_QUEUE_FILE_NAME = "task_queue.json"
_MEMORY_FILE_NAME = "knowledge.jsonl"


@dataclass(slots=True)
class TaskExec:
    """Manages task queue building, subagent dispatch tracking, and task completion."""

    state: RuntimeState

    def load_plan(self, adapter: ArtifactAdapter) -> WorkflowPlan:
        """Load and validate the finalized workflow plan from the artifact adapter."""
        self._require_approved_reviews()
        plan_path = adapter.plan_dir / _PLAN_FILE_NAME
        if not plan_path.exists():
            raise ValueError(
                f"Missing workflow plan file: {adapter._relative_path(plan_path)}"
            )

        content = plan_path.read_text(encoding="utf-8")
        plan = parse_plan(content)
        validate_plan(plan)
        if plan.workflow_id != self.state.workflow_id:
            raise ValueError("Workflow plan workflow_id does not match runtime state")

        logger.info(
            "plan_task_plan_loaded",
            workflow_id=plan.workflow_id,
            task_count=len(plan.tasks),
        )
        return plan

    def build_todo_queue(self, plan: WorkflowPlan) -> list[TaskRecord]:
        """Build a dependency-resolved queue of task records from the workflow plan."""
        tasks_by_id = {task.task_id: task for task in plan.tasks}
        remaining_dependencies = {
            task.task_id: set(task.dependencies) for task in plan.tasks
        }
        queue: list[TaskRecord] = []
        resolved: set[str] = set()

        for task in plan.tasks:
            for dependency in task.dependencies:
                if dependency not in tasks_by_id:
                    raise ValueError(
                        f"Task '{task.task_id}' depends on unknown task '{dependency}'"
                    )

        while remaining_dependencies:
            ready_task_ids = sorted(
                task_id
                for task_id, dependencies in remaining_dependencies.items()
                if dependencies <= resolved
            )
            if not ready_task_ids:
                cycle = ", ".join(sorted(remaining_dependencies))
                logger.warning(
                    "plan_task_dependency_cycle_detected",
                    workflow_id=self.state.workflow_id,
                    cycle_ids=sorted(remaining_dependencies),
                )
                raise ValueError(f"Task queue contains cyclic dependencies: {cycle}")

            for task_id in ready_task_ids:
                queue.append(self.normalize_task(tasks_by_id[task_id]))
                resolved.add(task_id)
                del remaining_dependencies[task_id]

        return queue

    def normalize_task(self, plan_task: PlanTask) -> TaskRecord:
        """Convert a PlanTask to a TaskRecord with execution status initialized."""
        if not plan_task.acceptance_criteria:
            raise ValueError(
                f"Task '{plan_task.task_id}' must define non-empty acceptance_criteria"
            )

        return TaskRecord(
            task_id=plan_task.task_id,
            title=plan_task.title,
            description=plan_task.description,
            dependencies=list(plan_task.dependencies),
            acceptance_criteria=list(plan_task.acceptance_criteria),
            execution_hints=list(plan_task.execution_hints),
            status="pending",
        )

    def initialize_task_queue(
        self, state: RuntimeState, adapter: ArtifactAdapter
    ) -> RuntimeState:
        """Load the plan and build the todo queue, then update state to TASK_RUNNING."""
        _ALLOWED_PHASES = {"PLAN_FINALIZED", "TASK_READY", "TASK_RUNNING", "TASK_BLOCKED"}
        if state.phase not in _ALLOWED_PHASES:
            raise ValueError(
                f"Cannot initialize task queue from phase {state.phase!r}; "
                f"workflow must be in one of: {sorted(_ALLOWED_PHASES)}"
            )
        plan = self.load_plan(adapter)
        queue = self.build_todo_queue(plan)
        current_task_id = queue[0].task_id if queue else None

        state.tasks = queue
        state.current_task_id = current_task_id
        state = self._transition_to_running(state)
        state.status = "active"
        adapter.write_state(state)
        self._write_task_queue_artifact(
            adapter.state_dir / _TASK_QUEUE_FILE_NAME, queue
        )
        logger.info(
            "plan_task_task_queue_initialized",
            workflow_id=state.workflow_id,
            task_count=len(queue),
            current_task_id=current_task_id,
            phase=state.phase,
        )
        return state

    def assemble_execution_context(
        self,
        state: RuntimeState,
        adapter: ArtifactAdapter,
        task_record: TaskRecord,
    ) -> dict[str, Any]:
        """Assemble a bounded execution context packet for subagent dispatch."""
        memory_entries = self._load_memory_entries(adapter, max_entries=5)
        return {
            "task_id": task_record.task_id,
            "title": task_record.title,
            "description": task_record.description,
            "acceptance_criteria": list(task_record.acceptance_criteria),
            "execution_hints": list(task_record.execution_hints),
            "workflow_id": state.workflow_id,
            "dependencies_completed": [
                task_id
                for task_id in state.completed_task_ids
                if task_id in task_record.dependencies
            ],
            "memory_entries": memory_entries,
        }

    def record_subagent_dispatch(
        self,
        state: RuntimeState,
        adapter: ArtifactAdapter,
        task_id: str,
        session_id: str,
    ) -> RuntimeState:
        """Record the subagent dispatch session and update task/state to running."""
        task = self._get_task_record(state, task_id)
        if task is None:
            raise ValueError(f"Unknown task_id for subagent dispatch: {task_id}")

        timestamp = self._utcnow_isoformat()
        task.status = "running"
        task.last_error = None
        state.current_task_id = task_id
        state = self._transition_to_running(state)
        state.status = "active"
        state.active_subagents.append(
            SubagentRecord(
                session_id=session_id,
                status="running",
                task_id=task_id,
                started_at=timestamp,
                completed_at=None,
            )
        )
        state.updated_at = timestamp
        adapter.write_state(state)
        logger.info(
            "plan_task_subagent_dispatched",
            workflow_id=state.workflow_id,
            task_id=task_id,
            session_id=session_id,
        )
        return state

    def record_task_completion(
        self,
        state: RuntimeState,
        adapter: ArtifactAdapter,
        task_id: str,
        evidence_refs: list[str] | None = None,
        summary: str = "",
    ) -> RuntimeState:
        """Mark a task completed, append memory, advance to next task or TASK_COMPLETED phase."""
        task = self._get_task_record(state, task_id)
        if task is None:
            raise ValueError(f"Unknown task_id for task completion: {task_id}")

        timestamp = self._utcnow_isoformat()
        task.status = "completed"
        task.last_error = None
        if task_id not in state.completed_task_ids:
            state.completed_task_ids.append(task_id)

        for record in state.active_subagents:
            if record.task_id == task_id and record.status in {"queued", "running"}:
                record.status = "succeeded"
                record.completed_at = timestamp

        evidence_list = list(evidence_refs or [])
        memory_path = adapter.memory_dir / _MEMORY_FILE_NAME
        line_number = self._count_jsonl_lines(memory_path) + 1
        adapter.append_memory(
            {
                "task_id": task_id,
                "summary": summary,
                "evidence_refs": evidence_list,
                "appended_at": timestamp,
            }
        )
        state.memory_refs.append(f"memory/{_MEMORY_FILE_NAME}#{line_number}")
        evidence_path = adapter.evidence_dir / f"task-{task_id}-result.json"
        evidence_data: dict[str, Any] = {
            "task_id": task_id,
            "summary": summary,
            "evidence_refs": evidence_list,
            "completed_at": timestamp,
        }
        self._write_text_atomic(
            evidence_path,
            json.dumps(evidence_data, ensure_ascii=False, indent=2) + "\n",
        )

        next_task_id = self._next_pending_task_id(state)
        state.current_task_id = next_task_id
        if next_task_id is None:
            state = WorkflowStateMachine().transition(state, "TASK_COMPLETED")
            state.status = "completed"
        else:
            state = self._transition_to_running(state)
            state.status = "active"

        state.updated_at = timestamp
        adapter.write_state(state)
        logger.info(
            "plan_task_task_completed",
            workflow_id=state.workflow_id,
            task_id=task_id,
            next_task_id=next_task_id,
            workflow_done=(next_task_id is None),
        )
        return state

    def check_circuit_breaker(
        self,
        state: RuntimeState,
        task_id: str,
        max_retries: int = 3,
    ) -> bool:
        """Return True if the task's retry budget has reached the maximum retries limit."""
        triggered = state.retry_budget.get(task_id, 0) >= max_retries
        if triggered:
            logger.warning(
                "plan_task_circuit_breaker_triggered",
                workflow_id=state.workflow_id,
                task_id=task_id,
                retry_count=state.retry_budget.get(task_id, 0),
                max_retries=max_retries,
            )
        return triggered

    def check_retry_budget_exhausted(
        self,
        state: RuntimeState,
        task_id: str,
        max_retries: int = 3,
    ) -> bool:
        """Return True if the task's retry budget is exhausted (alias for check_circuit_breaker)."""
        return self.check_circuit_breaker(state, task_id, max_retries=max_retries)

    def _require_approved_reviews(self) -> None:
        verdicts_by_phase = {
            verdict.phase: verdict.verdict for verdict in self.state.review_verdicts
        }
        required_phases = ("DRAFT_ADVISOR_REVIEW", "DRAFT_QA_REVIEW", "PLAN_QA_REVIEW")
        missing = [
            phase
            for phase in required_phases
            if verdicts_by_phase.get(phase) != "approved"
        ]
        if missing:
            formatted = ", ".join(missing)
            logger.warning(
                "plan_task_reviews_not_approved",
                workflow_id=self.state.workflow_id,
                missing_phases=missing,
            )
            raise ValueError(
                f"Task start requires approved review verdicts for: {formatted}"
            )

    def _write_task_queue_artifact(self, path: Path, queue: list[TaskRecord]) -> None:
        content = json.dumps(
            [asdict(task) for task in queue], ensure_ascii=False, indent=2
        )
        self._write_text_atomic(path, content + "\n")

    def _load_memory_entries(
        self, adapter: ArtifactAdapter, max_entries: int
    ) -> list[dict[str, Any]]:
        memory_path = adapter.memory_dir / _MEMORY_FILE_NAME
        if not memory_path.exists():
            return []

        entries: list[dict[str, Any]] = []
        for line in memory_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                entries.append(payload)
        return entries[-max_entries:]

    def _count_jsonl_lines(self, path: Path) -> int:
        if not path.exists():
            return 0
        return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line)

    def _get_task_record(self, state: RuntimeState, task_id: str) -> TaskRecord | None:
        for task in state.tasks:
            if task.task_id == task_id:
                return task
        return None

    def _next_pending_task_id(self, state: RuntimeState) -> str | None:
        completed = set(state.completed_task_ids)
        for task in state.tasks:
            if task.status == "completed":
                continue
            if set(task.dependencies) <= completed:
                return task.task_id
        return None

    def _transition_to_running(self, state: RuntimeState) -> RuntimeState:
        state_machine = WorkflowStateMachine()
        if state.phase == "PLAN_FINALIZED":
            state = state_machine.transition(state, "TASK_READY")
        if state.phase == "TASK_READY":
            return state_machine.transition(state, "TASK_RUNNING")
        if state.phase == "TASK_RUNNING":
            return state
        if "TASK_RUNNING" in self._allowed_transitions(state):
            return state_machine.transition(state, "TASK_RUNNING")
        return state

    def _allowed_transitions(self, state: RuntimeState) -> set[str]:
        return {t.target_state_id for t in _COMPILED_WORKFLOW.transitions_by_state.get(state.phase, ())}

    def _utcnow_isoformat(self) -> str:
        return datetime.datetime.now(datetime.UTC).isoformat()

    def _write_text_atomic(self, path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=path.parent,
                delete=False,
                prefix=f".{path.name}.",
                suffix=".tmp",
            ) as handle:
                handle.write(content)
                temp_path = handle.name

            os.replace(temp_path, path)
        finally:
            if temp_path is not None and os.path.exists(temp_path):
                os.unlink(temp_path)
