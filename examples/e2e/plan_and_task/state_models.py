"""Persisted runtime-state models for the plan-and-task example."""

import json
from dataclasses import asdict, dataclass, field
from typing import Any

from ecs_agent.logging import get_logger

from examples.e2e.plan_and_task.phase_graph import PLAN_TASK_PHASE_GRAPH

logger = get_logger(__name__)

_PHASE_VALUES = frozenset(PLAN_TASK_PHASE_GRAPH.phases_by_id)
_SUBAGENT_STATUS_VALUES = {"queued", "running", "succeeded", "failed", "stale"}
_REVIEW_VERDICT_VALUES = {"approved", "revise", "blocked"}


def _require_non_empty(value: str, *, field_name: str) -> None:
    if not value:
        raise ValueError(f"{field_name} must be a non-empty string")


@dataclass(slots=True)
class SubagentRecord:
    """Tracks a dispatched subagent's session, status, and timing."""

    session_id: str
    status: str
    task_id: str
    started_at: str | None
    completed_at: str | None

    def __post_init__(self) -> None:
        _require_non_empty(self.session_id, field_name="session_id")
        _require_non_empty(self.task_id, field_name="task_id")
        if self.status not in _SUBAGENT_STATUS_VALUES:
            raise ValueError(f"Invalid subagent status: {self.status}")


@dataclass(slots=True)
class ReviewVerdict:
    """Records a review decision with phase, verdict, notes, and citations."""

    phase: str
    verdict: str
    decided_at: str
    notes: str | None = None
    citations: list[str] = field(default_factory=list)
    evidence_refs: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.phase not in _PHASE_VALUES:
            raise ValueError(f"Invalid review phase: {self.phase}")
        if self.verdict not in _REVIEW_VERDICT_VALUES:
            raise ValueError(f"Invalid review verdict: {self.verdict}")
        _require_non_empty(self.decided_at, field_name="decided_at")


@dataclass(slots=True)
class TaskRecord:
    """Tracks the execution state and metadata for a single workflow task."""

    task_id: str
    title: str
    status: str
    description: str = ""
    dependencies: list[str] = field(default_factory=list)
    acceptance_criteria: list[str] = field(default_factory=list)
    execution_hints: list[str] = field(default_factory=list)
    retry_count: int = 0
    last_error: str | None = None

    def __post_init__(self) -> None:
        _require_non_empty(self.task_id, field_name="task_id")
        _require_non_empty(self.title, field_name="title")
        _require_non_empty(self.status, field_name="status")
        if self.retry_count < 0:
            raise ValueError("retry_count must be >= 0")


@dataclass(slots=True)
class RuntimeState:
    """Persisted runtime state for a workflow, including phase, tasks, and review verdicts."""

    workflow_id: str
    phase: str
    status: str
    active_plan_file: str
    current_task_id: str | None
    completed_task_ids: list[str]
    retry_budget: dict[str, int]
    review_verdicts: list[ReviewVerdict]
    active_subagents: list[SubagentRecord]
    memory_refs: list[str]
    last_checkpoint: str | None
    created_at: str
    updated_at: str
    abort_reason: str | None = None
    tasks: list[TaskRecord] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    confirmed_requirements: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        _require_non_empty(self.workflow_id, field_name="workflow_id")
        if self.phase not in _PHASE_VALUES:
            raise ValueError(f"Invalid runtime phase: {self.phase}")
        _require_non_empty(self.status, field_name="status")
        _require_non_empty(self.active_plan_file, field_name="active_plan_file")
        _require_non_empty(self.created_at, field_name="created_at")
        _require_non_empty(self.updated_at, field_name="updated_at")
        for task_id, budget in self.retry_budget.items():
            _require_non_empty(task_id, field_name="retry_budget task_id")
            if budget < 0:
                raise ValueError("retry_budget values must be >= 0")

    def upsert_verdict(self, verdict: "ReviewVerdict") -> None:
        if verdict.verdict == "approved":
            object.__setattr__(verdict, "notes", None)
        for i, existing in enumerate(self.review_verdicts):
            if existing.phase == verdict.phase:
                if existing.verdict == "approved":
                    return
                self.review_verdicts[i] = verdict
                return
        self.review_verdicts.append(verdict)

    def to_dict(self) -> dict[str, Any]:
        """Convert the runtime state to a dictionary representation."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert the runtime state to a JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n"

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RuntimeState":
        """Deserialize a runtime state from a dictionary, validating all required fields."""
        try:
            review_verdicts = [
                ReviewVerdict(**item) for item in payload["review_verdicts"]
            ]
            active_subagents = [
                SubagentRecord(**item) for item in payload["active_subagents"]
            ]
            tasks = [TaskRecord(**item) for item in payload.get("tasks", [])]
            return cls(
                workflow_id=payload["workflow_id"],
                phase=payload["phase"],
                status=payload["status"],
                active_plan_file=payload["active_plan_file"],
                current_task_id=payload["current_task_id"],
                completed_task_ids=list(payload["completed_task_ids"]),
                retry_budget=dict(payload["retry_budget"]),
                review_verdicts=review_verdicts,
                active_subagents=active_subagents,
                memory_refs=list(payload["memory_refs"]),
                last_checkpoint=payload["last_checkpoint"],
                abort_reason=payload.get("abort_reason"),
                created_at=payload["created_at"],
                updated_at=payload["updated_at"],
                tasks=tasks,
                open_questions=list(payload.get("open_questions", [])),
                confirmed_requirements=list(payload.get("confirmed_requirements", [])),
            )
        except KeyError as exc:
            logger.error("plan_task_runtime_state_missing_field", field=str(exc))
            raise ValueError(
                f"Invalid runtime state payload: missing field {exc.args[0]}"
            ) from exc
        except TypeError as exc:
            logger.error("plan_task_runtime_state_invalid_shape", exception=str(exc))
            raise ValueError(f"Invalid runtime state payload: {exc}") from exc

    @classmethod
    def from_json(cls, content: str) -> "RuntimeState":
        """Deserialize a runtime state from a JSON string."""
        try:
            payload = json.loads(content)
        except json.JSONDecodeError as exc:
            logger.error("plan_task_runtime_state_corrupt_json", exception=str(exc))
            raise ValueError(f"Corrupt runtime state JSON: {exc.msg}") from exc

        if not isinstance(payload, dict):
            raise ValueError("Invalid runtime state payload: expected a JSON object")

        return cls.from_dict(payload)
