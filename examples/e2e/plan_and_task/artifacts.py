"""Artifact persistence adapter for the plan-and-task example."""

import json
import os
import re
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ecs_agent.logging import get_logger

from .state_models import ReviewVerdict, RuntimeState, TaskRecord

logger = get_logger(__name__)

_PLAN_FILE_NAME = "workflow_plan.md"
_RUNTIME_STATE_FILE_NAME = "runtime_state.json"
_EVENTS_FILE_NAME = "events.jsonl"
_MEMORY_FILE_NAME = "knowledge.jsonl"
_PLAN_VERSION_PATTERN = re.compile(r"v(?P<version>\d+)_workflow_plan\.md")
_TASK_EXECUTION_PHASES = {
    "PLAN_FINALIZED",
    "TASK_READY",
    "TASK_RUNNING",
    "TASK_BLOCKED",
    "TASK_REPLAN",
    "TASK_COMPLETED",
    "TASK_ABORTED",
}


class ArtifactAdapter:
    """Provides read/write access to workflow artifacts on disk."""

    def __init__(self, *, base_dir: Path, workflow_id: str) -> None:
        self.base_dir = Path(base_dir)
        self.workflow_id = workflow_id
        self.workflow_root = self.base_dir / ".artifacts" / "workflows" / workflow_id
        self.plan_dir = self.workflow_root / "plan"
        self.plan_versions_dir = self.plan_dir / "plan_versions"
        self.state_dir = self.workflow_root / "state"
        self.memory_dir = self.workflow_root / "memory"
        self.evidence_dir = self.workflow_root / "evidence"
        self.review_dir = self.workflow_root / "review"
        self._ensure_layout()

    def write_plan(self, content: str) -> str:
        """Write workflow plan content atomically and preserve prior versions."""
        plan_path = self.plan_dir / _PLAN_FILE_NAME
        if plan_path.exists():
            current_content = plan_path.read_text(encoding="utf-8")
            version_path = self.plan_versions_dir / (
                f"v{self._next_plan_version()}_{_PLAN_FILE_NAME}"
            )
            self._write_text_atomic(version_path, current_content)

        self._write_text_atomic(plan_path, content)
        logger.info(
            "plan_task_plan_written",
            workflow_id=self.workflow_id,
            path=self._relative_path(plan_path),
        )
        return self._relative_path(plan_path)

    def write_state(self, state: RuntimeState) -> str:
        """Persist the runtime state to JSON file with validation and atomic write."""
        if state.workflow_id != self.workflow_id:
            raise ValueError(
                "Runtime state workflow_id does not match artifact adapter workflow_id"
            )

        state_path = self.state_dir / _RUNTIME_STATE_FILE_NAME
        self._write_text_atomic(state_path, state.to_json())
        logger.info(
            "plan_task_runtime_state_written",
            workflow_id=self.workflow_id,
            path=self._relative_path(state_path),
        )
        return self._relative_path(state_path)

    def read_state(self) -> RuntimeState:
        """Load and validate runtime state from the persisted JSON file."""
        state_path = self.state_dir / _RUNTIME_STATE_FILE_NAME
        if not state_path.exists():
            raise ValueError(
                f"Missing runtime state file: {self._relative_path(state_path)}"
            )

        try:
            content = state_path.read_text(encoding="utf-8")
            state = RuntimeState.from_json(content)
        except ValueError:
            raise
        except OSError as exc:
            logger.error(
                "plan_task_runtime_state_read_failed",
                workflow_id=self.workflow_id,
                exception=str(exc),
            )
            raise ValueError(f"Failed to read runtime state file: {exc}") from exc

        # Only validate plan file existence during task execution phases
        # During planning phases, workflow_plan.md is not yet created
        if state.phase in _TASK_EXECUTION_PHASES:
            plan_path = self.workflow_root / state.active_plan_file
            if not plan_path.exists():
                raise ValueError(
                    f"Runtime state references missing plan file: {state.active_plan_file}"
                )

        return state

    def append_event(self, event: dict[str, Any]) -> str:
        """Append an event record to the events journal in JSONL format."""
        events_path = self.state_dir / _EVENTS_FILE_NAME
        self._append_json_line_atomic(events_path, event)
        return self._relative_path(events_path)

    def append_memory(self, entry: dict[str, Any]) -> str:
        """Append a knowledge/memory entry to the memory journal in JSONL format."""
        memory_path = self.memory_dir / _MEMORY_FILE_NAME
        self._append_json_line_atomic(memory_path, entry)
        return self._relative_path(memory_path)

    def write_review_verdict(self, phase: str, verdict: ReviewVerdict) -> str:
        """Persist a review verdict as a JSON artifact with phase-based slug filename."""
        if verdict.phase != phase:
            raise ValueError("Review verdict phase must match the target phase")

        filename = f"{self._slugify(phase)}_verdict.json"
        review_path = self.review_dir / filename
        content = json.dumps(asdict(verdict), ensure_ascii=False, indent=2) + "\n"
        self._write_text_atomic(review_path, content)
        return self._relative_path(review_path)

    def mark_stale_subagents(self, state: RuntimeState) -> list[str]:
        """Mark in-flight subagents as stale and requeue their tasks, returning requeue list."""
        requeue_task_ids: list[str] = []
        for record in state.active_subagents:
            if record.status not in {"queued", "running"}:
                continue

            record.status = "stale"
            state.retry_budget[record.task_id] = (
                state.retry_budget.get(record.task_id, 0) + 1
            )
            task = self._task_for(state=state, task_id=record.task_id)
            if task is not None:
                task.retry_count += 1
                if task.status != "completed":
                    task.status = "pending"
            if state.current_task_id is None:
                state.current_task_id = record.task_id
            if record.task_id not in requeue_task_ids:
                requeue_task_ids.append(record.task_id)

        return requeue_task_ids

    def _ensure_layout(self) -> None:
        self.plan_versions_dir.mkdir(parents=True, exist_ok=True)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        self.review_dir.mkdir(parents=True, exist_ok=True)

    def _next_plan_version(self) -> int:
        versions = [0]
        for path in self.plan_versions_dir.glob(f"v*_{_PLAN_FILE_NAME}"):
            match = _PLAN_VERSION_PATTERN.fullmatch(path.name)
            if match is not None:
                versions.append(int(match.group("version")))
        return max(versions) + 1

    def _append_json_line_atomic(self, path: Path, payload: dict[str, Any]) -> None:
        existing = ""
        if path.exists():
            existing = path.read_text(encoding="utf-8")
        line = json.dumps(payload, ensure_ascii=False) + "\n"
        self._write_text_atomic(path, existing + line)

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

    def _relative_path(self, path: Path) -> str:
        return str(path.relative_to(self.workflow_root))

    def _slugify(self, value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")

    def _task_for(self, *, state: RuntimeState, task_id: str) -> TaskRecord | None:
        for task in state.tasks:
            if task.task_id == task_id:
                return task
        return None
