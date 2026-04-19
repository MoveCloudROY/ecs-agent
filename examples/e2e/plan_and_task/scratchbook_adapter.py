"""Scratchbook-backed artifact adapter for the plan-and-task example.

Drop-in replacement for ArtifactAdapter. Same public surface; stores artifacts
under ``scratchbook/<workflow_id>/`` instead of ``.artifacts/workflows/<workflow_id>/``.
Uses ScratchbookService for structured I/O and ArtifactRegistry for canonical artifact IDs.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ecs_agent.logging import get_logger
from ecs_agent.scratchbook.artifact_registry import ArtifactKind, ArtifactRegistry
from ecs_agent.scratchbook.prompt_definition import (
    ScratchbookArtifactPromptDef,
    ScratchbookPromptConfig,
)
from ecs_agent.scratchbook.service import ScratchbookService

from .state_models import ReviewVerdict, RuntimeState, TaskRecord

logger = get_logger(__name__)

_PLAN_FILE_NAME = "workflow_plan.md"
_RUNTIME_STATE_FILE_NAME = "runtime_state.json"
_EVENTS_FILE_NAME = "events.jsonl"
_MEMORY_FILE_NAME = "knowledge.jsonl"
_DRAFT_PHASES = frozenset(
    {"DRAFT_INTERVIEW", "DRAFT_ADVISOR_REVIEW", "DRAFT_QA_REVIEW"}
)
_WRITE_PLAN_PHASES = frozenset({"WRITE_PLAN", "PLAN_QA_REVIEW"})
_PLANNING_PHASES = _DRAFT_PHASES | _WRITE_PLAN_PHASES
_TASK_EXECUTION_PHASES = frozenset(
    {
        "PLAN_FINALIZED",
        "TASK_READY",
        "TASK_RUNNING",
        "TASK_BLOCKED",
        "TASK_REPLAN",
        "TASK_COMPLETED",
        "TASK_ABORTED",
    }
)


def _write_text_atomic(path: Path, content: str) -> None:
    """Write content to path using a temp-file + os.replace for atomicity."""
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


class PlanTaskScratchbookAdapter:
    """Scratchbook-backed read/write access to workflow artifacts on disk.

    Exposes the same public surface as ArtifactAdapter so that controller.py
    and task_exec.py require only an import change, not logic changes.

    Storage root: ``<base_dir>/scratchbook/<workflow_id>/``
    """

    def __init__(self, *, base_dir: Path, workflow_id: str) -> None:
        self.base_dir = Path(base_dir)
        self.workflow_id = workflow_id
        self._registry = ArtifactRegistry(root=self.base_dir)
        self._service = ScratchbookService(root=self.base_dir / "scratchbook")
        self.workflow_root = self.base_dir / "scratchbook" / workflow_id
        self.plan_dir = self.workflow_root / "plan"
        self.state_dir = self.workflow_root / "state"
        self.memory_dir = self.workflow_root / "memory"
        self.evidence_dir = self.workflow_root / "evidence"
        self.review_dir = self.workflow_root / "review"
        self._ensure_layout()

    def write_plan(self, content: str) -> str:
        plan_path = self.plan_dir / _PLAN_FILE_NAME
        _write_text_atomic(plan_path, content)
        logger.info(
            "plan_task_plan_written",
            workflow_id=self.workflow_id,
            path=self._relative_path(plan_path),
        )
        return self._relative_path(plan_path)

    def write_state(self, state: RuntimeState) -> str:
        if state.workflow_id != self.workflow_id:
            raise ValueError(
                "Runtime state workflow_id does not match artifact adapter workflow_id"
            )
        self._service.write_index(
            _RUNTIME_STATE_FILE_NAME,
            f"{self.workflow_id}/state",
            state.to_dict(),
        )
        logger.info("plan_task_runtime_state_written", workflow_id=self.workflow_id)
        return f"state/{_RUNTIME_STATE_FILE_NAME}"

    def read_state(self) -> RuntimeState:
        state_path = self.state_dir / _RUNTIME_STATE_FILE_NAME
        if not state_path.exists():
            raise ValueError(
                f"Missing runtime state file: {self._relative_path(state_path)}"
            )
        data = self._service.read_index(
            _RUNTIME_STATE_FILE_NAME, f"{self.workflow_id}/state"
        )
        if data is None:
            raise ValueError(
                f"Corrupt runtime state JSON at state/{_RUNTIME_STATE_FILE_NAME}"
            )
        try:
            state = RuntimeState.from_dict(data)
        except (ValueError, KeyError, TypeError) as exc:
            raise ValueError(f"Failed to parse runtime state: {exc}") from exc

        if state.phase in _DRAFT_PHASES:
            draft_path = self.plan_dir / "draft.md"
            if not draft_path.exists():
                raise ValueError(
                    f"Runtime state references missing draft file: plan/draft.md"
                )
        elif state.phase in _WRITE_PLAN_PHASES:
            draft_path = self.plan_dir / "draft.md"
            if not draft_path.exists():
                raise ValueError(
                    f"Runtime state references missing draft file: plan/draft.md"
                )
        elif state.phase in _TASK_EXECUTION_PHASES:
            plan_path = self.workflow_root / state.active_plan_file
            if not plan_path.exists():
                raise ValueError(
                    f"Runtime state references missing plan file: {state.active_plan_file}"
                )
        logger.debug(
            "plan_task_state_loaded",
            workflow_id=self.workflow_id,
            phase=state.phase,
        )
        return state

    def append_event(self, event: dict[str, Any]) -> str:
        self._service.append_log(
            _EVENTS_FILE_NAME,
            f"{self.workflow_id}/state",
            json.dumps(event, ensure_ascii=False) + "\n",
        )
        logger.debug(
            "plan_task_event_appended",
            workflow_id=self.workflow_id,
            event_type=event.get("type"),
        )
        return f"state/{_EVENTS_FILE_NAME}"

    def append_memory(self, entry: dict[str, Any]) -> str:
        self._service.append_log(
            _MEMORY_FILE_NAME,
            f"{self.workflow_id}/memory",
            json.dumps(entry, ensure_ascii=False) + "\n",
        )
        logger.debug(
            "plan_task_memory_appended",
            workflow_id=self.workflow_id,
            task_id=entry.get("task_id"),
        )
        return f"memory/{_MEMORY_FILE_NAME}"

    def write_review_verdict(self, phase: str, verdict: ReviewVerdict) -> str:
        if verdict.phase != phase:
            raise ValueError("Review verdict phase must match the target phase")
        content = json.dumps(asdict(verdict), ensure_ascii=False, indent=2) + "\n"
        result = self._registry.persist(kind=ArtifactKind.TOOL, content=content)
        filename = f"{self._slugify(phase)}_verdict.json"
        self._service.write_index(
            filename, f"{self.workflow_id}/review", asdict(verdict)
        )
        logger.info(
            "plan_task_review_verdict_written",
            workflow_id=self.workflow_id,
            phase=phase,
            artifact_id=result.descriptor.artifact_id,
        )
        return f"review/{filename}"

    def write_draft(self, content: str) -> str:
        draft_path = self.plan_dir / "draft.md"
        _write_text_atomic(draft_path, content)
        logger.info(
            "plan_task_draft_written",
            workflow_id=self.workflow_id,
            path=self._relative_path(draft_path),
        )
        return self._relative_path(draft_path)

    def read_draft(self) -> str | None:
        draft_path = self.plan_dir / "draft.md"
        if not draft_path.exists():
            return None
        return draft_path.read_text(encoding="utf-8")

    def read_draft_description(self) -> str | None:
        draft_path = self.plan_dir / "draft.md"
        if not draft_path.exists():
            return None
        raw = draft_path.read_text(encoding="utf-8")
        marker = "## Description\n"
        start = raw.find(marker)
        if start == -1:
            return None
        start += len(marker)
        end = raw.find("\n\n##", start)
        description = raw[start:end].strip() if end != -1 else raw[start:].strip()
        return description or None

    def mark_stale_subagents(self, state: RuntimeState) -> list[str]:
        """Mark in-flight subagents as stale and requeue their tasks."""
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
        if requeue_task_ids:
            logger.info(
                "plan_task_subagents_marked_stale",
                workflow_id=state.workflow_id,
                stale_count=len(requeue_task_ids),
                task_ids=requeue_task_ids,
            )
        return requeue_task_ids

    def _ensure_layout(self) -> None:
        self.plan_dir.mkdir(parents=True, exist_ok=True)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        self.review_dir.mkdir(parents=True, exist_ok=True)

    def _relative_path(self, path: Path) -> str:
        return str(path.relative_to(self.workflow_root))

    def _slugify(self, value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")

    def _task_for(self, *, state: RuntimeState, task_id: str) -> TaskRecord | None:
        for task in state.tasks:
            if task.task_id == task_id:
                return task
        return None


def build_scratchbook_prompt_config(workflow_id: str) -> ScratchbookPromptConfig:
    root = f"scratchbook/{workflow_id}"
    return ScratchbookPromptConfig(
        overview_default_template=(
            "Scratchbook root: ${scratchbook_path}\nArtifact types:\n${artifact_types}"
        ),
        scratchbook_root_path=root,
        artifacts=[
            ScratchbookArtifactPromptDef(
                artifact_type_id="draft_plan",
                path=f"{root}/plan/draft.md",
                purpose="Working draft created at plan:start; used by advisor/QA reviews before finalization.",
                readonly=False,
                read_when="planning phase, before finalization",
            ),
            ScratchbookArtifactPromptDef(
                artifact_type_id="workflow_plan",
                path=f"{root}/plan/workflow_plan.md",
                purpose="The current approved workflow plan in Markdown.",
                readonly=False,
                read_when="planning, reviewing, or executing tasks",
            ),
            ScratchbookArtifactPromptDef(
                artifact_type_id="runtime_state",
                path=f"{root}/state/runtime_state.json",
                purpose="Workflow runtime state: phase, tasks, subagents.",
                readonly=True,
                read_when="any action that depends on current workflow phase",
            ),
            ScratchbookArtifactPromptDef(
                artifact_type_id="events_journal",
                path=f"{root}/state/events.jsonl",
                purpose="Append-only log of all workflow events.",
                readonly=True,
                read_when="debugging or reviewing workflow history",
            ),
            ScratchbookArtifactPromptDef(
                artifact_type_id="knowledge_memory",
                path=f"{root}/memory/knowledge.jsonl",
                purpose="Cross-task knowledge and memory entries.",
                readonly=True,
                read_when="starting a new task or needing prior context",
            ),
            ScratchbookArtifactPromptDef(
                artifact_type_id="review_verdict",
                path=f"{root}/review/",
                purpose="Advisor and QA review verdict JSON files.",
                readonly=True,
                read_when="checking review approval status before finalizing",
            ),
        ],
    )
