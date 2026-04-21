"""Plan interview and finalization controller for the E2E example."""

from __future__ import annotations

import datetime
import warnings
from enum import Enum
from pathlib import Path
from typing import Any

from ecs_agent.logging import get_logger

from examples.e2e.plan_and_task.scratchbook_adapter import (
    PlanTaskScratchbookAdapter as ArtifactAdapter,
)
from examples.e2e.plan_and_task.state_models import RuntimeState, ReviewVerdict
from examples.e2e.plan_and_task.state_machine import (
    VALID_TRANSITIONS,
    WorkflowStateMachine,
)

logger = get_logger(__name__)


class ResumeAction(Enum):
    TRIGGER_PLAN_WRITER = "trigger_plan_writer"


class PlanController:
    """Manage draft creation, review gating, and plan finalization."""

    def __init__(self) -> None:
        self._state_machine = WorkflowStateMachine()

    def handle_plan_start(
        self, adapter: ArtifactAdapter, description: str
    ) -> RuntimeState:
        """Create workflow namespace, write draft.md, return DRAFT_INTERVIEW state."""
        timestamp = self._utcnow_isoformat()
        draft_content = self._build_draft_markdown(
            description=description,
            workflow_id=adapter.workflow_id,
            timestamp=timestamp,
        )
        adapter.write_draft(draft_content)
        state = RuntimeState(
            workflow_id=adapter.workflow_id,
            phase="DRAFT_INTERVIEW",
            status="active",
            active_plan_file="plan/workflow_plan.md",
            current_task_id=None,
            completed_task_ids=[],
            retry_budget={},
            review_verdicts=[],
            active_subagents=[],
            memory_refs=[],
            last_checkpoint=None,
            created_at=timestamp,
            updated_at=timestamp,
            tasks=[],
            open_questions=[],
            confirmed_requirements=[],
        )
        adapter.write_state(state)
        logger.info(
            "plan_task_plan_started",
            workflow_id=adapter.workflow_id,
            phase=state.phase,
        )
        return state

    def handle_plan_finalize(
        self, state: RuntimeState, adapter: ArtifactAdapter
    ) -> RuntimeState:
        """Validate both advisor+qa verdicts=approved, write workflow_plan.md, transition to TASK_READY."""
        missing_phases = self._missing_approved_reviews(state.review_verdicts)
        if missing_phases:
            formatted = ", ".join(missing_phases)
            logger.warning(
                "plan_task_finalize_blocked",
                workflow_id=state.workflow_id,
                missing_phases=missing_phases,
            )
            raise ValueError(
                "Cannot finalize: missing or unapproved review verdicts for: "
                f"{formatted}"
            )

        timestamp = self._utcnow_isoformat()
        description = adapter.read_draft_description() or "Finalized workflow plan"
        plan_content = self._build_final_plan_markdown(
            workflow_id=adapter.workflow_id,
            description=description,
            timestamp=timestamp,
        )
        adapter.write_plan(plan_content)

        if state.phase == "DRAFT_INTERVIEW":
            state = self._state_machine.transition(state, "DRAFT_ADVISOR_REVIEW")
        if state.phase == "DRAFT_ADVISOR_REVIEW":
            state = self._state_machine.transition(state, "DRAFT_QA_REVIEW")
        if state.phase == "DRAFT_QA_REVIEW":
            state = self._state_machine.transition(state, "WRITE_PLAN")
        if state.phase == "WRITE_PLAN":
            state = self._state_machine.transition(state, "PLAN_QA_REVIEW")
        if state.phase == "PLAN_QA_REVIEW":
            state = self._state_machine.transition(state, "PLAN_FINALIZED")
        if state.phase == "PLAN_FINALIZED":
            state = self._state_machine.transition(state, "TASK_READY")
        state.status = "ready"
        state.updated_at = timestamp
        adapter.write_state(state)
        logger.info(
            "plan_task_plan_finalized",
            workflow_id=state.workflow_id,
            phase=state.phase,
        )
        return state

    def get_plan_status(self, state: RuntimeState) -> dict[str, Any]:
        """Return structured status dict with phase, status, workflow_id, current_task_id, etc."""
        return {
            "workflow_id": state.workflow_id,
            "phase": state.phase,
            "status": state.status,
            "active_plan_file": state.active_plan_file,
            "current_task_id": state.current_task_id,
            "completed_task_ids": list(state.completed_task_ids),
            "review_verdicts": [
                {
                    "phase": verdict.phase,
                    "verdict": verdict.verdict,
                    "decided_at": verdict.decided_at,
                    "notes": verdict.notes,
                }
                for verdict in state.review_verdicts
            ],
            "task_count": len(state.tasks),
            "active_subagent_count": len(state.active_subagents),
            "last_checkpoint": state.last_checkpoint,
            "created_at": state.created_at,
            "updated_at": state.updated_at,
        }

    def handle_advisor_review(
        self,
        state: RuntimeState,
        adapter: ArtifactAdapter,
        verdict_str: str,
        notes: str | None = None,
        citations: list[str] | None = None,
        evidence_refs: list[str] | None = None,
    ) -> RuntimeState:
        """Record advisor review verdict, persist artifact, update state phase and review_verdicts."""
        if verdict_str not in {"approved", "revise", "blocked"}:
            raise ValueError(f"Invalid verdict: {verdict_str!r}")
        timestamp = self._utcnow_isoformat()
        verdict = ReviewVerdict(
            phase="DRAFT_ADVISOR_REVIEW",
            verdict=verdict_str,
            decided_at=timestamp,
            notes=notes,
            citations=citations or [],
            evidence_refs=evidence_refs or [],
        )
        adapter.write_review_verdict("DRAFT_ADVISOR_REVIEW", verdict)
        state.upsert_verdict(verdict)
        if "DRAFT_ADVISOR_REVIEW" in self._allowed_transitions(state):
            state = self._state_machine.transition(state, "DRAFT_ADVISOR_REVIEW")
        state.updated_at = timestamp
        adapter.write_state(state)
        logger.info(
            "plan_task_advisor_review_recorded",
            workflow_id=state.workflow_id,
            verdict=verdict_str,
        )
        return state

    def handle_qa_review(
        self,
        state: RuntimeState,
        adapter: ArtifactAdapter,
        verdict_str: str,
        notes: str | None = None,
        citations: list[str] | None = None,
        evidence_refs: list[str] | None = None,
    ) -> RuntimeState:
        """Record QA review verdict, persist artifact, update state phase and review_verdicts."""
        if verdict_str not in {"approved", "revise", "blocked"}:
            raise ValueError(f"Invalid verdict: {verdict_str!r}")
        timestamp = self._utcnow_isoformat()
        verdict = ReviewVerdict(
            phase="DRAFT_QA_REVIEW",
            verdict=verdict_str,
            decided_at=timestamp,
            notes=notes,
            citations=citations or [],
            evidence_refs=evidence_refs or [],
        )
        adapter.write_review_verdict("DRAFT_QA_REVIEW", verdict)
        state.upsert_verdict(verdict)
        if "DRAFT_QA_REVIEW" in self._allowed_transitions(state):
            state = self._state_machine.transition(state, "DRAFT_QA_REVIEW")
        if verdict_str == "approved" and "WRITE_PLAN" in self._allowed_transitions(state):
            state = self._state_machine.transition(state, "WRITE_PLAN")
            logger.info(
                "plan_task_auto_transition_write_plan",
                workflow_id=state.workflow_id,
            )
        state.updated_at = timestamp
        adapter.write_state(state)
        logger.info(
            "plan_task_qa_review_recorded",
            workflow_id=state.workflow_id,
            verdict=verdict_str,
        )
        return state

    def handle_write_plan(
        self,
        state: RuntimeState,
        adapter: ArtifactAdapter,
    ) -> RuntimeState:
        """Transition to WRITE_PLAN phase so the planner can produce workflow_plan.md."""
        if state.phase != "DRAFT_QA_REVIEW":
            raise ValueError(
                f"handle_write_plan requires DRAFT_QA_REVIEW phase, got {state.phase}"
            )
        timestamp = self._utcnow_isoformat()
        state = self._state_machine.transition(state, "WRITE_PLAN")
        state.updated_at = timestamp
        adapter.write_state(state)
        logger.info(
            "plan_task_write_plan_started",
            workflow_id=state.workflow_id,
        )
        return state

    def handle_write_plan_completed(
        self,
        state: RuntimeState,
        adapter: ArtifactAdapter,
    ) -> RuntimeState:
        if state.phase != "WRITE_PLAN":
            raise ValueError(
                f"handle_write_plan_completed requires WRITE_PLAN phase, got {state.phase}"
            )
        timestamp = self._utcnow_isoformat()
        state = self._state_machine.transition(state, "PLAN_QA_REVIEW")
        state.updated_at = timestamp
        adapter.write_state(state)
        logger.info(
            "plan_task_write_plan_completed",
            workflow_id=state.workflow_id,
        )
        return state

    def handle_plan_qa_review(
        self,
        state: RuntimeState,
        adapter: ArtifactAdapter,
        verdict_str: str,
        notes: str | None = None,
        citations: list[str] | None = None,
        evidence_refs: list[str] | None = None,
    ) -> RuntimeState:
        """Record plan QA review verdict on the final workflow_plan.md, persist, update state."""
        if verdict_str not in {"approved", "revise", "blocked"}:
            raise ValueError(f"Invalid verdict: {verdict_str!r}")
        timestamp = self._utcnow_isoformat()
        verdict = ReviewVerdict(
            phase="PLAN_QA_REVIEW",
            verdict=verdict_str,
            decided_at=timestamp,
            notes=notes,
            citations=citations or [],
            evidence_refs=evidence_refs or [],
        )
        adapter.write_review_verdict("PLAN_QA_REVIEW", verdict)
        state.upsert_verdict(verdict)
        if "PLAN_QA_REVIEW" in self._allowed_transitions(state):
            state = self._state_machine.transition(state, "PLAN_QA_REVIEW")
        if verdict_str == "approved" and "PLAN_FINALIZED" in self._allowed_transitions(state):
            state = self._state_machine.transition(state, "PLAN_FINALIZED")
            logger.info(
                "plan_task_auto_transition_plan_finalized",
                workflow_id=state.workflow_id,
            )
        state.updated_at = timestamp
        adapter.write_state(state)
        logger.info(
            "plan_task_plan_qa_review_recorded",
            workflow_id=state.workflow_id,
            verdict=verdict_str,
        )
        return state

    def reconcile_after_resume(
        self, state: RuntimeState, adapter: ArtifactAdapter
    ) -> list[ResumeAction]:
        verdicts_by_phase = {v.phase: v.verdict for v in state.review_verdicts}
        actions: list[ResumeAction] = []

        if state.phase == "DRAFT_QA_REVIEW":
            if verdicts_by_phase.get("DRAFT_QA_REVIEW") == "approved":
                state = self._state_machine.transition(state, "WRITE_PLAN")
                adapter.write_state(state)
                logger.info(
                    "plan_task_auto_transition_write_plan",
                    workflow_id=state.workflow_id,
                    source="reconcile_after_resume",
                )
                actions.append(ResumeAction.TRIGGER_PLAN_WRITER)

        elif state.phase == "WRITE_PLAN":
            actions.append(ResumeAction.TRIGGER_PLAN_WRITER)

        elif state.phase == "PLAN_QA_REVIEW":
            if verdicts_by_phase.get("PLAN_QA_REVIEW") == "approved":
                state = self._state_machine.transition(state, "PLAN_FINALIZED")
                adapter.write_state(state)
                logger.info(
                    "plan_task_auto_transition_plan_finalized",
                    workflow_id=state.workflow_id,
                    source="reconcile_after_resume",
                )

        return actions

    def handle_task_abort(
        self, state: RuntimeState, adapter: ArtifactAdapter, reason: str
    ) -> RuntimeState:
        """Abort the current task and transition to TASK_ABORTED terminal state."""
        self._require_reason(reason)
        self._require_plan_artifact(adapter, state)

        timestamp = self._utcnow_isoformat()
        updated_state = self._state_machine.transition(state, "TASK_ABORTED")
        updated_state.status = "aborted"
        updated_state.abort_reason = reason
        updated_state.last_checkpoint = reason
        updated_state.updated_at = timestamp
        self._append_task_event(
            adapter,
            event_type="task_aborted",
            state=updated_state,
            reason=reason,
            scope_changed=False,
        )
        adapter.write_state(updated_state)
        logger.info(
            "plan_task_aborted",
            workflow_id=state.workflow_id,
            task_id=state.current_task_id,
            reason=reason,
        )
        return updated_state

    def handle_task_replan(
        self,
        state: RuntimeState,
        adapter: ArtifactAdapter,
        reason: str,
        scope_changed: bool = False,
    ) -> RuntimeState:
        """Request a replan and optionally trigger advisor/QA review if scope changed."""
        self._require_reason(reason)
        self._require_plan_artifact(adapter, state)

        timestamp = self._utcnow_isoformat()
        updated_state = self._state_machine.transition(state, "TASK_REPLAN")
        updated_state.last_checkpoint = reason
        updated_state.abort_reason = None
        self._append_task_event(
            adapter,
            event_type="task_replan_requested",
            state=updated_state,
            reason=reason,
            scope_changed=scope_changed,
        )

        if scope_changed:
            updated_state.review_verdicts = []
            updated_state = self._state_machine.transition(
                updated_state, "DRAFT_ADVISOR_REVIEW"
            )
            updated_state.status = "needs_review"
        else:
            updated_state = self._state_machine.transition(
                updated_state, "TASK_RUNNING"
            )
            updated_state.status = "active"

        updated_state.updated_at = timestamp
        adapter.write_state(updated_state)
        logger.info(
            "plan_task_replan_handled",
            workflow_id=state.workflow_id,
            task_id=state.current_task_id,
            scope_changed=scope_changed,
        )
        return updated_state

    def handle_task_resume(
        self, state: RuntimeState, adapter: ArtifactAdapter
    ) -> RuntimeState:
        """Resume a blocked or replanned task by transitioning back to TASK_RUNNING."""
        if self._state_machine.is_terminal(state.phase):
            raise ValueError(f"Cannot resume terminal workflow phase: {state.phase}")
        if state.phase not in {"TASK_BLOCKED", "TASK_REPLAN"}:
            raise ValueError(
                f"Workflow phase is not resumable via /task:resume: {state.phase}"
            )

        self._require_plan_artifact(adapter, state)
        timestamp = self._utcnow_isoformat()
        updated_state = self._state_machine.transition(state, "TASK_RUNNING")
        updated_state.status = "active"
        updated_state.abort_reason = None
        updated_state.updated_at = timestamp
        self._append_task_event(
            adapter,
            event_type="task_resumed",
            state=updated_state,
            reason=updated_state.last_checkpoint,
            scope_changed=False,
        )
        adapter.write_state(updated_state)
        logger.info(
            "plan_task_resumed",
            workflow_id=state.workflow_id,
            task_id=state.current_task_id,
            phase=state.phase,
        )
        return updated_state

    def _missing_approved_reviews(self, verdicts: list[ReviewVerdict]) -> list[str]:
        verdicts_by_phase = {verdict.phase: verdict.verdict for verdict in verdicts}
        required_phases = ("DRAFT_ADVISOR_REVIEW", "DRAFT_QA_REVIEW", "PLAN_QA_REVIEW")
        return [
            phase
            for phase in required_phases
            if verdicts_by_phase.get(phase) != "approved"
        ]

    def _build_draft_markdown(
        self, *, description: str, workflow_id: str, timestamp: str
    ) -> str:
        return (
            f"# Draft Plan: {description}\n\n"
            "Status: draft\n"
            f"Workflow: {workflow_id}\n"
            f"Created: {timestamp}\n\n"
            "## Description\n"
            f"{description}\n\n"
            "## Scope\n"
            "(to be filled during interview — what is in and out of scope)\n\n"
            "## Confirmed Requirements\n"
            "(to be filled during interview — list each confirmed requirement)\n\n"
            "## Constraints\n"
            "(to be filled during interview — technical, budget, time constraints)\n\n"
            "## Risks\n"
            "(to be filled during interview — known risks and mitigations)\n\n"
            "## Acceptance Criteria\n"
            "(to be filled during interview — how will success be measured)\n\n"
            "## Open Questions\n"
            "(to be filled during interview — unresolved questions that need answers)\n"
        )

    def _build_final_plan_markdown(
        self, *, workflow_id: str, description: str, timestamp: str
    ) -> str:
        return f"""---
workflow_id: {workflow_id}
title: Finalized Plan
description: {description}
status: finalized
created_at: \"{timestamp}\"
finalized_at: \"{timestamp}\"
---

## Tasks

### Task: task-001
```yaml
task_id: task-001
title: Initial Task
description: First task in the workflow.
dependencies: []
acceptance_criteria:
  - Plan description is documented in workflow artifacts
  - Runtime state reflects TASK_COMPLETED phase
  - Evidence artifact is written to evidence/ directory
execution_hints: []
```
"""

    _PLANNING_PHASES = frozenset(
        {"DRAFT_INTERVIEW", "DRAFT_ADVISOR_REVIEW", "DRAFT_QA_REVIEW", "WRITE_PLAN", "PLAN_QA_REVIEW"}
    )

    def _require_plan_artifact(
        self, adapter: ArtifactAdapter, state: RuntimeState
    ) -> None:
        if state.phase in self._PLANNING_PHASES:
            return
        plan_path = adapter.workflow_root / state.active_plan_file
        if not plan_path.exists():
            logger.warning(
                "plan_task_plan_artifact_missing",
                workflow_id=state.workflow_id,
                path=state.active_plan_file,
            )
            raise ValueError(f"Missing plan artifact: {state.active_plan_file}")

    def _require_reason(self, reason: str) -> None:
        if not reason.strip():
            raise ValueError("Task control reason must be a non-empty string")

    def _append_task_event(
        self,
        adapter: ArtifactAdapter,
        *,
        event_type: str,
        state: RuntimeState,
        reason: str | None,
        scope_changed: bool,
    ) -> None:
        adapter.append_event(
            {
                "type": event_type,
                "workflow_id": state.workflow_id,
                "task_id": state.current_task_id,
                "phase": state.phase,
                "reason": reason,
                "scope_changed": scope_changed,
                "evidence_refs": list(state.memory_refs),
                "timestamp": self._utcnow_isoformat(),
            }
        )

    def _allowed_transitions(self, state: RuntimeState) -> set[str]:
        return VALID_TRANSITIONS.get(state.phase, set())

    def _utcnow_isoformat(self) -> str:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            return datetime.datetime.utcnow().isoformat()
