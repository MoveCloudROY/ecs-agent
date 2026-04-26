"""Workflow state machine for the plan-and-task E2E example."""

from __future__ import annotations

import datetime

from ecs_agent.logging import get_logger

from examples.e2e.plan_and_task.scratchbook_adapter import (
    PlanTaskScratchbookAdapter as ArtifactAdapter,
)
from examples.e2e.plan_and_task.state_models import RuntimeState

logger = get_logger(__name__)

VALID_TRANSITIONS: dict[str, set[str]] = {
    "IDLE": {"DRAFT_INTERVIEW"},
    "DRAFT_INTERVIEW": {"DRAFT_ADVISOR_REVIEW", "DRAFT_QA_REVIEW"},
    "DRAFT_ADVISOR_REVIEW": {"DRAFT_QA_REVIEW", "DRAFT_INTERVIEW"},
    "DRAFT_QA_REVIEW": {"WRITE_PLAN", "DRAFT_INTERVIEW"},
    "WRITE_PLAN": {"PLAN_QA_REVIEW"},
    "PLAN_QA_REVIEW": {"PLAN_FINALIZED", "WRITE_PLAN"},
    "PLAN_FINALIZED": {"TASK_READY"},
    "TASK_READY": {"TASK_RUNNING"},
    "TASK_RUNNING": {"TASK_COMPLETED", "TASK_BLOCKED", "TASK_REPLAN", "TASK_ABORTED"},
    "TASK_BLOCKED": {"TASK_RUNNING", "TASK_REPLAN", "TASK_ABORTED"},
    "TASK_REPLAN": {"DRAFT_INTERVIEW", "DRAFT_ADVISOR_REVIEW", "TASK_RUNNING"},
    "TASK_COMPLETED": set(),
    "TASK_ABORTED": set(),
}

_TERMINAL_PHASES: frozenset[str] = frozenset({"TASK_COMPLETED", "TASK_ABORTED"})


class WorkflowStateMachine:
    """Manages valid state transitions and restart semantics for the workflow controller."""

    def transition(self, state: RuntimeState, to_phase: str) -> RuntimeState:
        """Validate and apply a state transition.

        Args:
            state: Current runtime state.
            to_phase: Target phase to transition to.

        Returns:
            Updated RuntimeState with new phase and timestamp.

        Raises:
            ValueError: If the transition is invalid.
        """
        allowed = VALID_TRANSITIONS.get(state.phase, set())
        if to_phase not in allowed:
            raise ValueError(f"Invalid transition: {state.phase} → {to_phase}")
        state.phase = to_phase
        state.status = "active" if to_phase not in _TERMINAL_PHASES else "completed"
        state.updated_at = self._utcnow_isoformat()
        logger.info(
            "plan_task_state_transition",
            workflow_id=state.workflow_id,
            to_phase=to_phase,
        )
        return state

    def is_terminal(self, phase: str) -> bool:
        """Return True if the phase is a terminal state (no further transitions possible)."""
        return phase in _TERMINAL_PHASES

    def can_resume(self, phase: str) -> bool:
        """Return True if the workflow can be resumed from this phase."""
        return phase not in _TERMINAL_PHASES and phase != "IDLE"

    def requires_continuation(self, state: RuntimeState) -> bool:
        """Return True if the workflow was started but is not yet complete."""
        return not self.is_terminal(state.phase) and state.phase not in {"IDLE"}

    def handle_restart(
        self, state: RuntimeState, adapter: ArtifactAdapter
    ) -> RuntimeState:
        """Mark stale in-flight subagents, requeue blocked tasks, and persist updated state.

        Called when a new process restores persisted state. Any subagent that was
        queued/running is marked stale; if any stale subagents existed and the
        current phase was TASK_RUNNING, the phase is moved to TASK_BLOCKED so the
        controller knows to requeue work.

        Args:
            state: Current runtime state loaded from persistence.
            adapter: Artifact adapter for writing updated state.

        Returns:
            Updated RuntimeState.
        """
        stale_task_ids = adapter.mark_stale_subagents(state)
        if stale_task_ids and state.phase == "TASK_RUNNING":
            self._force_phase(state, "TASK_BLOCKED")
            logger.info(
                "plan_task_restart_blocked",
                workflow_id=state.workflow_id,
                stale_task_ids=stale_task_ids,
            )
        state.updated_at = self._utcnow_isoformat()
        adapter.write_state(state)
        logger.info(
            "plan_task_restart_complete",
            workflow_id=state.workflow_id,
            phase=state.phase,
            stale_count=len(stale_task_ids),
        )
        return state

    def _force_phase(self, state: RuntimeState, phase: str) -> None:
        """Forcibly set the phase to the target phase without validating transitions.

        This is an administrative-only bypass for exceptional recovery scenarios, such as
        marking in-flight tasks as blocked after a restart. Normal phase transitions must
        use the transition() method, which validates against VALID_TRANSITIONS.

        Args:
            state: Current runtime state to modify.
            phase: Target phase to set (bypass validation).
        """
        state.phase = phase

    def _utcnow_isoformat(self) -> str:
        return datetime.datetime.now(datetime.UTC).isoformat()
