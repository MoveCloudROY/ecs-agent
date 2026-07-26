"""Plan interview and finalization controller for the E2E example."""

from __future__ import annotations

import datetime
from enum import Enum
from typing import Any

from ecs_agent.components import PhaseApprovalsComponent, PhaseComponent
from ecs_agent.core import World
from ecs_agent.logging import get_logger
from ecs_agent.phases import PhaseGraph, advance, force, record_approval
from ecs_agent.types import EntityId

from examples.e2e.plan_and_task.phase_graph import (
    PLAN_TASK_PHASE_GRAPH,
    REVIEW_VERDICTS,
)
from examples.e2e.plan_and_task.phase_sync import derive_status, save_state
from examples.e2e.plan_and_task.scratchbook_adapter import (
    PlanTaskScratchbookAdapter as ArtifactAdapter,
)
from examples.e2e.plan_and_task.state_models import (
    RuntimeState,
    ReviewVerdict,
    missing_approvals,
)

logger = get_logger(__name__)

def _derive_finalize_hops(graph: PhaseGraph) -> dict[str, str]:
    """Happy-path walk from the first planning phase to TASK_READY.

    Each hop follows the phase's approved gate target when one exists,
    otherwise the first entry of `to` (the graph's forward-edge convention).
    TASK_READY is the finalize destination — a domain constant, not routing.
    """
    hops: dict[str, str] = {}
    phase = graph.phases_by_id[graph.initial].to[0]
    for _ in range(len(graph.phases_by_id)):
        if phase == "TASK_READY":
            break
        spec = graph.phases_by_id[phase]
        approved = spec.approval.verdicts.get("approved") if spec.approval else None
        hops[phase] = approved or spec.to[0]
        phase = hops[phase]
    else:
        raise AssertionError("finalize walk did not reach TASK_READY")
    gated = {p for p, s in graph.phases_by_id.items() if s.approval is not None}
    if not gated <= hops.keys():
        raise AssertionError(
            "approval-gated phases off the finalize walk: "
            f"{sorted(gated - hops.keys())}"
        )
    return hops


_FINALIZE_HOPS: dict[str, str] = _derive_finalize_hops(PLAN_TASK_PHASE_GRAPH)


class ResumeAction(Enum):
    TRIGGER_PLAN_WRITER = "trigger_plan_writer"


class PlanController:
    """Manage draft creation, review gating, and plan finalization.

    Transitions run through ecs_agent.phases; PhaseComponent is the in-memory
    authority (read via current_phase()), and RuntimeState.phase/graph_hash/
    status are stamped from it at persist time by phase_sync.save_state().
    """

    def __init__(self, world: World, entity_id: EntityId) -> None:
        self._world = world
        self._entity_id = entity_id

    # -- phase plumbing ------------------------------------------------------

    def _save(self, state: RuntimeState, adapter: ArtifactAdapter) -> None:
        save_state(self._world, self._entity_id, state, adapter)

    async def _advance(self, to_phase: str, *, reason: str) -> None:
        await advance(self._world, self._entity_id, to_phase, reason=reason)

    def current_phase(self) -> str:
        component = self._world.get_component(self._entity_id, PhaseComponent)
        return (
            component.phase
            if component is not None
            else PLAN_TASK_PHASE_GRAPH.initial
        )

    # -- handlers ------------------------------------------------------------

    async def handle_plan_start(
        self, adapter: ArtifactAdapter, description: str
    ) -> RuntimeState:
        """Create workflow namespace, write draft.md, enter DRAFT_INTERVIEW."""
        timestamp = self._utcnow_isoformat()
        draft_content = self._build_draft_markdown(
            description=description,
            workflow_id=adapter.workflow_id,
            timestamp=timestamp,
        )
        adapter.write_draft(draft_content)
        # A new workflow starts with a clean audit ledger; without this, a
        # previous workflow's approvals would leak into latest_verdicts().
        self._world.remove_component(self._entity_id, PhaseApprovalsComponent)
        # force(): a new workflow may start from any prior phase, including
        # terminal ones left by a previous workflow in this process.
        await force(self._world, self._entity_id, "DRAFT_INTERVIEW", reason="plan:start")
        state = RuntimeState(
            workflow_id=adapter.workflow_id,
            phase="DRAFT_INTERVIEW",
            status=derive_status(
                "DRAFT_INTERVIEW", abort_reason=None, review_verdicts=[]
            ),
            active_plan_file="plan/workflow_plan.md",
            current_task_id=None,
            review_verdicts=[],
            active_subagents=[],
            memory_refs=[],
            last_checkpoint=None,
            created_at=timestamp,
            updated_at=timestamp,
            graph_hash=PLAN_TASK_PHASE_GRAPH.structure_hash,
            tasks=[],
        )
        self._save(state, adapter)
        logger.info(
            "plan_task_plan_started",
            workflow_id=adapter.workflow_id,
            phase=state.phase,
        )
        return state

    async def handle_plan_finalize(
        self, state: RuntimeState, adapter: ArtifactAdapter
    ) -> RuntimeState:
        """Validate advisor+qa+plan-qa approvals, confirm the plan, walk to TASK_READY.

        The ``plan_writer`` subagent produces ``workflow_plan.md`` during the
        WRITE_PLAN phase; finalization must **not** rewrite it (doing so would
        clobber the real multi-task plan with a stub). It only gates on the three
        approvals, confirms the finalized plan artifact exists, and walks the
        phase forward — the task queue is built from that plan by ``TaskExec``.
        """
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

        plan_path = adapter.plan_dir / "workflow_plan.md"
        if not plan_path.exists():
            logger.warning(
                "plan_task_plan_artifact_missing",
                workflow_id=state.workflow_id,
                path="plan/workflow_plan.md",
            )
            raise ValueError(
                "Cannot finalize: plan/workflow_plan.md does not exist. The plan "
                "writer must produce it (WRITE_PLAN) before finalization."
            )

        timestamp = self._utcnow_isoformat()
        while (phase := self.current_phase()) in _FINALIZE_HOPS:
            await self._advance(_FINALIZE_HOPS[phase], reason="plan:finalize")
        state.updated_at = timestamp
        self._save(state, adapter)
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

    async def handle_advisor_review(
        self,
        state: RuntimeState,
        adapter: ArtifactAdapter,
        verdict_str: str,
        notes: str | None = None,
    ) -> RuntimeState:
        return await self._handle_review(
            state,
            adapter,
            review_phase="DRAFT_ADVISOR_REVIEW",
            verdict_str=verdict_str,
            notes=notes,
            log_event="plan_task_advisor_review_recorded",
        )

    async def handle_qa_review(
        self,
        state: RuntimeState,
        adapter: ArtifactAdapter,
        verdict_str: str,
        notes: str | None = None,
    ) -> RuntimeState:
        return await self._handle_review(
            state,
            adapter,
            review_phase="DRAFT_QA_REVIEW",
            verdict_str=verdict_str,
            notes=notes,
            log_event="plan_task_qa_review_recorded",
        )

    async def handle_plan_qa_review(
        self,
        state: RuntimeState,
        adapter: ArtifactAdapter,
        verdict_str: str,
        notes: str | None = None,
    ) -> RuntimeState:
        return await self._handle_review(
            state,
            adapter,
            review_phase="PLAN_QA_REVIEW",
            verdict_str=verdict_str,
            notes=notes,
            log_event="plan_task_plan_qa_review_recorded",
        )

    async def _handle_review(
        self,
        state: RuntimeState,
        adapter: ArtifactAdapter,
        *,
        review_phase: str,
        verdict_str: str,
        notes: str | None,
        log_event: str,
    ) -> RuntimeState:
        """Shared review flow: gate, record, persist, and route a review verdict.

        Order matters: validate the verdict string, reject review phases that
        are neither current nor adjacent (reachability gate), refuse to advance
        forward out of a review phase whose own verdict is not yet approved
        (approval gate), upsert into the state ledger (sticky approvals win),
        and only then write the verdict artifact and route via the phase's
        gate. A verdict discarded by the sticky-approval rule leaves no trace:
        no artifact write, no approval ledger append, no phase change, no state
        persist.

        Raises:
            ValueError: invalid verdict string; review_phase unreachable from
                the current phase; or advancing into review_phase would leave a
                current review phase that is not approved.
        """
        if verdict_str not in REVIEW_VERDICTS:
            raise ValueError(f"Invalid verdict: {verdict_str!r}")
        current = self.current_phase()
        current_spec = PLAN_TASK_PHASE_GRAPH.phases_by_id[current]
        allowed = current_spec.to
        if current != review_phase:
            if review_phase not in allowed:
                raise ValueError(
                    f"Cannot record {review_phase} verdict while in phase {current!r}"
                )
            # Forward progression out of a review phase is gated on that phase's
            # own approval. Entering the next review phase to record its verdict
            # must not vault the workflow past an advisor/QA review still sitting
            # on a 'revise'/'blocked' verdict — only an 'approved' review yields.
            if current_spec.approval is not None and not self._review_approved(
                state, current
            ):
                raise ValueError(
                    f"Cannot record {review_phase} verdict while {current} "
                    "is not approved"
                )
        timestamp = self._utcnow_isoformat()
        verdict = ReviewVerdict(
            phase=review_phase,
            verdict=verdict_str,
            decided_at=timestamp,
            notes=notes,
        )
        applied = state.upsert_verdict(verdict)
        if not applied:
            logger.info(
                "plan_task_review_verdict_ignored",
                workflow_id=state.workflow_id,
                phase=review_phase,
                verdict=verdict_str,
            )
            return state
        adapter.write_review_verdict(review_phase, verdict)

        if current != review_phase and review_phase in allowed:
            await self._advance(review_phase, reason=f"verdict:{review_phase}")
        if self.current_phase() == review_phase:
            # verdict.notes, not the raw parameter: the sticky rule cleared
            # notes for approved verdicts, and both ledgers must agree.
            await record_approval(
                self._world,
                self._entity_id,
                verdict_str,
                notes=verdict.notes,
                decided_at=timestamp,
            )

        state.updated_at = timestamp
        self._save(state, adapter)
        logger.info(log_event, workflow_id=state.workflow_id, verdict=verdict_str)
        return state

    async def handle_write_plan(
        self, state: RuntimeState, adapter: ArtifactAdapter
    ) -> RuntimeState:
        """Transition to WRITE_PLAN so the planner can produce workflow_plan.md.

        The manual counterpart to the DRAFT_QA_REVIEW ApprovalGate: it mirrors
        the gate's contract, so it only yields to WRITE_PLAN once QA approved the
        draft. Invoked while QA is still on 'revise'/'blocked', it must refuse —
        never bypass the gate and vault a rejected draft into the writer.

        Because approving DRAFT_QA_REVIEW auto-advances to WRITE_PLAN (via the
        gate on the live path, or gate-replay on resume), by the time this runs
        the phase is usually already WRITE_PLAN. That is an idempotent success:
        no transition, but the caller still re-emits the write-plan trigger so
        the plan_writer (re)runs. Only a phase that is neither DRAFT_QA_REVIEW nor
        WRITE_PLAN is an error.
        """
        current = self.current_phase()
        if current == "WRITE_PLAN":
            logger.info(
                "plan_task_write_plan_retriggered", workflow_id=state.workflow_id
            )
            return state
        if current != "DRAFT_QA_REVIEW":
            raise ValueError(
                "handle_write_plan requires DRAFT_QA_REVIEW or WRITE_PLAN phase, "
                f"got {current}"
            )
        if not self._review_approved(state, "DRAFT_QA_REVIEW"):
            raise ValueError(
                "Cannot write plan while DRAFT_QA_REVIEW is not approved"
            )
        timestamp = self._utcnow_isoformat()
        await self._advance("WRITE_PLAN", reason="plan:write")
        state.updated_at = timestamp
        self._save(state, adapter)
        logger.info("plan_task_write_plan_started", workflow_id=state.workflow_id)
        return state

    async def handle_write_plan_completed(
        self, state: RuntimeState, adapter: ArtifactAdapter
    ) -> RuntimeState:
        current = self.current_phase()
        if current != "WRITE_PLAN":
            raise ValueError(
                f"handle_write_plan_completed requires WRITE_PLAN phase, got {current}"
            )
        timestamp = self._utcnow_isoformat()
        await self._advance("PLAN_QA_REVIEW", reason="plan_writer:completed")
        state.updated_at = timestamp
        self._save(state, adapter)
        logger.info("plan_task_write_plan_completed", workflow_id=state.workflow_id)
        return state

    async def handle_task_abort(
        self, state: RuntimeState, adapter: ArtifactAdapter, reason: str
    ) -> RuntimeState:
        """Abort the current task and transition to TASK_ABORTED terminal state."""
        self._require_reason(reason)
        self._require_plan_artifact(adapter, state)

        timestamp = self._utcnow_isoformat()
        # abort_reason must be set before the save below: the persist-time
        # snapshot derives status, and "aborted" requires the reason present.
        state.abort_reason = reason
        await self._advance("TASK_ABORTED", reason=f"task:abort:{reason[:80]}")
        state.last_checkpoint = reason
        state.updated_at = timestamp
        self._save(state, adapter)
        logger.info(
            "plan_task_aborted",
            workflow_id=state.workflow_id,
            task_id=state.current_task_id,
            reason=reason,
        )
        return state

    async def handle_task_replan(
        self,
        state: RuntimeState,
        adapter: ArtifactAdapter,
        reason: str,
        scope_changed: bool = False,
    ) -> RuntimeState:
        """Request a replan and optionally force advisor/QA re-review on scope change."""
        self._require_reason(reason)
        self._require_plan_artifact(adapter, state)

        timestamp = self._utcnow_isoformat()
        await self._advance("TASK_REPLAN", reason=f"task:replan:{reason[:80]}")
        state.last_checkpoint = reason
        state.abort_reason = None

        if scope_changed:
            state.review_verdicts = []
            self._world.remove_component(self._entity_id, PhaseApprovalsComponent)
            await self._advance("DRAFT_ADVISOR_REVIEW", reason="replan:scope_changed")
        else:
            await self._advance("TASK_RUNNING", reason="replan:same_scope")

        state.updated_at = timestamp
        self._save(state, adapter)
        logger.info(
            "plan_task_replan_handled",
            workflow_id=state.workflow_id,
            task_id=state.current_task_id,
            scope_changed=scope_changed,
        )
        return state

    async def handle_task_resume(
        self, state: RuntimeState, adapter: ArtifactAdapter
    ) -> RuntimeState:
        """Resume a blocked or replanned task by transitioning back to TASK_RUNNING."""
        current = self.current_phase()
        if PLAN_TASK_PHASE_GRAPH.phases_by_id[current].terminal:
            raise ValueError(f"Cannot resume terminal workflow phase: {current}")
        if current not in {"TASK_BLOCKED", "TASK_REPLAN"}:
            raise ValueError(
                f"Workflow phase is not resumable via /task:resume: {current}"
            )

        self._require_plan_artifact(adapter, state)
        timestamp = self._utcnow_isoformat()
        await self._advance("TASK_RUNNING", reason="task:resume")
        state.abort_reason = None
        state.updated_at = timestamp
        self._save(state, adapter)
        logger.info(
            "plan_task_resumed",
            workflow_id=state.workflow_id,
            task_id=state.current_task_id,
            phase=state.phase,
        )
        return state

    def _missing_approved_reviews(self, verdicts: list[ReviewVerdict]) -> list[str]:
        return missing_approvals(verdicts)

    def _review_approved(self, state: RuntimeState, phase: str) -> bool:
        """True when the latest recorded verdict for `phase` is 'approved'.

        The state ledger holds at most one verdict per phase (upsert), so the
        last match is the effective verdict; approvals are sticky.
        """
        latest: str | None = None
        for verdict in state.review_verdicts:
            if verdict.phase == phase:
                latest = verdict.verdict
        return latest == "approved"

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

    _PLANNING_PHASES = frozenset(
        {"DRAFT_INTERVIEW", "DRAFT_ADVISOR_REVIEW", "DRAFT_QA_REVIEW", "WRITE_PLAN", "PLAN_QA_REVIEW"}
    )

    def _require_plan_artifact(
        self, adapter: ArtifactAdapter, state: RuntimeState
    ) -> None:
        if self.current_phase() in self._PLANNING_PHASES:
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

    def _utcnow_isoformat(self) -> str:
        return datetime.datetime.now(datetime.UTC).isoformat()
