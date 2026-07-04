"""Phase graph for the plan-and-task E2E example (legacy workflow_spec removed in Stage 3)."""

from __future__ import annotations

from ecs_agent.phases import ApprovalGate, PhaseSpec, build_graph

from examples.e2e.plan_and_task.prompts import (
    IDLE_MAIN_AGENT_SYSTEM_PROMPT,
    PLAN_MAIN_AGENT_SYSTEM_PROMPT,
    TASK_MAIN_AGENT_SYSTEM_PROMPT,
)

_IDLE = {"main": IDLE_MAIN_AGENT_SYSTEM_PROMPT}
_PLAN = {"main": PLAN_MAIN_AGENT_SYSTEM_PROMPT}
_TASK = {"main": TASK_MAIN_AGENT_SYSTEM_PROMPT}

# Verdict routing preserved from PlanController's pre-migration behavior:
# advisor verdicts only record; QA approval advances to WRITE_PLAN; plan-QA
# approval advances to PLAN_FINALIZED; revise/blocked always stay put.
_STAY: dict[str, str | None] = {"approved": None, "revise": None, "blocked": None}

PLAN_TASK_PHASE_GRAPH = build_graph(
    "plan-task",
    initial="IDLE",
    phases=[
        PhaseSpec(phase_id="IDLE", prompts=_IDLE, to=("DRAFT_INTERVIEW",)),
        PhaseSpec(
            phase_id="DRAFT_INTERVIEW",
            prompts=_PLAN,
            to=("DRAFT_ADVISOR_REVIEW", "DRAFT_QA_REVIEW"),
        ),
        PhaseSpec(
            phase_id="DRAFT_ADVISOR_REVIEW",
            prompts=_PLAN,
            to=("DRAFT_QA_REVIEW", "DRAFT_INTERVIEW"),
            approval=ApprovalGate(verdicts=dict(_STAY)),
        ),
        PhaseSpec(
            phase_id="DRAFT_QA_REVIEW",
            prompts=_PLAN,
            to=("WRITE_PLAN", "DRAFT_INTERVIEW"),
            approval=ApprovalGate(
                verdicts={"approved": "WRITE_PLAN", "revise": None, "blocked": None}
            ),
        ),
        PhaseSpec(phase_id="WRITE_PLAN", prompts=_PLAN, to=("PLAN_QA_REVIEW",)),
        PhaseSpec(
            phase_id="PLAN_QA_REVIEW",
            prompts=_PLAN,
            to=("PLAN_FINALIZED", "WRITE_PLAN"),
            approval=ApprovalGate(
                verdicts={"approved": "PLAN_FINALIZED", "revise": None, "blocked": None}
            ),
        ),
        PhaseSpec(phase_id="PLAN_FINALIZED", prompts=_PLAN, to=("TASK_READY",)),
        PhaseSpec(phase_id="TASK_READY", prompts=_PLAN, to=("TASK_RUNNING",)),
        PhaseSpec(
            phase_id="TASK_RUNNING",
            prompts=_TASK,
            to=("TASK_COMPLETED", "TASK_BLOCKED", "TASK_REPLAN", "TASK_ABORTED"),
            on_resume="TASK_BLOCKED",
        ),
        PhaseSpec(
            phase_id="TASK_BLOCKED",
            prompts=_TASK,
            to=("TASK_RUNNING", "TASK_REPLAN", "TASK_ABORTED"),
        ),
        PhaseSpec(
            phase_id="TASK_REPLAN",
            prompts=_TASK,
            to=("DRAFT_INTERVIEW", "DRAFT_ADVISOR_REVIEW", "TASK_RUNNING"),
        ),
        PhaseSpec(phase_id="TASK_COMPLETED", prompts=_TASK, terminal=True),
        PhaseSpec(phase_id="TASK_ABORTED", prompts=_TASK, terminal=True),
    ],
)
