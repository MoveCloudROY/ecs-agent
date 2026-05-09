"""Workflow spec for the plan-and-task E2E example."""

from __future__ import annotations

from ecs_agent.workflows import PromptProfileSpec, absent, workflow
from ecs_agent.workflows._components import WorkflowRuntimeComponent

from examples.e2e.plan_and_task.prompts import (
    IDLE_MAIN_AGENT_SYSTEM_PROMPT,
    PLAN_MAIN_AGENT_SYSTEM_PROMPT,
    TASK_MAIN_AGENT_SYSTEM_PROMPT,
)

_SENTINEL = absent(WorkflowRuntimeComponent)

PLAN_TASK_WORKFLOW_SPEC = workflow(
    workflow_id="plan-task",
    initial="IDLE",
    profiles={
        "main": {
            "idle_main": PromptProfileSpec(
                profile_id="idle_main",
                prompt=IDLE_MAIN_AGENT_SYSTEM_PROMPT,
            ),
            "plan_main": PromptProfileSpec(
                profile_id="plan_main",
                prompt=PLAN_MAIN_AGENT_SYSTEM_PROMPT,
            ),
            "task_main": PromptProfileSpec(
                profile_id="task_main",
                prompt=TASK_MAIN_AGENT_SYSTEM_PROMPT,
            ),
        }
    },
    states={
        "IDLE": {
            "bind": {"main": "idle_main"},
            "go": {"DRAFT_INTERVIEW": _SENTINEL},
        },
        "DRAFT_INTERVIEW": {
            "bind": {"main": "plan_main"},
            "go": {
                "DRAFT_ADVISOR_REVIEW": _SENTINEL,
                "DRAFT_QA_REVIEW": _SENTINEL,
            },
        },
        "DRAFT_ADVISOR_REVIEW": {
            "bind": {"main": "plan_main"},
            "go": {
                "DRAFT_QA_REVIEW": _SENTINEL,
                "DRAFT_INTERVIEW": _SENTINEL,
            },
        },
        "DRAFT_QA_REVIEW": {
            "bind": {"main": "plan_main"},
            "go": {
                "WRITE_PLAN": _SENTINEL,
                "DRAFT_INTERVIEW": _SENTINEL,
            },
        },
        "WRITE_PLAN": {
            "bind": {"main": "plan_main"},
            "go": {"PLAN_QA_REVIEW": _SENTINEL},
        },
        "PLAN_QA_REVIEW": {
            "bind": {"main": "plan_main"},
            "go": {
                "PLAN_FINALIZED": _SENTINEL,
                "WRITE_PLAN": _SENTINEL,
            },
        },
        "PLAN_FINALIZED": {
            "bind": {"main": "plan_main"},
            "go": {"TASK_READY": _SENTINEL},
        },
        "TASK_READY": {
            "bind": {"main": "plan_main"},
            "go": {"TASK_RUNNING": _SENTINEL},
        },
        "TASK_RUNNING": {
            "bind": {"main": "task_main"},
            "go": {
                "TASK_COMPLETED": _SENTINEL,
                "TASK_BLOCKED": _SENTINEL,
                "TASK_REPLAN": _SENTINEL,
                "TASK_ABORTED": _SENTINEL,
            },
        },
        "TASK_BLOCKED": {
            "bind": {"main": "task_main"},
            "go": {
                "TASK_RUNNING": _SENTINEL,
                "TASK_REPLAN": _SENTINEL,
                "TASK_ABORTED": _SENTINEL,
            },
        },
        "TASK_REPLAN": {
            "bind": {"main": "task_main"},
            "go": {
                "DRAFT_INTERVIEW": _SENTINEL,
                "DRAFT_ADVISOR_REVIEW": _SENTINEL,
                "TASK_RUNNING": _SENTINEL,
            },
        },
        "TASK_COMPLETED": {
            "bind": {"main": "task_main"},
            "go": {},
        },
        "TASK_ABORTED": {
            "bind": {"main": "task_main"},
            "go": {},
        },
    },
)
