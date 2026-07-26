"""Agent-drivable debug harness for the plan-and-task example.

A headless, scriptable, introspectable third front end for the plan-and-task
world (alongside the stdin REPL and the Textual TUI). See ``README.md`` and
``docs/rfc/2026-07-26-plan-task-debug-harness.md``.

Library entry point::

    from examples.e2e.plan_and_task.debug import PlanTaskDebugSession

CLI entry point::

    python -m examples.e2e.plan_and_task.debug
"""

from __future__ import annotations

from examples.e2e.plan_and_task.debug.policies import (
    AnswerPolicy,
    AutoAnswerPolicy,
    CallbackAnswerPolicy,
    ScriptedAnswerPolicy,
)
from examples.e2e.plan_and_task.debug.session import (
    PhaseTransitionRecord,
    PlanTaskDebugSession,
    QuestionRecord,
    StateSnapshot,
    SubagentRunRecord,
    ToolCallRecord,
    TurnResult,
)

__all__ = [
    "PlanTaskDebugSession",
    "TurnResult",
    "StateSnapshot",
    "ToolCallRecord",
    "SubagentRunRecord",
    "PhaseTransitionRecord",
    "QuestionRecord",
    "AnswerPolicy",
    "AutoAnswerPolicy",
    "ScriptedAnswerPolicy",
    "CallbackAnswerPolicy",
]
