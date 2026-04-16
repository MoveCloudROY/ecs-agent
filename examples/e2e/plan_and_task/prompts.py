"""Prompt helpers for the plan-and-task planner flow."""

from __future__ import annotations


def build_draft_prompt(description: str, questions: list[str]) -> str:
    """Build the planner prompt that turns interview notes into a draft."""
    question_lines = "\n".join(f"- {question}" for question in questions) or "- None"
    return (
        "Draft a workflow plan from the interview context below.\n\n"
        f"Description:\n{description}\n\n"
        "Open questions:\n"
        f"{question_lines}\n\n"
        "Produce a concise draft with assumptions, open questions, and confirmed requirements."
    )


def build_advisor_prompt(draft_content: str) -> str:
    """Build the advisor review prompt for the current draft."""
    return (
        "Review the following workflow draft as an advisor.\n"
        "Decide whether it is ready for QA review, and explain any revisions still needed.\n\n"
        f"Draft:\n{draft_content}"
    )


def build_qa_prompt(draft_content: str, advisor_verdict: str) -> str:
    """Build the QA review prompt for the current draft."""
    return (
        "Review the workflow draft as QA.\n"
        f"Advisor verdict: {advisor_verdict}\n"
        "Confirm that the plan is specific, testable, and execution-ready.\n\n"
        f"Draft:\n{draft_content}"
    )


_ADVISOR_PROMPT_EXAMPLE = build_advisor_prompt("<current draft content>")
_QA_PROMPT_EXAMPLE = build_qa_prompt(
    "<current draft content>",
    "<advisor verdict>",
)


PLAN_INTERVIEW_SYSTEM_PROMPT = f"""You are the planning interviewer for the plan-and-task workflow.

Ask concise follow-up questions to clarify scope, constraints, risks, and acceptance criteria.
Summarize confirmed requirements faithfully. Do not invent implementation details.

Available tools:
${{_installed_tools}}

Available subagents:
${{_installed_subagents}}

When the workflow draft is ready for advisor review:
1. Call subagent(category="advisor", prompt=<advisor review prompt>) using the advisor prompt format below.
2. The advisor verdict is recorded automatically — do NOT call any record_verdict tool.

Advisor prompt format:
{_ADVISOR_PROMPT_EXAMPLE}

When advisor review is approved and the workflow draft is ready for QA:
1. Call subagent(category="qa", prompt=<qa review prompt>) using the QA prompt format below.
2. The QA verdict is recorded automatically — do NOT call any record_verdict tool.

QA prompt format:
{_QA_PROMPT_EXAMPLE}
"""
