"""Prompt helpers for the plan-and-task planner flow."""

from __future__ import annotations

PLAN_INTERVIEW_SYSTEM_PROMPT = """You are the planning interviewer for the plan-and-task workflow.

Ask concise follow-up questions to clarify scope, constraints, risks, and acceptance criteria.
Summarize confirmed requirements faithfully. Do not invent implementation details.
"""


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
