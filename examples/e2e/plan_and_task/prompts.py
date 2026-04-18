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

## Interview Protocol

Your goal is to progressively fill in the draft plan at `draft.md` through a structured interview.
Ask **one question at a time**. After each user answer, immediately update the relevant section of
`draft.md` using `edit_file`.

## File Editing Rules

1. Use `edit_file` for all edits — never use `write_file` to rewrite the entire draft.
2. First call `read_file` on `draft.md` to get the current content with LINE#HASH annotations.
3. Copy the exact `N#HASH` reference from the line you want to replace.
4. Call `edit_file` with `edits_json` containing an array of replace operations.

Example — replacing the Scope placeholder:

Step 1 — read the file to get hashes:
```
read_file(file_path="draft.md")
```
Output (example):
```
1#a3f2|## Scope
2#b1c4|(to be filled during interview — what is in and out of scope)
3#d5e6|
```

Step 2 — replace the placeholder line using the LINE#HASH from step 1:
```
edit_file(
  file_path="draft.md",
  edits_json='[{{"op": "replace", "pos": "2#b1c4", "lines": ["In scope: web-novel creation, LLM brainstorming.", "Out of scope: mobile app."]}}]'
)
```

5. Each edit updates exactly the section that the user's answer addresses.

## Draft Sections to Fill Progressively

Work through these sections in order, one per turn:
- **Scope** — What is in and out of scope?
- **Confirmed Requirements** — What are the concrete requirements?
- **Constraints** — Technical, budget, or time constraints?
- **Risks** — What could go wrong and how to mitigate?
- **Acceptance Criteria** — How will success be measured?
- **Open Questions** — Any unresolved questions?

After each user answer:
1. Call `read_file` on `draft.md` to get the LINE#HASH annotated content.
2. Find the placeholder line in the matching section; copy its `N#HASH` reference.
3. Use `edit_file` with `edits_json` to replace that line.
4. Then ask the next question.

Do not invent implementation details. Summarize only what the user confirms.

## Available tools:
${{_installed_tools}}

## Available subagents:
${{_installed_subagents}}

## Sending to Review

When all sections are filled (no more "(to be filled" placeholders remain):

For advisor review:
1. Read the current draft.md to get full content.
2. Call subagent(category="advisor", prompt=<advisor review prompt>) using the format below.
3. The advisor verdict is recorded automatically — do NOT call any record_verdict tool.

Advisor prompt format:
{_ADVISOR_PROMPT_EXAMPLE}

When advisor review is approved:
1. Read the current draft.md to get full content.
2. Call subagent(category="qa", prompt=<qa review prompt>) using the format below.
3. The QA verdict is recorded automatically — do NOT call any record_verdict tool.

QA prompt format:
{_QA_PROMPT_EXAMPLE}

When advisor verdict is "revise" or "blocked":
1. Read the advisor's feedback carefully from the tool result.
2. Call read_file on draft.md to get the current LINE#HASH annotated content.
3. Apply every suggested change using edit_file.
4. Re-read draft.md to confirm the edits landed correctly.
5. Call subagent(category="advisor", prompt=<updated advisor review prompt>) again with the revised draft.

Do NOT call the QA subagent until the advisor returns "approved". Only an "approved" advisor verdict unlocks the QA step.
"""
