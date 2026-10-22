"""Live smoke test for the plan-and-task debug harness.

Select a profile with --live; see docs/testing.md.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from examples.e2e.plan_and_task.debug import PlanTaskDebugSession

from tests.support.models import LiveModelFactory


@pytest.mark.live(
    formats=["openai_chat_completions", "openai_responses", "anthropic_messages"],
    capabilities=["text", "tools"],
)
@pytest.mark.workflow
@pytest.mark.asyncio
async def test_live_debug_session_starts_and_drafts(
    tmp_path: Path, *, live_model_factory: LiveModelFactory
) -> None:
    model = live_model_factory.create()
    async with await PlanTaskDebugSession.build(
        model,
        base_dir=tmp_path,
        surface_questions=True,
        max_turn_seconds=live_model_factory.profile.timeouts.case_seconds,
        close_model=False,
    ) as session:
        started = await session.send(
            "/plan:start Draft a small CLI todo app in Python with add/list/done. "
            "Ask me one initial clarification using ask_question and wait for my answer. "
            "This session only covers the draft interview; do not start reviews or implementation."
        )
        assert started.ok, started.note
        assert started.kind == "question"
        assert started.snapshot.phase == "DRAFT_INTERVIEW"
        assert started.snapshot.workflow_id
        assert "plan/draft.md" in started.snapshot.artifacts
        assert session.runner_exception is None

        # Answer one interview question without automatically approving reviews.
        follow = await session.answer(
            [
                "Use Python's standard library and a local JSON file. Update the draft "
                "with these choices, then ask me one follow-up question. "
                "Do not start reviews or implementation yet."
            ]
        )
        assert follow.ok, follow.note
        assert follow.snapshot.phase == "DRAFT_INTERVIEW"
        assert session.runner_exception is None
        # Draft should be non-empty prose by now.
        assert session.read_artifact("plan/draft.md").strip()
