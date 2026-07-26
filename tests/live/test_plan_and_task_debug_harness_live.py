"""Live smoke test for the plan-and-task debug harness.

Requires ``LLM_API_KEY`` (skips otherwise). Drives a couple of real turns
through ``PlanTaskDebugSession`` in batch mode (auto-answered ask_question) and
asserts the workflow starts, produces a draft, and never wedges the runner.

Uses chat-completions by default (tool-heavy flow; see project notes on the
rutaceae gateway); override with ``LLM_API_FORMAT``.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from ecs_agent.providers.claude_model import ClaudeModel
from ecs_agent.providers.config import ApiFormat, ProviderConfig
from ecs_agent.providers.openai_model import OpenAIModel
from ecs_agent.providers.protocol import LLMModel
from examples.e2e.plan_and_task.debug import AutoAnswerPolicy, PlanTaskDebugSession
from tests.live.api_format import resolve_live_api_format


def _build_live_model(api_key: str) -> LLMModel:
    api_format = resolve_live_api_format(default=ApiFormat.OPENAI_CHAT_COMPLETIONS)
    if api_format is None:
        pytest.skip(f"Unrecognized LLM_API_FORMAT: {os.getenv('LLM_API_FORMAT')!r}")
    if api_format is ApiFormat.ANTHROPIC_MESSAGES:
        base_url = os.getenv("LLM_BASE_URL") or os.getenv("ANTHROPIC_BASE_URL") or ""
        model = os.getenv("LLM_MODEL") or os.getenv("ANTHROPIC_MODEL") or ""
        return ClaudeModel(
            config=ProviderConfig(
                provider_id="anthropic",
                base_url=base_url,
                api_key=os.getenv("ANTHROPIC_API_KEY") or api_key,
                api_format=ApiFormat.ANTHROPIC_MESSAGES,
            ),
            model=model,
        )
    base_url = os.getenv("LLM_BASE_URL", "https://api.rutaceae.com/v1")
    model = os.getenv("LLM_MODEL") or "gpt-5.6-sol"
    return OpenAIModel(
        config=ProviderConfig(
            provider_id="openai",
            base_url=base_url,
            api_key=api_key,
            api_format=api_format,
        ),
        model=model,
    )


@pytest.mark.asyncio
async def test_live_debug_session_starts_and_drafts(
    live_api_key: str, tmp_path: Path
) -> None:
    model = _build_live_model(live_api_key)
    async with await PlanTaskDebugSession.build(
        model,
        base_dir=tmp_path,
        answer_policy=AutoAnswerPolicy(),
        max_turn_seconds=180.0,
        close_model=True,
    ) as session:
        started = await session.send(
            "/plan:start Build a small CLI todo app in Python with add/list/done"
        )
        assert started.snapshot.phase == "DRAFT_INTERVIEW"
        assert started.snapshot.workflow_id
        assert "plan/draft.md" in started.snapshot.artifacts
        assert session.runner_exception is None

        # One interview turn: the planner should drive the draft / ask a question.
        follow = await session.send("Looks good — proceed with your recommendations.")
        assert follow.ok, follow.note
        assert session.runner_exception is None
        # Draft should be non-empty prose by now.
        assert session.read_artifact("plan/draft.md").strip()
