"""Integration tests for the plan-and-task debug harness (FakeModel-driven).

These exercise the third front end (``PlanTaskDebugSession``) end to end over a
real ``World`` built by ``build_plan_task_world`` — turn boundaries, tool-call
and phase-transition recording, the surface-question/answer round trip, answer
policies, snapshots, artifact reads, and the JSONL CLI dispatch.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ecs_agent.providers.fake_model import FakeModel
from ecs_agent.types import CompletionResult, Message, ToolCall
from examples.e2e.plan_and_task.ask_tool import AskQuestion, AskOption, QuestionAnswer
from examples.e2e.plan_and_task.debug import (
    AutoAnswerPolicy,
    PlanTaskDebugSession,
    ScriptedAnswerPolicy,
)
from examples.e2e.plan_and_task.debug.policies import CallbackAnswerPolicy


def _resp(content: str = "", tool_calls: list[ToolCall] | None = None) -> CompletionResult:
    return CompletionResult(
        message=Message(role="assistant", content=content, tool_calls=tool_calls or [])
    )


def _ask_call(call_id: str = "c1") -> ToolCall:
    return ToolCall(
        id=call_id,
        name="ask_question",
        arguments={
            "questions": [
                {
                    "header": "Storage",
                    "question": "Which datastore?",
                    "options": [
                        {"label": "Postgres (recommended)"},
                        {"label": "Redis"},
                    ],
                }
            ]
        },
    )


async def test_send_reaches_boundary_and_records_assistant(tmp_path: Path) -> None:
    model = FakeModel(responses=[_resp("hello from the agent")])
    async with await PlanTaskDebugSession.build(model, base_dir=tmp_path) as s:
        result = await s.send("hi there", timeout=30)

    assert result.kind == "turn"
    assert result.ok is True
    assert result.sent == "hi there"
    assert "hello from the agent" in result.assistant_messages
    assert result.snapshot.phase == "IDLE"
    # Result must be JSON-serializable for the CLI.
    json.dumps(result.to_dict())


async def test_plan_start_creates_draft_and_transitions(tmp_path: Path) -> None:
    # response[0] -> workflow-id slug derivation; response[1] -> turn reasoning.
    model = FakeModel(responses=[_resp("todo-app-demo"), _resp("Drafting the plan.")])
    async with await PlanTaskDebugSession.build(model, base_dir=tmp_path) as s:
        result = await s.send("/plan:start Build a CLI todo app", timeout=30)

        assert result.snapshot.phase == "DRAFT_INTERVIEW"
        assert result.snapshot.workflow_id == "todo-app-demo"
        transitions = [(t.from_phase, t.to_phase) for t in result.phase_transitions]
        assert ("IDLE", "DRAFT_INTERVIEW") in transitions
        draft = s.read_artifact("plan/draft.md")
        assert draft.strip()
        assert "plan/draft.md" in result.snapshot.artifacts


async def test_surface_question_then_answer_round_trip(tmp_path: Path) -> None:
    model = FakeModel(
        responses=[_resp("Let me ask.", [_ask_call()]), _resp("Proceeding.")]
    )
    async with await PlanTaskDebugSession.build(
        model, base_dir=tmp_path, surface_questions=True
    ) as s:
        r1 = await s.send("help me plan", timeout=30)
        assert r1.kind == "question"
        assert [q.header for q in r1.pending_question] == ["Storage"]
        assert [q.header for q in r1.questions_asked] == ["Storage"]

        r2 = await s.answer([1], timeout=30)
        assert r2.kind == "turn"
        assert "Proceeding." in r2.assistant_messages

    # The selected (recommended) option flowed back into the tool result.
    completions = [
        e for e in s.events(kinds=["ToolExecutionCompletedEvent"])
    ]
    assert completions
    assert "Postgres (recommended)" in completions[-1]["result_preview"]


async def test_auto_answer_policy_completes_in_one_turn(tmp_path: Path) -> None:
    model = FakeModel(
        responses=[_resp("Asking.", [_ask_call()]), _resp("Done.")]
    )
    async with await PlanTaskDebugSession.build(
        model, base_dir=tmp_path, answer_policy=AutoAnswerPolicy()
    ) as s:
        result = await s.send("plan it", timeout=30)

    assert result.kind == "turn"
    assert "Done." in result.assistant_messages
    aq = [t for t in result.tool_calls if t.name == "ask_question"]
    assert aq and aq[0].success is True
    assert aq[0].result is not None
    assert "Postgres (recommended)" in aq[0].result


async def test_scripted_answer_policy_uses_custom_text(tmp_path: Path) -> None:
    scripted = ScriptedAnswerPolicy(
        responses=[
            [
                QuestionAnswer(
                    header="Storage",
                    question="Which datastore?",
                    custom_text="Use SQLite for the demo",
                )
            ]
        ]
    )
    model = FakeModel(responses=[_resp("Asking.", [_ask_call()]), _resp("Ok.")])
    async with await PlanTaskDebugSession.build(
        model, base_dir=tmp_path, answer_policy=scripted
    ) as s:
        result = await s.send("plan it", timeout=30)

    aq = [t for t in result.tool_calls if t.name == "ask_question"][0]
    assert "Use SQLite for the demo" in (aq.result or "")


async def test_callback_answer_policy_receives_questions() -> None:
    seen: list[str] = []

    def cb(questions: list[AskQuestion]) -> list[QuestionAnswer] | None:
        seen.extend(q.header for q in questions)
        return [
            QuestionAnswer(header=q.header, question=q.question, custom_text="ok")
            for q in questions
        ]

    policy = CallbackAnswerPolicy(cb)
    answers = policy.answer(
        [AskQuestion(question="Q?", header="H", options=[AskOption("A"), AskOption("B")])]
    )
    assert seen == ["H"]
    assert answers is not None and answers[0].custom_text == "ok"


async def test_auto_policy_picks_recommended_option() -> None:
    policy = AutoAnswerPolicy()
    q = AskQuestion(
        question="Pick",
        header="H",
        options=[AskOption("First"), AskOption("Second (recommended)")],
    )
    answers = policy.answer([q])
    assert answers is not None
    assert answers[0].selected == ["Second (recommended)"]


async def test_events_and_snapshot_shapes(tmp_path: Path) -> None:
    model = FakeModel(responses=[_resp("hi")])
    async with await PlanTaskDebugSession.build(model, base_dir=tmp_path) as s:
        await s.send("hello", timeout=30)
        turn0 = s.events(turn=0)
        assert turn0
        assert all("event" in e and "t" in e for e in turn0)
        snap = s.snapshot()
        assert set(snap.cumulative_usage) == {
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
        }


async def test_status_command_makes_no_llm_call(tmp_path: Path) -> None:
    """Read-only /plan:status completes a turn without a model round-trip."""
    model = FakeModel(responses=[_resp("should-not-be-consumed")])
    async with await PlanTaskDebugSession.build(model, base_dir=tmp_path) as s:
        result = await s.send("/plan:status", timeout=30)

    assert result.kind == "turn"
    assert result.usage["total_tokens"] == 0
    assert s.events(turn=0, kinds=["LLMInvocationEvent"]) == []
    assert any("No active workflow" in m for m in result.assistant_messages)


async def test_read_artifact_rejects_path_escape(tmp_path: Path) -> None:
    model = FakeModel(responses=[_resp("todo-app"), _resp("drafting")])
    async with await PlanTaskDebugSession.build(model, base_dir=tmp_path) as s:
        await s.send("/plan:start Build a thing", timeout=30)
        with pytest.raises(ValueError):
            s.read_artifact("../../../etc/passwd")


async def test_terminal_after_exit(tmp_path: Path) -> None:
    model = FakeModel(responses=[_resp("bye")])
    async with await PlanTaskDebugSession.build(model, base_dir=tmp_path) as s:
        result = await s.send("exit", timeout=30)
        # "exit" sets a TerminalComponent; the world winds down.
        assert result.kind in ("turn", "terminal")
        # A follow-up send finds the world terminal.
        follow = await s.send("anything", timeout=10)
        assert follow.kind == "terminal"
        assert s.finished is True


# -- CLI dispatch ---------------------------------------------------------


async def test_cli_dispatch_and_fake_loader(tmp_path: Path) -> None:
    from examples.e2e.plan_and_task.debug.cli import _dispatch, _load_fake_model

    script = tmp_path / "script.json"
    script.write_text(
        json.dumps(
            {
                "responses": [
                    {
                        "content": "Asking.",
                        "tool_calls": [
                            {
                                "id": "c1",
                                "name": "ask_question",
                                "arguments": {
                                    "questions": [
                                        {
                                            "header": "Storage",
                                            "question": "Which?",
                                            "options": [
                                                {"label": "Postgres (recommended)"},
                                                {"label": "Redis"},
                                            ],
                                        }
                                    ]
                                },
                            }
                        ],
                    },
                    "Proceeding.",
                ]
            }
        ),
        encoding="utf-8",
    )
    model = _load_fake_model(script)
    async with await PlanTaskDebugSession.build(
        model, base_dir=tmp_path, surface_questions=True
    ) as s:
        sent = await _dispatch(s, {"cmd": "send", "text": "plan it"})
        assert sent["kind"] == "question"
        answered = await _dispatch(s, {"cmd": "answer", "answers": [1]})
        assert answered["kind"] == "turn"
        snap = await _dispatch(s, {"cmd": "snapshot"})
        assert snap["ok"] is True and "phase" in snap["snapshot"]
        events = await _dispatch(s, {"cmd": "events", "turn": 0})
        assert events["ok"] is True and events["events"]
        bad = await _dispatch(s, {"cmd": "nope"})
        assert bad["ok"] is False
