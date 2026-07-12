"""The interactive loop must hand control back to the user after an answered
``ask_question``, whatever the resumed turn does.

Regression: when the model narrates + calls ask_question in one message and the
resumed (second) turn errors or stalls, ReasoningSystem fires no
ReasoningCompleteEvent and leaves no PendingToolCallsComponent, so it silently
re-invokes the same failing call every tick. The session then looks frozen with
the spinner spinning forever right after the user answered — it never returns to
the input prompt. The bridge now re-arms the prompt on an agent error so control
returns to the user.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest

from examples.e2e.plan_and_task.tui.app import QuestionScreen


class _Model:
    """Streams narration + an ask_question first; the resume response is pluggable."""

    def __init__(self, resume: str) -> None:
        self.calls: list[list[tuple[str, str]]] = []
        self.model_id = "scripted"
        self._resume = resume

    async def complete(self, messages, tools=None, stream=False, response_format=None):  # type: ignore[no-untyped-def]
        from ecs_agent.types import CompletionResult, Message, Usage

        self.calls.append([(m.role, m.content or "") for m in messages])
        idx = len(self.calls) - 1
        if not stream:
            return CompletionResult(
                message=Message(role="assistant", content="scripted"),
                usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            )
        return self._ask_stream() if idx == 0 else self._resume_stream()

    async def _ask_stream(self):  # type: ignore[no-untyped-def]
        from ecs_agent.types import StreamDelta, ToolCall, Usage

        # Narration and the tool call arrive in the same assistant message,
        # exactly as the real interview turn does.
        for char in "I've written the Scope into the draft. ":
            yield StreamDelta(content=char)
        yield StreamDelta(
            tool_calls=[
                ToolCall(
                    id="c1",
                    name="ask_question",
                    arguments={
                        "questions": [
                            {
                                "header": "Scope",
                                "question": "Confirm the three dimensions?",
                                "options": [{"label": "Confirm"}, {"label": "Tweak"}],
                            }
                        ]
                    },
                )
            ]
        )
        yield StreamDelta(
            finish_reason="tool_calls",
            usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )

    async def _resume_stream(self):  # type: ignore[no-untyped-def]
        from ecs_agent.types import StreamDelta, Usage

        if self._resume == "error":
            raise RuntimeError("gateway blew up on resume")
        if self._resume == "stall":
            # Partial content flows, then the connection goes silent: the read
            # timeout fires mid-stream, exactly like a dead gateway on the
            # resumed turn. ReasoningSystem must record the partial + error and
            # the frontend must re-arm — not hang.
            yield StreamDelta(content="Starting")
            raise httpx.ReadTimeout("stream went silent mid-resume")
        for char in self._resume:  # may be "" → empty assistant message
            yield StreamDelta(content=char)
        yield StreamDelta(
            finish_reason="stop",
            usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )


async def _drive(resume: str, tmp_path) -> dict:
    from textual.widgets import RadioButton

    from ecs_agent.core import Runner
    from examples.e2e.plan_and_task.main import build_plan_task_world
    from examples.e2e.plan_and_task.tui.session import create_tui_session

    model = _Model(resume)
    world, agent_id, _adapter_ref, runtime_state = await build_plan_task_world(
        model=model, base_dir=tmp_path
    )
    session = create_tui_session(world, agent_id, runtime_state)

    async def wait_for(predicate, timeout: float = 8.0) -> bool:
        deadline = asyncio.get_running_loop().time() + timeout
        while not predicate():
            if asyncio.get_running_loop().time() > deadline:
                return False
            await asyncio.sleep(0.02)
        return True

    async with session.app.run_test() as pilot:
        runner_task = asyncio.create_task(Runner().run(world, max_ticks=None))
        await wait_for(lambda: session.bridge.input_pending)
        session.bridge.submit_input("hi")

        await wait_for(lambda: isinstance(session.app.screen, QuestionScreen))
        for _ in range(10):
            await pilot.pause()
            if session.app.screen.query(RadioButton):
                break
        session.app.screen.query(RadioButton).first().value = True
        await pilot.pause()
        await pilot.press("ctrl+s")

        second_call = await wait_for(lambda: len(model.calls) >= 2, timeout=6.0)
        # The one behaviour every resume must satisfy: control returns to the
        # user (a fresh input prompt is armed) and the runner is still alive.
        returned_to_idle = await wait_for(
            lambda: session.bridge.input_pending, timeout=6.0
        )
        outcome = {
            "second_call": second_call,
            "returned_to_idle": returned_to_idle,
            "runner_done_early": runner_task.done(),
        }
        session.bridge.request_quit()
        try:
            await asyncio.wait_for(runner_task, timeout=8)
        except (asyncio.TimeoutError, Exception):
            runner_task.cancel()
    return outcome


@pytest.mark.asyncio
async def test_resume_with_prose_returns_to_prompt(tmp_path) -> None:
    outcome = await _drive("Great — starting the plan now.", tmp_path)
    assert outcome == {
        "second_call": True,
        "returned_to_idle": True,
        "runner_done_early": False,
    }, outcome


@pytest.mark.asyncio
async def test_resume_empty_message_returns_to_prompt(tmp_path) -> None:
    outcome = await _drive("", tmp_path)
    assert outcome == {
        "second_call": True,
        "returned_to_idle": True,
        "runner_done_early": False,
    }, outcome


@pytest.mark.asyncio
async def test_resume_error_returns_to_prompt_instead_of_spinning(tmp_path) -> None:
    # The regression: a failed resume must not spin forever — the prompt is
    # re-armed so the user regains control.
    outcome = await _drive("error", tmp_path)
    assert outcome["returned_to_idle"], outcome
    assert not outcome["runner_done_early"], outcome


@pytest.mark.asyncio
async def test_resume_mid_stream_stall_returns_to_prompt(tmp_path) -> None:
    # The primary motivating case: a stall (ReadTimeout) after partial content
    # surfaces as an error and hands control back — it does not hang the turn.
    outcome = await _drive("stall", tmp_path)
    assert outcome["returned_to_idle"], outcome
    assert not outcome["runner_done_early"], outcome
