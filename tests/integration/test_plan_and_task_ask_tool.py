"""Tests for the plan-and-task ``ask_question`` interactive tool and its TUI wiring."""

from __future__ import annotations

import asyncio
import json

import pytest

from ecs_agent.components import ToolRegistryComponent
from ecs_agent.core import World
from ecs_agent.types import EntityId
from examples.e2e.plan_and_task.ask_tool import (
    AskOption,
    AskQuestion,
    QuestionAnswer,
    UserQuestionRequestedEvent,
    format_answers,
    install_ask_question_tool,
    make_ask_question_handler,
    parse_questions,
)
from examples.e2e.plan_and_task.phase_graph import PLAN_TASK_PHASE_GRAPH
from examples.e2e.plan_and_task.tui.app import PlanTaskTuiApp, QuestionScreen
from examples.e2e.plan_and_task.tui.bridge import PlanTaskTuiBridge
from examples.e2e.plan_and_task.tui.view_model import PlanTaskViewModel, UiChange


class TestParseQuestions:
    def test_free_text_question(self) -> None:
        parsed = parse_questions(
            [{"header": "Name", "question": "What should the service be called?"}]
        )
        assert len(parsed) == 1
        assert parsed[0].header == "Name"
        assert parsed[0].options == []
        assert not parsed[0].multi_select

    def test_multiple_choice_question(self) -> None:
        parsed = parse_questions(
            [
                {
                    "header": "Storage",
                    "question": "Which datastore?",
                    "options": [
                        {"label": "Postgres", "description": "relational"},
                        {"label": "Redis"},
                    ],
                }
            ]
        )
        assert [o.label for o in parsed[0].options] == ["Postgres", "Redis"]
        assert parsed[0].options[0].description == "relational"
        assert parsed[0].options[1].description == ""

    def test_empty_list_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-empty list"):
            parse_questions([])

    def test_too_many_questions_rejected(self) -> None:
        with pytest.raises(ValueError, match="at most"):
            parse_questions(
                [{"header": f"h{i}", "question": "q"} for i in range(5)]
            )

    def test_missing_header_rejected(self) -> None:
        with pytest.raises(ValueError, match="header"):
            parse_questions([{"question": "q"}])

    def test_single_option_rejected(self) -> None:
        with pytest.raises(ValueError, match="2-4"):
            parse_questions(
                [{"header": "h", "question": "q", "options": [{"label": "only"}]}]
            )

    def test_multi_select_without_options_rejected(self) -> None:
        with pytest.raises(ValueError, match="multi_select requires options"):
            parse_questions(
                [{"header": "h", "question": "q", "multi_select": True}]
            )

    def test_option_missing_label_rejected(self) -> None:
        with pytest.raises(ValueError, match="label"):
            parse_questions(
                [
                    {
                        "header": "h",
                        "question": "q",
                        "options": [{"label": "ok"}, {"description": "no label"}],
                    }
                ]
            )


class TestFormatAnswers:
    def test_single_selection(self) -> None:
        payload = json.loads(
            format_answers(
                [QuestionAnswer(header="Storage", question="q", selected=["Postgres"])]
            )
        )
        assert payload["answers"][0]["answer"] == "Postgres"

    def test_multi_selection_joined(self) -> None:
        payload = json.loads(
            format_answers(
                [QuestionAnswer(header="h", question="q", selected=["A", "B"])]
            )
        )
        assert payload["answers"][0]["answer"] == "A; B"

    def test_custom_text_appended(self) -> None:
        payload = json.loads(
            format_answers(
                [
                    QuestionAnswer(
                        header="h", question="q", selected=["A"], custom_text="note"
                    )
                ]
            )
        )
        assert payload["answers"][0]["answer"] == "A; note"

    def test_no_answer_placeholder(self) -> None:
        payload = json.loads(
            format_answers([QuestionAnswer(header="h", question="q")])
        )
        assert payload["answers"][0]["answer"] == "(no answer)"


class TestAskQuestionHandler:
    def _world(self) -> tuple[World, EntityId]:
        world = World()
        entity_id = world.create_entity()
        world.add_component(entity_id, ToolRegistryComponent(tools={}, handlers={}))
        return world, entity_id

    async def test_invalid_arguments_return_error(self) -> None:
        world, entity_id = self._world()
        handler = make_ask_question_handler(world, entity_id)
        result = await handler(questions=[])
        assert result.startswith("Error:")

    async def test_no_frontend_returns_error(self) -> None:
        world, entity_id = self._world()
        handler = make_ask_question_handler(world, entity_id)
        result = await handler(
            questions=[{"header": "h", "question": "q"}]
        )
        assert result.startswith("Error: no interactive front end")

    async def test_publishes_event_and_returns_answers(self) -> None:
        world, entity_id = self._world()
        handler = make_ask_question_handler(world, entity_id)

        async def frontend(event: UserQuestionRequestedEvent) -> None:
            assert event.entity_id == entity_id
            answers = [
                QuestionAnswer(
                    header=q.header, question=q.question, selected=["picked"]
                )
                for q in event.questions
            ]
            event.answer_future.set_result(answers)

        world.event_bus.subscribe(UserQuestionRequestedEvent, frontend)
        result = await handler(
            questions=[{"header": "Storage", "question": "Which datastore?"}]
        )
        payload = json.loads(result)
        assert payload["answers"][0]["answer"] == "picked"

    async def test_dismissed_question_returns_cancelled(self) -> None:
        world, entity_id = self._world()
        handler = make_ask_question_handler(world, entity_id)

        async def frontend(event: UserQuestionRequestedEvent) -> None:
            event.answer_future.set_result(None)

        world.event_bus.subscribe(UserQuestionRequestedEvent, frontend)
        result = await handler(questions=[{"header": "h", "question": "q"}])
        assert json.loads(result)["cancelled"] is True

    async def test_install_registers_tool_and_schema(self) -> None:
        world, entity_id = self._world()
        install_ask_question_tool(world, entity_id)
        registry = world.get_component(entity_id, ToolRegistryComponent)
        assert registry is not None
        assert "ask_question" in registry.tools
        assert "ask_question" in registry.handlers
        # ask_question must be a barrier (not concurrency-safe): it blocks on
        # the user, so it must never overlap with sibling tool calls.
        assert not registry.tools["ask_question"].concurrency_safe


PHASE_IDS = tuple(PLAN_TASK_PHASE_GRAPH.phases_by_id)
AGENT = EntityId(1)


def _questions() -> list[AskQuestion]:
    return [
        AskQuestion(
            question="Which datastore should the service use?",
            header="Storage",
            options=[
                AskOption(label="Postgres", description="relational"),
                AskOption(label="Redis", description="in-memory"),
            ],
        )
    ]


class TestViewModelQuestion:
    async def test_fold_question_event_emits_question_section(self) -> None:
        vm = PlanTaskViewModel(agent_id=AGENT, phase_ids=PHASE_IDS)
        future: asyncio.Future[list[QuestionAnswer] | None] = (
            asyncio.get_running_loop().create_future()
        )
        changes = vm.apply_event(
            UserQuestionRequestedEvent(
                entity_id=AGENT, questions=_questions(), answer_future=future
            )
        )
        question_changes = [c for c in changes if c.section == "question"]
        assert len(question_changes) == 1
        assert question_changes[0].questions[0].header == "Storage"
        # A system transcript line records that a question was raised.
        assert vm.transcript[-1].kind == "system"
        assert "Storage" in vm.transcript[-1].text

    async def test_question_event_for_other_entity_ignored(self) -> None:
        vm = PlanTaskViewModel(agent_id=AGENT, phase_ids=PHASE_IDS)
        future: asyncio.Future[list[QuestionAnswer] | None] = (
            asyncio.get_running_loop().create_future()
        )
        changes = vm.apply_event(
            UserQuestionRequestedEvent(
                entity_id=EntityId(999),
                questions=_questions(),
                answer_future=future,
            )
        )
        assert changes == []


class TestBridgeQuestion:
    async def _build(self) -> tuple[World, EntityId, PlanTaskViewModel, object, list]:
        world = World()
        agent_id = world.create_entity()
        vm = PlanTaskViewModel(agent_id=agent_id, phase_ids=PHASE_IDS)
        seen: list = []
        bridge = PlanTaskTuiBridge(
            world=world,
            agent_id=agent_id,
            view_model=vm,
            runtime_state_ref=[None],
            on_change=seen.append,
        )
        bridge.attach()
        return world, agent_id, vm, bridge, seen

    async def test_submit_answers_resolves_question_future(self) -> None:
        world, agent_id, _vm, bridge, seen = await self._build()
        future: asyncio.Future[list[QuestionAnswer] | None] = (
            asyncio.get_running_loop().create_future()
        )
        await world.event_bus.publish(
            UserQuestionRequestedEvent(
                entity_id=agent_id, questions=_questions(), answer_future=future
            )
        )
        assert bridge.question_pending
        assert any(c.section == "question" for c in seen)

        answers = [
            QuestionAnswer(header="Storage", question="q", selected=["Postgres"])
        ]
        assert bridge.submit_answers(answers)
        assert future.result() == answers
        assert not bridge.question_pending

    async def test_submit_answers_without_pending_question_rejected(self) -> None:
        _world, _agent_id, _vm, bridge, _seen = await self._build()
        assert not bridge.submit_answers([])

    async def test_request_quit_resolves_pending_question(self) -> None:
        world, agent_id, _vm, bridge, _seen = await self._build()
        future: asyncio.Future[list[QuestionAnswer] | None] = (
            asyncio.get_running_loop().create_future()
        )
        await world.event_bus.publish(
            UserQuestionRequestedEvent(
                entity_id=agent_id, questions=_questions(), answer_future=future
            )
        )
        bridge.request_quit()
        assert future.done()
        assert future.result() is None


class _CapturingSink:
    input_pending = True

    def __init__(self) -> None:
        self.answers: list[list[QuestionAnswer] | None] = []

    def submit_input(self, text: str) -> bool:
        return True

    def submit_answers(self, answers: list[QuestionAnswer] | None) -> bool:
        self.answers.append(answers)
        return True

    def request_quit(self) -> None:
        return None


class TestQuestionModal:
    async def test_free_text_answer_submits_via_enter(self) -> None:
        from textual.widgets import Input

        vm = PlanTaskViewModel(agent_id=AGENT, phase_ids=PHASE_IDS)
        sink = _CapturingSink()
        app = PlanTaskTuiApp(view_model=vm, sink=sink)
        async with app.run_test(size=(100, 40)) as pilot:
            app.dispatch_change(
                UiChange(
                    section="question",
                    questions=[
                        AskQuestion(question="Name it?", header="Name")
                    ],
                )
            )
            await pilot.pause()
            assert isinstance(app.screen, QuestionScreen)
            app.screen.query_one("#q0-input", Input).value = "billing-service"
            await pilot.press("enter")
            await pilot.pause()
        assert len(sink.answers) == 1
        answer = sink.answers[0]
        assert answer is not None
        assert answer[0].custom_text == "billing-service"

    async def test_radio_selection_is_collected(self) -> None:
        from textual.widgets import RadioButton

        vm = PlanTaskViewModel(agent_id=AGENT, phase_ids=PHASE_IDS)
        sink = _CapturingSink()
        app = PlanTaskTuiApp(view_model=vm, sink=sink)
        async with app.run_test(size=(100, 40)) as pilot:
            app.dispatch_change(
                UiChange(section="question", questions=_questions())
            )
            await pilot.pause()
            app.screen.query(RadioButton).first().value = True
            await pilot.pause()
            await pilot.press("ctrl+s")
            await pilot.pause()
        assert len(sink.answers) == 1
        answer = sink.answers[0]
        assert answer is not None
        assert answer[0].selected == ["Postgres"]

    async def test_cancel_dismisses_with_none(self) -> None:
        vm = PlanTaskViewModel(agent_id=AGENT, phase_ids=PHASE_IDS)
        sink = _CapturingSink()
        app = PlanTaskTuiApp(view_model=vm, sink=sink)
        async with app.run_test(size=(100, 40)) as pilot:
            app.dispatch_change(
                UiChange(section="question", questions=_questions())
            )
            await pilot.pause()
            await pilot.press("escape")
            await pilot.pause()
        assert sink.answers == [None]

    async def test_submit_button_click_sends_answer(self) -> None:
        """Clicking the Submit button (the primary path) collects and sends."""
        from textual.widgets import RadioButton

        vm = PlanTaskViewModel(agent_id=AGENT, phase_ids=PHASE_IDS)
        sink = _CapturingSink()
        app = PlanTaskTuiApp(view_model=vm, sink=sink)
        async with app.run_test(size=(100, 40)) as pilot:
            app.dispatch_change(
                UiChange(section="question", questions=_questions())
            )
            for _ in range(8):
                await pilot.pause()
            app.screen.query(RadioButton).first().value = True
            await pilot.pause()
            await pilot.click("#submit")
            await pilot.pause()
        assert len(sink.answers) == 1
        answer = sink.answers[0]
        assert answer is not None
        assert answer[0].selected == ["Postgres"]


class TestAskQuestionThroughTui:
    async def test_handler_resolves_via_modal(self) -> None:
        from textual.widgets import RadioButton

        world = World()
        agent_id = world.create_entity()
        world.add_component(agent_id, ToolRegistryComponent(tools={}, handlers={}))
        install_ask_question_tool(world, agent_id)

        vm = PlanTaskViewModel(agent_id=agent_id, phase_ids=PHASE_IDS)
        app_holder: list[PlanTaskTuiApp] = []
        bridge = PlanTaskTuiBridge(
            world=world,
            agent_id=agent_id,
            view_model=vm,
            runtime_state_ref=[None],
            on_change=lambda change: app_holder[0].dispatch_change(change),
        )
        app = PlanTaskTuiApp(view_model=vm, sink=bridge)
        app_holder.append(app)
        bridge.attach()

        registry = world.get_component(agent_id, ToolRegistryComponent)
        assert registry is not None
        handler = registry.handlers["ask_question"]

        async with app.run_test(size=(100, 40)) as pilot:
            task = asyncio.create_task(
                handler(
                    questions=[
                        {
                            "header": "Storage",
                            "question": "Which datastore?",
                            "options": [
                                {"label": "Postgres"},
                                {"label": "Redis"},
                            ],
                        }
                    ]
                )
            )
            for _ in range(100):
                if isinstance(app.screen, QuestionScreen):
                    break
                await pilot.pause()
            assert isinstance(app.screen, QuestionScreen)
            app.screen.query(RadioButton).first().value = True
            await pilot.pause()
            await pilot.press("ctrl+s")
            result = await asyncio.wait_for(task, timeout=5)

        payload = json.loads(result)
        assert payload["answers"][0]["answer"] == "Postgres"
        assert not bridge.question_pending


_LONG_DESC = (
    "This is a fairly long option description that explains the trade-offs in "
    "enough detail that it cannot possibly fit on a single truncated line."
)


def _tall_questions() -> list[AskQuestion]:
    return [
        AskQuestion(
            question=f"Q{i}: " + ("Pick the datastore and justify the choice. " * 3),
            header=f"Header {i}",
            options=[
                AskOption(label="Postgres", description=_LONG_DESC),
                AskOption(label="Redis", description=_LONG_DESC),
                AskOption(label="SQLite", description=_LONG_DESC),
            ],
        )
        for i in range(3)
    ]


class TestQuestionModalLayout:
    async def test_submit_button_visible_with_tall_content(self) -> None:
        """Long, multi-question modals must keep Submit on screen.

        Regression: a fixed-height body pushed the buttons off-screen so the
        user could not submit, leaving the ask_question future unresolved and
        hanging the agent loop.
        """
        from textual.widgets import Button

        vm = PlanTaskViewModel(agent_id=AGENT, phase_ids=PHASE_IDS)
        app = PlanTaskTuiApp(view_model=vm, sink=_CapturingSink())
        async with app.run_test(size=(80, 20)) as pilot:
            app.dispatch_change(
                UiChange(section="question", questions=_tall_questions())
            )
            for _ in range(8):
                await pilot.pause()
            assert isinstance(app.screen, QuestionScreen)
            submit = app.screen.query_one("#submit", Button)
            assert app.screen.region.contains_region(submit.region), (
                "Submit button must be fully on screen even with tall content"
            )

    async def test_option_descriptions_render_as_wrapping_legend(self) -> None:
        """Descriptions appear in a multi-line legend, not a truncated label."""
        from textual.widgets import RadioButton, Static

        vm = PlanTaskViewModel(agent_id=AGENT, phase_ids=PHASE_IDS)
        app = PlanTaskTuiApp(view_model=vm, sink=_CapturingSink())
        async with app.run_test(size=(80, 30)) as pilot:
            app.dispatch_change(
                UiChange(section="question", questions=_tall_questions())
            )
            for _ in range(8):
                await pilot.pause()
            # Radio labels stay short (single line, no embedded description).
            labels = [str(rb.label) for rb in app.screen.query(RadioButton)]
            assert "Postgres" in labels
            assert all(_LONG_DESC not in label for label in labels)
            # The description shows fully in a wrapping legend that occupies
            # multiple rows.
            legends = app.screen.query(".option-legend")
            assert len(legends) == 3
            assert all(
                isinstance(node, Static) and node.region.height > 1
                for node in legends
            )


class _ScriptedStreamModel:
    """Streaming model that calls ask_question once, then answers in prose."""

    def __init__(self) -> None:
        self.calls: list[list[tuple[str, str]]] = []
        self.model_id = "scripted"

    async def complete(
        self, messages, tools=None, stream=False, response_format=None
    ):  # type: ignore[no-untyped-def]
        from ecs_agent.types import CompletionResult, Message, Usage

        self.calls.append([(m.role, m.content or "") for m in messages])
        idx = len(self.calls) - 1
        if not stream:
            return CompletionResult(
                message=Message(role="assistant", content="scripted"),
                usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            )
        return self._toolcall_stream() if idx == 0 else self._text_stream("Done.")

    async def _toolcall_stream(self):  # type: ignore[no-untyped-def]
        from ecs_agent.types import StreamDelta, ToolCall, Usage

        yield StreamDelta(
            tool_calls=[
                ToolCall(
                    id="c1",
                    name="ask_question",
                    arguments={
                        "questions": [
                            {
                                "header": "Storage",
                                "question": "Which datastore?",
                                "options": [
                                    {"label": "Postgres"},
                                    {"label": "Redis"},
                                ],
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

    async def _text_stream(self, text: str):  # type: ignore[no-untyped-def]
        from ecs_agent.types import StreamDelta, Usage

        for char in text:
            yield StreamDelta(content=char)
        yield StreamDelta(
            finish_reason="stop",
            usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )


class TestAskQuestionFeedbackEndToEnd:
    async def test_answer_is_fed_back_to_the_model(self, tmp_path: object) -> None:
        """The modal answer must reach the model's next call as a tool result.

        Full loop: build_plan_task_world + create_tui_session + a real Runner,
        with a streaming model that calls ask_question. Answering the modal must
        feed the answer back so the model is invoked again with the tool result.
        """
        from pathlib import Path

        from textual.widgets import RadioButton

        from ecs_agent.core import Runner
        from examples.e2e.plan_and_task.main import build_plan_task_world
        from examples.e2e.plan_and_task.tui.session import create_tui_session

        assert isinstance(tmp_path, Path)
        model = _ScriptedStreamModel()
        world, agent_id, _adapter_ref, runtime_state = await build_plan_task_world(
            model=model, base_dir=tmp_path
        )
        session = create_tui_session(world, agent_id, runtime_state)

        async def wait_for(predicate: object, timeout: float = 8.0) -> None:
            assert callable(predicate)
            deadline = asyncio.get_running_loop().time() + timeout
            while not predicate():
                if asyncio.get_running_loop().time() > deadline:
                    raise AssertionError("timed out")
                await asyncio.sleep(0.02)

        async with session.app.run_test() as pilot:
            runner_task = asyncio.create_task(Runner().run(world, max_ticks=None))
            await wait_for(lambda: session.bridge.input_pending)
            session.bridge.submit_input("hi")

            await wait_for(
                lambda: isinstance(session.app.screen, QuestionScreen)
            )
            for _ in range(10):
                await pilot.pause()
                if session.app.screen.query(RadioButton):
                    break
            session.app.screen.query(RadioButton).first().value = True
            await pilot.pause()
            await pilot.press("ctrl+s")

            await wait_for(lambda: len(model.calls) >= 2, timeout=6.0)
            session.bridge.request_quit()
            try:
                await asyncio.wait_for(runner_task, timeout=8)
            except asyncio.TimeoutError:
                runner_task.cancel()

        # The model's second call must include the ask_question tool result
        # carrying the chosen answer.
        second_call = model.calls[1]
        tool_messages = [content for role, content in second_call if role == "tool"]
        assert tool_messages, "answer was not fed back to the model"
        assert "Postgres" in tool_messages[-1]


class TestInteractiveModelStoreDefault:
    """Stored-response chaining must be off by default for the interactive flow.

    ``ask_question`` (and the input prompt) can pause a turn for minutes; a
    server-side ``previous_response_id`` chain can expire during that pause and,
    with the Responses stream having no read timeout, stall the next turn
    forever. The adapter always sends the full history, so chaining saves no
    tokens here — it stays off unless explicitly opted in.
    """

    def test_store_chaining_off_by_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from examples.e2e.plan_and_task.main import build_model_from_env

        monkeypatch.setenv("LLM_API_KEY", "sk-test")
        monkeypatch.setenv("LLM_API_FORMAT", "openai_responses")
        monkeypatch.delenv("LLM_ENABLE_STORE", raising=False)
        model = build_model_from_env()
        assert model._provider_config.enable_store is False

    def test_store_chaining_opt_in(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from examples.e2e.plan_and_task.main import build_model_from_env

        monkeypatch.setenv("LLM_API_KEY", "sk-test")
        monkeypatch.setenv("LLM_API_FORMAT", "openai_responses")
        monkeypatch.setenv("LLM_ENABLE_STORE", "1")
        model = build_model_from_env()
        assert model._provider_config.enable_store is True
