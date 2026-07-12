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
