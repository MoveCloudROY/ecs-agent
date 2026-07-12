"""Interactive ``ask_question`` tool for the plan-and-task example.

Lets the main agent pause mid-turn and put one or more structured questions to
the user — each optionally multiple-choice — then resume with the answers as
the tool result. The tool is UI-agnostic: it publishes a
:class:`UserQuestionRequestedEvent` carrying an ``asyncio.Future`` and blocks
on it. A front end (the Textual TUI bridge, or the stdin runtime) presents the
questions and resolves the future with the collected answers.

The data models here (:class:`AskOption`, :class:`AskQuestion`,
:class:`QuestionAnswer`) are shared by the tool handler, the view model, and the
TUI modal so all three agree on the question/answer shape.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from ecs_agent.core.world import World
from ecs_agent.logging import get_logger
from ecs_agent.types import EntityId, ToolSchema

logger = get_logger(__name__)

_MAX_QUESTIONS = 4
_MIN_OPTIONS = 2
_MAX_OPTIONS = 4


@dataclass(slots=True)
class AskOption:
    """One selectable choice for a question."""

    label: str
    description: str = ""


@dataclass(slots=True)
class AskQuestion:
    """A single question put to the user.

    ``options`` is empty for a free-text question; when present it holds
    ``_MIN_OPTIONS``–``_MAX_OPTIONS`` choices. ``multi_select`` allows more than
    one option to be chosen. The user may always supply free-form text in
    addition to (or instead of) the offered options.
    """

    question: str
    header: str
    options: list[AskOption] = field(default_factory=list)
    multi_select: bool = False


@dataclass(slots=True)
class QuestionAnswer:
    """The user's answer to one :class:`AskQuestion`.

    ``selected`` holds the chosen option labels (possibly several when the
    question was multi-select); ``custom_text`` holds any free-form text the
    user typed. Both may be populated, e.g. picking an option and adding a note.
    """

    header: str
    question: str
    selected: list[str] = field(default_factory=list)
    custom_text: str | None = None

    def as_text(self) -> str:
        """Flatten the answer into a single human/LLM-readable string."""
        parts = list(self.selected)
        if self.custom_text:
            parts.append(self.custom_text)
        return "; ".join(parts) if parts else "(no answer)"


@dataclass(slots=True)
class UserQuestionRequestedEvent:
    """Emitted by ``ask_question`` when the agent needs the user to answer.

    A front end subscribes, presents ``questions``, and resolves
    ``answer_future`` with the collected :class:`QuestionAnswer` list — or
    ``None`` if the user dismissed the prompt without answering.
    """

    entity_id: EntityId
    questions: list[AskQuestion]
    answer_future: asyncio.Future[list[QuestionAnswer] | None]


def build_ask_question_schema(tool_name: str = "ask_question") -> ToolSchema:
    """Declarative schema for the interactive ``ask_question`` tool."""
    return ToolSchema(
        name=tool_name,
        description=(
            "Ask the user one or more questions and block until they answer. Use "
            "this to resolve a genuine decision that is the user's to make and that "
            "you cannot settle from context or a sensible default — e.g. clarifying "
            "ambiguous requirements during the planning interview.\n\n"
            "Each question has a short `header` (a chip label), the full `question` "
            "text, and either free-form input or 2-4 multiple-choice `options`. The "
            "user can always type a custom answer on top of the offered options.\n\n"
            "WHEN TO CALL:\n"
            "  - A requirement is ambiguous and the answer changes what you build.\n"
            "  - You are choosing between real alternatives with different trade-offs.\n"
            "  Do NOT use it for things you can decide yourself or look up.\n\n"
            "INTERFACE:\n"
            "  questions (required) — 1-4 question objects. Each: {header, question, "
            "[options], [multi_select]}. Each option: {label, [description]}.\n\n"
            "RETURNS: JSON {answers: [{header, question, answer}]}, where `answer` is "
            "the user's chosen label(s) and/or typed text. A `cancelled` payload is "
            "returned if the user dismisses the prompt.\n\n"
            "EXAMPLE:\n"
            '  ask_question(questions=[{"header": "Storage", "question": "Which '
            'datastore should the service use?", "options": [{"label": "Postgres", '
            '"description": "Relational, strong consistency"}, {"label": "Redis", '
            '"description": "In-memory, ephemeral"}]}])'
        ),
        parameters={
            "type": "object",
            "properties": {
                "questions": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": _MAX_QUESTIONS,
                    "description": "The questions to ask (1-4).",
                    "items": {
                        "type": "object",
                        "properties": {
                            "question": {
                                "type": "string",
                                "description": "The full question text to show the user.",
                            },
                            "header": {
                                "type": "string",
                                "description": "Short label (a few words) shown as a chip/heading.",
                            },
                            "multi_select": {
                                "type": "boolean",
                                "description": "Allow selecting more than one option. Default false.",
                            },
                            "options": {
                                "type": "array",
                                "minItems": _MIN_OPTIONS,
                                "maxItems": _MAX_OPTIONS,
                                "description": "2-4 choices; omit for a free-text question.",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "label": {
                                            "type": "string",
                                            "description": "The choice text the user selects.",
                                        },
                                        "description": {
                                            "type": "string",
                                            "description": "Optional one-line explanation of the choice.",
                                        },
                                    },
                                    "required": ["label"],
                                },
                            },
                        },
                        "required": ["question", "header"],
                    },
                }
            },
            "required": ["questions"],
        },
    )


def parse_questions(raw: object) -> list[AskQuestion]:
    """Validate and normalize the ``questions`` argument into dataclasses.

    Raises:
        ValueError: when the payload is structurally invalid. The message is
            surfaced to the model verbatim as an ``Error: ...`` tool result so
            it can correct the call.
    """
    if not isinstance(raw, list) or not raw:
        raise ValueError("questions must be a non-empty list of question objects.")
    if len(raw) > _MAX_QUESTIONS:
        raise ValueError(f"at most {_MAX_QUESTIONS} questions may be asked at once.")

    questions: list[AskQuestion] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"questions[{index}] must be an object.")
        question = item.get("question")
        header = item.get("header")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"questions[{index}].question must be a non-empty string.")
        if not isinstance(header, str) or not header.strip():
            raise ValueError(f"questions[{index}].header must be a non-empty string.")

        options = _parse_options(item.get("options"), index)
        multi_select = bool(item.get("multi_select", False))
        if multi_select and not options:
            raise ValueError(
                f"questions[{index}].multi_select requires options to select from."
            )
        questions.append(
            AskQuestion(
                question=question.strip(),
                header=header.strip(),
                options=options,
                multi_select=multi_select,
            )
        )
    return questions


def _parse_options(raw: object, question_index: int) -> list[AskOption]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError(f"questions[{question_index}].options must be a list.")
    if not raw:
        return []
    if not (_MIN_OPTIONS <= len(raw) <= _MAX_OPTIONS):
        raise ValueError(
            f"questions[{question_index}].options must have "
            f"{_MIN_OPTIONS}-{_MAX_OPTIONS} entries when provided."
        )
    options: list[AskOption] = []
    for opt_index, option in enumerate(raw):
        if not isinstance(option, dict):
            raise ValueError(
                f"questions[{question_index}].options[{opt_index}] must be an object."
            )
        label = option.get("label")
        if not isinstance(label, str) or not label.strip():
            raise ValueError(
                f"questions[{question_index}].options[{opt_index}].label "
                "must be a non-empty string."
            )
        description = option.get("description")
        options.append(
            AskOption(
                label=label.strip(),
                description=description.strip()
                if isinstance(description, str)
                else "",
            )
        )
    return options


def format_answers(answers: list[QuestionAnswer]) -> str:
    """Render collected answers as the JSON tool result the model reads."""
    return json.dumps(
        {
            "answers": [
                {
                    "header": answer.header,
                    "question": answer.question,
                    "answer": answer.as_text(),
                }
                for answer in answers
            ]
        },
        ensure_ascii=False,
    )


def make_ask_question_handler(
    world: World, entity_id: EntityId
) -> Callable[..., Awaitable[str]]:
    """Build the ``ask_question`` handler bound to ``entity_id``.

    The handler validates the arguments, publishes a
    :class:`UserQuestionRequestedEvent`, and awaits the front end's answer.
    """

    async def ask_question(questions: object = None) -> str:
        try:
            parsed = parse_questions(questions)
        except ValueError as exc:
            logger.warning("ask_question_invalid_arguments", exception=str(exc))
            return f"Error: {exc}"

        if not world.event_bus.has_subscribers(UserQuestionRequestedEvent):
            logger.warning("ask_question_no_frontend", entity_id=entity_id)
            return (
                "Error: no interactive front end is attached to receive questions. "
                "Answer the question yourself or proceed with a sensible default."
            )

        loop = asyncio.get_running_loop()
        future: asyncio.Future[list[QuestionAnswer] | None] = loop.create_future()
        logger.info(
            "ask_question_requested",
            entity_id=entity_id,
            question_count=len(parsed),
            headers=[q.header for q in parsed],
        )
        await world.event_bus.publish(
            UserQuestionRequestedEvent(
                entity_id=entity_id, questions=parsed, answer_future=future
            )
        )

        try:
            answers = await future
        except asyncio.CancelledError:
            logger.info("ask_question_cancelled", entity_id=entity_id)
            return json.dumps(
                {"cancelled": True, "reason": "session ended before the user answered"}
            )

        if answers is None:
            logger.info("ask_question_dismissed", entity_id=entity_id)
            return json.dumps(
                {"cancelled": True, "reason": "user dismissed the questions"}
            )
        logger.info(
            "ask_question_answered", entity_id=entity_id, answer_count=len(answers)
        )
        return format_answers(answers)

    return ask_question


def install_ask_question_tool(
    world: World, entity_id: EntityId, tool_name: str = "ask_question"
) -> None:
    """Register the ``ask_question`` tool on ``entity_id``'s tool registry."""
    from ecs_agent.components import ToolRegistryComponent

    registry = world.get_component(entity_id, ToolRegistryComponent)
    if registry is None:
        raise ValueError(f"Entity {entity_id} missing ToolRegistryComponent")
    registry.tools[tool_name] = build_ask_question_schema(tool_name)
    registry.handlers[tool_name] = make_ask_question_handler(world, entity_id)
    logger.info("ask_question_tool_installed", entity_id=entity_id, tool_name=tool_name)


__all__ = [
    "AskOption",
    "AskQuestion",
    "QuestionAnswer",
    "UserQuestionRequestedEvent",
    "build_ask_question_schema",
    "parse_questions",
    "format_answers",
    "make_ask_question_handler",
    "install_ask_question_tool",
]
