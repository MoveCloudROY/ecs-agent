"""Answer policies for the ``ask_question`` tool in headless debug runs.

When the debug session is *not* surfacing questions to its caller, an
``AnswerPolicy`` decides how to resolve each ``UserQuestionRequestedEvent`` so a
whole plan-and-task flow can run unattended (batch / deterministic mode).

The three built-ins cover the common needs:

- :class:`AutoAnswerPolicy` — pick the option marked ``(recommended)`` (or the
  first option), and a fixed canned string for free-text questions.
- :class:`ScriptedAnswerPolicy` — return pre-baked answers, one per
  ``ask_question`` call, falling back to a wrapped policy when exhausted.
- :class:`CallbackAnswerPolicy` — delegate to a caller-supplied function.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from examples.e2e.plan_and_task.ask_tool import AskQuestion, QuestionAnswer

_RECOMMENDED_MARKER = "(recommended)"
_DEFAULT_FREE_TEXT = (
    "Use your best judgment and proceed with the recommended approach."
)


@runtime_checkable
class AnswerPolicy(Protocol):
    """Resolves a batch of ``ask_question`` questions into answers.

    Returning ``None`` signals a dismissal (the tool result reports the user
    dismissed the prompt).
    """

    def answer(self, questions: list[AskQuestion]) -> list[QuestionAnswer] | None:
        """Produce one :class:`QuestionAnswer` per question, or ``None``."""
        ...


def _pick_option(question: AskQuestion) -> str:
    """Return the label of the recommended option, else the first option."""
    for option in question.options:
        if _RECOMMENDED_MARKER in option.label.lower():
            return option.label
    return question.options[0].label


@dataclass(slots=True)
class AutoAnswerPolicy:
    """Auto-resolve every question with the recommended/first option.

    Free-text questions (no options) get ``free_text``. This lets an entire
    interview run without a human while still exercising the real tool round
    trip.
    """

    free_text: str = _DEFAULT_FREE_TEXT

    def answer(self, questions: list[AskQuestion]) -> list[QuestionAnswer] | None:
        answers: list[QuestionAnswer] = []
        for question in questions:
            if question.options:
                answers.append(
                    QuestionAnswer(
                        header=question.header,
                        question=question.question,
                        selected=[_pick_option(question)],
                    )
                )
            else:
                answers.append(
                    QuestionAnswer(
                        header=question.header,
                        question=question.question,
                        custom_text=self.free_text,
                    )
                )
        return answers


@dataclass(slots=True)
class ScriptedAnswerPolicy:
    """Return pre-baked answers, one entry per ``ask_question`` invocation.

    ``responses`` is consumed in order; each entry is the full answer list (or
    ``None`` to dismiss) for one call. When exhausted, ``fallback`` handles the
    remaining calls (defaults to :class:`AutoAnswerPolicy`).
    """

    responses: list[list[QuestionAnswer] | None]
    fallback: AnswerPolicy = field(default_factory=AutoAnswerPolicy)
    _index: int = 0

    def answer(self, questions: list[AskQuestion]) -> list[QuestionAnswer] | None:
        if self._index < len(self.responses):
            response = self.responses[self._index]
            self._index += 1
            return response
        return self.fallback.answer(questions)


@dataclass(slots=True)
class CallbackAnswerPolicy:
    """Delegate answering to a caller-supplied function."""

    fn: Callable[[list[AskQuestion]], list[QuestionAnswer] | None]

    def answer(self, questions: list[AskQuestion]) -> list[QuestionAnswer] | None:
        return self.fn(questions)


__all__ = [
    "AnswerPolicy",
    "AutoAnswerPolicy",
    "ScriptedAnswerPolicy",
    "CallbackAnswerPolicy",
]
