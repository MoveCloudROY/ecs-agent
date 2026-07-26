"""Headless, scriptable, introspectable front end for the plan-and-task world.

``PlanTaskDebugSession`` is a third front end for the same event/future contract
the stdin REPL (``runtime.setup_interactive_input``) and the Textual TUI
(``tui.PlanTaskTuiBridge``) implement — but drivable by an agent or a test
instead of a human. It reuses ``build_plan_task_world`` verbatim and attaches the
same ``UserInputSystem`` (−15) + ``TerminalCleanupSystem`` wiring, so a bug seen
here is a real bug.

Usage::

    async with await PlanTaskDebugSession.build(model=model) as session:
        result = await session.send("/plan:start Build a demo")
        print(result.snapshot.phase)          # -> "DRAFT_INTERVIEW"
        print(session.read_artifact("plan/draft.md"))

Every ``send`` runs exactly one turn (resolve the pending user-input future, then
wait for the next turn boundary — the next input request, a surfaced
``ask_question``, or a terminal/timeout) and returns a :class:`TurnResult`
capturing what happened.
"""

from __future__ import annotations

import asyncio
import contextlib
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ecs_agent.accounting.models import LLMInvocationEvent
from ecs_agent.components import UserInputComponent
from ecs_agent.components.definitions import (
    ConversationComponent,
    PhaseComponent,
    TerminalComponent,
)
from ecs_agent.core import Runner, World
from ecs_agent.logging import get_logger
from ecs_agent.providers.protocol import LLMModel
from ecs_agent.systems import TerminalCleanupSystem
from ecs_agent.systems.user_input import UserInputSystem
from ecs_agent.types import (
    CompactionCompleteEvent,
    DelegationCompletedEvent,
    DelegationStartedEvent,
    EntityId,
    ErrorOccurredEvent,
    PhaseChangedEvent,
    PromptReplacementEvent,
    ReasoningCompleteEvent,
    RunCompletedEvent,
    SubagentStreamDeltaEvent,
    SubagentStreamEndEvent,
    SubagentStreamStartEvent,
    ToolExecutionCompletedEvent,
    ToolExecutionStartedEvent,
    UserInputReceivedEvent,
    UserInputRequestedEvent,
)
from examples.e2e.plan_and_task.ask_tool import (
    AskQuestion,
    QuestionAnswer,
    UserQuestionRequestedEvent,
)
from examples.e2e.plan_and_task.main import build_plan_task_world
from examples.e2e.plan_and_task.phase_graph import REVIEW_VERDICTS
from examples.e2e.plan_and_task.scratchbook_adapter import (
    PlanTaskScratchbookAdapter as ArtifactAdapter,
)
from examples.e2e.plan_and_task.state_models import RuntimeState
from examples.e2e.plan_and_task.debug.policies import AnswerPolicy, AutoAnswerPolicy

logger = get_logger(__name__)

_INPUT_PROMPT = "You> "
_EXIT_WORDS = frozenset({"exit", "quit"})
_REVIEWER_NAMES = frozenset({"advisor", "qa", "plan_qa"})
_DEFAULT_MAX_TURN_SECONDS = 180.0

# Reviewer replies end with a `VERDICT: <token>` line; a bare-word scan is the
# fallback. Mirrors ``main._extract_verdict_from_result`` without importing it.
_VERDICT_LINE_PATTERN = re.compile(
    r"^[ \t]*verdict[ \t]*:[ \t]*(" + "|".join(REVIEW_VERDICTS) + r")\b",
    re.IGNORECASE | re.MULTILINE,
)
_VERDICT_WORD_PATTERN = re.compile(
    r"\b(" + "|".join(REVIEW_VERDICTS) + r")\b", re.IGNORECASE
)

# Every event type the recorder subscribes to (superset of what a TurnResult
# needs) so ``session.events()`` is a faithful, replayable trace of the turn.
_RECORDED_EVENTS: tuple[type, ...] = (
    UserInputRequestedEvent,
    UserInputReceivedEvent,
    UserQuestionRequestedEvent,
    PromptReplacementEvent,
    ToolExecutionStartedEvent,
    ToolExecutionCompletedEvent,
    DelegationStartedEvent,
    DelegationCompletedEvent,
    SubagentStreamStartEvent,
    SubagentStreamDeltaEvent,
    SubagentStreamEndEvent,
    PhaseChangedEvent,
    LLMInvocationEvent,
    CompactionCompleteEvent,
    ReasoningCompleteEvent,
    ErrorOccurredEvent,
)


def _extract_verdict(result: str) -> str | None:
    """Extract a review verdict token from a reviewer subagent result."""
    marker = _VERDICT_LINE_PATTERN.findall(result)
    if marker:
        return str(marker[-1]).lower()
    word = _VERDICT_WORD_PATTERN.search(result)
    return str(word.group(1)).lower() if word else None


# --------------------------------------------------------------------------
# Result / snapshot data models
# --------------------------------------------------------------------------


@dataclass(slots=True)
class ToolCallRecord:
    """One tool call the main agent made during a turn."""

    name: str
    arguments: dict[str, Any]
    result: str | None = None
    success: bool | None = None
    duration_seconds: float | None = None


@dataclass(slots=True)
class SubagentRunRecord:
    """One delegated subagent run that completed during a turn."""

    name: str
    task: str
    result: str
    success: bool
    verdict: str | None = None
    duration_seconds: float | None = None


@dataclass(slots=True)
class PhaseTransitionRecord:
    """One committed phase transition during a turn."""

    from_phase: str
    to_phase: str
    reason: str
    forced: bool
    tick: int


@dataclass(slots=True)
class QuestionRecord:
    """An ``ask_question`` question, as asked by the agent."""

    header: str
    question: str
    options: list[str] = field(default_factory=list)
    multi_select: bool = False

    @classmethod
    def from_ask(cls, q: AskQuestion) -> QuestionRecord:
        return cls(
            header=q.header,
            question=q.question,
            options=[o.label for o in q.options],
            multi_select=q.multi_select,
        )


@dataclass(slots=True)
class StateSnapshot:
    """A structured read of the world at a point in time."""

    phase: str | None
    status: str | None
    workflow_id: str | None
    current_task_id: str | None
    review_verdicts: list[dict[str, Any]]
    tasks: list[dict[str, Any]]
    phase_history: list[dict[str, Any]]
    pending_question: list[QuestionRecord]
    conversation_messages: int
    cumulative_usage: dict[str, int]
    artifacts: list[str]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["pending_question"] = [asdict(q) for q in self.pending_question]
        return payload


@dataclass(slots=True)
class TurnResult:
    """Everything observable about a single driven turn."""

    kind: str  # "turn" | "question" | "terminal" | "timeout"
    ok: bool
    sent: str | None
    assistant_messages: list[str]
    tool_calls: list[ToolCallRecord]
    subagents: list[SubagentRunRecord]
    phase_transitions: list[PhaseTransitionRecord]
    questions_asked: list[QuestionRecord]
    errors: list[str]
    usage: dict[str, int]
    pending_question: list[QuestionRecord]
    snapshot: StateSnapshot
    note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "ok": self.ok,
            "sent": self.sent,
            "assistant_messages": self.assistant_messages,
            "tool_calls": [asdict(t) for t in self.tool_calls],
            "subagents": [asdict(s) for s in self.subagents],
            "phase_transitions": [asdict(p) for p in self.phase_transitions],
            "questions_asked": [asdict(q) for q in self.questions_asked],
            "errors": self.errors,
            "usage": self.usage,
            "pending_question": [asdict(q) for q in self.pending_question],
            "snapshot": self.snapshot.to_dict(),
            "note": self.note,
        }


class PlanTaskDebugSession:
    """Drives a plan-and-task world one turn at a time and records everything."""

    def __init__(
        self,
        world: World,
        agent_id: EntityId,
        adapter_ref: list[ArtifactAdapter | None],
        runtime_state_ref: list[RuntimeState | None],
        *,
        answer_policy: AnswerPolicy | None = None,
        surface_questions: bool = False,
        max_turn_seconds: float = _DEFAULT_MAX_TURN_SECONDS,
        close_model: bool = False,
    ) -> None:
        self._world = world
        self._agent_id = agent_id
        self._adapter_ref = adapter_ref
        self._runtime_state_ref = runtime_state_ref
        self._answer_policy: AnswerPolicy = answer_policy or AutoAnswerPolicy()
        self._surface_questions = surface_questions
        self._max_turn_seconds = max_turn_seconds
        self._close_model = close_model

        self._log: list[tuple[float, object]] = []
        self._turn_starts: list[int] = []
        self._conv_mark = 0
        self._pending_input: asyncio.Future[str] | None = None
        self._pending_question: (
            tuple[UserQuestionRequestedEvent, asyncio.Future[list[QuestionAnswer] | None]]
            | None
        ) = None
        self._wake = asyncio.Event()
        self._finished = False
        self._runner_exc: BaseException | None = None
        self._runner_task: asyncio.Task[None] | None = None
        self._started = False

    # -- construction -----------------------------------------------------

    @classmethod
    async def build(
        cls,
        model: LLMModel,
        *,
        base_dir: Path | None = None,
        answer_policy: AnswerPolicy | None = None,
        surface_questions: bool = False,
        enable_tool_sink: bool = False,
        max_turn_seconds: float = _DEFAULT_MAX_TURN_SECONDS,
        close_model: bool = False,
    ) -> PlanTaskDebugSession:
        """Build a world via ``build_plan_task_world`` and wrap it in a session."""
        world, agent_id, adapter_ref, runtime_state_ref = await build_plan_task_world(
            model, base_dir, enable_tool_sink=enable_tool_sink
        )
        session = cls(
            world,
            agent_id,
            adapter_ref,
            runtime_state_ref,
            answer_policy=answer_policy,
            surface_questions=surface_questions,
            max_turn_seconds=max_turn_seconds,
            close_model=close_model,
        )
        await session.start()
        return session

    async def start(self) -> None:
        """Attach the front end and launch the runner as a background task."""
        if self._started:
            return
        self._started = True
        bus = self._world.event_bus
        for event_type in _RECORDED_EVENTS:
            bus.subscribe(event_type, self._record)
        bus.subscribe(UserInputRequestedEvent, self._on_input_requested)
        bus.subscribe(UserQuestionRequestedEvent, self._on_question_requested)
        bus.subscribe(ReasoningCompleteEvent, self._on_reasoning_complete)
        bus.subscribe(ErrorOccurredEvent, self._on_agent_error)

        # Same wiring as the stdin runtime / TUI bridge: input runs before
        # normalization (−15 < −10) and reasoning_complete terminals are cleared
        # so the session keeps ticking.
        self._world.register_system(
            TerminalCleanupSystem(priority=1, clear_reasons=("reasoning_complete",)),
            priority=1,
        )
        self._world.register_system(UserInputSystem(priority=-15), priority=-15)
        if self._world.get_component(self._agent_id, UserInputComponent) is None:
            self._world.add_component(
                self._agent_id, UserInputComponent(prompt=_INPUT_PROMPT)
            )

        runner = Runner()
        self._runner_task = asyncio.create_task(
            runner.run(self._world, max_ticks=None)
        )
        self._runner_task.add_done_callback(self._on_runner_done)

    # -- context manager --------------------------------------------------

    async def __aenter__(self) -> PlanTaskDebugSession:
        await self.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Terminate the world and stop the runner task cleanly."""
        if not self._world.has_component(self._agent_id, TerminalComponent):
            self._world.add_component(
                self._agent_id, TerminalComponent(reason="debug_session_closed")
            )
        # Unblock anything the world is parked on so the tick can settle.
        if self._pending_input is not None and not self._pending_input.done():
            self._pending_input.set_result("exit")
        if self._pending_question is not None:
            _event, future = self._pending_question
            if not future.done():
                future.set_result(None)
        self._pending_input = None
        self._pending_question = None

        task = self._runner_task
        if task is not None and not task.done():
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(asyncio.shield(task), timeout=10.0)
            if not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
        if self._close_model:
            with contextlib.suppress(Exception):
                await self._world_model_aclose()

    async def _world_model_aclose(self) -> None:
        from ecs_agent.components import LLMComponent

        llm = self._world.get_component(self._agent_id, LLMComponent)
        if llm is not None and hasattr(llm.model, "aclose"):
            await llm.model.aclose()

    # -- driving ----------------------------------------------------------

    async def send(self, text: str, *, timeout: float | None = None) -> TurnResult:
        """Send one user input and run to the next turn boundary."""
        deadline = time.monotonic() + (timeout or self._max_turn_seconds)
        boundary = await self._await_boundary(deadline)
        if boundary != "input":
            # Already terminal (or stalled) before we could send.
            self._conv_mark = self._current_conv_len()
            return self._build_result("terminal", sent=text, mark=len(self._log))

        mark = len(self._log)
        self._turn_starts.append(mark)
        self._conv_mark = self._current_conv_len()
        future = self._pending_input
        assert future is not None
        self._pending_input = None
        self._wake.clear()
        if not future.done():
            future.set_result(text)
        return await self._run_to_boundary(text, mark, deadline)

    async def answer(
        self,
        answers: list[int | str | dict[str, Any]] | None = None,
        *,
        timeout: float | None = None,
    ) -> TurnResult:
        """Resolve a surfaced ``ask_question`` and continue the current turn.

        ``answers`` is one entry per question: an int (1-based option index), a
        string (custom free text), or a dict ``{selected: [...], custom_text:
        ...}``. ``None`` dismisses the prompt.
        """
        if self._pending_question is None:
            raise ValueError("No ask_question is awaiting an answer.")
        deadline = time.monotonic() + (timeout or self._max_turn_seconds)
        event, future = self._pending_question
        self._pending_question = None
        resolved = _resolve_answers(event.questions, answers)
        mark = len(self._log)
        self._conv_mark = self._current_conv_len()
        self._wake.clear()
        if not future.done():
            future.set_result(resolved)
        return await self._run_to_boundary(None, mark, deadline)

    async def _run_to_boundary(
        self, sent: str | None, mark: int, deadline: float
    ) -> TurnResult:
        boundary = await self._await_boundary(deadline)
        kind = {
            "input": "turn",
            "question": "question",
            "terminal": "terminal",
            "timeout": "timeout",
        }[boundary]
        return self._build_result(kind, sent=sent, mark=mark)

    async def _await_boundary(self, deadline: float) -> str:
        """Block until the next boundary; return which kind was reached."""
        while True:
            if self._finished:
                return "terminal"
            if self._surface_questions and self._pending_question is not None:
                return "question"
            if self._pending_input is not None:
                return "input"
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return "timeout"
            try:
                await asyncio.wait_for(self._wake.wait(), timeout=remaining)
            except asyncio.TimeoutError:
                return "timeout"
            self._wake.clear()

    # -- event handlers ---------------------------------------------------

    async def _record(self, event: object) -> None:
        self._log.append((time.monotonic(), event))

    async def _on_input_requested(self, event: UserInputRequestedEvent) -> None:
        if event.entity_id != self._agent_id:
            return
        self._pending_input = event.input_future
        self._wake.set()

    async def _on_question_requested(
        self, event: UserQuestionRequestedEvent
    ) -> None:
        if event.entity_id != self._agent_id:
            return
        if self._surface_questions:
            self._pending_question = (event, event.answer_future)
            self._wake.set()
            return
        # Batch mode: resolve immediately via the answer policy.
        if not event.answer_future.done():
            event.answer_future.set_result(
                self._answer_policy.answer(event.questions)
            )

    async def _on_reasoning_complete(self, event: ReasoningCompleteEvent) -> None:
        if event.entity_id != self._agent_id:
            return
        self._world.add_component(
            self._agent_id, UserInputComponent(prompt=_INPUT_PROMPT)
        )

    async def _on_agent_error(self, event: ErrorOccurredEvent) -> None:
        if event.entity_id != self._agent_id:
            return
        # A failed turn fires no ReasoningCompleteEvent, so re-arm input to
        # return control — exactly as the stdin runtime / TUI bridge do.
        if self._pending_input is not None or self._pending_question is not None:
            return
        self._world.add_component(
            self._agent_id, UserInputComponent(prompt=_INPUT_PROMPT)
        )

    def _on_runner_done(self, task: asyncio.Task[None]) -> None:
        self._finished = True
        with contextlib.suppress(asyncio.CancelledError):
            self._runner_exc = task.exception()
        self._wake.set()

    # -- result building --------------------------------------------------

    def _build_result(self, kind: str, *, sent: str | None, mark: int) -> TurnResult:
        events = [event for _ts, event in self._log[mark:]]
        tool_calls = self._collect_tool_calls(events)
        subagents = self._collect_subagents(events)
        transitions = self._collect_transitions(events)
        questions_asked = self._collect_questions(events)
        errors = [
            f"{e.system_name}: {e.error}"
            for e in events
            if isinstance(e, ErrorOccurredEvent) and e.entity_id == self._agent_id
        ]
        if self._runner_exc is not None and kind == "terminal":
            errors.append(f"runner: {self._runner_exc!r}")
        usage = self._collect_usage(events)
        snapshot = self.snapshot()
        note: str | None = None
        if kind == "timeout":
            note = f"No turn boundary within {self._max_turn_seconds:.0f}s (stall)."
        elif kind == "terminal":
            note = "World reached a terminal state; no further turns."
        return TurnResult(
            kind=kind,
            ok=kind in ("turn", "question"),
            sent=sent,
            assistant_messages=self._new_assistant_messages(),
            tool_calls=tool_calls,
            subagents=subagents,
            phase_transitions=transitions,
            questions_asked=questions_asked,
            errors=errors,
            usage=usage,
            pending_question=snapshot.pending_question,
            snapshot=snapshot,
            note=note,
        )

    def _collect_tool_calls(self, events: list[object]) -> list[ToolCallRecord]:
        started: dict[str, ToolExecutionStartedEvent] = {}
        order: list[str] = []
        for e in events:
            if isinstance(e, ToolExecutionStartedEvent) and e.entity_id == self._agent_id:
                started[e.tool_call.id] = e
                order.append(e.tool_call.id)
        completed: dict[str, ToolExecutionCompletedEvent] = {
            e.tool_call_id: e
            for e in events
            if isinstance(e, ToolExecutionCompletedEvent)
            and e.entity_id == self._agent_id
        }
        records: list[ToolCallRecord] = []
        for call_id in order:
            start = started[call_id]
            done = completed.get(call_id)
            args = start.tool_call.arguments
            records.append(
                ToolCallRecord(
                    name=start.tool_call.name,
                    arguments=args if isinstance(args, dict) else {"_raw": args},
                    result=None if done is None else done.result,
                    success=None if done is None else done.success,
                    duration_seconds=None if done is None else done.duration_seconds,
                )
            )
        return records

    def _collect_subagents(self, events: list[object]) -> list[SubagentRunRecord]:
        records: list[SubagentRunRecord] = []
        for e in events:
            if not isinstance(e, DelegationCompletedEvent):
                continue
            if e.entity_id != self._agent_id:
                continue
            verdict = (
                _extract_verdict(e.result)
                if e.success and e.subagent_name in _REVIEWER_NAMES
                else None
            )
            records.append(
                SubagentRunRecord(
                    name=e.subagent_name,
                    task=e.task,
                    result=e.result,
                    success=e.success,
                    verdict=verdict,
                    duration_seconds=e.duration_seconds,
                )
            )
        return records

    def _collect_transitions(
        self, events: list[object]
    ) -> list[PhaseTransitionRecord]:
        return [
            PhaseTransitionRecord(
                from_phase=e.from_phase,
                to_phase=e.to_phase,
                reason=e.reason,
                forced=e.forced,
                tick=e.tick,
            )
            for e in events
            if isinstance(e, PhaseChangedEvent) and e.entity_id == self._agent_id
        ]

    def _collect_questions(self, events: list[object]) -> list[QuestionRecord]:
        records: list[QuestionRecord] = []
        for e in events:
            if isinstance(e, UserQuestionRequestedEvent) and e.entity_id == self._agent_id:
                records.extend(QuestionRecord.from_ask(q) for q in e.questions)
        return records

    def _collect_usage(self, events: list[object]) -> dict[str, int]:
        totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        for e in events:
            if isinstance(e, LLMInvocationEvent):
                totals["prompt_tokens"] += e.usage.prompt_tokens or 0
                totals["completion_tokens"] += e.usage.completion_tokens or 0
                totals["total_tokens"] += e.usage.total_tokens or 0
        return totals

    def _current_conv_len(self) -> int:
        conv = self._world.get_component(self._agent_id, ConversationComponent)
        return len(conv.messages) if conv is not None else 0

    def _new_assistant_messages(self) -> list[str]:
        # Assistant messages appended since the baseline captured at turn start.
        conv = self._world.get_component(self._agent_id, ConversationComponent)
        if conv is None:
            return []
        return [
            m.content
            for m in conv.messages[self._conv_mark :]
            if m.role == "assistant" and m.content
        ]

    # -- inspection -------------------------------------------------------

    def snapshot(self) -> StateSnapshot:
        """Read the current world state into a structured snapshot."""
        phase_comp = self._world.get_component(self._agent_id, PhaseComponent)
        state = self._runtime_state_ref[0]
        conv = self._world.get_component(self._agent_id, ConversationComponent)
        adapter = self._adapter_ref[0]

        pending: list[QuestionRecord] = []
        if self._pending_question is not None:
            pending = [
                QuestionRecord.from_ask(q)
                for q in self._pending_question[0].questions
            ]

        history: list[dict[str, Any]] = []
        if phase_comp is not None:
            for entry in phase_comp.history[-8:]:
                history.append(entry if isinstance(entry, dict) else {"entry": entry})

        return StateSnapshot(
            phase=phase_comp.phase if phase_comp is not None else None,
            status=state.status if state is not None else None,
            workflow_id=(
                state.workflow_id
                if state is not None
                else (adapter.workflow_id if adapter is not None else None)
            ),
            current_task_id=state.current_task_id if state is not None else None,
            review_verdicts=[
                {"phase": v.phase, "verdict": v.verdict, "notes": v.notes}
                for v in (state.review_verdicts if state is not None else [])
            ],
            tasks=[
                {
                    "task_id": t.task_id,
                    "status": t.status,
                    "retry_count": t.retry_count,
                    "dependencies": list(t.dependencies),
                }
                for t in (state.tasks if state is not None else [])
            ],
            phase_history=history,
            pending_question=pending,
            conversation_messages=len(conv.messages) if conv is not None else 0,
            cumulative_usage=self._cumulative_usage(),
            artifacts=self._list_artifacts(),
        )

    def _cumulative_usage(self) -> dict[str, int]:
        return self._collect_usage([event for _ts, event in self._log])

    def _list_artifacts(self) -> list[str]:
        adapter = self._adapter_ref[0]
        if adapter is None or not adapter.workflow_root.exists():
            return []
        root = adapter.workflow_root
        files = sorted(
            str(p.relative_to(root))
            for p in root.rglob("*")
            if p.is_file()
        )
        return files

    def read_artifact(self, relpath: str) -> str:
        """Read a file under the active workflow's scratchbook directory."""
        adapter = self._adapter_ref[0]
        if adapter is None:
            raise ValueError("No active workflow; start one with /plan:start first.")
        target = (adapter.workflow_root / relpath).resolve()
        root = adapter.workflow_root.resolve()
        if root not in target.parents and target != root:
            raise ValueError(f"Path escapes the workflow scratchbook: {relpath}")
        if not target.exists():
            raise ValueError(f"Artifact not found: {relpath}")
        return target.read_text(encoding="utf-8")

    def events(
        self, *, turn: int | None = None, kinds: list[str] | None = None
    ) -> list[dict[str, Any]]:
        """Return recorded events as JSON-friendly dicts.

        ``turn`` selects a single turn's slice (0-based; the Nth ``send``);
        ``kinds`` filters by event class name.
        """
        if turn is not None:
            if not 0 <= turn < len(self._turn_starts):
                return []
            start = self._turn_starts[turn]
            end = (
                self._turn_starts[turn + 1]
                if turn + 1 < len(self._turn_starts)
                else len(self._log)
            )
            entries = self._log[start:end]
        else:
            entries = list(self._log)
        out: list[dict[str, Any]] = []
        for ts, event in entries:
            name = type(event).__name__
            if kinds is not None and name not in kinds:
                continue
            out.append({"t": round(ts, 4), "event": name, **_event_summary(event)})
        return out

    @property
    def finished(self) -> bool:
        """True once the world has reached a terminal state."""
        return self._finished

    @property
    def runner_exception(self) -> BaseException | None:
        """The exception that terminated the runner task, if any."""
        return self._runner_exc


def _resolve_answers(
    questions: list[AskQuestion],
    answers: list[int | str | dict[str, Any]] | None,
) -> list[QuestionAnswer] | None:
    if answers is None:
        return None
    resolved: list[QuestionAnswer] = []
    for index, question in enumerate(questions):
        item: int | str | dict[str, Any] = (
            answers[index] if index < len(answers) else ""
        )
        resolved.append(_coerce_answer(question, item))
    return resolved


def _coerce_answer(
    question: AskQuestion, item: int | str | dict[str, Any]
) -> QuestionAnswer:
    selected: list[str] = []
    custom: str | None = None
    if isinstance(item, bool):
        # bool is an int subclass; treat as free text to avoid silent index use.
        custom = str(item)
    elif isinstance(item, int):
        if question.options and 1 <= item <= len(question.options):
            selected = [question.options[item - 1].label]
        else:
            custom = str(item)
    elif isinstance(item, str):
        match = _match_option_label(question, item)
        if match is not None:
            selected = [match]
        else:
            custom = item
    elif isinstance(item, dict):
        for label in item.get("selected", []) or []:
            match = _match_option_label(question, str(label))
            selected.append(match if match is not None else str(label))
        raw_custom = item.get("custom_text")
        custom = str(raw_custom) if raw_custom else None
    return QuestionAnswer(
        header=question.header,
        question=question.question,
        selected=selected,
        custom_text=custom,
    )


def _match_option_label(question: AskQuestion, value: str) -> str | None:
    for option in question.options:
        if option.label.lower() == value.strip().lower():
            return option.label
    return None


def _event_summary(event: object) -> dict[str, Any]:
    """Compact, JSON-safe projection of an event for ``events()``."""
    if isinstance(event, PhaseChangedEvent):
        return {
            "from": event.from_phase,
            "to": event.to_phase,
            "reason": event.reason,
            "forced": event.forced,
            "tick": event.tick,
        }
    if isinstance(event, ToolExecutionStartedEvent):
        return {"tool": event.tool_call.name, "id": event.tool_call.id}
    if isinstance(event, ToolExecutionCompletedEvent):
        return {
            "tool": event.tool_name,
            "success": event.success,
            "result_preview": (event.result or "")[:200],
        }
    if isinstance(event, DelegationStartedEvent):
        return {"subagent": event.subagent_name, "task": event.task[:200]}
    if isinstance(event, DelegationCompletedEvent):
        return {
            "subagent": event.subagent_name,
            "success": event.success,
            "result_preview": (event.result or "")[:200],
        }
    if isinstance(event, UserQuestionRequestedEvent):
        return {"headers": [q.header for q in event.questions]}
    if isinstance(event, ErrorOccurredEvent):
        return {"system": event.system_name, "error": event.error[:300]}
    if isinstance(event, LLMInvocationEvent):
        return {
            "model": event.model,
            "status": event.status,
            "total_tokens": event.usage.total_tokens,
        }
    if isinstance(event, PromptReplacementEvent):
        return {"kind": event.prompt_kind, "preview": event.rendered_text[:200]}
    if isinstance(event, ReasoningCompleteEvent):
        return {"model": event.model, "duration_ms": round(event.duration_ms, 1)}
    if isinstance(event, UserInputReceivedEvent):
        return {"text": event.text[:200]}
    if isinstance(event, CompactionCompleteEvent):
        return {
            "original_tokens": event.original_tokens,
            "compacted_tokens": event.compacted_tokens,
        }
    return {}


__all__ = [
    "PlanTaskDebugSession",
    "TurnResult",
    "StateSnapshot",
    "ToolCallRecord",
    "SubagentRunRecord",
    "PhaseTransitionRecord",
    "QuestionRecord",
]
