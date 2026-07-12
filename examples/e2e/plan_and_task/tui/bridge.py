"""Event-bus bridge between a plan-and-task ``World`` and the TUI.

Subscribes the ``PlanTaskViewModel`` reducer to every event the TUI renders,
owns the pending ``UserInputRequestedEvent`` future, and mirrors the REPL
loop of ``runtime.setup_interactive_input`` (re-arm ``UserInputComponent``
after each completed reasoning turn).
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import TYPE_CHECKING

from ecs_agent.accounting.models import LLMInvocationEvent
from ecs_agent.components import UserInputComponent
from ecs_agent.components.definitions import TerminalComponent
from ecs_agent.logging import get_logger
from ecs_agent.systems import TerminalCleanupSystem
from ecs_agent.systems.user_input import UserInputSystem
from ecs_agent.types import (
    CompactionCompleteEvent,
    DelegationCompletedEvent,
    DelegationStartedEvent,
    ErrorOccurredEvent,
    PhaseChangedEvent,
    PromptReplacementEvent,
    ReasoningCompleteEvent,
    StreamContentDeltaEvent,
    StreamContentStartEvent,
    StreamEndEvent,
    StreamReasoningDeltaEvent,
    StreamReasoningEndEvent,
    StreamStartEvent,
    SubagentStreamDeltaEvent,
    SubagentStreamEndEvent,
    ToolExecutionCompletedEvent,
    ToolExecutionStartedEvent,
    UserInputReceivedEvent,
    UserInputRequestedEvent,
)
from examples.e2e.plan_and_task.ask_tool import (
    QuestionAnswer,
    UserQuestionRequestedEvent,
)
from examples.e2e.plan_and_task.tui.view_model import PlanTaskViewModel, UiChange

if TYPE_CHECKING:
    from ecs_agent.core import World
    from ecs_agent.types import EntityId
    from examples.e2e.plan_and_task.state_models import RuntimeState

logger = get_logger(__name__)

_INPUT_PROMPT = "You> "
_EXIT_WORDS = frozenset({"exit", "quit"})

# Events whose handling may change the persisted runtime state, so the task
# panel is refreshed from ``runtime_state_ref`` right after they are folded.
_RUNTIME_REFRESH_EVENTS = (
    PhaseChangedEvent,
    DelegationCompletedEvent,
    PromptReplacementEvent,
)

_ROUTED_EVENTS = (
    StreamStartEvent,
    StreamReasoningDeltaEvent,
    StreamReasoningEndEvent,
    StreamContentStartEvent,
    StreamContentDeltaEvent,
    StreamEndEvent,
    UserInputReceivedEvent,
    UserInputRequestedEvent,
    PromptReplacementEvent,
    ToolExecutionStartedEvent,
    ToolExecutionCompletedEvent,
    DelegationStartedEvent,
    DelegationCompletedEvent,
    SubagentStreamDeltaEvent,
    SubagentStreamEndEvent,
    PhaseChangedEvent,
    LLMInvocationEvent,
    CompactionCompleteEvent,
    ErrorOccurredEvent,
    UserQuestionRequestedEvent,
)


class PlanTaskTuiBridge:
    """Routes ECS events into the view model and user input back into ECS."""

    def __init__(
        self,
        world: World,
        agent_id: EntityId,
        view_model: PlanTaskViewModel,
        runtime_state_ref: list[RuntimeState | None],
        on_change: Callable[[UiChange], None],
    ) -> None:
        self._world = world
        self._agent_id = agent_id
        self._view_model = view_model
        self._runtime_state_ref = runtime_state_ref
        self._on_change = on_change
        self._pending_future: asyncio.Future[str] | None = None
        self._pending_question_future: (
            asyncio.Future[list[QuestionAnswer] | None] | None
        ) = None

    @property
    def input_pending(self) -> bool:
        """True while the world is waiting on user input."""
        future = self._pending_future
        return future is not None and not future.done()

    @property
    def question_pending(self) -> bool:
        """True while the agent is blocked on an ``ask_question`` answer."""
        future = self._pending_question_future
        return future is not None and not future.done()

    def attach(self) -> None:
        """Subscribe to the world's event bus and install the input loop."""
        for event_type in _ROUTED_EVENTS:
            self._world.event_bus.subscribe(event_type, self._route_event)
        self._world.event_bus.subscribe(
            UserInputRequestedEvent, self._on_input_requested
        )
        self._world.event_bus.subscribe(
            UserQuestionRequestedEvent, self._on_question_requested
        )
        self._world.event_bus.subscribe(
            ReasoningCompleteEvent, self._on_reasoning_complete
        )
        self._world.event_bus.subscribe(
            ErrorOccurredEvent, self._on_agent_error
        )
        # Same wiring as runtime.setup_interactive_input: input runs before
        # normalization (-15 < -10) and reasoning_complete terminals are
        # cleared so the interactive session keeps ticking.
        self._world.register_system(
            TerminalCleanupSystem(priority=1, clear_reasons=("reasoning_complete",)),
            priority=1,
        )
        self._world.register_system(UserInputSystem(priority=-15), priority=-15)
        if self._world.get_component(self._agent_id, UserInputComponent) is None:
            self._world.add_component(
                self._agent_id, UserInputComponent(prompt=_INPUT_PROMPT)
            )

    def submit_input(self, text: str) -> bool:
        """Resolve the pending input future with ``text``.

        Returns False when nothing is waiting for input or the text is empty.
        ``exit``/``quit`` terminate the session like the stdin REPL does.
        """
        user_text = text.strip()
        if not user_text:
            return False
        future = self._pending_future
        if future is None or future.done():
            return False
        if user_text.lower() in _EXIT_WORDS:
            logger.info("plan_task_tui_user_exit", entity_id=int(self._agent_id))
            self._world.add_component(
                self._agent_id, TerminalComponent(reason="user_exit_command")
            )
        future.set_result(user_text)
        self._pending_future = None
        self._on_change(UiChange(section="input"))
        return True

    def submit_answers(
        self, answers: list[QuestionAnswer] | None
    ) -> bool:
        """Resolve a pending ``ask_question`` future with the user's answers.

        ``None`` signals the user dismissed the prompt. Returns False when no
        question is currently awaiting an answer.
        """
        future = self._pending_question_future
        if future is None or future.done():
            return False
        future.set_result(answers)
        self._pending_question_future = None
        self._on_change(UiChange(section="input"))
        return True

    def request_quit(self) -> None:
        """Terminate the session regardless of pending input state."""
        logger.info("plan_task_tui_quit_requested", entity_id=int(self._agent_id))
        self._world.add_component(
            self._agent_id, TerminalComponent(reason="user_exit_command")
        )
        future = self._pending_future
        if future is not None and not future.done():
            future.set_result("exit")
        self._pending_future = None
        question_future = self._pending_question_future
        if question_future is not None and not question_future.done():
            # Unblock the ask_question handler so the world tick can settle
            # before the runner tears down; None reads as "dismissed".
            question_future.set_result(None)
        self._pending_question_future = None

    # -- event handlers ---------------------------------------------------

    async def _route_event(self, event: object) -> None:
        for change in self._view_model.apply_event(event):
            self._on_change(change)
        if isinstance(event, _RUNTIME_REFRESH_EVENTS):
            state = self._runtime_state_ref[0]
            if state is not None:
                self._on_change(self._view_model.refresh_runtime(state))

    async def _on_input_requested(self, event: UserInputRequestedEvent) -> None:
        if event.entity_id != self._agent_id:
            return
        self._pending_future = event.input_future
        self._on_change(UiChange(section="input"))

    async def _on_question_requested(
        self, event: UserQuestionRequestedEvent
    ) -> None:
        if event.entity_id != self._agent_id:
            return
        # The view model (routed alongside) emits the UiChange that drives the
        # modal; this handler only captures the future the modal resolves.
        self._pending_question_future = event.answer_future

    async def _on_reasoning_complete(self, event: ReasoningCompleteEvent) -> None:
        if event.entity_id != self._agent_id:
            return
        self._world.add_component(
            self._agent_id, UserInputComponent(prompt=_INPUT_PROMPT)
        )

    async def _on_agent_error(self, event: ErrorOccurredEvent) -> None:
        if event.entity_id != self._agent_id:
            return
        # A turn that failed (the model call errored or stalled past its read
        # timeout) leaves no PendingToolCallsComponent and fires no
        # ReasoningCompleteEvent, so ReasoningSystem would silently re-invoke the
        # same failing call every tick — the session looks frozen with the
        # spinner spinning forever right after the user answered an
        # ask_question. Re-arm the input prompt so control returns to the user,
        # exactly as a completed turn does. Skip when input/question is already
        # pending so an active prompt or open modal is never clobbered.
        if self.input_pending or self.question_pending:
            return
        self._world.add_component(
            self._agent_id, UserInputComponent(prompt=_INPUT_PROMPT)
        )
