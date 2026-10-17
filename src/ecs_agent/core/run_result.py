"""Snapshot an agent run's semantic outcome independently of runner mechanics."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from ecs_agent.components import (
    ChildStubComponent,
    ConversationComponent,
    ErrorComponent,
    PendingToolCallsComponent,
    TerminalComponent,
)
from ecs_agent.outcomes import GenerationStatus
from ecs_agent.providers._message_contract import tool_span_error
from ecs_agent.types import EntityId, Message

if TYPE_CHECKING:
    from ecs_agent.core.world import World


class RunStatus(StrEnum):
    """Semantic outcome; exhausting a budget or merely stopping is not success."""

    COMPLETED = "completed"
    ERROR = "error"
    CANCELLED = "cancelled"
    INTERRUPTED = "interrupted"
    MAX_TICKS = "max_ticks"
    TIMEOUT = "timeout"
    BUDGET_EXHAUSTED = "budget_exhausted"
    INCOMPLETE = "incomplete"
    UNKNOWN = "unknown"


@dataclass(slots=True)
class RunResult:
    """Stable outcome of a run, including final or explicitly partial output."""

    run_id: str | None
    status: RunStatus
    reason: str
    ticks: int
    entity_id: EntityId | None = None
    final_message: Message | None = None
    partial_message: Message | None = None
    error: str | None = None

    @property
    def success(self) -> bool:
        """Whether the target produced a verified final result."""
        return self.status is RunStatus.COMPLETED


def snapshot_run_result(
    world: World,
    *,
    run_id: str | None,
    reason: str,
    ticks: int,
    target_entity: EntityId | None = None,
    error: str | None = None,
) -> RunResult:
    """Capture a result before event subscribers can mutate terminal state."""
    terminals = [
        (entity, terminal)
        for entity, (terminal,) in world.query(TerminalComponent)
        if not world.has_component(entity, ChildStubComponent)
        and (target_entity is None or entity == target_entity)
    ]
    # A failure must not be hidden by an unrelated successful terminal.
    terminals.sort(key=lambda item: item[1].reason == "reasoning_complete")
    entity = target_entity
    terminal = None
    if terminals and reason == "terminal_component":
        entity, terminal = terminals[0]
        reason = terminal.reason
        error = terminal.error or error

    partial = None
    latest_is_assistant = False
    history_error = None
    if entity is not None:
        conversation = world.get_component(entity, ConversationComponent)
        if conversation is not None:
            history_error = tool_span_error(conversation.messages)
            latest_is_assistant = bool(conversation.messages and conversation.messages[-1].role == "assistant")
            partial = next((
                deepcopy(message) for message in reversed(conversation.messages)
                if message.role == "assistant"
            ), None)
        pending = world.get_component(entity, PendingToolCallsComponent)
        pending_error = world.get_component(entity, ErrorComponent)
        if pending_error is not None:
            error = pending_error.error
    else:
        pending = None

    status = RunStatus.UNKNOWN
    final = None
    if reason.endswith("timeout"):
        status = RunStatus.TIMEOUT
    elif reason == "budget_exhausted":
        status = RunStatus.BUDGET_EXHAUSTED
    elif error is not None or reason.endswith("_error") or reason in {"provider_exhausted", "exception"}:
        status = RunStatus.ERROR
    elif reason == "external_cancellation" or reason.endswith("_cancelled"):
        status = RunStatus.CANCELLED
    elif reason == "interruption_component":
        status = RunStatus.INTERRUPTED
    elif reason == "max_ticks":
        status = RunStatus.MAX_TICKS
    elif reason.endswith(("_incomplete", "_filtered")):
        status = RunStatus.INCOMPLETE
    elif reason == "reasoning_complete":
        if (
            partial is not None
            and latest_is_assistant
            and history_error is None
            and partial.generation_status is GenerationStatus.COMPLETED
            and not partial.tool_calls
            and pending is None
            and (partial.content or partial.parts)
        ):
            status = RunStatus.COMPLETED
            final, partial = partial, None
        else:
            status = RunStatus.ERROR
            error = history_error or "Agent stopped without a valid final result or with pending tools"

    return RunResult(
        run_id=run_id, status=status, reason=reason, ticks=ticks,
        entity_id=entity, final_message=final, partial_message=partial,
        error=error,
    )
