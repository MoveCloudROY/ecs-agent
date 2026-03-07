"""Task state transition engine with explicit guardrails."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

from ecs_agent.logging import get_logger
from ecs_agent.types import TaskStatus

logger = get_logger(__name__)


class TaskStateTransitionError(ValueError):
    """Raised when a task transition violates explicit state rules."""


@dataclass(slots=True, frozen=True)
class TaskState:
    """Runtime task state tracked by the transition engine."""

    task_id: str
    status: TaskStatus
    retry_count: int = 0
    max_retries: int = 0
    blocked_until_manual: bool = False
    blocked_reason: str | None = None


@dataclass(slots=True, frozen=True)
class TransitionRequest:
    """Requested task transition metadata."""

    target_status: TaskStatus
    manual_action: bool = False
    reason: str | None = None


Guard = Callable[[TaskState, TransitionRequest], str | None]
Mutator = Callable[[TaskState, TransitionRequest], TaskState]


@dataclass(slots=True, frozen=True)
class TransitionRule:
    """Transition rule composed of pure guards and mutator."""

    guards: tuple[Guard, ...] = ()
    mutator: Mutator | None = None


def _guard_manual_action_required(
    state: TaskState, request: TransitionRequest
) -> str | None:
    _ = state
    if request.manual_action:
        return None
    return "manual action required"


def _guard_reason_required(state: TaskState, request: TransitionRequest) -> str | None:
    _ = state
    if request.reason:
        return None
    return "reason is required"


def _guard_retry_available(state: TaskState, request: TransitionRequest) -> str | None:
    _ = request
    if state.retry_count < state.max_retries:
        return None
    return "no retries remaining"


def _clear_block_metadata(state: TaskState, request: TransitionRequest) -> TaskState:
    _ = request
    return replace(state, blocked_until_manual=False, blocked_reason=None)


def _mark_failed(state: TaskState, request: TransitionRequest) -> TaskState:
    _ = request
    return replace(
        state,
        retry_count=state.retry_count + 1,
        blocked_until_manual=False,
        blocked_reason=None,
    )


def _mark_blocked_until_manual(
    state: TaskState, request: TransitionRequest
) -> TaskState:
    return replace(
        state,
        blocked_until_manual=True,
        blocked_reason=request.reason,
    )


_TRANSITION_RULES: dict[tuple[TaskStatus, TaskStatus], TransitionRule] = {
    (TaskStatus.PENDING, TaskStatus.READY): TransitionRule(
        mutator=_clear_block_metadata
    ),
    (TaskStatus.PENDING, TaskStatus.BLOCKED): TransitionRule(
        guards=(_guard_reason_required,),
        mutator=_mark_blocked_until_manual,
    ),
    (TaskStatus.READY, TaskStatus.RUNNING): TransitionRule(
        mutator=_clear_block_metadata
    ),
    (TaskStatus.READY, TaskStatus.BLOCKED): TransitionRule(
        guards=(_guard_reason_required,),
        mutator=_mark_blocked_until_manual,
    ),
    (TaskStatus.RUNNING, TaskStatus.COMPLETED): TransitionRule(
        mutator=_clear_block_metadata
    ),
    (TaskStatus.RUNNING, TaskStatus.FAILED): TransitionRule(
        guards=(_guard_reason_required,),
        mutator=_mark_failed,
    ),
    (TaskStatus.RUNNING, TaskStatus.BLOCKED): TransitionRule(
        guards=(_guard_reason_required,),
        mutator=_mark_blocked_until_manual,
    ),
    (TaskStatus.BLOCKED, TaskStatus.READY): TransitionRule(
        guards=(_guard_manual_action_required,),
        mutator=_clear_block_metadata,
    ),
    (TaskStatus.FAILED, TaskStatus.READY): TransitionRule(
        guards=(_guard_manual_action_required, _guard_retry_available),
        mutator=_clear_block_metadata,
    ),
}


def transition_task_state(state: TaskState, request: TransitionRequest) -> TaskState:
    """Apply one deterministic transition or raise TaskStateTransitionError."""

    rule = _TRANSITION_RULES.get((state.status, request.target_status))
    if rule is None:
        allowed = sorted(
            target.value
            for source, target in _TRANSITION_RULES
            if source is state.status
        )
        message = (
            f"illegal transition for task '{state.task_id}': "
            f"{state.status.value} -> {request.target_status.value}; "
            f"allowed={allowed}"
        )
        logger.warning(
            "task_state_transition_rejected",
            task_id=state.task_id,
            from_status=state.status.value,
            to_status=request.target_status.value,
            reason="illegal_transition",
            allowed=allowed,
        )
        raise TaskStateTransitionError(message)

    for guard in rule.guards:
        violation = guard(state, request)
        if violation is None:
            continue
        message = (
            f"illegal transition for task '{state.task_id}': "
            f"{state.status.value} -> {request.target_status.value}; "
            f"reason={violation}"
        )
        logger.warning(
            "task_state_transition_rejected",
            task_id=state.task_id,
            from_status=state.status.value,
            to_status=request.target_status.value,
            reason=violation,
            manual_action=request.manual_action,
            retry_count=state.retry_count,
            max_retries=state.max_retries,
        )
        raise TaskStateTransitionError(message)

    next_state = replace(state, status=request.target_status)
    if rule.mutator is not None:
        next_state = rule.mutator(next_state, request)

    logger.info(
        "task_state_transition_applied",
        task_id=state.task_id,
        from_status=state.status.value,
        to_status=next_state.status.value,
        manual_action=request.manual_action,
        retry_count=next_state.retry_count,
        max_retries=next_state.max_retries,
        blocked_until_manual=next_state.blocked_until_manual,
    )
    return next_state


def block_task_due_to_upstream_failure(
    state: TaskState,
    *,
    dependency_task_id: str,
) -> TaskState:
    """Block task until manual action because an upstream dependency failed."""

    return transition_task_state(
        state,
        TransitionRequest(
            target_status=TaskStatus.BLOCKED,
            reason=f"upstream dependency failed: {dependency_task_id}",
        ),
    )


def manual_unblock_task(state: TaskState, *, reason: str) -> TaskState:
    """Manually clear blocked state and move task back to ready."""

    _ = reason
    return transition_task_state(
        state,
        TransitionRequest(target_status=TaskStatus.READY, manual_action=True),
    )


def manual_retry_task(state: TaskState) -> TaskState:
    """Manually retry a failed task when retries are still available."""

    return transition_task_state(
        state,
        TransitionRequest(target_status=TaskStatus.READY, manual_action=True),
    )


__all__ = [
    "TaskState",
    "TaskStateTransitionError",
    "TransitionRequest",
    "TransitionRule",
    "block_task_due_to_upstream_failure",
    "manual_retry_task",
    "manual_unblock_task",
    "transition_task_state",
]
