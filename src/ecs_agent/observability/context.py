"""Async-safe observability context helpers."""

from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass


_trace_id: ContextVar[str | None] = ContextVar("ecs_agent_trace_id", default=None)
_run_id: ContextVar[str | None] = ContextVar("ecs_agent_run_id", default=None)
_observation_stack: ContextVar[tuple[str, ...]] = ContextVar(
    "ecs_agent_observation_stack",
    default=(),
)


@dataclass(slots=True)
class RunContextToken:
    """Tokens needed to reset trace and run context."""

    trace_id: Token[str | None]
    run_id: Token[str | None]
    observation_stack: Token[tuple[str, ...]]


@dataclass(slots=True)
class ObservationContextToken:
    """Token needed to reset the observation stack."""

    stack: Token[tuple[str, ...]]


def set_run_context(*, trace_id: str, run_id: str) -> RunContextToken:
    """Set run-scoped identifiers for the current async context."""
    trace_token = _trace_id.set(trace_id)
    run_token = _run_id.set(run_id)
    stack_token = _observation_stack.set(())
    return RunContextToken(
        trace_id=trace_token,
        run_id=run_token,
        observation_stack=stack_token,
    )


def reset_run_context(token: RunContextToken) -> None:
    """Reset run-scoped identifiers using a token from set_run_context."""
    _observation_stack.reset(token.observation_stack)
    _run_id.reset(token.run_id)
    _trace_id.reset(token.trace_id)


def current_trace_id() -> str | None:
    """Return the current trace ID, if one has been set."""
    return _trace_id.get()


def current_run_id() -> str | None:
    """Return the current run ID, if one has been set."""
    return _run_id.get()


def push_observation(observation_id: str) -> ObservationContextToken:
    """Push an observation ID onto the current async context stack."""
    token = _observation_stack.set((*_observation_stack.get(), observation_id))
    return ObservationContextToken(stack=token)


def reset_observation(token: ObservationContextToken) -> None:
    """Reset the observation stack using a token from push_observation."""
    _observation_stack.reset(token.stack)


def current_observation_stack() -> tuple[str, ...]:
    """Return the current observation stack."""
    return _observation_stack.get()


def current_observation_id() -> str | None:
    """Return the most recent observation ID, if the stack is non-empty."""
    stack = _observation_stack.get()
    if not stack:
        return None
    return stack[-1]


def current_parent_observation_id() -> str | None:
    """Return the parent observation ID for the current stack, if any."""
    stack = _observation_stack.get()
    if len(stack) < 2:
        return None
    return stack[-2]


__all__ = [
    "ObservationContextToken",
    "RunContextToken",
    "current_observation_id",
    "current_observation_stack",
    "current_parent_observation_id",
    "current_run_id",
    "current_trace_id",
    "push_observation",
    "reset_observation",
    "reset_run_context",
    "set_run_context",
]
