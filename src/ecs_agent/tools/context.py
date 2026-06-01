"""Internal tool execution context helpers."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

from ecs_agent.components import ToolRuntimeStateComponent, ToolStateNamespace
from ecs_agent.core.world import World
from ecs_agent.types import EntityId


@dataclass(slots=True, frozen=True)
class ToolExecutionContext:
    """Framework-internal context for one tool call."""

    world: World
    entity_id: EntityId
    tool_name: str
    tool_call_id: str | None = None
    workspace_root: str | None = None
    run_id: str | None = None
    parent_entity_id: EntityId | None = None


_CURRENT_TOOL_CONTEXT: ContextVar[ToolExecutionContext | None] = ContextVar(
    "ecs_agent_current_tool_execution_context",
    default=None,
)


def current_tool_context() -> ToolExecutionContext:
    """Return the active internal tool execution context."""
    context = _CURRENT_TOOL_CONTEXT.get()
    if context is None:
        raise RuntimeError("ToolExecutionContext is not available")
    return context


@contextmanager
def use_tool_context(context: ToolExecutionContext) -> Iterator[None]:
    """Temporarily bind a tool execution context."""
    token = _CURRENT_TOOL_CONTEXT.set(context)
    try:
        yield
    finally:
        _CURRENT_TOOL_CONTEXT.reset(token)


def current_tool_runtime_state() -> ToolRuntimeStateComponent:
    """Return the active entity's runtime tool state component."""
    context = current_tool_context()
    component = context.world.get_component(context.entity_id, ToolRuntimeStateComponent)
    if component is None:
        component = ToolRuntimeStateComponent()
        context.world.add_component(context.entity_id, component)
    return component


def tool_state_namespace(name: str) -> ToolStateNamespace:
    """Return a stable namespace from the active entity's tool runtime state."""
    component = current_tool_runtime_state()
    return component.namespaces.setdefault(name, ToolStateNamespace())


__all__ = [
    "ToolExecutionContext",
    "current_tool_context",
    "current_tool_runtime_state",
    "tool_state_namespace",
    "use_tool_context",
]
