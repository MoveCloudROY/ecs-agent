from __future__ import annotations

import pytest

from ecs_agent.components import ToolRuntimeStateComponent
from ecs_agent.core import World
from ecs_agent.tools.context import (
    ToolExecutionContext,
    current_tool_context,
    current_tool_runtime_state,
    tool_state_namespace,
    use_tool_context,
)


def test_current_tool_context_requires_active_context() -> None:
    with pytest.raises(RuntimeError, match="ToolExecutionContext"):
        current_tool_context()


def test_use_tool_context_sets_and_resets_context() -> None:
    world = World()
    entity_id = world.create_entity()
    context = ToolExecutionContext(
        world=world,
        entity_id=entity_id,
        tool_name="read_file",
        tool_call_id="call-1",
    )

    with use_tool_context(context):
        assert current_tool_context() is context

    with pytest.raises(RuntimeError, match="ToolExecutionContext"):
        current_tool_context()


def test_current_tool_runtime_state_lazily_adds_component() -> None:
    world = World()
    entity_id = world.create_entity()
    context = ToolExecutionContext(world=world, entity_id=entity_id, tool_name="tool")

    with use_tool_context(context):
        component = current_tool_runtime_state()

    assert world.get_component(entity_id, ToolRuntimeStateComponent) is component


def test_tool_state_namespace_returns_stable_namespace() -> None:
    world = World()
    entity_id = world.create_entity()
    context = ToolExecutionContext(world=world, entity_id=entity_id, tool_name="tool")

    with use_tool_context(context):
        first = tool_state_namespace("file")
        first.values["marker"] = "value"
        second = tool_state_namespace("file")

    assert second is first
    assert second.values == {"marker": "value"}
