"""TDD tests for configurable child-world runtime profiles (Task 12).

A subagent's child-world system set is chosen by a serialization-safe profile name
(SubagentConfig.runtime_profile, default "default"). Profiles are registered in a
process-level registry mapping name -> builder(ChildProfileContext) -> list[ChildSystemSpec].
"""

from __future__ import annotations

import pytest

from ecs_agent.components import LLMComponent, ToolRegistryComponent
from ecs_agent.core import World
from ecs_agent.providers.fake_model import FakeModel
from ecs_agent.systems.compaction import CompactionSystem
from ecs_agent.systems.error_handling import ErrorHandlingSystem
from ecs_agent.systems.reasoning import ReasoningSystem
from ecs_agent.systems.subagent import SubagentSystem
from ecs_agent.systems.subagent.child_world import ChildWorldBuilder
from ecs_agent.systems.subagent.runtime_profiles import (
    ChildProfileContext,
    ChildSystemSpec,
    register_child_runtime_profile,
    resolve_child_runtime_profile,
)
from ecs_agent.systems.system_prompt_render_system import SystemPromptRenderSystem
from ecs_agent.types import SubagentConfig


def _registered_system_types(child_world: World) -> list[type]:
    child_world._systems.apply_queued_operations()
    return [type(entry.system) for entry in child_world._systems._systems]


def _parent_world() -> tuple[World, int]:
    world = World()
    parent = world.create_entity()
    world.add_component(
        parent,
        LLMComponent(model=FakeModel(responses=[]), system_prompt="parent"),
    )
    world.add_component(parent, ToolRegistryComponent(tools={}, handlers={}))
    return world, parent


# --- (a) default profile content ------------------------------------------------


def test_default_profile_without_compaction_reproduces_current_set() -> None:
    builder = resolve_child_runtime_profile("default")
    specs = builder(ChildProfileContext(parent_has_compaction=False))
    by_type = {type(spec.factory()).__name__: spec.priority for spec in specs}

    assert by_type["SystemPromptRenderSystem"] == -20
    assert by_type["ReasoningSystem"] == 0
    assert by_type["ErrorHandlingSystem"] == 99
    assert "CompactionSystem" not in by_type


def test_default_profile_with_compaction_adds_compaction_system() -> None:
    builder = resolve_child_runtime_profile("default")
    specs = builder(ChildProfileContext(parent_has_compaction=True))
    by_type = {type(spec.factory()).__name__: spec.priority for spec in specs}

    assert "CompactionSystem" in by_type
    assert by_type["CompactionSystem"] == -30


def test_none_runtime_profile_resolves_to_default() -> None:
    assert resolve_child_runtime_profile(None) is resolve_child_runtime_profile("default")


# --- (b) ChildWorldBuilder uses the default profile -----------------------------


def test_child_world_registers_default_profile_systems() -> None:
    world, parent = _parent_world()
    config = SubagentConfig(name="child", model=FakeModel(responses=[]))

    child_world, _ = ChildWorldBuilder().assemble_child_world(world, parent, config)
    types = _registered_system_types(child_world)

    assert SystemPromptRenderSystem in types
    assert ReasoningSystem in types
    assert ErrorHandlingSystem in types
    assert CompactionSystem not in types


# --- (c) a custom profile replaces the whole set --------------------------------


def test_custom_profile_replaces_child_system_set() -> None:
    register_child_runtime_profile(
        "reasoning_only_test",
        lambda ctx: [ChildSystemSpec(factory=lambda: ReasoningSystem(priority=0), priority=0)],
    )
    world, parent = _parent_world()
    config = SubagentConfig(
        name="child", model=FakeModel(responses=[]), runtime_profile="reasoning_only_test"
    )

    child_world, _ = ChildWorldBuilder().assemble_child_world(world, parent, config)
    types = _registered_system_types(child_world)

    assert ReasoningSystem in types
    assert SystemPromptRenderSystem not in types
    assert ErrorHandlingSystem not in types


# --- (d) serialization round-trip -----------------------------------------------


def test_runtime_profile_round_trips_through_serializer() -> None:
    from ecs_agent.components import SubagentRegistryComponent
    from ecs_agent.serialization import WorldSerializer

    world = World()
    entity = world.create_entity()
    model = FakeModel(responses=[])
    world.add_component(
        entity,
        SubagentRegistryComponent(
            subagents={
                "child": SubagentConfig(
                    name="child", model=model, runtime_profile="reasoning_only_test"
                )
            }
        ),
    )

    data = WorldSerializer.to_dict(world)
    restored = WorldSerializer.from_dict(
        data, providers={"default": model}, tool_handlers={}
    )

    reg = restored.get_component(entity, SubagentRegistryComponent)
    assert reg is not None
    assert reg.subagents["child"].runtime_profile == "reasoning_only_test"


# --- (e) unknown profile raises a clear error -----------------------------------


def test_unknown_profile_raises_clear_error() -> None:
    with pytest.raises(ValueError, match="runtime profile"):
        resolve_child_runtime_profile("does_not_exist_profile")


def test_assemble_child_world_unknown_profile_raises() -> None:
    world, parent = _parent_world()
    config = SubagentConfig(
        name="child", model=FakeModel(responses=[]), runtime_profile="nope_missing"
    )
    with pytest.raises(ValueError, match="runtime profile"):
        ChildWorldBuilder().assemble_child_world(world, parent, config)


def test_subagent_config_defaults_runtime_profile_to_none() -> None:
    config = SubagentConfig(name="c", model=FakeModel(responses=[]))
    assert config.runtime_profile is None
