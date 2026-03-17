"""Roundtrip serialization tests for prompt normalization components."""

from __future__ import annotations

from ecs_agent.components import (
    OneShotContextPoolComponent,
    PromptConfigComponent,
    PromptContributionsComponent,
    TurnStateComponent,
)
from ecs_agent.core.world import World
from ecs_agent.prompts.contracts import PromptSectionSpec
from ecs_agent.serialization import WorldSerializer, COMPONENT_REGISTRY


class DummyProvider:
    async def complete(self, messages, tools=None, stream=False, response_format=None):
        _ = (messages, tools, stream, response_format)
        raise NotImplementedError


def test_prompt_config_component_roundtrip() -> None:
    """PromptConfigComponent serializes and deserializes correctly."""
    world = World()
    entity = world.create_entity()

    # Create a PromptConfigComponent with all fields
    config = PromptConfigComponent(
        keyword_templates={
            "inject_coding": "template-1",
            "inject_context": "template-2",
        },
        enable_context_pool=True,
        context_pool_max_chars=16384,
    )
    world.add_component(entity, config)

    # Serialize and deserialize
    data = WorldSerializer.to_dict(world)
    providers = {"default": DummyProvider()}
    tool_handlers = {}
    world2 = WorldSerializer.from_dict(
        data, providers=providers, tool_handlers=tool_handlers
    )

    # Verify roundtrip
    config2 = world2.get_component(entity, PromptConfigComponent)
    assert config2 is not None
    assert config2.keyword_templates == {
        "inject_coding": "template-1",
        "inject_context": "template-2",
    }
    assert config2.enable_context_pool is True
    assert config2.context_pool_max_chars == 16384


def test_prompt_contributions_component_roundtrip() -> None:
    """PromptContributionsComponent with nested PromptSectionSpec serializes correctly."""
    world = World()
    entity = world.create_entity()

    # Create a PromptContributionsComponent with sections
    sections = [
        PromptSectionSpec(title="Context", lines=["line1", "line2"], priority=10),
        PromptSectionSpec(title="Rules", lines=["rule1"], priority=5),
    ]
    contributions = PromptContributionsComponent(sections=sections)
    world.add_component(entity, contributions)

    # Serialize and deserialize
    data = WorldSerializer.to_dict(world)
    providers = {"default": DummyProvider()}
    tool_handlers = {}
    world2 = WorldSerializer.from_dict(
        data, providers=providers, tool_handlers=tool_handlers
    )

    # Verify roundtrip
    contributions2 = world2.get_component(entity, PromptContributionsComponent)
    assert contributions2 is not None
    assert len(contributions2.sections) == 2

    # First section
    assert contributions2.sections[0].title == "Context"
    assert contributions2.sections[0].lines == ["line1", "line2"]
    assert contributions2.sections[0].priority == 10

    # Second section
    assert contributions2.sections[1].title == "Rules"
    assert contributions2.sections[1].lines == ["rule1"]
    assert contributions2.sections[1].priority == 5


def test_oneshot_context_pool_component_roundtrip() -> None:
    """OneShotContextPoolComponent serializes and deserializes correctly."""
    world = World()
    entity = world.create_entity()

    # Create a OneShotContextPoolComponent with items, state, and counter
    pool = OneShotContextPoolComponent(
        items=[
            (10, 1, "source_a", "content_a"),
            (5, 2, "source_b", "content_b"),
        ],
        state="reserved",
        reserved_turn_id="turn-123",
        _counter=42,
    )
    world.add_component(entity, pool)

    # Serialize and deserialize
    data = WorldSerializer.to_dict(world)
    providers = {"default": DummyProvider()}
    tool_handlers = {}
    world2 = WorldSerializer.from_dict(
        data, providers=providers, tool_handlers=tool_handlers
    )

    # Verify roundtrip
    pool2 = world2.get_component(entity, OneShotContextPoolComponent)
    assert pool2 is not None
    assert pool2.items == [
        (10, 1, "source_a", "content_a"),
        (5, 2, "source_b", "content_b"),
    ]
    assert pool2.state == "reserved"
    assert pool2.reserved_turn_id == "turn-123"
    assert pool2._counter == 42


def test_turn_state_component_roundtrip() -> None:
    """TurnStateComponent serializes and deserializes correctly."""
    world = World()
    entity = world.create_entity()

    # Create a TurnStateComponent with turn IDs
    turn_state = TurnStateComponent(
        current_turn_id="turn-100",
        last_injected_turn_id="turn-99",
    )
    world.add_component(entity, turn_state)

    # Serialize and deserialize
    data = WorldSerializer.to_dict(world)
    providers = {"default": DummyProvider()}
    tool_handlers = {}
    world2 = WorldSerializer.from_dict(
        data, providers=providers, tool_handlers=tool_handlers
    )

    # Verify roundtrip
    turn_state2 = world2.get_component(entity, TurnStateComponent)
    assert turn_state2 is not None
    assert turn_state2.current_turn_id == "turn-100"
    assert turn_state2.last_injected_turn_id == "turn-99"


def test_prompt_components_serializer_registered() -> None:
    """All 4 new prompt components are registered in COMPONENT_REGISTRY."""
    component_names = {
        PromptConfigComponent.__name__,
        PromptContributionsComponent.__name__,
        OneShotContextPoolComponent.__name__,
        TurnStateComponent.__name__,
    }

    for name in component_names:
        assert name in COMPONENT_REGISTRY, f"{name} not in COMPONENT_REGISTRY"
        assert COMPONENT_REGISTRY[name] is not None
