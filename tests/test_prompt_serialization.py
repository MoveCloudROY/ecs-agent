"""Roundtrip serialization tests for prompt normalization components."""

from __future__ import annotations

import json

from ecs_agent.components import (
    OneShotContextPoolComponent,
    UserPromptConfigComponent,
    RenderedSystemPromptComponent,
    RenderedUserPromptComponent,
    SystemPromptComponent,
    TurnStateComponent,
)
from ecs_agent.core.world import World
from ecs_agent.prompts.contracts import (
    PlaceholderSpec,
    SystemPromptConfigSpec,
    PromptSectionSpec,
    PromptTemplateSource,
)
from ecs_agent.serialization import WorldSerializer, COMPONENT_REGISTRY


class DummyProvider:
    async def complete(self, messages, tools=None, stream=False, response_format=None):
        _ = (messages, tools, stream, response_format)
        raise NotImplementedError


def test_prompt_config_component_roundtrip() -> None:
    world = World()
    entity = world.create_entity()

    config = UserPromptConfigComponent(
        triggers={
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
    config2 = world2.get_component(entity, UserPromptConfigComponent)
    assert config2 is not None
    assert config2.triggers == {
        "inject_coding": "template-1",
        "inject_context": "template-2",
    }
    assert config2.enable_context_pool is True
    assert config2.context_pool_max_chars == 16384


def test_system_prompt_component_roundtrip() -> None:
    """SystemPromptComponent with nested PromptSectionSpec serializes correctly."""
    world = World()
    entity = world.create_entity()

    sections = [
        PromptSectionSpec(title="Context", lines=["line1", "line2"], priority=10),
        PromptSectionSpec(title="Rules", lines=["rule1"], priority=5),
    ]
    system_prompt = SystemPromptComponent(
        template="Core ${toolSelection} ${exploreSection} ${librarianSection}",
        sections=sections,
        content="rendered content",
    )
    world.add_component(entity, system_prompt)

    # Serialize and deserialize
    data = WorldSerializer.to_dict(world)
    providers = {"default": DummyProvider()}
    tool_handlers = {}
    world2 = WorldSerializer.from_dict(
        data, providers=providers, tool_handlers=tool_handlers
    )

    # Verify roundtrip
    system_prompt2 = world2.get_component(entity, SystemPromptComponent)
    assert system_prompt2 is not None
    assert system_prompt2.template == system_prompt.template
    assert system_prompt2.content == "rendered content"
    assert len(system_prompt2.sections) == 2

    # First section
    assert system_prompt2.sections[0].title == "Context"
    assert system_prompt2.sections[0].lines == ["line1", "line2"]
    assert system_prompt2.sections[0].priority == 10

    # Second section
    assert system_prompt2.sections[1].title == "Rules"
    assert system_prompt2.sections[1].lines == ["rule1"]
    assert system_prompt2.sections[1].priority == 5


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


def test_rendered_system_prompt_component_roundtrip() -> None:
    world = World()
    entity = world.create_entity()

    rendered = RenderedSystemPromptComponent(
        text="rendered prompt",
        placeholder_snapshot={"installed_tools": "- bash"},
    )
    world.add_component(entity, rendered)

    data = WorldSerializer.to_dict(world)
    providers = {"default": DummyProvider()}
    world2 = WorldSerializer.from_dict(data, providers=providers, tool_handlers={})

    rendered2 = world2.get_component(entity, RenderedSystemPromptComponent)
    assert rendered2 is not None
    assert rendered2.text == "rendered prompt"
    assert rendered2.placeholder_snapshot == {"installed_tools": "- bash"}


def test_rendered_user_prompt_component_roundtrip() -> None:
    world = World()
    entity = world.create_entity()

    rendered = RenderedUserPromptComponent(
        text="rendered user prompt", turn_id="turn-42"
    )
    world.add_component(entity, rendered)

    data = WorldSerializer.to_dict(world)
    providers = {"default": DummyProvider()}
    world2 = WorldSerializer.from_dict(data, providers=providers, tool_handlers={})

    rendered2 = world2.get_component(entity, RenderedUserPromptComponent)
    assert rendered2 is not None
    assert rendered2.text == "rendered user prompt"
    assert rendered2.turn_id == "turn-42"


def test_prompt_config_spec_static_roundtrip() -> None:
    world = World()
    entity = world.create_entity()

    config = SystemPromptConfigSpec(
        template_source=PromptTemplateSource(inline="Hello ${name}"),
        placeholders=[PlaceholderSpec(name="name", value="world")],
    )
    world.add_component(entity, config)

    data = WorldSerializer.to_dict(world)
    json.loads(json.dumps(data))
    providers = {"default": DummyProvider()}
    world2 = WorldSerializer.from_dict(data, providers=providers, tool_handlers={})

    config2 = world2.get_component(entity, SystemPromptConfigSpec)
    assert config2 is not None
    assert config2.template_source.inline == "Hello ${name}"
    assert config2.template_source.file_path is None
    assert len(config2.placeholders) == 1
    assert config2.placeholders[0].name == "name"
    assert config2.placeholders[0].value == "world"


def test_prompt_components_serializer_registered() -> None:
    """All prompt components are registered in COMPONENT_REGISTRY."""
    component_names = {
        UserPromptConfigComponent.__name__,
        SystemPromptConfigSpec.__name__,
        SystemPromptComponent.__name__,
        OneShotContextPoolComponent.__name__,
        RenderedSystemPromptComponent.__name__,
        RenderedUserPromptComponent.__name__,
        TurnStateComponent.__name__,
    }

    for name in component_names:
        assert name in COMPONENT_REGISTRY, f"{name} not in COMPONENT_REGISTRY"
        assert COMPONENT_REGISTRY[name] is not None
