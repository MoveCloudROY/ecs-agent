"""Roundtrip serialization tests for prompt normalization components."""

from __future__ import annotations

import json

from ecs_agent.components import (
    ContextEntry,
    PromptContextQueueComponent,
    PromptContextReservationComponent,
    UserPromptConfigComponent,
    RenderedSystemPromptComponent,
    RenderedUserPromptComponent,
    SystemPromptComponent,
)
from ecs_agent.core.world import World
from ecs_agent.prompts.contracts import (
    PlaceholderSpec,
    SystemPromptConfigSpec,
    PromptTemplateSource,
    TriggerSpec,
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
        triggers=[
            TriggerSpec(
                pattern="inject_coding",
                match_mode="keyword",
                action="skill",
                content="template-1",
                priority=0,
            ),
            TriggerSpec(
                pattern="inject_context",
                match_mode="keyword",
                action="skill",
                content="template-2",
                priority=0,
            ),
        ],
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
    assert len(config2.triggers) == 2
    assert config2.triggers[0].pattern == "inject_coding"
    assert config2.triggers[0].content == "template-1"
    assert config2.triggers[1].pattern == "inject_context"
    assert config2.triggers[1].content == "template-2"
    assert config2.enable_context_pool is True
    assert config2.context_pool_max_chars == 16384



def test_prompt_context_queue_component_roundtrip() -> None:
    world = World()
    entity = world.create_entity()

    queue = PromptContextQueueComponent(
        entries=[
            ContextEntry(
                entry_id="entry-a",
                priority=10,
                source_label="source_a",
                content="content_a",
                registration_order=1,
            ),
            ContextEntry(
                entry_id="entry-b",
                priority=5,
                source_label="source_b",
                content="content_b",
                registration_order=2,
            ),
        ],
    )
    world.add_component(entity, queue)

    # Serialize and deserialize
    data = WorldSerializer.to_dict(world)
    providers = {"default": DummyProvider()}
    tool_handlers = {}
    world2 = WorldSerializer.from_dict(
        data, providers=providers, tool_handlers=tool_handlers
    )

    # Verify roundtrip
    queue2 = world2.get_component(entity, PromptContextQueueComponent)
    assert queue2 is not None
    assert [entry.entry_id for entry in queue2.entries] == ["entry-a", "entry-b"]
    assert [entry.priority for entry in queue2.entries] == [10, 5]
    assert [entry.source_label for entry in queue2.entries] == ["source_a", "source_b"]
    assert [entry.content for entry in queue2.entries] == ["content_a", "content_b"]
    assert [entry.registration_order for entry in queue2.entries] == [1, 2]


def test_prompt_context_reservation_component_roundtrip() -> None:
    world = World()
    entity = world.create_entity()

    reservation = PromptContextReservationComponent(
        reservation_id="reservation-100",
        created_at_tick=99,
        reserved_entries=[
            ContextEntry(
                entry_id="entry-reserved",
                priority=42,
                source_label="tool:search",
                content="reserved content",
                registration_order=11,
            )
        ],
    )
    world.add_component(entity, reservation)

    # Serialize and deserialize
    data = WorldSerializer.to_dict(world)
    providers = {"default": DummyProvider()}
    tool_handlers = {}
    world2 = WorldSerializer.from_dict(
        data, providers=providers, tool_handlers=tool_handlers
    )

    # Verify roundtrip
    reservation2 = world2.get_component(entity, PromptContextReservationComponent)
    assert reservation2 is not None
    assert reservation2.reservation_id == "reservation-100"
    assert reservation2.created_at_tick == 99
    assert len(reservation2.reserved_entries) == 1
    assert reservation2.reserved_entries[0].entry_id == "entry-reserved"
    assert reservation2.reserved_entries[0].source_label == "tool:search"


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
        text="rendered user prompt"
    )
    world.add_component(entity, rendered)

    data = WorldSerializer.to_dict(world)
    providers = {"default": DummyProvider()}
    world2 = WorldSerializer.from_dict(data, providers=providers, tool_handlers={})

    rendered2 = world2.get_component(entity, RenderedUserPromptComponent)
    assert rendered2 is not None
    assert rendered2.text == "rendered user prompt"


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
        ContextEntry.__name__,
        PromptContextQueueComponent.__name__,
        PromptContextReservationComponent.__name__,
        RenderedSystemPromptComponent.__name__,
        RenderedUserPromptComponent.__name__,
    }

    for name in component_names:
        assert name in COMPONENT_REGISTRY, f"{name} not in COMPONENT_REGISTRY"
        assert COMPONENT_REGISTRY[name] is not None
