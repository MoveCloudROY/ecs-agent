"""Tests for subagent types and SubagentSystem."""

from __future__ import annotations

from ecs_agent.components.definitions import SubagentRegistryComponent
from ecs_agent.core.world import World
from ecs_agent.providers.fake_provider import FakeProvider
from ecs_agent.types import (
    CompletionResult,
    DelegationCompletedEvent,
    DelegationStartedEvent,
    EntityId,
    Message,
    SubagentConfig,
)


def test_subagent_config_dataclass() -> None:
    """Verify SubagentConfig has all required fields."""
    provider = FakeProvider(
        responses=[CompletionResult(message=Message(role="assistant", content="done"))]
    )

    config = SubagentConfig(
        name="researcher",
        provider=provider,
        model="fake",
        system_prompt="You research things",
        skills=["web-search", "read-file"],
        max_ticks=5,
    )

    assert config.name == "researcher"
    assert config.provider is provider
    assert config.model == "fake"
    assert config.system_prompt == "You research things"
    assert config.skills == ["web-search", "read-file"]
    assert config.max_ticks == 5


def test_subagent_config_defaults() -> None:
    """Verify SubagentConfig has sensible defaults."""
    provider = FakeProvider(responses=[])

    config = SubagentConfig(
        name="default-agent",
        provider=provider,
        model="fake",
    )

    assert config.system_prompt == ""
    assert config.skills == []
    assert config.max_ticks == 10


def test_subagent_registry_component_defaults() -> None:
    """Verify SubagentRegistryComponent starts with empty registry."""
    registry = SubagentRegistryComponent()
    assert registry.subagents == {}


def test_subagent_registry_register_and_lookup() -> None:
    """Register a SubagentConfig and look it up by name."""
    provider = FakeProvider(
        responses=[
            CompletionResult(message=Message(role="assistant", content="result"))
        ]
    )

    registry = SubagentRegistryComponent()
    config = SubagentConfig(
        name="researcher",
        provider=provider,
        model="fake",
        system_prompt="You research things",
    )

    registry.subagents["researcher"] = config

    retrieved = registry.subagents["researcher"]
    assert retrieved.name == "researcher"
    assert retrieved.provider is provider
    assert retrieved.max_ticks == 10


def test_delegation_started_event() -> None:
    """Verify DelegationStartedEvent has required fields."""
    world = World()
    entity = world.create_entity()

    event = DelegationStartedEvent(
        entity_id=entity,
        subagent_name="researcher",
        task="Find the latest AI research papers",
    )

    assert event.entity_id == entity
    assert event.subagent_name == "researcher"
    assert event.task == "Find the latest AI research papers"


def test_delegation_completed_event() -> None:
    """Verify DelegationCompletedEvent has required fields."""
    world = World()
    entity = world.create_entity()

    event = DelegationCompletedEvent(
        entity_id=entity,
        subagent_name="researcher",
        result="Found 5 papers from 2024",
    )

    assert event.entity_id == entity
    assert event.subagent_name == "researcher"
    assert event.result == "Found 5 papers from 2024"
