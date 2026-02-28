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


# ──────────────────────────────────────────────────────────────────────────────
# Task 12: SubagentSystem + Delegate Tool Tests
# ──────────────────────────────────────────────────────────────────────────────

import pytest

from ecs_agent.components import (
    ConversationComponent,
    LLMComponent,
    OwnerComponent,
    TerminalComponent,
    ToolRegistryComponent,
)
from ecs_agent.core.runner import Runner


def test_delegate_tool_schema() -> None:
    """Verify delegate tool has correct parameters."""
    from ecs_agent.systems.subagent import SubagentSystem

    world = World()
    entity = world.create_entity()
    world.add_component(entity, SubagentRegistryComponent())
    world.add_component(entity, ToolRegistryComponent(tools={}, handlers={}))

    system = SubagentSystem()
    # Process should register the delegate tool
    import asyncio

    asyncio.run(system.process(world))

    registry = world.get_component(entity, ToolRegistryComponent)
    assert registry is not None
    assert "delegate" in registry.tools
    delegate_schema = registry.tools["delegate"]
    assert delegate_schema.name == "delegate"
    assert "subagent_name" in delegate_schema.parameters["properties"]
    assert "task" in delegate_schema.parameters["properties"]
    assert delegate_schema.parameters["required"] == ["subagent_name", "task"]


async def test_delegate_creates_child_entity() -> None:
    """Verify delegate creates new entity with correct components."""
    from ecs_agent.systems.subagent import SubagentSystem

    world = World()
    parent_entity = world.create_entity()

    # Set up parent with registry
    provider = FakeProvider(
        responses=[CompletionResult(message=Message(role="assistant", content="Result"))]
    )
    config = SubagentConfig(
        name="test-agent", provider=provider, model="fake", system_prompt="Test prompt"
    )
    registry = SubagentRegistryComponent(subagents={"test-agent": config})
    world.add_component(parent_entity, registry)
    world.add_component(parent_entity, ToolRegistryComponent(tools={}, handlers={}))

    system = SubagentSystem()
    await system.process(world)

    # Get the delegate handler
    tool_registry = world.get_component(parent_entity, ToolRegistryComponent)
    assert tool_registry is not None
    delegate_handler = tool_registry.handlers["delegate"]

    # Count entities before delegation
    entities_before = len([e for e in world.query(LLMComponent)])

    # Call delegate
    result = await delegate_handler(subagent_name="test-agent", task="Do something")

    # Verify child entity was created
    entities_after = len([e for e in world.query(LLMComponent)])
    assert entities_after > entities_before, "Child entity should be created"

    # Find the child entity (has OwnerComponent pointing to parent)
    child_entity = None
    for entity_id, components in world.query(OwnerComponent):
        owner_comp = components[0]
        assert isinstance(owner_comp, OwnerComponent)
        if owner_comp.owner_id == parent_entity:
            child_entity = entity_id
            break

    assert child_entity is not None, "Child entity with OwnerComponent should exist"

    # Verify child has correct components
    child_llm = world.get_component(child_entity, LLMComponent)
    assert child_llm is not None
    assert child_llm.model == "fake"
    assert child_llm.system_prompt == "Test prompt"

    child_conv = world.get_component(child_entity, ConversationComponent)
    assert child_conv is not None
    assert len(child_conv.messages) >= 1
    assert child_conv.messages[0].role == "user"
    assert child_conv.messages[0].content == "Do something"


async def test_delegate_runs_child_to_completion() -> None:
    """Verify delegate runs child entity until TerminalComponent."""
    from ecs_agent.systems.subagent import SubagentSystem

    world = World()
    parent_entity = world.create_entity()

    # Set up parent with registry
    provider = FakeProvider(
        responses=[CompletionResult(message=Message(role="assistant", content="Completed"))]
    )
    config = SubagentConfig(
        name="test-agent", provider=provider, model="fake", max_ticks=2
    )
    registry = SubagentRegistryComponent(subagents={"test-agent": config})
    world.add_component(parent_entity, registry)
    world.add_component(parent_entity, ToolRegistryComponent(tools={}, handlers={}))

    system = SubagentSystem()
    await system.process(world)

    # Get delegate handler
    tool_registry = world.get_component(parent_entity, ToolRegistryComponent)
    assert tool_registry is not None
    delegate_handler = tool_registry.handlers["delegate"]

    # Call delegate
    result = await delegate_handler(subagent_name="test-agent", task="Do task")

    # Find child entity
    child_entity = None
    for entity_id, components in world.query(OwnerComponent):
        owner_comp = components[0]
        assert isinstance(owner_comp, OwnerComponent)
        if owner_comp.owner_id == parent_entity:
            child_entity = entity_id
            break

    assert child_entity is not None

    # Verify child has TerminalComponent (runner completed)
    terminal = world.get_component(child_entity, TerminalComponent)
    assert terminal is not None, "Child should have TerminalComponent after completion"


async def test_delegate_returns_last_assistant_message() -> None:
    """Verify delegate returns the last assistant message content."""
    from ecs_agent.systems.subagent import SubagentSystem

    world = World()
    parent_entity = world.create_entity()

    # Set up parent with registry
    provider = FakeProvider(
        responses=[
            CompletionResult(
                message=Message(role="assistant", content="This is the final answer")
            )
        ]
    )
    config = SubagentConfig(name="test-agent", provider=provider, model="fake")
    registry = SubagentRegistryComponent(subagents={"test-agent": config})
    world.add_component(parent_entity, registry)
    world.add_component(parent_entity, ToolRegistryComponent(tools={}, handlers={}))

    system = SubagentSystem()
    await system.process(world)

    tool_registry = world.get_component(parent_entity, ToolRegistryComponent)
    assert tool_registry is not None
    delegate_handler = tool_registry.handlers["delegate"]

    result = await delegate_handler(subagent_name="test-agent", task="Answer this")

    assert isinstance(result, str)
    assert result == "This is the final answer"


async def test_delegate_publishes_events() -> None:
    """Verify DelegationStartedEvent and DelegationCompletedEvent are published."""
    from ecs_agent.systems.subagent import SubagentSystem

    world = World()
    parent_entity = world.create_entity()

    # Set up parent with registry
    provider = FakeProvider(
        responses=[CompletionResult(message=Message(role="assistant", content="Done"))]
    )
    config = SubagentConfig(name="test-agent", provider=provider, model="fake")
    registry = SubagentRegistryComponent(subagents={"test-agent": config})
    world.add_component(parent_entity, registry)
    world.add_component(parent_entity, ToolRegistryComponent(tools={}, handlers={}))

    # Subscribe to events
    started_events: list[DelegationStartedEvent] = []
    completed_events: list[DelegationCompletedEvent] = []

    async def on_started(event: DelegationStartedEvent) -> None:
        started_events.append(event)

    async def on_completed(event: DelegationCompletedEvent) -> None:
        completed_events.append(event)

    world.event_bus.subscribe(DelegationStartedEvent, on_started)
    world.event_bus.subscribe(DelegationCompletedEvent, on_completed)

    system = SubagentSystem()
    await system.process(world)

    tool_registry = world.get_component(parent_entity, ToolRegistryComponent)
    assert tool_registry is not None
    delegate_handler = tool_registry.handlers["delegate"]

    await delegate_handler(subagent_name="test-agent", task="Do work")

    assert len(started_events) == 1
    assert started_events[0].entity_id == parent_entity
    assert started_events[0].subagent_name == "test-agent"
    assert started_events[0].task == "Do work"

    assert len(completed_events) == 1
    assert completed_events[0].entity_id == parent_entity
    assert completed_events[0].subagent_name == "test-agent"
    assert isinstance(completed_events[0].result, str)


async def test_delegate_unknown_subagent_returns_error() -> None:
    """Verify delegate with unknown name returns error string, does not raise."""
    from ecs_agent.systems.subagent import SubagentSystem

    world = World()
    parent_entity = world.create_entity()

    registry = SubagentRegistryComponent()  # Empty registry
    world.add_component(parent_entity, registry)
    world.add_component(parent_entity, ToolRegistryComponent(tools={}, handlers={}))

    system = SubagentSystem()
    await system.process(world)

    tool_registry = world.get_component(parent_entity, ToolRegistryComponent)
    assert tool_registry is not None
    delegate_handler = tool_registry.handlers["delegate"]

    # Should not raise, should return error string
    result = await delegate_handler(subagent_name="unknown-agent", task="Do task")

    assert isinstance(result, str)
    assert "error" in result.lower() or "unknown" in result.lower()


async def test_subagent_system_registers_delegate_tool() -> None:
    """SubagentSystem.process() ensures delegate tool is registered."""
    from ecs_agent.systems.subagent import SubagentSystem

    world = World()
    entity = world.create_entity()
    world.add_component(entity, SubagentRegistryComponent())
    world.add_component(entity, ToolRegistryComponent(tools={}, handlers={}))

    # Before processing, no delegate tool
    registry = world.get_component(entity, ToolRegistryComponent)
    assert registry is not None
    assert "delegate" not in registry.tools

    # After processing, delegate tool is registered
    system = SubagentSystem()
    await system.process(world)

    registry = world.get_component(entity, ToolRegistryComponent)
    assert registry is not None
    assert "delegate" in registry.tools
    assert "delegate" in registry.handlers


async def test_delegate_with_skills_installs_skills() -> None:
    """SubagentConfig with skills list, verify skills installed on child entity."""
    pytest.skip("Skills installation requires SkillManager integration - defer to integration tests")
    # This test is complex and requires full SkillManager setup
    # We'll test the basic flow and defer full skill installation to Task 15
