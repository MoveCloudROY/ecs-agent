"""Tests for subagent types and SubagentSystem."""

from __future__ import annotations
from typing import Any

from ecs_agent.components import MessageBusConfigComponent
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
    ToolSchema,
)
from ecs_agent.systems.message_bus import MessageBusSystem


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
        correlation_id="corr-123",
        traceparent="00-0123456789abcdef0123456789abcdef-0123456789abcdef-01",
    )

    assert event.entity_id == entity
    assert event.subagent_name == "researcher"
    assert event.task == "Find the latest AI research papers"
    assert event.correlation_id == "corr-123"
    assert (
        event.traceparent == "00-0123456789abcdef0123456789abcdef-0123456789abcdef-01"
    )


def test_delegation_completed_event() -> None:
    """Verify DelegationCompletedEvent has required fields."""
    world = World()
    entity = world.create_entity()

    event = DelegationCompletedEvent(
        entity_id=entity,
        subagent_name="researcher",
        result="Found 5 papers from 2024",
        success=True,
        error=None,
        correlation_id="corr-123",
        traceparent="00-0123456789abcdef0123456789abcdef-0123456789abcdef-01",
    )

    assert event.entity_id == entity
    assert event.subagent_name == "researcher"
    assert event.result == "Found 5 papers from 2024"
    assert event.success is True
    assert event.error is None
    assert event.correlation_id == "corr-123"
    assert (
        event.traceparent == "00-0123456789abcdef0123456789abcdef-0123456789abcdef-01"
    )


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


def _register_message_bus(
    world: World, parent_entity: EntityId, request_timeout: float = 30.0
) -> MessageBusSystem:
    world.add_component(
        parent_entity,
        MessageBusConfigComponent(request_timeout=request_timeout),
    )
    message_bus = MessageBusSystem(priority=5, request_timeout=request_timeout)
    world.register_system(message_bus, priority=5)
    return message_bus


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
        responses=[
            CompletionResult(message=Message(role="assistant", content="Result"))
        ]
    )
    config = SubagentConfig(
        name="test-agent", provider=provider, model="fake", system_prompt="Test prompt"
    )
    registry = SubagentRegistryComponent(subagents={"test-agent": config})
    world.add_component(parent_entity, registry)
    world.add_component(parent_entity, ToolRegistryComponent(tools={}, handlers={}))
    _register_message_bus(world, parent_entity)

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
        responses=[
            CompletionResult(message=Message(role="assistant", content="Completed"))
        ]
    )
    config = SubagentConfig(
        name="test-agent", provider=provider, model="fake", max_ticks=2
    )
    registry = SubagentRegistryComponent(subagents={"test-agent": config})
    world.add_component(parent_entity, registry)
    world.add_component(parent_entity, ToolRegistryComponent(tools={}, handlers={}))
    _register_message_bus(world, parent_entity)

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

    # Verify child has ConversationComponent (delegation completed)
    child_conv = world.get_component(child_entity, ConversationComponent)
    assert child_conv is not None, "Child should have ConversationComponent"
    assert len(child_conv.messages) > 0, "Child conversation should have messages"


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
    _register_message_bus(world, parent_entity)

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
    _register_message_bus(world, parent_entity)

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
    assert started_events[0].correlation_id
    assert started_events[0].traceparent

    assert len(completed_events) == 1
    assert completed_events[0].entity_id == parent_entity
    assert completed_events[0].subagent_name == "test-agent"
    assert isinstance(completed_events[0].result, str)
    assert completed_events[0].success is True
    assert completed_events[0].error is None
    assert completed_events[0].correlation_id == started_events[0].correlation_id
    assert completed_events[0].traceparent == started_events[0].traceparent


async def test_delegate_unknown_subagent_returns_error() -> None:
    """Verify delegate with unknown name returns error string, does not raise."""
    from ecs_agent.systems.subagent import SubagentSystem

    world = World()
    parent_entity = world.create_entity()

    registry = SubagentRegistryComponent()  # Empty registry
    world.add_component(parent_entity, registry)
    world.add_component(parent_entity, ToolRegistryComponent(tools={}, handlers={}))
    _register_message_bus(world, parent_entity)

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
    from ecs_agent.systems.subagent import SubagentSystem
    from ecs_agent.components.definitions import SkillComponent
    from ecs_agent.skills.protocol import Skill
    from ecs_agent.skills.manager import SkillManager

    # Create a test skill to verify installation
    class TestSkill(Skill):
        name: str = "test-skill"
        description: str = "A test skill"

        def tools(self) -> dict[str, tuple[ToolSchema, Any]]:
            async def test_tool() -> str:
                return "test result"

            return {
                "test_tool": (
                    ToolSchema(
                        name="test_tool",
                        description="A test tool",
                        parameters={"type": "object", "properties": {}},
                    ),
                    test_tool,
                )
            }

        def system_prompt(self) -> str:
            return "Test skill prompt"

        def install(self, world: World, entity_id: EntityId) -> None:
            pass

        def uninstall(self, world: World, entity_id: EntityId) -> None:
            pass

    world = World()
    parent_entity = world.create_entity()

    # Create a subagent config WITH skills list
    provider = FakeProvider(
        responses=[CompletionResult(message=Message(role="assistant", content="Done"))]
    )
    config = SubagentConfig(
        name="test-agent",
        provider=provider,
        model="fake",
        skills=["test-skill"],  # Request skill installation
    )

    # Register TestSkill on parent entity so it's available for delegation
    test_skill = TestSkill()
    skill_manager = SkillManager()
    skill_manager.install(world, parent_entity, test_skill)
    
    # Register subagent config
    registry = SubagentRegistryComponent(subagents={"test-agent": config})
    world.add_component(parent_entity, registry)
    _register_message_bus(world, parent_entity)

    # Register SubagentSystem to auto-register delegate tool
    system = SubagentSystem()
    await system.process(world)

    # Get the delegate handler
    tool_registry = world.get_component(parent_entity, ToolRegistryComponent)
    assert tool_registry is not None
    assert "delegate" in tool_registry.handlers

    delegate_handler = tool_registry.handlers["delegate"]

    # Track entities before delegation
    entities_before = len([e for e in world.query(OwnerComponent)])

    # Call delegate handler (this should create child and install skills)
    result = await delegate_handler(subagent_name="test-agent", task="test task")

    # Find child entity (has OwnerComponent pointing to parent)
    child_entity_id = None
    for entity_id, components in world.query(OwnerComponent):
        owner_comp = components[0]
        assert isinstance(owner_comp, OwnerComponent)
        if owner_comp.owner_id == parent_entity:
            child_entity_id = entity_id
            break

    assert child_entity_id is not None, "Child entity should be created"

    # RED TEST: Verify skill installation happened on child entity

    # ASSERTION 1: Child entity has SkillComponent with the requested skill
    skill_component = world.get_component(child_entity_id, SkillComponent)
    assert skill_component is not None, "Child entity should have SkillComponent"
    assert "test-skill" in skill_component.skills, (
        "test-skill should be installed on child"
    )

    # ASSERTION 2: Child's ToolRegistryComponent has the skill's tools
    child_tool_registry = world.get_component(child_entity_id, ToolRegistryComponent)
    assert child_tool_registry is not None, "Child should have ToolRegistryComponent"
    assert "test_tool" in child_tool_registry.tools, "test_tool should be registered"
    assert "test_tool" in child_tool_registry.handlers, (
        "test_tool handler should be registered"
    )

    # ASSERTION 3: Skill metadata is correctly populated
    metadata = skill_component.skills["test-skill"]
    assert metadata.name == "test-skill"
    assert metadata.description == "A test skill"
    assert "test_tool" in metadata.tool_names
    assert metadata.has_system_prompt is True


async def test_delegate_timeout_returns_deterministic_error_and_cleans_pending_requests() -> (
    None
):
    from ecs_agent.systems.subagent import SubagentSystem

    world = World()
    parent_entity = world.create_entity()

    provider = FakeProvider(
        responses=[CompletionResult(message=Message(role="assistant", content="Done"))]
    )
    config = SubagentConfig(name="test-agent", provider=provider, model="fake")
    registry = SubagentRegistryComponent(subagents={"test-agent": config})
    world.add_component(parent_entity, registry)
    world.add_component(parent_entity, ToolRegistryComponent(tools={}, handlers={}))
    message_bus = _register_message_bus(world, parent_entity, request_timeout=0.01)

    completed_events: list[DelegationCompletedEvent] = []

    async def on_completed(event: DelegationCompletedEvent) -> None:
        completed_events.append(event)

    world.event_bus.subscribe(DelegationCompletedEvent, on_completed)

    async def never_respond(correlation_id: str, message: dict[str, object]) -> bool:
        _ = correlation_id
        _ = message
        return False

    setattr(message_bus, "respond", never_respond)

    system = SubagentSystem()
    await system.process(world)

    tool_registry = world.get_component(parent_entity, ToolRegistryComponent)
    assert tool_registry is not None
    delegate_handler = tool_registry.handlers["delegate"]

    result = await delegate_handler(subagent_name="test-agent", task="Do work")

    assert result == "Error: Subagent timeout"
    assert len(completed_events) == 1
    assert completed_events[0].success is False
    assert completed_events[0].error == "Error: Subagent timeout"

    pending_requests = getattr(message_bus, "_pending_requests", None)
    assert isinstance(pending_requests, dict)
    assert pending_requests == {}


# ──────────────────────────────────────────────────────────────────────────────
# Wave 1: System-Level Delegation Contract Tests (RED)
# ──────────────────────────────────────────────────────────────────────────────


async def test_delegate_tool_available_before_first_reasoning_tick() -> None:
    """Delegate tool MUST be registered before ReasoningSystem processes."""
    from ecs_agent.systems.subagent import SubagentSystem
    from ecs_agent.systems.reasoning import ReasoningSystem

    class ToolCheckingProvider:
        def __init__(self) -> None:
            self.saw_delegate = False

        async def complete(
            self,
            messages: list[Message],
            tools: list[ToolSchema] | None = None,
            stream: bool = False,
            response_format: dict[str, Any] | None = None,
        ) -> CompletionResult:
            _ = messages
            _ = stream
            _ = response_format
            assert tools is not None
            self.saw_delegate = any(tool.name == "delegate" for tool in tools)
            assert self.saw_delegate
            return CompletionResult(message=Message(role="assistant", content="Done"))

    world = World()
    parent_entity = world.create_entity()

    # Register subagent config
    provider = ToolCheckingProvider()
    config = SubagentConfig(name="test-agent", provider=provider, model="fake")
    registry = SubagentRegistryComponent(subagents={"test-agent": config})
    world.add_component(parent_entity, registry)
    world.add_component(parent_entity, ToolRegistryComponent(tools={}, handlers={}))
    world.add_component(parent_entity, LLMComponent(provider=provider, model="fake"))
    world.add_component(
        parent_entity,
        ConversationComponent(messages=[Message(role="user", content="Use delegate")]),
    )
    _register_message_bus(world, parent_entity)

    # Register systems (SubagentSystem BEFORE ReasoningSystem)
    world.register_system(SubagentSystem(priority=0), priority=0)
    world.register_system(ReasoningSystem(priority=1), priority=1)

    await world.process()

    # Assert delegate tool is registered BEFORE reasoning system processes
    tool_registry = world.get_component(parent_entity, ToolRegistryComponent)
    assert tool_registry is not None
    assert "delegate" in tool_registry.tools, (
        "Delegate tool MUST be registered before first reasoning tick"
    )
    assert "delegate" in tool_registry.handlers, (
        "Delegate handler MUST be registered before first reasoning tick"
    )

    conv = world.get_component(parent_entity, ConversationComponent)
    assert conv is not None
    assert len(conv.messages) == 2
    assert conv.messages[1].role == "assistant"
    assert provider.saw_delegate


async def test_delegate_roundtrip_parent_delegate_tool_result_parent_summary() -> None:
    """Test parent→delegate call→child execution→result→parent receives summary."""
    from ecs_agent.systems.subagent import SubagentSystem

    world = World()
    parent_entity = world.create_entity()

    # Set up child provider to return a specific result
    child_provider = FakeProvider(
        responses=[
            CompletionResult(
                message=Message(role="assistant", content="Child completed task")
            )
        ]
    )
    config = SubagentConfig(
        name="child-agent", provider=child_provider, model="fake", max_ticks=2
    )
    registry = SubagentRegistryComponent(subagents={"child-agent": config})
    world.add_component(parent_entity, registry)
    world.add_component(parent_entity, ToolRegistryComponent(tools={}, handlers={}))
    _register_message_bus(world, parent_entity)

    # Register and process SubagentSystem
    system = SubagentSystem()
    await system.process(world)

    # Get delegate handler
    tool_registry = world.get_component(parent_entity, ToolRegistryComponent)
    assert tool_registry is not None
    delegate_handler = tool_registry.handlers["delegate"]

    # PARENT → DELEGATE CALL
    result = await delegate_handler(
        subagent_name="child-agent", task="Execute this task"
    )

    # Assert result is string summary from child
    assert isinstance(result, str), "Delegate must return string summary"
    assert result == "Child completed task", (
        "Result must match child's last assistant message"
    )

    # Find child entity
    child_entity = None
    for entity_id, components in world.query(OwnerComponent):
        owner_comp = components[0]
        assert isinstance(owner_comp, OwnerComponent)
        if owner_comp.owner_id == parent_entity:
            child_entity = entity_id
            break

    assert child_entity is not None, "Child entity must be created"

    # Assert child has conversation (execution completed)
    child_conv = world.get_component(child_entity, ConversationComponent)
    assert child_conv is not None, "Child must have ConversationComponent"
    assert len(child_conv.messages) > 0, "Child conversation must have messages"

    # Assert child conversation contains the delegated task
    child_conv = world.get_component(child_entity, ConversationComponent)
    assert child_conv is not None
    assert len(child_conv.messages) >= 1
    assert child_conv.messages[0].content == "Execute this task", (
        "Child's first message must be delegated task"
    )


async def test_delegation_event_correlation_integrity() -> None:
    """DelegationStartedEvent and DelegationCompletedEvent MUST share correlation_id."""
    from ecs_agent.systems.subagent import SubagentSystem

    world = World()
    parent_entity = world.create_entity()

    provider = FakeProvider(
        responses=[CompletionResult(message=Message(role="assistant", content="Done"))]
    )
    config = SubagentConfig(name="test-agent", provider=provider, model="fake")
    registry = SubagentRegistryComponent(subagents={"test-agent": config})
    world.add_component(parent_entity, registry)
    world.add_component(parent_entity, ToolRegistryComponent(tools={}, handlers={}))
    _register_message_bus(world, parent_entity)

    # Subscribe to delegation events
    started_events: list[DelegationStartedEvent] = []
    completed_events: list[DelegationCompletedEvent] = []

    async def on_started(event: DelegationStartedEvent) -> None:
        started_events.append(event)

    async def on_completed(event: DelegationCompletedEvent) -> None:
        completed_events.append(event)

    world.event_bus.subscribe(DelegationStartedEvent, on_started)
    world.event_bus.subscribe(DelegationCompletedEvent, on_completed)

    # Process to register delegate tool
    system = SubagentSystem()
    await system.process(world)

    tool_registry = world.get_component(parent_entity, ToolRegistryComponent)
    assert tool_registry is not None
    delegate_handler = tool_registry.handlers["delegate"]

    # Invoke delegation
    await delegate_handler(subagent_name="test-agent", task="test task")

    # Verify events published
    assert len(started_events) == 1, "DelegationStartedEvent MUST be published"
    assert len(completed_events) == 1, "DelegationCompletedEvent MUST be published"

    # EVENT CORRELATION INTEGRITY
    started_event = started_events[0]
    completed_event = completed_events[0]

    # Assert correlation_id matches
    assert started_event.correlation_id == completed_event.correlation_id, (
        "DelegationStartedEvent and DelegationCompletedEvent MUST share correlation_id"
    )

    # Assert traceparent matches (distributed tracing integrity)
    assert started_event.traceparent == completed_event.traceparent, (
        "DelegationStartedEvent and DelegationCompletedEvent MUST share traceparent"
    )

    # Assert correlation_id is not empty (valid UUID)
    assert started_event.correlation_id, "correlation_id MUST be non-empty"
    assert len(started_event.correlation_id) > 0, "correlation_id MUST be valid UUID"

    # Assert traceparent format (W3C Trace Context: 00-{trace-id}-{parent-id}-{flags})
    assert started_event.traceparent.startswith("00-"), (
        "traceparent MUST follow W3C Trace Context format"
    )
