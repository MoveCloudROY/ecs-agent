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
    InheritancePolicy,
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
    # Verify inheritance_policy has correct defaults
    assert config.inheritance_policy is not None
    assert config.inheritance_policy.enabled is True
    assert config.inheritance_policy.inherit_system_prompt is True
    assert config.inheritance_policy.inherit_tools == []
    assert config.inheritance_policy.inherit_permissions is False
    assert config.inheritance_policy.allow_delegate_tool is True
    assert config.inheritance_policy.tool_conflict_policy == "skip"
    assert config.inheritance_policy.missing_skill_policy == "warn"


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
    PermissionComponent,
    TerminalComponent,
    ToolRegistryComponent,
)
from ecs_agent.components.definitions import SkillComponent
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


# ──────────────────────────────────────────────────────────────────────────────
# Task 2: Explicit Delegate Installer API Tests (RED)
# ──────────────────────────────────────────────────────────────────────────────


async def test_install_delegate_tool_with_default_name() -> None:
    """Test explicit installer with default tool name 'delegate'."""
    from ecs_agent.systems.subagent import SubagentSystem

    world = World()
    entity = world.create_entity()

    provider = FakeProvider(
        responses=[CompletionResult(message=Message(role="assistant", content="Done"))]
    )
    config = SubagentConfig(name="test-agent", provider=provider, model="fake")
    registry = SubagentRegistryComponent(subagents={"test-agent": config})
    world.add_component(entity, registry)
    world.add_component(entity, ToolRegistryComponent(tools={}, handlers={}))

    system = SubagentSystem()
    # This method doesn't exist yet - RED!
    system.install_delegate_tool(world, entity)

    tool_registry = world.get_component(entity, ToolRegistryComponent)
    assert tool_registry is not None
    assert "delegate" in tool_registry.handlers, (
        "Expected 'delegate' handler registered with default tool name"
    )
    assert "delegate" in tool_registry.tools, (
        "Expected 'delegate' tool schema registered with default tool name"
    )


async def test_install_delegate_tool_with_custom_name() -> None:
    """Test explicit installer with custom tool name."""
    from ecs_agent.systems.subagent import SubagentSystem

    world = World()
    entity = world.create_entity()

    provider = FakeProvider(
        responses=[CompletionResult(message=Message(role="assistant", content="Done"))]
    )
    config = SubagentConfig(name="test-agent", provider=provider, model="fake")
    registry = SubagentRegistryComponent(subagents={"test-agent": config})
    world.add_component(entity, registry)
    world.add_component(entity, ToolRegistryComponent(tools={}, handlers={}))

    system = SubagentSystem()
    # This method doesn't exist yet - RED!
    system.install_delegate_tool(world, entity, tool_name="custom_delegate")

    tool_registry = world.get_component(entity, ToolRegistryComponent)
    assert tool_registry is not None
    assert "custom_delegate" in tool_registry.handlers, (
        "Expected 'custom_delegate' handler registered with custom tool name"
    )
    assert "custom_delegate" in tool_registry.tools, (
        "Expected 'custom_delegate' tool schema registered with custom tool name"
    )
    # Ensure default name was NOT used
    assert "delegate" not in tool_registry.handlers, (
        "Expected 'delegate' NOT registered when custom tool name provided"
    )


async def test_install_delegate_tool_idempotency() -> None:
    """Test calling install_delegate_tool twice with same params succeeds (idempotent)."""
    from ecs_agent.systems.subagent import SubagentSystem

    world = World()
    entity = world.create_entity()

    provider = FakeProvider(
        responses=[CompletionResult(message=Message(role="assistant", content="Done"))]
    )
    config = SubagentConfig(name="test-agent", provider=provider, model="fake")
    registry = SubagentRegistryComponent(subagents={"test-agent": config})
    world.add_component(entity, registry)
    world.add_component(entity, ToolRegistryComponent(tools={}, handlers={}))

    system = SubagentSystem()
    # This method doesn't exist yet - RED!
    system.install_delegate_tool(world, entity)
    # Call again - should not raise, should be idempotent
    system.install_delegate_tool(world, entity)

    tool_registry = world.get_component(entity, ToolRegistryComponent)
    assert tool_registry is not None
    assert "delegate" in tool_registry.handlers, (
        "Expected 'delegate' handler after idempotent install"
    )


async def test_install_delegate_tool_no_overwrite_by_default() -> None:
    """Test that existing handler is NOT replaced when override=False (default)."""
    from ecs_agent.systems.subagent import SubagentSystem

    world = World()
    entity = world.create_entity()

    provider = FakeProvider(
        responses=[CompletionResult(message=Message(role="assistant", content="Done"))]
    )
    config = SubagentConfig(name="test-agent", provider=provider, model="fake")
    registry = SubagentRegistryComponent(subagents={"test-agent": config})
    world.add_component(entity, registry)

    # Pre-register a custom 'delegate' handler
    async def custom_handler(subagent_name: str, task: str) -> str:
        return "custom_result"

    delegate_schema = ToolSchema(
        name="delegate",
        description="Custom delegate",
        parameters={"type": "object", "properties": {}, "required": []},
    )
    tool_registry = ToolRegistryComponent(
        tools={"delegate": delegate_schema}, handlers={"delegate": custom_handler}
    )
    world.add_component(entity, tool_registry)

    system = SubagentSystem()
    # This method doesn't exist yet - RED!
    # Default override=False, should NOT replace existing handler
    system.install_delegate_tool(world, entity)

    tool_registry = world.get_component(entity, ToolRegistryComponent)
    assert tool_registry is not None
    # Handler should still be the custom one (not replaced)
    result = await tool_registry.handlers["delegate"](subagent_name="test", task="test")
    assert result == "custom_result", (
        "Expected custom handler preserved (not overwritten) when override=False"
    )


async def test_install_delegate_tool_overwrite_when_override_true() -> None:
    """Test that existing handler IS replaced when override=True."""
    from ecs_agent.systems.subagent import SubagentSystem

    world = World()
    entity = world.create_entity()

    provider = FakeProvider(
        responses=[CompletionResult(message=Message(role="assistant", content="Done"))]
    )
    config = SubagentConfig(name="test-agent", provider=provider, model="fake")
    registry = SubagentRegistryComponent(subagents={"test-agent": config})
    world.add_component(entity, registry)

    # Pre-register a custom 'delegate' handler
    async def custom_handler(subagent_name: str, task: str) -> str:
        return "custom_result"

    delegate_schema = ToolSchema(
        name="delegate",
        description="Custom delegate",
        parameters={"type": "object", "properties": {}, "required": []},
    )
    tool_registry = ToolRegistryComponent(
        tools={"delegate": delegate_schema}, handlers={"delegate": custom_handler}
    )
    world.add_component(entity, tool_registry)
    _register_message_bus(world, entity)

    system = SubagentSystem()
    # This method doesn't exist yet - RED!
    # override=True, should REPLACE existing handler
    system.install_delegate_tool(world, entity, override=True)

    tool_registry = world.get_component(entity, ToolRegistryComponent)
    assert tool_registry is not None
    # Handler should be replaced with system's handler (different behavior)
    result = await tool_registry.handlers["delegate"](
        subagent_name="test-agent", task="test"
    )
    # System handler should execute delegation, not return "custom_result"
    assert result != "custom_result", (
        "Expected system handler replaced custom handler when override=True"
    )
    # Verify it's the real delegate handler by checking result format
    assert isinstance(result, str), "Delegate handler must return string"


async def test_backward_compatible_auto_registration_still_works() -> None:
    """Test that existing process() auto-registration behavior is preserved."""
    from ecs_agent.systems.subagent import SubagentSystem

    world = World()
    entity = world.create_entity()

    provider = FakeProvider(
        responses=[CompletionResult(message=Message(role="assistant", content="Done"))]
    )
    config = SubagentConfig(name="test-agent", provider=provider, model="fake")
    registry = SubagentRegistryComponent(subagents={"test-agent": config})
    world.add_component(entity, registry)
    world.add_component(entity, ToolRegistryComponent(tools={}, handlers={}))

    system = SubagentSystem()
    # This is the EXISTING behavior - must still work!
    await system.process(world)

    tool_registry = world.get_component(entity, ToolRegistryComponent)
    assert tool_registry is not None
    assert "delegate" in tool_registry.handlers, (
        "Backward compatibility: process() must still auto-register delegate tool"
    )
    assert "delegate" in tool_registry.tools, (
        "Backward compatibility: process() must still register delegate schema"
    )


def _shared_tool_schema() -> ToolSchema:
    return ToolSchema(
        name="shared_tool",
        description="Shared tool for inheritance tests",
        parameters={"type": "object", "properties": {}, "required": []},
    )


def _find_child_entity(world: World, parent_entity: EntityId) -> EntityId:
    for entity_id, components in world.query(OwnerComponent):
        owner_comp = components[0]
        assert isinstance(owner_comp, OwnerComponent)
        if owner_comp.owner_id == parent_entity:
            return entity_id
    raise AssertionError("Expected delegated child entity to exist")


async def _delegate_with_policy(
    *,
    policy: InheritancePolicy,
    parent_system_prompt: str = "parent-system",
    child_system_prompt: str = "",
    child_has_permission: bool = False,
) -> tuple[World, EntityId, EntityId, str]:
    from ecs_agent.systems.subagent import SubagentSystem

    world = World()
    parent_entity = world.create_entity()

    async def parent_shared_tool_handler() -> str:
        return "parent-handler"

    parent_provider = FakeProvider(
        responses=[
            CompletionResult(message=Message(role="assistant", content="Parent done"))
        ]
    )
    parent_registry = ToolRegistryComponent(
        tools={"shared_tool": _shared_tool_schema()},
        handlers={"shared_tool": parent_shared_tool_handler},
    )
    world.add_component(parent_entity, parent_registry)
    world.add_component(
        parent_entity,
        LLMComponent(
            provider=parent_provider,
            model="fake",
            system_prompt=parent_system_prompt,
        ),
    )
    if child_has_permission:
        world.add_component(
            parent_entity,
            PermissionComponent(allowed_tools=["shared_tool"], denied_tools=[]),
        )

    child_provider = FakeProvider(
        responses=[
            CompletionResult(message=Message(role="assistant", content="Child done"))
        ]
    )
    config = SubagentConfig(
        name="child",
        provider=child_provider,
        model="fake",
        system_prompt=child_system_prompt,
        inheritance_policy=policy,
    )
    world.add_component(
        parent_entity,
        SubagentRegistryComponent(subagents={"child": config}),
    )
    _register_message_bus(world, parent_entity)

    system = SubagentSystem()
    await system.process(world)

    parent_tools = world.get_component(parent_entity, ToolRegistryComponent)
    assert parent_tools is not None
    delegate_handler = parent_tools.handlers["delegate"]
    result = await delegate_handler(subagent_name="child", task="Run delegated task")

    child_entity = _find_child_entity(world, parent_entity)
    return world, parent_entity, child_entity, result


async def test_inheritance_policy_conflict_skip_keeps_child_tool() -> None:
    policy = InheritancePolicy(
        enabled=True,
        inherit_tools=["shared_tool"],
        tool_conflict_policy="skip",
    )
    world, _, child_entity, _ = await _delegate_with_policy(policy=policy)

    child_tools = world.get_component(child_entity, ToolRegistryComponent)
    assert child_tools is not None, (
        "Skip policy test requires child ToolRegistryComponent to be present"
    )
    assert child_tools.handlers["shared_tool"] is not None, (
        "Skip policy should preserve child handler and not drop tool registration"
    )


async def test_inheritance_policy_conflict_error_raises() -> None:
    policy = InheritancePolicy(
        enabled=True,
        inherit_tools=["shared_tool"],
        tool_conflict_policy="error",
    )

    with pytest.raises(ValueError, match="shared_tool"):
        await _delegate_with_policy(policy=policy)


async def test_inheritance_policy_conflict_override_replaces_child_tool() -> None:
    policy = InheritancePolicy(
        enabled=True,
        inherit_tools=["shared_tool"],
        tool_conflict_policy="override",
    )
    world, _, child_entity, _ = await _delegate_with_policy(policy=policy)

    child_tools = world.get_component(child_entity, ToolRegistryComponent)
    assert child_tools is not None, (
        "Override policy test requires child ToolRegistryComponent to be present"
    )
    assert "shared_tool" in child_tools.tools, (
        "Override policy should keep shared_tool registered after replacement"
    )


async def test_inheritance_policy_allow_delegate_tool_false_blocks_delegate() -> None:
    policy = InheritancePolicy(
        enabled=True,
        inherit_tools=["delegate"],
        allow_delegate_tool=False,
    )
    world, _, child_entity, _ = await _delegate_with_policy(policy=policy)

    child_tools = world.get_component(child_entity, ToolRegistryComponent)
    assert child_tools is not None, (
        "Recursion guard test requires child ToolRegistryComponent to be present"
    )
    assert "delegate" not in child_tools.tools, (
        "allow_delegate_tool=False must block delegate tool inheritance"
    )


async def test_inheritance_policy_allow_delegate_tool_true_inherits_delegate() -> None:
    policy = InheritancePolicy(
        enabled=True,
        inherit_tools=["delegate"],
        allow_delegate_tool=True,
    )
    world, _, child_entity, _ = await _delegate_with_policy(policy=policy)

    child_tools = world.get_component(child_entity, ToolRegistryComponent)
    assert child_tools is not None, (
        "allow_delegate_tool=True test requires child ToolRegistryComponent"
    )
    assert "delegate" in child_tools.tools, (
        "allow_delegate_tool=True should allow delegate inheritance for controlled recursion"
    )


async def test_inheritance_policy_explicit_child_system_prompt_authoritative() -> None:
    policy = InheritancePolicy(enabled=True, inherit_system_prompt=True)
    world, _, child_entity, _ = await _delegate_with_policy(
        policy=policy,
        parent_system_prompt="parent prompt",
        child_system_prompt="child prompt",
    )

    child_llm = world.get_component(child_entity, LLMComponent)
    assert child_llm is not None
    assert child_llm.system_prompt == "child prompt", (
        "Child explicit system_prompt must remain authoritative over inherited prompt"
    )


async def test_inheritance_policy_whitelist_only_inherits_named_tools() -> None:
    policy = InheritancePolicy(
        enabled=True,
        inherit_tools=["shared_tool"],
    )
    world, _, child_entity, _ = await _delegate_with_policy(policy=policy)

    child_tools = world.get_component(child_entity, ToolRegistryComponent)
    assert child_tools is not None, (
        "Whitelist test requires child ToolRegistryComponent to be present"
    )
    assert "shared_tool" in child_tools.tools, (
        "Whitelist inheritance should copy only named tools to child"
    )
    assert "delegate" not in child_tools.tools, (
        "Whitelist inheritance must not copy tools omitted from inherit_tools"
    )


async def test_inheritance_policy_enabled_false_skips_all_inheritance() -> None:
    policy = InheritancePolicy(
        enabled=False,
        inherit_system_prompt=True,
        inherit_tools=["shared_tool", "delegate"],
        inherit_permissions=True,
    )
    world, _, child_entity, _ = await _delegate_with_policy(
        policy=policy,
        parent_system_prompt="parent prompt",
        child_system_prompt="",
        child_has_permission=True,
    )

    child_llm = world.get_component(child_entity, LLMComponent)
    assert child_llm is not None
    assert child_llm.system_prompt == "", (
        "enabled=False must disable system_prompt inheritance"
    )

    child_tools = world.get_component(child_entity, ToolRegistryComponent)
    if child_tools is not None:
        assert "shared_tool" not in child_tools.tools, (
            "enabled=False must disable tool inheritance"
        )
    child_perm = world.get_component(child_entity, PermissionComponent)
    assert child_perm is None, "enabled=False must disable permission inheritance"


async def test_inheritance_policy_inherit_system_prompt_when_child_empty() -> None:
    policy = InheritancePolicy(enabled=True, inherit_system_prompt=True)
    world, _, child_entity, _ = await _delegate_with_policy(
        policy=policy,
        parent_system_prompt="parent inherited prompt",
        child_system_prompt="",
    )

    child_llm = world.get_component(child_entity, LLMComponent)
    assert child_llm is not None
    assert child_llm.system_prompt == "parent inherited prompt", (
        "inherit_system_prompt=True should copy parent prompt when child prompt is empty"
    )


async def test_inheritance_policy_inherit_permissions_true_copies_permission_component() -> (
    None
):
    policy = InheritancePolicy(enabled=True, inherit_permissions=True)
    world, _, child_entity, _ = await _delegate_with_policy(
        policy=policy,
        child_has_permission=True,
    )

    child_perm = world.get_component(child_entity, PermissionComponent)
    assert child_perm is not None, (
        "inherit_permissions=True should copy parent PermissionComponent to child"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Task 4: SkillManager-Aligned Skill Inheritance Tests (RED)
# ──────────────────────────────────────────────────────────────────────────────


async def test_subagent_skills_skill_manager_inherited_tools_available() -> None:
    """Child can execute tools from inherited skills (SkillManager semantics)."""
    from ecs_agent.systems.subagent import SubagentSystem
    from ecs_agent.skills.manager import SkillManager
    from ecs_agent.skills.protocol import Skill

    # Define a test skill with a tool
    class ParentSkill(Skill):
        name: str = "parent-skill"
        description: str = "Skill installed on parent"

        def tools(self) -> dict[str, tuple[ToolSchema, Any]]:
            async def parent_tool() -> str:
                return "parent_tool_result"

            return {
                "parent_tool": (
                    ToolSchema(
                        name="parent_tool",
                        description="A tool from parent skill",
                        parameters={"type": "object", "properties": {}},
                    ),
                    parent_tool,
                )
            }

        def system_prompt(self) -> str:
            return "Parent skill system prompt"

        def install(self, world: World, entity_id: EntityId) -> None:
            pass

        def uninstall(self, world: World, entity_id: EntityId) -> None:
            pass

    world = World()
    parent_entity = world.create_entity()

    # Install skill on parent using SkillManager
    skill_manager = SkillManager()
    parent_skill = ParentSkill()
    skill_manager.install(world, parent_entity, parent_skill)

    # Configure child to inherit parent skill
    child_provider = FakeProvider(
        responses=[CompletionResult(message=Message(role="assistant", content="Done"))]
    )
    config = SubagentConfig(
        name="child",
        provider=child_provider,
        model="fake",
        inheritance_policy=InheritancePolicy(
            enabled=True,
            inherit_tools=["parent_tool"],  # Inherit from parent skill
        ),
    )

    registry = SubagentRegistryComponent(subagents={"child": config})
    world.add_component(parent_entity, registry)
    _register_message_bus(world, parent_entity)

    # Delegation logic doesn't exist yet - RED!
    system = SubagentSystem()
    await system.process(world)

    tool_registry = world.get_component(parent_entity, ToolRegistryComponent)
    assert tool_registry is not None
    delegate_handler = tool_registry.handlers["delegate"]

    # Delegate task
    result = await delegate_handler(subagent_name="child", task="test task")

    # Find child entity
    child_entity = _find_child_entity(world, parent_entity)

    # RED TEST: Verify child has SkillComponent with parent skill
    child_skills = world.get_component(child_entity, SkillComponent)
    assert child_skills is not None, (
        "Child entity should have SkillComponent after skill inheritance"
    )
    assert "parent-skill" in child_skills.skills, (
        "parent-skill should be installed on child via SkillManager semantics"
    )

    # RED TEST: Verify child can call parent_tool
    child_tools = world.get_component(child_entity, ToolRegistryComponent)
    assert child_tools is not None
    assert "parent_tool" in child_tools.handlers, (
        "Child should have inherited parent_tool handler"
    )
    assert "parent_tool" in child_tools.tools, (
        "Child should have inherited parent_tool schema"
    )

    # RED TEST: Verify tool is actually callable
    tool_result = await child_tools.handlers["parent_tool"]()
    assert tool_result == "parent_tool_result", (
        "Inherited tool should execute and return expected result"
    )


async def test_subagent_skills_skill_manager_requested_tools_available() -> None:
    """Child can execute tools from requested skills (SubagentConfig.skills)."""
    from ecs_agent.systems.subagent import SubagentSystem
    from ecs_agent.skills.manager import SkillManager
    from ecs_agent.skills.protocol import Skill

    # Define a skill that child requests explicitly
    class RequestedSkill(Skill):
        name: str = "requested-skill"
        description: str = "Skill requested by child"

        def tools(self) -> dict[str, tuple[ToolSchema, Any]]:
            async def requested_tool() -> str:
                return "requested_tool_result"

            return {
                "requested_tool": (
                    ToolSchema(
                        name="requested_tool",
                        description="Tool from requested skill",
                        parameters={"type": "object", "properties": {}},
                    ),
                    requested_tool,
                )
            }

        def system_prompt(self) -> str:
            return "Requested skill prompt"

        def install(self, world: World, entity_id: EntityId) -> None:
            pass

        def uninstall(self, world: World, entity_id: EntityId) -> None:
            pass

    world = World()
    parent_entity = world.create_entity()

    # Install skill on parent using SkillManager
    skill_manager = SkillManager()
    requested_skill = RequestedSkill()
    skill_manager.install(world, parent_entity, requested_skill)

    # Configure child to request skill explicitly
    child_provider = FakeProvider(
        responses=[CompletionResult(message=Message(role="assistant", content="Done"))]
    )
    config = SubagentConfig(
        name="child",
        provider=child_provider,
        model="fake",
        skills=["requested-skill"],  # Request skill installation
    )

    registry = SubagentRegistryComponent(subagents={"child": config})
    world.add_component(parent_entity, registry)
    _register_message_bus(world, parent_entity)

    # Delegation logic doesn't exist yet - RED!
    system = SubagentSystem()
    await system.process(world)

    tool_registry = world.get_component(parent_entity, ToolRegistryComponent)
    assert tool_registry is not None
    delegate_handler = tool_registry.handlers["delegate"]

    # Delegate task
    result = await delegate_handler(subagent_name="child", task="test task")

    # Find child entity
    child_entity = _find_child_entity(world, parent_entity)

    # RED TEST: Verify child has SkillComponent with requested skill
    child_skills = world.get_component(child_entity, SkillComponent)
    assert child_skills is not None, (
        "Child entity should have SkillComponent with requested skills"
    )
    assert "requested-skill" in child_skills.skills, (
        "requested-skill should be installed on child via SkillManager.install()"
    )

    # RED TEST: Verify child can call requested_tool
    child_tools = world.get_component(child_entity, ToolRegistryComponent)
    assert child_tools is not None
    assert "requested_tool" in child_tools.handlers, (
        "Child should have requested_tool handler from SkillManager.install()"
    )
    assert "requested_tool" in child_tools.tools, (
        "Child should have requested_tool schema from SkillManager.install()"
    )

    # RED TEST: Verify tool is actually callable
    tool_result = await child_tools.handlers["requested_tool"]()
    assert tool_result == "requested_tool_result", (
        "Requested skill tool should execute via SkillManager lifecycle"
    )


async def test_subagent_skills_missing_skill_warn_policy() -> None:
    """When missing skill + warn policy, logs warning and continues."""
    from ecs_agent.systems.subagent import SubagentSystem
    import logging
    from unittest.mock import patch

    world = World()
    parent_entity = world.create_entity()

    # Parent has NO skills installed
    world.add_component(parent_entity, ToolRegistryComponent(tools={}, handlers={}))

    # Child config requests nonexistent skill with warn policy
    child_provider = FakeProvider(
        responses=[CompletionResult(message=Message(role="assistant", content="Done"))]
    )
    config = SubagentConfig(
        name="child",
        provider=child_provider,
        model="fake",
        skills=["nonexistent_skill"],  # Request missing skill
        inheritance_policy=InheritancePolicy(
            missing_skill_policy="warn"  # Key policy
        ),
    )

    registry = SubagentRegistryComponent(subagents={"child": config})
    world.add_component(parent_entity, registry)
    _register_message_bus(world, parent_entity)

    # Track warning logs
    with patch("ecs_agent.systems.subagent.logger") as mock_logger:
        system = SubagentSystem()
        await system.process(world)

        tool_registry = world.get_component(parent_entity, ToolRegistryComponent)
        assert tool_registry is not None
        delegate_handler = tool_registry.handlers["delegate"]

        # RED TEST: Delegation should complete despite missing skill
        result = await delegate_handler(subagent_name="child", task="test task")

        # RED TEST: Verify delegation completed (no exception raised)
        assert isinstance(result, str), (
            "Delegation with warn policy should return result string, not raise"
        )

        # RED TEST: Verify warning was logged (doesn't exist yet - RED!)
        # This assertion will fail because warning logic doesn't exist
        mock_logger.warning.assert_called_once()
        warning_call = mock_logger.warning.call_args
        assert "nonexistent_skill" in str(warning_call), (
            "Warning should mention the missing skill name"
        )


async def test_subagent_skills_missing_skill_error_policy() -> None:
    """When missing skill + error policy, raises error."""
    from ecs_agent.systems.subagent import SubagentSystem

    world = World()
    parent_entity = world.create_entity()

    # Parent has NO skills installed
    world.add_component(parent_entity, ToolRegistryComponent(tools={}, handlers={}))

    # Child config requests nonexistent skill with error policy
    child_provider = FakeProvider(
        responses=[CompletionResult(message=Message(role="assistant", content="Done"))]
    )
    config = SubagentConfig(
        name="child",
        provider=child_provider,
        model="fake",
        skills=["nonexistent_skill"],  # Request missing skill
        inheritance_policy=InheritancePolicy(
            missing_skill_policy="error"  # Key policy
        ),
    )

    registry = SubagentRegistryComponent(subagents={"child": config})
    world.add_component(parent_entity, registry)
    _register_message_bus(world, parent_entity)

    system = SubagentSystem()
    await system.process(world)

    tool_registry = world.get_component(parent_entity, ToolRegistryComponent)
    assert tool_registry is not None
    delegate_handler = tool_registry.handlers["delegate"]

    # RED TEST: Delegation should raise error for missing skill
    # This will fail because error-raising logic doesn't exist yet - RED!
    with pytest.raises((ValueError, KeyError), match="nonexistent_skill"):
        await delegate_handler(subagent_name="child", task="test task")


async def test_subagent_skills_skill_manager_install_uninstall_lifecycle() -> None:
    """Skills installed via SkillManager.install(), not manual dict copying."""
    from ecs_agent.systems.subagent import SubagentSystem
    from ecs_agent.skills.manager import SkillManager
    from ecs_agent.skills.protocol import Skill
    from unittest.mock import patch

    # Define a skill to track SkillManager.install() calls
    class LifecycleSkill(Skill):
        name: str = "lifecycle-skill"
        description: str = "Skill with lifecycle tracking"

        def tools(self) -> dict[str, tuple[ToolSchema, Any]]:
            async def lifecycle_tool() -> str:
                return "lifecycle_result"

            return {
                "lifecycle_tool": (
                    ToolSchema(
                        name="lifecycle_tool",
                        description="Lifecycle tool",
                        parameters={"type": "object", "properties": {}},
                    ),
                    lifecycle_tool,
                )
            }

        def system_prompt(self) -> str:
            return "Lifecycle prompt"

        def install(self, world: World, entity_id: EntityId) -> None:
            # Track install call
            pass

        def uninstall(self, world: World, entity_id: EntityId) -> None:
            pass

    world = World()
    parent_entity = world.create_entity()

    # Install skill on parent
    skill_manager = SkillManager()
    lifecycle_skill = LifecycleSkill()
    skill_manager.install(world, parent_entity, lifecycle_skill)

    # Child requests skill
    child_provider = FakeProvider(
        responses=[CompletionResult(message=Message(role="assistant", content="Done"))]
    )
    config = SubagentConfig(
        name="child",
        provider=child_provider,
        model="fake",
        skills=["lifecycle-skill"],
    )

    registry = SubagentRegistryComponent(subagents={"child": config})
    world.add_component(parent_entity, registry)
    _register_message_bus(world, parent_entity)

    # RED TEST: Verify SkillManager.install() is called (not manual dict copy)
    # This will fail because SkillManager integration doesn't exist - RED!
    with patch.object(SkillManager, "install", wraps=skill_manager.install) as mock_install:
        system = SubagentSystem()
        await system.process(world)

        tool_registry = world.get_component(parent_entity, ToolRegistryComponent)
        assert tool_registry is not None
        delegate_handler = tool_registry.handlers["delegate"]

        await delegate_handler(subagent_name="child", task="test task")

        # Verify SkillManager.install() was called for child entity
        child_entity = _find_child_entity(world, parent_entity)

        # This assertion will fail - current implementation uses dict copying
        mock_install.assert_called()
        install_calls = mock_install.call_args_list
        child_install_calls = [
            call for call in install_calls
            if call[0][1] == child_entity  # entity_id arg
        ]
        assert len(child_install_calls) > 0, (
            "SkillManager.install() should be called for child entity, not manual dict copy"
        )
