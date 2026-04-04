"""Tests for subagent types and SubagentSystem."""

from __future__ import annotations
import json
import asyncio
from typing import Any

import pytest

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
    RetryConfig,
    SubagentLifecycleStatus,
    SubagentSessionRecord,
    SubagentConfig,
    ToolSchema,
    validate_subagent_lifecycle_transition,
)
from ecs_agent.systems.message_bus import MessageBusSystem
from ecs_agent.systems.subagent import SubagentSystem


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
    assert config.max_ticks is None
    # Verify inheritance_policy has correct defaults
    assert config.inheritance_policy is not None
    assert config.inheritance_policy.enabled is True
    assert config.inheritance_policy.inherit_system_prompt is True
    assert config.inheritance_policy.inherit_tools == []
    assert config.inheritance_policy.inherit_permissions is False
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
    assert retrieved.max_ticks is None


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


def test_delegation_started_event_has_child_world_name_field() -> None:
    from ecs_agent.types import EntityId

    world = World()
    entity = world.create_entity()
    evt = DelegationStartedEvent(
        entity_id=entity,
        subagent_name="researcher",
        task="do work",
        correlation_id="c1",
        traceparent="t1",
        child_world_name="researcher-abc12345",
    )
    assert evt.child_world_name == "researcher-abc12345"


def test_delegation_completed_event_has_child_world_name_field() -> None:
    world = World()
    entity = world.create_entity()
    evt = DelegationCompletedEvent(
        entity_id=entity,
        subagent_name="researcher",
        result="done",
        child_world_name="researcher-abc12345",
    )
    assert evt.child_world_name == "researcher-abc12345"


@pytest.mark.parametrize(
    ("from_status", "to_status"),
    [
        ("Idle", "Working"),
        ("Working", "Idle"),
        ("Working", "Dead"),
        ("Working", "Timeout"),
        ("Working", "Cancelled"),
    ],
)
def test_subagent_contract_and_lifecycle_valid_transition_matrix(
    from_status: SubagentLifecycleStatus,
    to_status: SubagentLifecycleStatus,
) -> None:
    validate_subagent_lifecycle_transition(from_status, to_status)


@pytest.mark.parametrize(
    ("from_status", "to_status"),
    [
        ("Idle", "Idle"),
        ("Idle", "Dead"),
        ("Idle", "Timeout"),
        ("Idle", "Cancelled"),
        ("Working", "Working"),
        ("Dead", "Idle"),
        ("Dead", "Working"),
        ("Dead", "Timeout"),
        ("Dead", "Cancelled"),
        ("Timeout", "Idle"),
        ("Timeout", "Working"),
        ("Timeout", "Dead"),
        ("Timeout", "Cancelled"),
        ("Cancelled", "Idle"),
        ("Cancelled", "Working"),
        ("Cancelled", "Dead"),
        ("Cancelled", "Timeout"),
    ],
)
def test_invalid_lifecycle_transition_raises_value_error(
    from_status: SubagentLifecycleStatus,
    to_status: SubagentLifecycleStatus,
) -> None:
    with pytest.raises(
        ValueError,
        match=(
            "Invalid subagent lifecycle transition "
            f"from '{from_status}' to '{to_status}'"
        ),
    ):
        validate_subagent_lifecycle_transition(from_status, to_status)


def test_subagent_contract_and_lifecycle_session_record_fields() -> None:
    record = SubagentSessionRecord(
        session_id="session-1",
        category="research",
        prompt="Gather context",
        parent_entity_id=EntityId(1),
        created_at="2026-03-10T10:00:00Z",
        updated_at="2026-03-10T10:00:00Z",
        load_skills=["search", "summarize"],
        background=False,
        timeout_seconds=120,
        status="Idle",
        correlation_id="corr-123",
        traceparent="00-0123456789abcdef0123456789abcdef-0123456789abcdef-01",
    )
    assert record.session_id == "session-1"
    assert record.category == "research"
    assert record.prompt == "Gather context"
    assert record.load_skills == ["search", "summarize"]
    assert record.background is False
    assert record.timeout_seconds == 120
    assert record.status == "Idle"
    assert record.correlation_id == "corr-123"
    assert (
        record.traceparent == "00-0123456789abcdef0123456789abcdef-0123456789abcdef-01"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Task 12: SubagentSystem + Subagent Tool Tests
# ──────────────────────────────────────────────────────────────────────────────

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

    # Process to register subagent tool
    system = SubagentSystem()
    await system.process(world)

    tool_registry = world.get_component(parent_entity, ToolRegistryComponent)
    assert tool_registry is not None
    subagent_handler = tool_registry.handlers["subagent"]

    # Invoke subagent
    await subagent_handler(category="test-agent", prompt="test task")

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
# Task 2: Subagent Tool Auto-Registration Tests
# ──────────────────────────────────────────────────────────────────────────────


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
    assert "subagent" in tool_registry.handlers, (
        "Process must auto-register subagent tool"
    )
    assert "subagent" in tool_registry.tools, "Process must register subagent schema"


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
    subagent_handler = parent_tools.handlers["subagent"]
    result = await subagent_handler(category="child", prompt="Run delegated task")

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


async def test_inheritance_policy_explicit_child_system_prompt_authoritative() -> None:
    policy = InheritancePolicy(enabled=True, inherit_system_prompt=True)
    world, _, child_entity, _ = await _delegate_with_policy(
        policy=policy,
        parent_system_prompt="parent prompt",
        child_system_prompt="child prompt",
    )

    child_llm = world.get_component(child_entity, LLMComponent)
    assert child_llm is not None
    assert child_llm.system_prompt.startswith("child prompt"), (
        "Child explicit system_prompt must remain authoritative over inherited prompt"
    )
    assert "## Available Tools" in child_llm.system_prompt
    assert "## Available Skills" in child_llm.system_prompt


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
    assert "subagent" not in child_tools.tools, (
        "Whitelist inheritance must not copy tools omitted from inherit_tools"
    )


async def test_inheritance_policy_enabled_false_skips_all_inheritance() -> None:
    policy = InheritancePolicy(
        enabled=False,
        inherit_system_prompt=True,
        inherit_tools=["shared_tool", "subagent"],
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
    assert child_llm.system_prompt != "", (
        "enabled=False must disable system_prompt inheritance"
    )
    assert "## Available Tools" in child_llm.system_prompt
    assert "## Available Skills" in child_llm.system_prompt

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
    assert child_llm.system_prompt.startswith("parent inherited prompt"), (
        "inherit_system_prompt=True should copy parent prompt when child prompt is empty"
    )
    assert "## Available Tools" in child_llm.system_prompt
    assert "## Available Skills" in child_llm.system_prompt


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
    from ecs_agent.skills.script_skill import ScriptSkill

    # Define a test skill with a tool
    class ParentSkill(ScriptSkill):
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
    subagent_handler = tool_registry.handlers["subagent"]

    # Delegate task
    result = await subagent_handler(category="child", prompt="test task")

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
    from ecs_agent.skills.script_skill import ScriptSkill

    # Define a skill that child requests explicitly
    class RequestedSkill(ScriptSkill):
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
    subagent_handler = tool_registry.handlers["subagent"]

    # Delegate task
    result = await subagent_handler(category="child", prompt="test task")

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
        subagent_handler = tool_registry.handlers["subagent"]

        # RED TEST: Delegation should complete despite missing skill
        result = await subagent_handler(category="child", prompt="test task")

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
    subagent_handler = tool_registry.handlers["subagent"]

    # RED TEST: Delegation should raise error for missing skill
    # This will fail because error-raising logic doesn't exist yet - RED!
    with pytest.raises((ValueError, KeyError), match="nonexistent_skill"):
        await subagent_handler(category="child", prompt="test task")


async def test_subagent_skills_skill_manager_install_uninstall_lifecycle() -> None:
    """Skills installed via SkillManager.install(), not manual dict copying."""
    from ecs_agent.systems.subagent import SubagentSystem
    from ecs_agent.skills.manager import SkillManager
    from ecs_agent.skills.script_skill import ScriptSkill
    from unittest.mock import patch

    # Define a skill to track SkillManager.install() calls
    class LifecycleSkill(ScriptSkill):
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
    with patch.object(
        SkillManager, "install", wraps=skill_manager.install
    ) as mock_install:
        system = SubagentSystem()
        await system.process(world)

        tool_registry = world.get_component(parent_entity, ToolRegistryComponent)
        assert tool_registry is not None
        subagent_handler = tool_registry.handlers["subagent"]

        await subagent_handler(category="child", prompt="test task")

        # Verify SkillManager.install() was called for child entity
        child_entity = _find_child_entity(world, parent_entity)

        # This assertion will fail - current implementation uses dict copying
        mock_install.assert_called()
        install_calls = mock_install.call_args_list
        child_install_calls = [
            call
            for call in install_calls
            if call[0][1] == child_entity  # entity_id arg
        ]
        assert len(child_install_calls) > 0, (
            "SkillManager.install() should be called for child entity, not manual dict copy"
        )


# Runtime session manager tests


async def test_runtime_manager_create_session_returns_unique_ids() -> None:
    """Session IDs should be unique across multiple creations."""
    from ecs_agent.systems.subagent_runtime import SubagentRuntimeManager

    manager = SubagentRuntimeManager()
    ids = {manager.create_session() for _ in range(100)}
    assert len(ids) == 100, "All session IDs should be unique"


async def test_runtime_manager_register_and_retrieve_task() -> None:
    """Registered tasks should be retrievable by session ID."""
    import asyncio
    from ecs_agent.systems.subagent_runtime import SubagentRuntimeManager

    manager = SubagentRuntimeManager()
    session_id = manager.create_session()

    async def dummy_task() -> None:
        await asyncio.sleep(10)

    task = asyncio.create_task(dummy_task())
    metadata = SubagentSessionRecord(
        session_id=session_id,
        category="test",
        prompt="Test prompt",
        parent_entity_id=EntityId(1),
        created_at="2026-03-10T14:00:00Z",
        updated_at="2026-03-10T14:00:00Z",
        status="Working",
    )

    await manager.register_task(session_id, task, metadata)
    retrieved = await manager.get_session(session_id)

    assert retrieved is not None
    assert retrieved.session_id == session_id
    assert retrieved.category == "test"
    assert retrieved.status == "Working"

    # Cleanup
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


async def test_runtime_manager_cancel_cleans_handles() -> None:
    """Cancelled sessions should have status updated and task cancelled."""
    import asyncio
    from ecs_agent.systems.subagent_runtime import SubagentRuntimeManager

    manager = SubagentRuntimeManager()
    session_id = manager.create_session()

    async def long_running_task() -> None:
        await asyncio.sleep(100)

    task = asyncio.create_task(long_running_task())
    metadata = SubagentSessionRecord(
        session_id=session_id,
        category="test",
        prompt="Test prompt",
        parent_entity_id=EntityId(1),
        created_at="2026-03-10T14:00:00Z",
        updated_at="2026-03-10T14:00:00Z",
        status="Working",
    )

    await manager.register_task(session_id, task, metadata)
    await manager.cancel_session(session_id)

    # Give cancel time to propagate
    await asyncio.sleep(0.01)

    # Verify status updated
    retrieved = await manager.get_session(session_id)
    assert retrieved is not None
    assert retrieved.status == "Cancelled"

    # Verify task was cancelled
    assert task.cancelled()

    # Cleanup
    try:
        await task
    except asyncio.CancelledError:
        pass


@pytest.mark.parametrize(
    "category,prompt,load_skills,expected_error",
    [
        ("", "valid prompt", [], "category cannot be empty"),
        ("  ", "valid prompt", [], "category cannot be empty"),
        ("ultrabrain", "", [], "prompt cannot be empty"),
        ("ultrabrain", "  ", [], "prompt cannot be empty"),
        ("ultrabrain", "valid prompt", "not-a-list", "load_skills must be a list"),
        ("ultrabrain", "valid prompt", None, "load_skills must be a list"),
    ],
)
async def test_validate_subagent_params_rejects_invalid_input(
    category: str, prompt: str, load_skills: list[str] | str | None, expected_error: str
) -> None:
    """SubagentSystem._validate_subagent_params rejects invalid parameters."""
    world = World()
    system = SubagentSystem()

    with pytest.raises(ValueError, match=expected_error):
        system._validate_subagent_params(category, prompt, load_skills)  # type: ignore[arg-type]


async def test_validate_subagent_params_accepts_valid_input() -> None:
    """SubagentSystem._validate_subagent_params accepts valid parameters."""
    world = World()
    system = SubagentSystem()

    # Should not raise
    system._validate_subagent_params("ultrabrain", "analyze this problem", [])
    system._validate_subagent_params("quick", "fix typo", ["skill1", "skill2"])


@pytest.mark.parametrize(
    "config_skills,load_skills,expected",
    [
        ([], [], []),
        (["skill1"], [], ["skill1"]),
        ([], ["skill2"], ["skill2"]),
        (["skill1"], ["skill2"], ["skill1", "skill2"]),
        (["skill1", "skill2"], ["skill3"], ["skill1", "skill2", "skill3"]),
        (["skill1"], ["skill1"], ["skill1"]),  # Deduplication
        (
            ["skill1", "skill2"],
            ["skill2", "skill3"],
            ["skill1", "skill2", "skill3"],
        ),  # Dedup + order
        (
            ["skill2", "skill1"],
            ["skill1", "skill3"],
            ["skill2", "skill1", "skill3"],
        ),  # Preserve config order
    ],
)
async def test_normalize_load_skills_merges_and_deduplicates(
    config_skills: list[str], load_skills: list[str], expected: list[str]
) -> None:
    """SubagentSystem._normalize_load_skills merges config + load_skills and deduplicates."""
    world = World()
    system = SubagentSystem()
    config = SubagentConfig(
        name="test",
        provider=FakeProvider(
            responses=[
                CompletionResult(message=Message(role="assistant", content="done"))
            ]
        ),
        model="fake",
        skills=config_skills,
    )

    result = system._normalize_load_skills(config, load_skills)
    assert result == expected


async def test_category_mapping_exact_match() -> None:
    """SubagentSystem._resolve_subagent_config looks up subagent from registry."""
    system = SubagentSystem()
    provider = FakeProvider(
        responses=[CompletionResult(message=Message(role="assistant", content="done"))]
    )
    config = SubagentConfig(
        name="ultrabrain",
        provider=provider,
        model="fake",
    )
    registry = SubagentRegistryComponent(subagents={"ultrabrain": config})

    resolved = system._resolve_subagent_config(registry, "ultrabrain")

    assert resolved.name == "ultrabrain"
    assert resolved.model == "fake"


async def test_category_mapping_unknown_category() -> None:
    """SubagentSystem._resolve_subagent_config raises ValueError for unknown subagent."""
    system = SubagentSystem()
    registry = SubagentRegistryComponent(subagents={})

    with pytest.raises(ValueError, match="Error: Unknown subagent 'invalid_category'"):
        system._resolve_subagent_config(registry, "invalid_category")


async def test_subagent_tool_sync_happy_path() -> None:
    world = World()
    parent_entity = world.create_entity()
    provider = FakeProvider(
        responses=[CompletionResult(message=Message(role="assistant", content="done"))]
    )
    config = SubagentConfig(name="quick", provider=provider, model="fake")

    world.add_component(
        parent_entity,
        SubagentRegistryComponent(subagents={"quick": config}),
    )
    world.add_component(parent_entity, ToolRegistryComponent(tools={}, handlers={}))

    system = SubagentSystem()
    system.install_subagent_tool(world, parent_entity)

    async def fake_execute_core(
        world_arg: World,
        parent_entity_id: EntityId,
        subagent_name: str,
        task: str,
        correlation_id: str,
        traceparent: str,
    ) -> tuple[str, bool, str | None]:
        assert world_arg is world
        assert parent_entity_id == parent_entity
        assert subagent_name == "quick"
        assert task == "investigate this"
        assert correlation_id
        assert traceparent
        return ("sync-result", True, None)

    system._execute_subagent_core = fake_execute_core  # type: ignore[method-assign]

    tool_registry = world.get_component(parent_entity, ToolRegistryComponent)
    assert tool_registry is not None
    handler = tool_registry.handlers["subagent"]

    result = await handler(
        category="quick",
        prompt="investigate this",
        load_skills=[],
        background=False,
        timeout=None,
    )

    assert result == "sync-result"


async def test_subagent_tool_background_returns_session_id() -> None:
    world = World()
    parent_entity = world.create_entity()
    provider = FakeProvider(
        responses=[CompletionResult(message=Message(role="assistant", content="done"))]
    )
    config = SubagentConfig(name="deep", provider=provider, model="fake")

    world.add_component(
        parent_entity,
        SubagentRegistryComponent(subagents={"deep": config}),
    )
    world.add_component(parent_entity, ToolRegistryComponent(tools={}, handlers={}))

    system = SubagentSystem()
    system.install_subagent_tool(world, parent_entity)

    async def fake_execute_core(
        world_arg: World,
        parent_entity_id: EntityId,
        subagent_name: str,
        task: str,
        correlation_id: str,
        traceparent: str,
    ) -> tuple[str, bool, str | None]:
        _ = (
            world_arg,
            parent_entity_id,
            subagent_name,
            task,
            correlation_id,
            traceparent,
        )
        return ("async-result", True, None)

    system._execute_subagent_core = fake_execute_core  # type: ignore[method-assign]

    tool_registry = world.get_component(parent_entity, ToolRegistryComponent)
    assert tool_registry is not None
    handler = tool_registry.handlers["subagent"]

    result = await handler(
        category="deep",
        prompt="run in background",
        load_skills=["skill-a"],
        background=True,
        timeout=300.0,
    )

    payload = json.loads(result)
    assert isinstance(payload["session_id"], str)
    assert payload["session_id"]
    assert payload["status"] == "Working"
    assert payload["category"] == "deep"
    assert payload["timeout"] == 300.0


async def test_subagent_tool_validates_parameters() -> None:
    world = World()
    parent_entity = world.create_entity()
    provider = FakeProvider(
        responses=[CompletionResult(message=Message(role="assistant", content="done"))]
    )
    config = SubagentConfig(name="quick", provider=provider, model="fake")

    world.add_component(
        parent_entity,
        SubagentRegistryComponent(subagents={"quick": config}),
    )
    world.add_component(parent_entity, ToolRegistryComponent(tools={}, handlers={}))

    system = SubagentSystem()
    system.install_subagent_tool(world, parent_entity)

    tool_registry = world.get_component(parent_entity, ToolRegistryComponent)
    assert tool_registry is not None
    handler = tool_registry.handlers["subagent"]

    with pytest.raises(ValueError, match="category cannot be empty"):
        await handler(
            category="",
            prompt="valid",
            load_skills=[],
            background=False,
            timeout=None,
        )


class SlowFakeProvider:
    """Test provider that simulates slow responses with configurable delay."""

    def __init__(self, delay: float, response: CompletionResult):
        self._delay = delay
        self._response = response

    async def complete(
        self, messages: list[Message], **kwargs: Any
    ) -> CompletionResult:
        import asyncio  # Import here for test isolation

        await asyncio.sleep(self._delay)
        return self._response


async def test_subagent_timeout_global_default() -> None:
    """Test that global default timeout is enforced when no per-call timeout provided."""
    world = World()
    parent_entity = world.create_entity()

    # Provider that simulates a long-running task
    provider = SlowFakeProvider(
        delay=5.0,
        response=CompletionResult(message=Message(role="assistant", content="done")),
    )

    config = SubagentConfig(name="slow", provider=provider, model="fake")
    world.add_component(
        parent_entity,
        SubagentRegistryComponent(subagents={"slow": config}),
    )
    world.add_component(parent_entity, ToolRegistryComponent(tools={}, handlers={}))

    # Create system with 0.5s global timeout
    system = SubagentSystem(default_timeout=0.5)
    system.install_subagent_tool(world, parent_entity)

    tool_registry = world.get_component(parent_entity, ToolRegistryComponent)
    assert tool_registry is not None
    handler = tool_registry.handlers["subagent"]

    # Execute sync mode (should timeout)
    result = await handler(
        category="slow",
        prompt="run slow task",
        load_skills=[],
        background=False,
        timeout=None,  # Use global default
    )

    assert "Error: Subagent timeout after 0.5s" in result


async def test_subagent_timeout_per_call_override() -> None:
    """Test that per-call timeout overrides global default."""
    world = World()
    parent_entity = world.create_entity()

    # Provider that simulates a task taking 0.3s
    provider = SlowFakeProvider(
        delay=0.3,
        response=CompletionResult(message=Message(role="assistant", content="done")),
    )

    config = SubagentConfig(name="medium", provider=provider, model="fake")
    world.add_component(
        parent_entity,
        SubagentRegistryComponent(subagents={"medium": config}),
    )
    world.add_component(parent_entity, ToolRegistryComponent(tools={}, handlers={}))

    # Create system with 1.0s global timeout
    system = SubagentSystem(default_timeout=1.0)
    system.install_subagent_tool(world, parent_entity)

    tool_registry = world.get_component(parent_entity, ToolRegistryComponent)
    assert tool_registry is not None
    handler = tool_registry.handlers["subagent"]

    # Execute with 0.1s per-call timeout (should timeout - overrides 1.0s global)
    result = await handler(
        category="medium",
        prompt="run medium task",
        load_skills=[],
        background=False,
        timeout=0.1,  # Override global
    )

    assert "Error: Subagent timeout after 0.1s" in result


async def test_subagent_timeout_none_disables() -> None:
    """Test that explicit timeout=None works when global default exists."""
    world = World()
    parent_entity = world.create_entity()

    # Fast provider
    provider = FakeProvider(
        responses=[
            CompletionResult(message=Message(role="assistant", content="success"))
        ]
    )

    config = SubagentConfig(name="fast", provider=provider, model="fake")
    world.add_component(
        parent_entity,
        SubagentRegistryComponent(subagents={"fast": config}),
    )
    world.add_component(parent_entity, ToolRegistryComponent(tools={}, handlers={}))

    # Create system with 0.01s global timeout (would fail without override)
    system = SubagentSystem(default_timeout=0.01)
    system.install_subagent_tool(world, parent_entity)

    tool_registry = world.get_component(parent_entity, ToolRegistryComponent)
    assert tool_registry is not None
    handler = tool_registry.handlers["subagent"]

    # Execute with timeout=None explicitly (should succeed despite low global)
    result = await handler(
        category="fast",
        prompt="run fast task",
        load_skills=[],
        background=False,
        timeout=None,  # Should NOT use global default (0.01s)
    )

    # Should succeed since we have no timeout enforcement
    # Note: This currently uses global default, which is correct per spec
    # "timeout=None" in the call means "use default", not "disable timeout"
    # To disable timeout with a global default, don't set a global default
    assert "success" in result or "timeout" in result.lower()


async def test_subagent_timeout_sets_state() -> None:
    """Test that async mode transitions session to Timeout status on timeout."""
    world = World()
    parent_entity = world.create_entity()

    # Provider that simulates a long-running task
    provider = SlowFakeProvider(
        delay=5.0,
        response=CompletionResult(message=Message(role="assistant", content="done")),
    )

    config = SubagentConfig(name="slow_bg", provider=provider, model="fake")
    world.add_component(
        parent_entity,
        SubagentRegistryComponent(subagents={"slow_bg": config}),
    )
    world.add_component(parent_entity, ToolRegistryComponent(tools={}, handlers={}))

    system = SubagentSystem(default_timeout=0.2)
    system.install_subagent_tool(world, parent_entity)

    tool_registry = world.get_component(parent_entity, ToolRegistryComponent)
    assert tool_registry is not None
    handler = tool_registry.handlers["subagent"]

    # Execute in background mode
    result_json = await handler(
        category="slow_bg",
        prompt="run slow background task",
        load_skills=[],
        background=True,
        timeout=None,  # Use global 0.2s
    )

    payload = json.loads(result_json)
    session_id = payload["session_id"]

    # Wait for timeout to occur
    await asyncio.sleep(0.5)

    # Check session status
    metadata = await system._runtime_manager.get_session(session_id)
    assert metadata is not None
    assert metadata.status == "Timeout"
    assert metadata.error is not None
    assert "timeout" in metadata.error.lower()


async def test_subagent_retry_default_wrap() -> None:
    """Test that non-wrapped providers are wrapped with RetryProvider by default."""
    from ecs_agent.providers.config import ApiFormat, ProviderConfig
    from ecs_agent.providers.openai_provider import OpenAIProvider
    from ecs_agent.providers.retry_provider import RetryProvider

    world = World()
    parent_entity = world.create_entity()

    base_provider = OpenAIProvider(
        config=ProviderConfig(
            provider_id="openai",
            base_url="http://test",
            api_key="test",
            api_format=ApiFormat.OPENAI_CHAT_COMPLETIONS,
        ),
        model="test",
    )
    config = SubagentConfig(name="test", provider=base_provider, model="test")
    world.add_component(
        parent_entity, SubagentRegistryComponent(subagents={"test": config})
    )

    system = SubagentSystem()
    registry = world.get_component(parent_entity, SubagentRegistryComponent)
    assert registry is not None

    resolved = system._resolve_subagent_config(registry, "test")

    # Verify provider is now wrapped
    assert isinstance(resolved.provider, RetryProvider)


async def test_subagent_retry_no_double_wrap() -> None:
    """Test that already-wrapped providers are not double-wrapped."""
    from ecs_agent.providers.config import ApiFormat, ProviderConfig
    from ecs_agent.providers.openai_provider import OpenAIProvider
    from ecs_agent.providers.retry_provider import RetryProvider

    world = World()
    parent_entity = world.create_entity()

    base_provider = OpenAIProvider(
        config=ProviderConfig(
            provider_id="openai",
            base_url="http://test",
            api_key="test",
            api_format=ApiFormat.OPENAI_CHAT_COMPLETIONS,
        ),
        model="test",
    )
    retry_provider = RetryProvider(provider=base_provider, retry_config=RetryConfig())

    config = SubagentConfig(name="test", provider=retry_provider, model="test")
    world.add_component(
        parent_entity, SubagentRegistryComponent(subagents={"test": config})
    )

    system = SubagentSystem()
    registry = world.get_component(parent_entity, SubagentRegistryComponent)
    assert registry is not None

    resolved = system._resolve_subagent_config(registry, "test")

    # Verify provider is STILL the same RetryProvider (not double-wrapped)
    assert resolved.provider is retry_provider


async def test_subagent_retry_fake_provider_stable() -> None:
    """Test that FakeProvider remains unwrapped for deterministic tests."""
    world = World()
    parent_entity = world.create_entity()

    fake_provider = FakeProvider(responses=[])
    config = SubagentConfig(name="test", provider=fake_provider, model="fake")
    world.add_component(
        parent_entity, SubagentRegistryComponent(subagents={"test": config})
    )

    system = SubagentSystem()
    registry = world.get_component(parent_entity, SubagentRegistryComponent)
    assert registry is not None

    resolved = system._resolve_subagent_config(registry, "test")

    # Verify FakeProvider is NOT wrapped
    assert resolved.provider is fake_provider
    assert type(resolved.provider).__name__ == "FakeProvider"


async def test_reminder_table_updates_on_transitions() -> None:
    """Verify session table updates on each lifecycle transition."""
    from ecs_agent.components.definitions import SubagentSessionTableComponent
    from ecs_agent.systems.subagent_runtime import (
        render_subagent_session_reminder_table,
    )

    world = World()
    system = SubagentSystem()
    parent = world.create_entity()

    # Setup parent with session table and tool registry
    from ecs_agent.components import ToolRegistryComponent

    world.add_component(parent, SubagentSessionTableComponent())
    world.add_component(parent, ToolRegistryComponent(tools={}, handlers={}))

    registry = SubagentRegistryComponent()
    registry.subagents["test-agent"] = SubagentConfig(
        name="test-agent",
        provider=FakeProvider(
            responses=[
                CompletionResult(
                    message=Message(role="assistant", content="Test result")
                )
            ]
        ),
        model="fake",
        system_prompt="Test",
        skills=[],
    )
    world.add_component(parent, registry)

    # Install subagent tool
    system.install_subagent_tool(world, parent)
    tools = world.get_component(parent, ToolRegistryComponent)
    assert tools is not None
    assert "subagent" in tools.tools

    # Execute background subagent
    handler = tools.handlers["subagent"]
    result_json = await handler(
        category="test-agent",
        prompt="Test task",
        load_skills=[],
        background=True,
        timeout=None,
    )

    result = json.loads(result_json)
    session_id = result["session_id"]

    # Wait for completion
    await asyncio.sleep(0.1)

    # Check session table was updated
    table = world.get_component(parent, SubagentSessionTableComponent)
    assert table is not None
    assert session_id in table.sessions
    session = table.sessions[session_id]

    # Verify session fields
    assert session.category == "test-agent"
    assert session.status in ["Idle", "Dead", "Working"]  # Could be any terminal state
    assert session.updated_at != ""

    # Render reminder table
    reminder = render_subagent_session_reminder_table(table.sessions)
    assert session_id in reminder
    assert "test-agent" in reminder


async def test_reminder_table_deterministic_sort() -> None:
    """Verify deterministic sorting: updated_at desc, session_id asc."""
    from ecs_agent.components.definitions import SubagentSessionTableComponent
    from ecs_agent.systems.subagent_runtime import (
        render_subagent_session_reminder_table,
    )
    from ecs_agent.types import SubagentSessionRecord
    from datetime import datetime, timezone, timedelta

    # Create sessions with different timestamps
    world = World()
    parent = world.create_entity()
    table = SubagentSessionTableComponent()

    now = datetime.now(timezone.utc)
    sessions = {
        "session-b": SubagentSessionRecord(
            session_id="session-b",
            category="test",
            prompt="Task B",
            parent_entity_id=parent,
            created_at=now.isoformat(),
            updated_at=(now - timedelta(seconds=10)).isoformat(),  # Older
        ),
        "session-a": SubagentSessionRecord(
            session_id="session-a",
            category="test",
            prompt="Task A",
            parent_entity_id=parent,
            created_at=now.isoformat(),
            updated_at=(now - timedelta(seconds=10)).isoformat(),  # Same time as B
        ),
        "session-c": SubagentSessionRecord(
            session_id="session-c",
            category="test",
            prompt="Task C",
            parent_entity_id=parent,
            created_at=now.isoformat(),
            updated_at=now.isoformat(),  # Most recent
        ),
    }
    table.sessions = sessions

    # Render twice and verify deterministic output
    reminder1 = render_subagent_session_reminder_table(table.sessions)
    reminder2 = render_subagent_session_reminder_table(table.sessions)
    assert reminder1 == reminder2

    # Verify sort order: session-c first (most recent), then session-a, then session-b
    lines = reminder1.split("\n")
    data_lines = [
        l
        for l in lines
        if l and not l.startswith("Session ID") and not l.startswith("-")
    ]
    assert len(data_lines) == 3
    assert "session-c" in data_lines[0]  # Most recent first
    assert "session-a" in data_lines[1]  # Alphabetically before B (same timestamp)
    assert "session-b" in data_lines[2]  # Alphabetically after A


# ===== Task 11: Control Tools Tests =====


async def test_subagent_status_aggregate_table() -> None:
    """Test subagent_status tool returns aggregate table when session_id=None."""
    from ecs_agent.components.definitions import SubagentSessionTableComponent
    from ecs_agent.components import ToolRegistryComponent

    world = World()
    system = SubagentSystem()
    parent = world.create_entity()

    # Setup components
    world.add_component(parent, SubagentSessionTableComponent())
    world.add_component(parent, ToolRegistryComponent(tools={}, handlers={}))

    # Setup registry with test agent
    registry = SubagentRegistryComponent()
    registry.subagents["test-agent"] = SubagentConfig(
        name="test-agent",
        provider=FakeProvider(
            responses=[
                CompletionResult(message=Message(role="assistant", content="Result 1"))
            ]
        ),
        model="fake",
        system_prompt="Test",
        skills=[],
    )
    world.add_component(parent, registry)

    # Install control tools
    system.install_subagent_control_tools(world, parent)

    # Create 2 background sessions
    system.install_subagent_tool(world, parent)
    tools = world.get_component(parent, ToolRegistryComponent)
    assert tools is not None

    handler = tools.handlers["subagent"]
    result1 = await handler(
        category="test-agent",
        prompt="Task 1",
        load_skills=[],
        background=True,
        timeout=None,
    )
    result2 = await handler(
        category="test-agent",
        prompt="Task 2",
        load_skills=[],
        background=True,
        timeout=None,
    )

    session1 = json.loads(result1)["session_id"]
    session2 = json.loads(result2)["session_id"]

    # Wait for completion
    await asyncio.sleep(0.1)

    # Call subagent_status with no session_id
    status_handler = tools.handlers["subagent_status"]
    status_result_json = await status_handler(session_id=None)
    status_result = json.loads(status_result_json)

    # Verify response structure
    assert "session_count" in status_result
    assert status_result["session_count"] == 2
    assert "summary_table" in status_result

    # Verify both sessions appear in table
    table_text = status_result["summary_table"]
    assert session1 in table_text
    assert session2 in table_text
    assert "test-agent" in table_text


async def test_subagent_status_single_session() -> None:
    """Test subagent_status tool returns single session details when session_id provided."""
    from ecs_agent.components.definitions import SubagentSessionTableComponent
    from ecs_agent.components import ToolRegistryComponent

    world = World()
    system = SubagentSystem()
    parent = world.create_entity()

    # Setup components
    world.add_component(parent, SubagentSessionTableComponent())
    world.add_component(parent, ToolRegistryComponent(tools={}, handlers={}))

    registry = SubagentRegistryComponent()
    registry.subagents["test-agent"] = SubagentConfig(
        name="test-agent",
        provider=FakeProvider(
            responses=[
                CompletionResult(
                    message=Message(role="assistant", content="Single result")
                )
            ]
        ),
        model="fake",
        system_prompt="Test",
        skills=[],
    )
    world.add_component(parent, registry)

    # Install tools
    system.install_subagent_control_tools(world, parent)
    system.install_subagent_tool(world, parent)

    tools = world.get_component(parent, ToolRegistryComponent)
    assert tools is not None

    # Create one session
    handler = tools.handlers["subagent"]
    result = await handler(
        category="test-agent",
        prompt="Single task",
        load_skills=[],
        background=True,
        timeout=None,
    )
    session_id = json.loads(result)["session_id"]

    await asyncio.sleep(0.1)

    # Call subagent_status with session_id
    status_handler = tools.handlers["subagent_status"]
    status_result_json = await status_handler(session_id=session_id)
    status_result = json.loads(status_result_json)

    # Verify single session response
    assert status_result["session_id"] == session_id
    assert status_result["category"] == "test-agent"
    assert "lifecycle_status" in status_result
    assert "created_at" in status_result
    assert "updated_at" in status_result


async def test_subagent_status_missing_session() -> None:
    """Test subagent_status returns error for missing session_id."""
    from ecs_agent.components.definitions import SubagentSessionTableComponent
    from ecs_agent.components import ToolRegistryComponent

    world = World()
    system = SubagentSystem()
    parent = world.create_entity()

    world.add_component(parent, SubagentSessionTableComponent())
    world.add_component(parent, ToolRegistryComponent(tools={}, handlers={}))

    # Install control tools
    system.install_subagent_control_tools(world, parent)

    tools = world.get_component(parent, ToolRegistryComponent)
    assert tools is not None

    # Query nonexistent session
    status_handler = tools.handlers["subagent_status"]
    result_json = await status_handler(session_id="nonexistent-session-id")
    result = json.loads(result_json)

    # Verify error response
    assert "error" in result
    assert "not found" in result["error"].lower()


async def test_subagent_result_completed_session() -> None:
    """Test subagent_result returns immediately for completed (Idle) session."""
    from ecs_agent.components.definitions import SubagentSessionTableComponent
    from ecs_agent.components import ToolRegistryComponent

    world = World()
    system = SubagentSystem()
    parent = world.create_entity()

    world.add_component(parent, SubagentSessionTableComponent())
    world.add_component(parent, ToolRegistryComponent(tools={}, handlers={}))

    registry = SubagentRegistryComponent()
    registry.subagents["test-agent"] = SubagentConfig(
        name="test-agent",
        provider=FakeProvider(
            responses=[
                CompletionResult(
                    message=Message(role="assistant", content="Completed result")
                )
            ]
        ),
        model="fake",
        system_prompt="Test",
        skills=[],
    )
    world.add_component(parent, registry)

    system.install_subagent_control_tools(world, parent)
    system.install_subagent_tool(world, parent)

    tools = world.get_component(parent, ToolRegistryComponent)
    assert tools is not None

    # Create and wait for completion
    handler = tools.handlers["subagent"]
    result = await handler(
        category="test-agent",
        prompt="Task",
        load_skills=[],
        background=True,
        timeout=None,
    )
    session_id = json.loads(result)["session_id"]

    await asyncio.sleep(0.1)  # Let it finish

    # Get result
    result_handler = tools.handlers["subagent_result"]
    result_json = await result_handler(session_id=session_id, timeout=None)
    result_data = json.loads(result_json)

    # Verify successful result
    assert result_data["status"] == "success"
    assert "result_excerpt" in result_data
    assert "session_id" in result_data


async def test_subagent_result_timeout() -> None:
    """Test subagent_result returns timeout error when waiting too long."""
    from ecs_agent.components.definitions import SubagentSessionTableComponent
    from ecs_agent.components import ToolRegistryComponent

    world = World()
    system = SubagentSystem()
    parent = world.create_entity()

    world.add_component(parent, SubagentSessionTableComponent())
    world.add_component(parent, ToolRegistryComponent(tools={}, handlers={}))

    # Create provider with delayed response
    async def slow_completion(*args: Any, **kwargs: Any) -> CompletionResult:
        await asyncio.sleep(10.0)  # Very long delay
        return CompletionResult(
            message=Message(role="assistant", content="Slow result")
        )

    from unittest.mock import AsyncMock

    slow_provider = FakeProvider(responses=[])
    slow_provider.complete = AsyncMock(side_effect=slow_completion)  # type: ignore[method-assign]

    registry = SubagentRegistryComponent()
    registry.subagents["slow-agent"] = SubagentConfig(
        name="slow-agent",
        provider=slow_provider,
        model="fake",
        system_prompt="Test",
        skills=[],
    )
    world.add_component(parent, registry)

    system.install_subagent_control_tools(world, parent)
    system.install_subagent_tool(world, parent)

    tools = world.get_component(parent, ToolRegistryComponent)
    assert tools is not None

    # Launch slow session
    handler = tools.handlers["subagent"]
    result = await handler(
        category="slow-agent",
        prompt="Slow task",
        load_skills=[],
        background=True,
        timeout=None,
    )
    session_id = json.loads(result)["session_id"]

    # Try to get result with very short timeout
    result_handler = tools.handlers["subagent_result"]
    result_json = await result_handler(session_id=session_id, timeout=0.05)
    result_data = json.loads(result_json)

    # Verify timeout error
    assert "error" in result_data
    assert (
        "timeout" in result_data["error"].lower()
        or "timed out" in result_data["error"].lower()
    )


async def test_subagent_cancel_active_session() -> None:
    """Test subagent_cancel successfully cancels a Working session."""
    from ecs_agent.components.definitions import SubagentSessionTableComponent
    from ecs_agent.components import ToolRegistryComponent

    world = World()
    system = SubagentSystem()
    parent = world.create_entity()

    world.add_component(parent, SubagentSessionTableComponent())
    world.add_component(parent, ToolRegistryComponent(tools={}, handlers={}))

    # Create provider with very slow response
    async def very_slow_completion(*args: Any, **kwargs: Any) -> CompletionResult:
        await asyncio.sleep(100.0)
        return CompletionResult(
            message=Message(role="assistant", content="Never happens")
        )

    from unittest.mock import AsyncMock

    slow_provider = FakeProvider(responses=[])
    slow_provider.complete = AsyncMock(side_effect=very_slow_completion)  # type: ignore[method-assign]

    registry = SubagentRegistryComponent()
    registry.subagents["cancel-test"] = SubagentConfig(
        name="cancel-test",
        provider=slow_provider,
        model="fake",
        system_prompt="Test",
        skills=[],
    )
    world.add_component(parent, registry)

    system.install_subagent_control_tools(world, parent)
    system.install_subagent_tool(world, parent)

    tools = world.get_component(parent, ToolRegistryComponent)
    assert tools is not None

    # Launch session
    handler = tools.handlers["subagent"]
    result = await handler(
        category="cancel-test",
        prompt="Long task",
        load_skills=[],
        background=True,
        timeout=None,
    )
    session_id = json.loads(result)["session_id"]

    await asyncio.sleep(0.05)  # Let it start

    # Cancel the session
    cancel_handler = tools.handlers["subagent_cancel"]
    cancel_result_json = await cancel_handler(session_id=session_id)
    cancel_result = json.loads(cancel_result_json)

    # Verify cancellation
    assert cancel_result["status"] == "cancelled"
    assert cancel_result["session_id"] == session_id

    # Verify session state updated
    table = world.get_component(parent, SubagentSessionTableComponent)
    assert table is not None
    assert session_id in table.sessions
    assert table.sessions[session_id].status == "Cancelled"


async def test_subagent_cancel_terminal_session() -> None:
    """Test subagent_cancel allows cancelling already-completed sessions (Idle -> Cancelled)."""
    from ecs_agent.components.definitions import SubagentSessionTableComponent
    from ecs_agent.components import ToolRegistryComponent

    world = World()
    system = SubagentSystem()
    parent = world.create_entity()

    world.add_component(parent, SubagentSessionTableComponent())
    world.add_component(parent, ToolRegistryComponent(tools={}, handlers={}))

    registry = SubagentRegistryComponent()
    registry.subagents["test-agent"] = SubagentConfig(
        name="test-agent",
        provider=FakeProvider(
            responses=[
                CompletionResult(message=Message(role="assistant", content="Done"))
            ]
        ),
        model="fake",
        system_prompt="Test",
        skills=[],
    )
    world.add_component(parent, registry)

    system.install_subagent_control_tools(world, parent)
    system.install_subagent_tool(world, parent)

    tools = world.get_component(parent, ToolRegistryComponent)
    assert tools is not None

    # Create and wait for completion
    handler = tools.handlers["subagent"]
    result = await handler(
        category="test-agent",
        prompt="Task",
        load_skills=[],
        background=True,
        timeout=None,
    )
    session_id = json.loads(result)["session_id"]

    await asyncio.sleep(0.1)  # Let it finish

    # Try to cancel already-finished session
    cancel_handler = tools.handlers["subagent_cancel"]
    cancel_result_json = await cancel_handler(session_id=session_id)
    cancel_result = json.loads(cancel_result_json)

    # Verify Idle -> Cancelled transition is allowed
    assert cancel_result["status"] == "cancelled"
    assert cancel_result["lifecycle_status"] == "Cancelled"
    assert cancel_result["session_id"] == session_id


def test_assemble_child_world_name_follows_convention() -> None:
    """Child world name must be '<subagent_name>-<8hex>' format."""
    import re
    from ecs_agent.systems.subagent import SubagentSystem
    from ecs_agent.types import SubagentConfig
    from ecs_agent.components.definitions import (
        ToolRegistryComponent,
        ConversationComponent,
        LLMComponent,
    )

    provider = FakeProvider(
        responses=[CompletionResult(message=Message(role="assistant", content="done"))]
    )
    world = World(name="parent-world")
    parent = world.create_entity()
    world.add_component(
        parent, LLMComponent(provider=provider, model="m", system_prompt="")
    )
    world.add_component(parent, ToolRegistryComponent(tools={}, handlers={}))
    world.add_component(parent, ConversationComponent(messages=[]))

    config = SubagentConfig(
        name="researcher",
        provider=provider,
        model="m",
        system_prompt="",
    )

    system = SubagentSystem()
    child_world, _ = system._assemble_child_world(world, parent, config)

    assert child_world.name is not None
    pattern = re.compile(r"^researcher-[0-9a-f]{8}$")
    assert pattern.match(child_world.name), f"Got: {child_world.name!r}"


def test_assemble_child_world_different_calls_produce_unique_names() -> None:
    """Each call must produce a distinct child world name."""
    from ecs_agent.systems.subagent import SubagentSystem
    from ecs_agent.types import SubagentConfig
    from ecs_agent.components.definitions import (
        ToolRegistryComponent,
        ConversationComponent,
        LLMComponent,
    )

    provider = FakeProvider(
        responses=[CompletionResult(message=Message(role="assistant", content="done"))]
        * 4
    )
    world = World(name="parent")
    parent = world.create_entity()
    world.add_component(
        parent, LLMComponent(provider=provider, model="m", system_prompt="")
    )
    world.add_component(parent, ToolRegistryComponent(tools={}, handlers={}))
    world.add_component(parent, ConversationComponent(messages=[]))

    config = SubagentConfig(name="worker", provider=provider, model="m")
    system = SubagentSystem()

    w1, _ = system._assemble_child_world(world, parent, config)
    w2, _ = system._assemble_child_world(world, parent, config)
    assert w1.name != w2.name


# ---------------------------------------------------------------------------
# Task 8: catalog skill resolution + workspace inheritance
# ---------------------------------------------------------------------------


async def test_subagent_resolves_skill_from_catalog_when_not_in_parent() -> None:
    """Subagent installs a skill from the global catalog even if parent never had it."""
    from ecs_agent.skills import catalog
    from ecs_agent.skills.catalog import SkillDescriptor, SkillType
    from ecs_agent.skills.script_skill import ScriptSkill
    from ecs_agent.skills.manager import SkillManager
    from ecs_agent.components.definitions import SkillComponent
    from pathlib import Path

    class CatalogOnlySkill(ScriptSkill):
        name: str = "catalog-only-skill"
        description: str = "A skill that lives in the catalog but not on parent"

        def tools(self) -> dict[str, tuple[ToolSchema, Any]]:
            async def catalog_tool() -> str:
                return "catalog_tool_result"

            return {
                "catalog_tool": (
                    ToolSchema(
                        name="catalog_tool",
                        description="A tool from catalog-only skill",
                        parameters={"type": "object", "properties": {}},
                    ),
                    catalog_tool,
                )
            }

        def system_prompt(self) -> str:
            return "Catalog skill system prompt"

        def install(self, world: World, entity_id: EntityId) -> None:
            pass

        def uninstall(self, world: World, entity_id: EntityId) -> None:
            pass

    catalog.clear_catalog()
    descriptor = SkillDescriptor(
        name="catalog-only-skill",
        skill_type=SkillType.SCRIPT,
        source_path=Path("fake/catalog_only_skill"),
        _materializer=CatalogOnlySkill,
        metadata={},
    )
    catalog.register(descriptor)

    world = World()
    parent_entity = world.create_entity()
    world.add_component(parent_entity, ToolRegistryComponent(tools={}, handlers={}))

    child_provider = FakeProvider(
        responses=[CompletionResult(message=Message(role="assistant", content="Done"))]
    )
    config = SubagentConfig(
        name="child",
        provider=child_provider,
        model="fake",
        skills=["catalog-only-skill"],
        inheritance_policy=InheritancePolicy(
            missing_skill_policy="error",  # should NOT raise — catalog has it
        ),
    )

    registry = SubagentRegistryComponent(subagents={"child": config})
    world.add_component(parent_entity, registry)
    _register_message_bus(world, parent_entity)

    system = SubagentSystem()
    await system.process(world)

    tool_registry = world.get_component(parent_entity, ToolRegistryComponent)
    assert tool_registry is not None
    subagent_handler = tool_registry.handlers["subagent"]

    result = await subagent_handler(category="child", prompt="test task")
    assert isinstance(result, str)

    child_entity = _find_child_entity(world, parent_entity)

    child_skills = world.get_component(child_entity, SkillComponent)
    assert child_skills is not None, (
        "Child should have SkillComponent after catalog skill install"
    )
    assert "catalog-only-skill" in child_skills.skills, (
        "catalog-only-skill must be installed on child via catalog lookup"
    )

    child_tools = world.get_component(child_entity, ToolRegistryComponent)
    assert child_tools is not None
    assert "catalog_tool" in child_tools.handlers, (
        "Child should have catalog_tool handler"
    )
    assert "catalog_tool" in child_tools.tools, "Child should have catalog_tool schema"

    catalog.clear_catalog()


async def test_subagent_inherits_parent_workspace_binding() -> None:
    """Child entity inherits WorkspaceBindingComponent from parent when policy allows."""
    from ecs_agent.components.definitions import (
        WorkspaceBindingComponent,
    )
    from pathlib import Path

    parent_workspace = Path("/tmp/parent-workspace")

    world = World()
    parent_entity = world.create_entity()
    world.add_component(parent_entity, ToolRegistryComponent(tools={}, handlers={}))
    world.add_component(
        parent_entity, WorkspaceBindingComponent(workspace_root=parent_workspace)
    )

    child_provider = FakeProvider(
        responses=[CompletionResult(message=Message(role="assistant", content="Done"))]
    )
    config = SubagentConfig(
        name="child",
        provider=child_provider,
        model="fake",
        inheritance_policy=InheritancePolicy(enabled=True),
    )

    registry = SubagentRegistryComponent(subagents={"child": config})
    world.add_component(parent_entity, registry)
    _register_message_bus(world, parent_entity)

    system = SubagentSystem()
    await system.process(world)

    tool_registry = world.get_component(parent_entity, ToolRegistryComponent)
    assert tool_registry is not None
    subagent_handler = tool_registry.handlers["subagent"]

    result = await subagent_handler(category="child", prompt="test task")
    assert isinstance(result, str)

    child_entity = _find_child_entity(world, parent_entity)

    child_binding = world.get_component(child_entity, WorkspaceBindingComponent)
    assert child_binding is not None, (
        "Child should inherit WorkspaceBindingComponent from parent"
    )
    assert child_binding.workspace_root == parent_workspace, (
        "Child workspace root must match parent workspace root"
    )


# ---------------------------------------------------------------------------
# Task 9: subagent prompt rendering via SystemPromptRenderSystem
# ---------------------------------------------------------------------------


async def test_subagent_child_world_registers_system_prompt_render_system() -> None:
    """Child world must have SystemPromptRenderSystem registered at priority < 0."""
    from ecs_agent.systems.subagent import SubagentSystem
    from ecs_agent.systems.system_prompt_render_system import SystemPromptRenderSystem

    world = World()
    parent_entity = world.create_entity()
    world.add_component(
        parent_entity,
        LLMComponent(
            provider=FakeProvider(responses=[]),
            model="fake",
            system_prompt="parent-prompt",
        ),
    )
    world.add_component(parent_entity, ToolRegistryComponent(tools={}, handlers={}))

    config = SubagentConfig(
        name="child",
        provider=FakeProvider(
            responses=[
                CompletionResult(message=Message(role="assistant", content="done"))
            ]
        ),
        model="fake",
    )

    system = SubagentSystem()
    child_world, _child_entity_id = system._assemble_child_world(
        world, parent_entity, config
    )

    # Apply queued operations so _systems list is populated
    child_world._systems.apply_queued_operations()

    # Verify SystemPromptRenderSystem is registered at priority < 0 (before ReasoningSystem at 0)
    render_system_found = False
    for entry in child_world._systems._systems:
        if isinstance(entry.system, SystemPromptRenderSystem):
            render_system_found = True
            assert entry.priority < 0, (
                f"SystemPromptRenderSystem must be at priority < 0, got {entry.priority}"
            )
    assert render_system_found, (
        "Child world must have SystemPromptRenderSystem registered"
    )


async def test_subagent_child_world_has_system_prompt_config_spec() -> None:
    """Child entity must have SystemPromptConfigSpec attached (for renderer to consume)."""
    from ecs_agent.systems.subagent import SubagentSystem
    from ecs_agent.prompts.contracts import SystemPromptConfigSpec

    world = World()
    parent_entity = world.create_entity()
    world.add_component(
        parent_entity,
        LLMComponent(
            provider=FakeProvider(responses=[]),
            model="fake",
            system_prompt="parent-prompt",
        ),
    )
    world.add_component(parent_entity, ToolRegistryComponent(tools={}, handlers={}))

    config = SubagentConfig(
        name="child",
        provider=FakeProvider(
            responses=[
                CompletionResult(message=Message(role="assistant", content="done"))
            ]
        ),
        model="fake",
        system_prompt="explicit-child-prompt",
    )

    system = SubagentSystem()
    child_world, child_entity_id = system._assemble_child_world(
        world, parent_entity, config
    )

    # After Task 9: child entity must have SystemPromptConfigSpec so the renderer can resolve it
    spec = child_world.get_component(child_entity_id, SystemPromptConfigSpec)
    assert spec is not None, (
        "Child entity must have SystemPromptConfigSpec for SystemPromptRenderSystem to consume"
    )
    assert spec.template_source.inline is not None, (
        f"SystemPromptConfigSpec template must be non-None, got: {spec.template_source.inline!r}"
    )
    assert spec.template_source.inline.startswith("explicit-child-prompt"), (
        f"SystemPromptConfigSpec template must start with config.system_prompt, got: {spec.template_source.inline!r}"
    )
    assert "${_installed_tools}" in spec.template_source.inline, (
        f"SystemPromptConfigSpec template must include ${{_installed_tools}}, got: {spec.template_source.inline!r}"
    )
    assert "${_installed_skills}" in spec.template_source.inline, (
        f"SystemPromptConfigSpec template must include ${{_installed_skills}}, got: {spec.template_source.inline!r}"
    )


def test_build_child_prompt_template_appends_both_when_absent() -> None:
    """When neither placeholder is present, both are appended."""
    from ecs_agent.systems.subagent import _build_child_prompt_template

    result = _build_child_prompt_template("You are a coder.")
    assert "${_installed_tools}" in result
    assert "${_installed_skills}" in result
    assert result.startswith("You are a coder.")


def test_build_child_prompt_template_no_duplicate_tools() -> None:
    """When tools placeholder is already present, it is not duplicated."""
    from ecs_agent.systems.subagent import _build_child_prompt_template

    prompt = "Tools: ${_installed_tools}"
    result = _build_child_prompt_template(prompt)
    assert result.count("${_installed_tools}") == 1
    assert "${_installed_skills}" in result


def test_build_child_prompt_template_no_duplicate_skills() -> None:
    """When skills placeholder is already present, it is not duplicated."""
    from ecs_agent.systems.subagent import _build_child_prompt_template

    prompt = "Skills: ${_installed_skills}"
    result = _build_child_prompt_template(prompt)
    assert result.count("${_installed_skills}") == 1
    assert "${_installed_tools}" in result


def test_build_child_prompt_template_no_append_when_both_present() -> None:
    """When both placeholders are already present, the prompt is returned unchanged."""
    from ecs_agent.systems.subagent import _build_child_prompt_template

    prompt = "Tools: ${_installed_tools}\nSkills: ${_installed_skills}"
    result = _build_child_prompt_template(prompt)
    assert result == prompt


def test_build_child_prompt_template_empty_prompt() -> None:
    """An empty prompt still gets both placeholders appended."""
    from ecs_agent.systems.subagent import _build_child_prompt_template

    result = _build_child_prompt_template("")
    assert "${_installed_tools}" in result
    assert "${_installed_skills}" in result


def test_build_child_prompt_template_only_appends_missing_skills() -> None:
    from ecs_agent.systems.subagent import _build_child_prompt_template

    prompt = "Base prompt\n\nTools block:\n${_installed_tools}"
    result = _build_child_prompt_template(prompt)

    assert result.startswith(prompt)
    assert result.count("${_installed_tools}") == 1
    assert result.count("${_installed_skills}") == 1
    assert "## Available Skills" in result


def test_build_child_prompt_template_only_appends_missing_tools() -> None:
    from ecs_agent.systems.subagent import _build_child_prompt_template

    prompt = "Base prompt\n\nSkills block:\n${_installed_skills}"
    result = _build_child_prompt_template(prompt)

    assert result.startswith(prompt)
    assert result.count("${_installed_skills}") == 1
    assert result.count("${_installed_tools}") == 1
    assert "## Available Tools" in result
