"""Tests for enhanced event types used by skills, tool execution, and MCP."""

import pytest

from ecs_agent.core import World
from ecs_agent.types import (
    EntityId,
    ToolCall,
    ToolExecutionStartedEvent,
    ToolExecutionCompletedEvent,
    SkillInstalledEvent,
    SkillUninstalledEvent,
    SkillDiscoveryEvent,
    MCPConnectedEvent,
    MCPDisconnectedEvent,
    MCPToolCallEvent,
)


class TestToolExecutionStartedEvent:
    """Tests for ToolExecutionStartedEvent."""

    def test_instantiate_with_entity_id_and_tool_call(self) -> None:
        """Tool execution started event can be created."""
        entity_id = EntityId(1)
        tool_call = ToolCall(id="tc1", name="test_tool", arguments={"arg": "value"})

        event = ToolExecutionStartedEvent(entity_id=entity_id, tool_call=tool_call)

        assert event.entity_id == entity_id
        assert event.tool_call == tool_call
        assert event.tool_call.id == "tc1"

    def test_has_slots(self) -> None:
        """ToolExecutionStartedEvent uses slots."""
        entity_id = EntityId(1)
        tool_call = ToolCall(id="tc1", name="test_tool", arguments={})
        event = ToolExecutionStartedEvent(entity_id=entity_id, tool_call=tool_call)

        # Verify slots by checking __slots__ exists
        assert hasattr(event, "__slots__")


class TestToolExecutionCompletedEvent:
    """Tests for ToolExecutionCompletedEvent."""

    def test_instantiate_with_success_true(self) -> None:
        """Tool execution completed event with success=true."""
        entity_id = EntityId(2)
        event = ToolExecutionCompletedEvent(
            entity_id=entity_id,
            tool_call_id="tc1",
            result="Execution successful",
            success=True,
        )

        assert event.entity_id == entity_id
        assert event.tool_call_id == "tc1"
        assert event.result == "Execution successful"
        assert event.success is True

    def test_instantiate_with_success_false(self) -> None:
        """Tool execution completed event with success=false."""
        entity_id = EntityId(3)
        event = ToolExecutionCompletedEvent(
            entity_id=entity_id,
            tool_call_id="tc2",
            result="Error: Unknown tool",
            success=False,
        )

        assert event.success is False
        assert "Error" in event.result


class TestSkillInstalledEvent:
    """Tests for SkillInstalledEvent."""

    def test_instantiate_with_skill_name_and_tools(self) -> None:
        """Skill installed event can be created."""
        entity_id = EntityId(4)
        skill_name = "builtin_tools"
        tool_names = ["read_file", "write_file", "edit_file"]

        event = SkillInstalledEvent(
            entity_id=entity_id, skill_name=skill_name, tool_names=tool_names
        )

        assert event.entity_id == entity_id
        assert event.skill_name == skill_name
        assert event.tool_names == tool_names
        assert len(event.tool_names) == 3

    def test_instantiate_with_empty_tool_names(self) -> None:
        """Skill installed event with no tools."""
        entity_id = EntityId(5)
        event = SkillInstalledEvent(
            entity_id=entity_id, skill_name="empty_skill", tool_names=[]
        )

        assert event.tool_names == []


class TestSkillUninstalledEvent:
    """Tests for SkillUninstalledEvent."""

    def test_instantiate_with_skill_name(self) -> None:
        """Skill uninstalled event can be created."""
        entity_id = EntityId(6)
        skill_name = "builtin_tools"

        event = SkillUninstalledEvent(entity_id=entity_id, skill_name=skill_name)

        assert event.entity_id == entity_id
        assert event.skill_name == skill_name


class TestSkillDiscoveryEvent:
    """Tests for SkillDiscoveryEvent."""

    def test_instantiate_with_skills_found(self) -> None:
        """Skill discovery event with successful discovery."""
        source = "builtin"
        skills_found = ["file_ops", "bash"]
        errors = []

        event = SkillDiscoveryEvent(
            source=source, skills_found=skills_found, errors=errors
        )

        assert event.source == source
        assert event.skills_found == skills_found
        assert event.errors == []

    def test_instantiate_with_discovery_errors(self) -> None:
        """Skill discovery event with errors."""
        source = "remote_provider"
        event = SkillDiscoveryEvent(
            source=source,
            skills_found=["skill_a"],
            errors=["Failed to load skill_b: timeout"],
        )

        assert len(event.errors) == 1
        assert "timeout" in event.errors[0]


class TestMCPConnectedEvent:
    """Tests for MCPConnectedEvent."""

    def test_instantiate_with_server_name(self) -> None:
        """MCP connected event can be created."""
        server_name = "filesystem_server"

        event = MCPConnectedEvent(server_name=server_name)

        assert event.server_name == server_name


class TestMCPDisconnectedEvent:
    """Tests for MCPDisconnectedEvent."""

    def test_instantiate_with_server_name(self) -> None:
        """MCP disconnected event can be created."""
        server_name = "filesystem_server"

        event = MCPDisconnectedEvent(server_name=server_name)

        assert event.server_name == server_name


class TestMCPToolCallEvent:
    """Tests for MCPToolCallEvent."""

    def test_instantiate_with_success_true(self) -> None:
        """MCP tool call event with success=true."""
        server_name = "filesystem_server"
        tool_name = "read_file"

        event = MCPToolCallEvent(
            server_name=server_name, tool_name=tool_name, success=True
        )

        assert event.server_name == server_name
        assert event.tool_name == tool_name
        assert event.success is True

    def test_instantiate_with_success_false(self) -> None:
        """MCP tool call event with success=false."""
        event = MCPToolCallEvent(
            server_name="fs", tool_name="write_file", success=False
        )

        assert event.success is False


class TestEventTypesInWorld:
    """Integration tests: events work with World."""

    async def test_publish_tool_execution_started_event(self) -> None:
        """EventBus can publish ToolExecutionStartedEvent."""
        world = World()
        entity_id = world.create_entity()
        events_received: list[ToolExecutionStartedEvent] = []

        async def handler(event: ToolExecutionStartedEvent) -> None:
            events_received.append(event)

        world.event_bus.subscribe(ToolExecutionStartedEvent, handler)

        tool_call = ToolCall(id="tc1", name="test", arguments={})
        event = ToolExecutionStartedEvent(entity_id=entity_id, tool_call=tool_call)
        await world.event_bus.publish(event)

        assert len(events_received) == 1
        assert events_received[0].entity_id == entity_id

    async def test_publish_tool_execution_completed_event(self) -> None:
        """EventBus can publish ToolExecutionCompletedEvent."""
        world = World()
        entity_id = world.create_entity()
        events_received: list[ToolExecutionCompletedEvent] = []

        async def handler(event: ToolExecutionCompletedEvent) -> None:
            events_received.append(event)

        world.event_bus.subscribe(ToolExecutionCompletedEvent, handler)

        event = ToolExecutionCompletedEvent(
            entity_id=entity_id, tool_call_id="tc1", result="ok", success=True
        )
        await world.event_bus.publish(event)

        assert len(events_received) == 1
        assert events_received[0].success is True
