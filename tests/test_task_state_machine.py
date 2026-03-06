"""Tests for TaskComponent model field validation and state machine transitions.

This module tests the task schema defined in Task 7:
- Required attributes enforcement (description, expected_output, assigned_agent, tools, context_dependencies)
- Runtime status metadata (task_id, status, priority)
- Optional fields (output_schema, max_retries)
- Field defaults and validation
"""

import pytest

from ecs_agent.components import TaskComponent
from ecs_agent.types import EntityId, TaskStatus


class TestTaskComponentModelFields:
    """Test TaskComponent model field validation."""

    def test_required_fields_present(self):
        """Test TaskComponent enforces all required fields."""
        # All required fields must be provided
        comp = TaskComponent(
            description="Test task",
            expected_output="Expected result",
            assigned_agent=None,
            tools=["tool1"],
            context_dependencies=["dep1"],
            task_id="task-001",
            status=TaskStatus.PENDING,
        )
        assert comp.description == "Test task"
        assert comp.expected_output == "Expected result"
        assert comp.assigned_agent is None
        assert comp.tools == ["tool1"]
        assert comp.context_dependencies == ["dep1"]
        assert comp.task_id == "task-001"
        assert comp.status == TaskStatus.PENDING

    def test_description_is_required(self):
        """Test that description is mandatory and cannot be omitted."""
        with pytest.raises(TypeError, match="description"):
            TaskComponent(  # type: ignore[call-arg]
                expected_output="Result",
                assigned_agent=None,
                tools=[],
                context_dependencies=[],
                task_id="t1",
                status=TaskStatus.PENDING,
            )

    def test_expected_output_is_required(self):
        """Test that expected_output is mandatory (not optional in v1)."""
        with pytest.raises(TypeError, match="expected_output"):
            TaskComponent(  # type: ignore[call-arg]
                description="Task",
                assigned_agent=None,
                tools=[],
                context_dependencies=[],
                task_id="t1",
                status=TaskStatus.PENDING,
            )

    def test_assigned_agent_is_required_field(self):
        """Test that assigned_agent is required (can be None, but field must be present)."""
        with pytest.raises(TypeError, match="assigned_agent"):
            TaskComponent(  # type: ignore[call-arg]
                description="Task",
                expected_output="Result",
                tools=[],
                context_dependencies=[],
                task_id="t1",
                status=TaskStatus.PENDING,
            )

    def test_tools_is_required(self):
        """Test that tools is mandatory."""
        with pytest.raises(TypeError, match="tools"):
            TaskComponent(  # type: ignore[call-arg]
                description="Task",
                expected_output="Result",
                assigned_agent=None,
                context_dependencies=[],
                task_id="t1",
                status=TaskStatus.PENDING,
            )

    def test_context_dependencies_is_required(self):
        """Test that context_dependencies is mandatory."""
        with pytest.raises(TypeError, match="context_dependencies"):
            TaskComponent(  # type: ignore[call-arg]
                description="Task",
                expected_output="Result",
                assigned_agent=None,
                tools=[],
                task_id="t1",
                status=TaskStatus.PENDING,
            )

    def test_task_id_is_required(self):
        """Test that task_id is mandatory for stable identification."""
        with pytest.raises(TypeError, match="task_id"):
            TaskComponent(  # type: ignore[call-arg]
                description="Task",
                expected_output="Result",
                assigned_agent=None,
                tools=[],
                context_dependencies=[],
                status=TaskStatus.PENDING,
            )

    def test_status_is_required(self):
        """Test that status is mandatory runtime metadata."""
        with pytest.raises(TypeError, match="status"):
            TaskComponent(  # type: ignore[call-arg]
                description="Task",
                expected_output="Result",
                assigned_agent=None,
                tools=[],
                context_dependencies=[],
                task_id="t1",
            )


class TestTaskComponentDefaults:
    """Test TaskComponent default values for optional fields."""

    def test_priority_defaults_to_zero(self):
        """Test priority has default value of 0."""
        comp = TaskComponent(
            description="Task",
            expected_output="Result",
            assigned_agent=None,
            tools=[],
            context_dependencies=[],
            task_id="t1",
            status=TaskStatus.PENDING,
        )
        assert comp.priority == 0

    def test_output_schema_defaults_to_none(self):
        """Test output_schema is optional and defaults to None."""
        comp = TaskComponent(
            description="Task",
            expected_output="Result",
            assigned_agent=None,
            tools=[],
            context_dependencies=[],
            task_id="t1",
            status=TaskStatus.PENDING,
        )
        assert comp.output_schema is None

    def test_max_retries_defaults_to_zero(self):
        """Test max_retries has default value of 0."""
        comp = TaskComponent(
            description="Task",
            expected_output="Result",
            assigned_agent=None,
            tools=[],
            context_dependencies=[],
            task_id="t1",
            status=TaskStatus.PENDING,
        )
        assert comp.max_retries == 0

    def test_optional_fields_can_be_set(self):
        """Test that optional fields can be explicitly set."""
        schema = {"type": "object", "properties": {"result": {"type": "string"}}}
        comp = TaskComponent(
            description="Task",
            expected_output="Result",
            assigned_agent=None,
            tools=[],
            context_dependencies=[],
            task_id="t1",
            status=TaskStatus.PENDING,
            priority=5,
            output_schema=schema,
            max_retries=3,
        )
        assert comp.priority == 5
        assert comp.output_schema == schema
        assert comp.max_retries == 3


class TestTaskComponentAssignedAgentUnion:
    """Test assigned_agent field union type (EntityId | str | None)."""

    def test_assigned_agent_none(self):
        """Test assigned_agent can be None for unassigned tasks."""
        comp = TaskComponent(
            description="Task",
            expected_output="Result",
            assigned_agent=None,
            tools=[],
            context_dependencies=[],
            task_id="t1",
            status=TaskStatus.PENDING,
        )
        assert comp.assigned_agent is None

    def test_assigned_agent_entity_id(self):
        """Test assigned_agent can be EntityId for entity references."""
        agent_id = EntityId(42)
        comp = TaskComponent(
            description="Task",
            expected_output="Result",
            assigned_agent=agent_id,
            tools=[],
            context_dependencies=[],
            task_id="t1",
            status=TaskStatus.PENDING,
        )
        assert comp.assigned_agent == agent_id
        assert isinstance(comp.assigned_agent, int)

    def test_assigned_agent_string(self):
        """Test assigned_agent can be string for subagent names."""
        comp = TaskComponent(
            description="Task",
            expected_output="Result",
            assigned_agent="subagent_researcher",
            tools=[],
            context_dependencies=[],
            task_id="t1",
            status=TaskStatus.PENDING,
        )
        assert comp.assigned_agent == "subagent_researcher"
        assert isinstance(comp.assigned_agent, str)


class TestTaskComponentOutputSchema:
    """Test output_schema field for structured DoD support."""

    def test_output_schema_none_by_default(self):
        """Test output_schema is None when not provided (text expected_output only)."""
        comp = TaskComponent(
            description="Task",
            expected_output="Plain text DoD",
            assigned_agent=None,
            tools=[],
            context_dependencies=[],
            task_id="t1",
            status=TaskStatus.PENDING,
        )
        assert comp.output_schema is None

    def test_output_schema_supports_dict(self):
        """Test output_schema can store a dict schema (Pydantic-like JSON schema)."""
        schema = {
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "status": {"type": "string", "enum": ["pass", "fail"]},
            },
            "required": ["summary", "status"],
        }
        comp = TaskComponent(
            description="Task",
            expected_output="Structured output",
            assigned_agent=None,
            tools=[],
            context_dependencies=[],
            task_id="t1",
            status=TaskStatus.PENDING,
            output_schema=schema,
        )
        assert comp.output_schema == schema
        assert "properties" in comp.output_schema
        assert comp.output_schema["properties"]["summary"]["type"] == "string"

    def test_output_schema_does_not_replace_expected_output(self):
        """Test output_schema is supplementary, expected_output is still required."""
        # Both expected_output (text) and output_schema (structured) can coexist
        schema = {"type": "object"}
        comp = TaskComponent(
            description="Task",
            expected_output="Human-readable DoD",
            assigned_agent=None,
            tools=[],
            context_dependencies=[],
            task_id="t1",
            status=TaskStatus.PENDING,
            output_schema=schema,
        )
        assert comp.expected_output == "Human-readable DoD"
        assert comp.output_schema == schema


class TestTaskComponentRuntimeMetadata:
    """Test runtime status metadata fields."""

    def test_task_id_stable_identifier(self):
        """Test task_id provides stable identification across sessions."""
        comp = TaskComponent(
            description="Task",
            expected_output="Result",
            assigned_agent=None,
            tools=[],
            context_dependencies=[],
            task_id="task-stable-001",
            status=TaskStatus.PENDING,
        )
        assert comp.task_id == "task-stable-001"

    def test_status_enum_field(self):
        """Test status uses TaskStatus enum for state machine transitions."""
        comp = TaskComponent(
            description="Task",
            expected_output="Result",
            assigned_agent=None,
            tools=[],
            context_dependencies=[],
            task_id="t1",
            status=TaskStatus.RUNNING,
        )
        assert comp.status == TaskStatus.RUNNING
        assert isinstance(comp.status, TaskStatus)

    def test_priority_integer_field(self):
        """Test priority field stores integer for task scheduling."""
        comp = TaskComponent(
            description="High priority task",
            expected_output="Result",
            assigned_agent=None,
            tools=[],
            context_dependencies=[],
            task_id="t1",
            status=TaskStatus.PENDING,
            priority=10,
        )
        assert comp.priority == 10
        assert isinstance(comp.priority, int)

    def test_max_retries_integer_field(self):
        """Test max_retries field stores integer for retry policy."""
        comp = TaskComponent(
            description="Task",
            expected_output="Result",
            assigned_agent=None,
            tools=[],
            context_dependencies=[],
            task_id="t1",
            status=TaskStatus.PENDING,
            max_retries=5,
        )
        assert comp.max_retries == 5
        assert isinstance(comp.max_retries, int)


class TestTaskComponentDataclassProperties:
    """Test TaskComponent follows dataclass conventions."""

    def test_dataclass_slots(self):
        """Test TaskComponent uses slots for memory efficiency."""
        assert hasattr(TaskComponent, "__slots__")

    def test_component_is_immutable_by_intent(self):
        """Test TaskComponent fields can be mutated (frozen=False by default)."""
        # Note: TaskComponent is not frozen, fields can be mutated
        # This is intentional for runtime status updates
        comp = TaskComponent(
            description="Task",
            expected_output="Result",
            assigned_agent=None,
            tools=[],
            context_dependencies=[],
            task_id="t1",
            status=TaskStatus.PENDING,
        )
        # Status transitions are allowed
        comp.status = TaskStatus.RUNNING
        assert comp.status == TaskStatus.RUNNING
