"""Roundtrip serialization tests for task and scratchbook components."""

from __future__ import annotations

from typing import Any

from ecs_agent.components import (
    ScratchbookIndexComponent,
    ScratchbookRefComponent,
    TaskComponent,
)
from ecs_agent.core.world import World
from ecs_agent.serialization import WorldSerializer
from ecs_agent.types import EntityId, ScratchbookRef, TaskStatus


class DummyProvider:
    async def complete(self, messages, tools=None, stream=False, response_format=None):
        _ = (messages, tools, stream, response_format)
        raise NotImplementedError


def test_task_component_roundtrip() -> None:
    """TaskComponent serializes and deserializes correctly."""
    world = World()
    entity = world.create_entity()

    # Create a TaskComponent with EntityId assigned_agent
    task = TaskComponent(
        description="Process the data",
        expected_output="Processed result",
        assigned_agent=EntityId(2),
        tools=["tool_a", "tool_b"],
        context_dependencies=["dep1", "dep2"],
        task_id="task-001",
        status=TaskStatus.RUNNING,
        priority=5,
        output_schema={"type": "object", "properties": {"result": {"type": "string"}}},
        max_retries=3,
    )
    world.add_component(entity, task)

    # Serialize and deserialize
    data = WorldSerializer.to_dict(world)
    providers = {"default": DummyProvider()}
    tool_handlers = {}
    world2 = WorldSerializer.from_dict(
        data, providers=providers, tool_handlers=tool_handlers
    )

    # Verify roundtrip
    task2 = world2.get_component(entity, TaskComponent)
    assert task2 is not None
    assert task2.description == "Process the data"
    assert task2.expected_output == "Processed result"
    assert task2.assigned_agent == EntityId(2)
    assert task2.tools == ["tool_a", "tool_b"]
    assert task2.context_dependencies == ["dep1", "dep2"]
    assert task2.task_id == "task-001"
    assert task2.status == TaskStatus.RUNNING
    assert task2.priority == 5
    assert task2.output_schema == {
        "type": "object",
        "properties": {"result": {"type": "string"}},
    }
    assert task2.max_retries == 3


def test_task_component_with_string_assigned_agent() -> None:
    """TaskComponent with string assigned_agent serializes correctly."""
    world = World()
    entity = world.create_entity()

    task = TaskComponent(
        description="Subagent task",
        expected_output="Subagent result",
        assigned_agent="subagent-name",  # String instead of EntityId
        tools=["tool_x"],
        context_dependencies=[],
        task_id="task-002",
        status=TaskStatus.PENDING,
    )
    world.add_component(entity, task)

    # Serialize and deserialize
    data = WorldSerializer.to_dict(world)
    providers = {"default": DummyProvider()}
    tool_handlers = {}
    world2 = WorldSerializer.from_dict(
        data, providers=providers, tool_handlers=tool_handlers
    )

    # Verify roundtrip with string agent
    task2 = world2.get_component(entity, TaskComponent)
    assert task2 is not None
    assert task2.assigned_agent == "subagent-name"
    assert task2.status == TaskStatus.PENDING


def test_task_component_with_none_assigned_agent() -> None:
    """TaskComponent with None assigned_agent serializes correctly."""
    world = World()
    entity = world.create_entity()

    task = TaskComponent(
        description="Unassigned task",
        expected_output="Result",
        assigned_agent=None,
        tools=[],
        context_dependencies=[],
        task_id="task-003",
        status=TaskStatus.PENDING,
    )
    world.add_component(entity, task)

    # Serialize and deserialize
    data = WorldSerializer.to_dict(world)
    providers = {"default": DummyProvider()}
    tool_handlers = {}
    world2 = WorldSerializer.from_dict(
        data, providers=providers, tool_handlers=tool_handlers
    )

    # Verify roundtrip with None agent
    task2 = world2.get_component(entity, TaskComponent)
    assert task2 is not None
    assert task2.assigned_agent is None


def test_scratchbook_ref_component_roundtrip() -> None:
    """ScratchbookRefComponent serializes and deserializes correctly."""
    world = World()
    entity = world.create_entity()

    ref = ScratchbookRefComponent(
        artifact_id="artifact-001",
        category="reasoning",
        content_hash="abc123def456",
        timestamp="2026-03-07T10:30:00Z",
    )
    world.add_component(entity, ref)

    # Serialize and deserialize
    data = WorldSerializer.to_dict(world)
    providers = {"default": DummyProvider()}
    tool_handlers = {}
    world2 = WorldSerializer.from_dict(
        data, providers=providers, tool_handlers=tool_handlers
    )

    # Verify roundtrip
    ref2 = world2.get_component(entity, ScratchbookRefComponent)
    assert ref2 is not None
    assert ref2.artifact_id == "artifact-001"
    assert ref2.category == "reasoning"
    assert ref2.content_hash == "abc123def456"
    assert ref2.timestamp == "2026-03-07T10:30:00Z"


def test_scratchbook_index_component_roundtrip() -> None:
    """ScratchbookIndexComponent with artifacts dict serializes correctly."""
    world = World()
    entity = world.create_entity()

    artifacts = {
        "art-1": ScratchbookRef(
            artifact_id="art-1",
            category="planning",
            content_hash="hash1",
            timestamp="2026-03-07T10:00:00Z",
        ),
        "art-2": ScratchbookRef(
            artifact_id="art-2",
            category="reasoning",
            content_hash="hash2",
            timestamp="2026-03-07T10:10:00Z",
        ),
    }
    index = ScratchbookIndexComponent(artifacts=artifacts)
    world.add_component(entity, index)

    # Serialize and deserialize
    data = WorldSerializer.to_dict(world)
    providers = {"default": DummyProvider()}
    tool_handlers = {}
    world2 = WorldSerializer.from_dict(
        data, providers=providers, tool_handlers=tool_handlers
    )

    # Verify roundtrip
    index2 = world2.get_component(entity, ScratchbookIndexComponent)
    assert index2 is not None
    assert len(index2.artifacts) == 2
    assert "art-1" in index2.artifacts
    assert "art-2" in index2.artifacts
    assert index2.artifacts["art-1"].artifact_id == "art-1"
    assert index2.artifacts["art-1"].category == "planning"
    assert index2.artifacts["art-2"].content_hash == "hash2"


def test_multiple_task_and_scratchbook_components() -> None:
    """Multiple entities with task and scratchbook components serialize correctly."""
    world = World()

    # Entity 1: Task + ScratchbookRef
    entity1 = world.create_entity()
    world.add_component(
        entity1,
        TaskComponent(
            description="Task 1",
            expected_output="Output 1",
            assigned_agent=EntityId(3),
            tools=["tool_1"],
            context_dependencies=[],
            task_id="task-1",
            status=TaskStatus.COMPLETED,
        ),
    )
    world.add_component(
        entity1,
        ScratchbookRefComponent(
            artifact_id="art-1",
            category="result",
            content_hash="hash1",
            timestamp="2026-03-07T10:00:00Z",
        ),
    )

    # Entity 2: Task + ScratchbookIndex
    entity2 = world.create_entity()
    world.add_component(
        entity2,
        TaskComponent(
            description="Task 2",
            expected_output="Output 2",
            assigned_agent="subagent",
            tools=["tool_2"],
            context_dependencies=["task-1"],
            task_id="task-2",
            status=TaskStatus.RUNNING,
        ),
    )
    world.add_component(
        entity2,
        ScratchbookIndexComponent(
            artifacts={
                "art-2": ScratchbookRef(
                    artifact_id="art-2",
                    category="work",
                    content_hash="hash2",
                    timestamp="2026-03-07T10:10:00Z",
                ),
            }
        ),
    )

    # Serialize and deserialize
    data = WorldSerializer.to_dict(world)
    providers = {"default": DummyProvider()}
    tool_handlers = {}
    world2 = WorldSerializer.from_dict(
        data, providers=providers, tool_handlers=tool_handlers
    )

    # Verify both entities exist and have correct components
    task1 = world2.get_component(entity1, TaskComponent)
    assert task1 is not None
    assert task1.task_id == "task-1"

    ref1 = world2.get_component(entity1, ScratchbookRefComponent)
    assert ref1 is not None
    assert ref1.artifact_id == "art-1"

    task2 = world2.get_component(entity2, TaskComponent)
    assert task2 is not None
    assert task2.task_id == "task-2"

    index2 = world2.get_component(entity2, ScratchbookIndexComponent)
    assert index2 is not None
    assert "art-2" in index2.artifacts


def test_backward_compatibility_legacy_payload_without_task_components() -> None:
    """World deserializes without error when task/scratchbook components missing."""
    # Simulate legacy payload without task/scratchbook components
    legacy_data = {
        "next_entity_id": 2,
        "entities": {"1": {"KVStoreComponent": {"store": {"key": "value"}}}},
        "_entity_registry": {},
        "_entity_tags": {},
    }

    # Should deserialize without error
    providers = {"default": DummyProvider()}
    tool_handlers = {}
    world = WorldSerializer.from_dict(
        legacy_data, providers=providers, tool_handlers=tool_handlers
    )

    # Verify legacy component still present
    assert world is not None
    entity = EntityId(1)
    from ecs_agent.components import KVStoreComponent

    kv = world.get_component(entity, KVStoreComponent)
    assert kv is not None
    assert kv.store == {"key": "value"}


def test_empty_artifacts_dict_roundtrip() -> None:
    """ScratchbookIndexComponent with empty artifacts dict serializes correctly."""
    world = World()
    entity = world.create_entity()

    index = ScratchbookIndexComponent(artifacts={})
    world.add_component(entity, index)

    # Serialize and deserialize
    data = WorldSerializer.to_dict(world)
    providers = {"default": DummyProvider()}
    tool_handlers = {}
    world2 = WorldSerializer.from_dict(
        data, providers=providers, tool_handlers=tool_handlers
    )

    # Verify empty dict is preserved
    index2 = world2.get_component(entity, ScratchbookIndexComponent)
    assert index2 is not None
    assert index2.artifacts == {}
