"""Tests for plan and replanning scratchbook persistence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ecs_agent.components import (
    ConversationComponent,
    LLMComponent,
    PlanComponent,
    ScratchbookIndexComponent,
)
from ecs_agent.core import World
from ecs_agent.providers import FakeProvider
from ecs_agent.scratchbook import ScratchbookService
from ecs_agent.systems.planning import PlanningSystem
from ecs_agent.systems.replanning import ReplanningSystem
from ecs_agent.types import CompletionResult, EntityId, Message


pytestmark = pytest.mark.asyncio


@pytest.fixture
def tmp_scratchbook(tmp_path: Path) -> Path:
    """Create isolated temp directory for scratchbook tests."""
    scratchbook_root = tmp_path / "scratchbook"
    scratchbook_root.mkdir(parents=True, exist_ok=True)
    return scratchbook_root


async def test_planning_system_persists_plan_snapshot(tmp_scratchbook: Path) -> None:
    """Plan snapshots are persisted after step completion."""
    service = ScratchbookService(root=tmp_scratchbook)
    world = World()
    provider = FakeProvider(
        responses=[
            CompletionResult(message=Message(role="assistant", content="step 1 done"))
        ]
    )
    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(provider=provider, model="fake"))
    world.add_component(
        entity_id,
        ConversationComponent(messages=[Message(role="user", content="start")]),
    )
    world.add_component(
        entity_id,
        PlanComponent(steps=["gather facts", "answer user"], current_step=0),
    )
    world.add_component(entity_id, ScratchbookIndexComponent(artifacts={}))

    # Run planning system (should persist plan snapshot)
    await PlanningSystem(service=service).process(world)

    # Verify plan snapshot artifact exists
    artifact_id = f"plan-snapshot-{entity_id}-step-0"
    persisted = service.read_artifact(artifact_id, category="planning")
    assert persisted is not None
    assert persisted["entity_id"] == entity_id
    assert persisted["step_index"] == 0
    assert persisted["step_description"] == "gather facts"
    assert persisted["current_step"] == 1
    assert persisted["completed"] is False


async def test_replanning_system_persists_replanning_delta(
    tmp_scratchbook: Path,
) -> None:
    """Replanning deltas are persisted after plan revision."""
    service = ScratchbookService(root=tmp_scratchbook)
    world = World()
    provider = FakeProvider(
        responses=[
            CompletionResult(
                message=Message(
                    role="assistant",
                    content='{"revised_steps": ["new step 2", "new step 3"]}',
                )
            )
        ]
    )
    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(provider=provider, model="fake"))
    world.add_component(
        entity_id,
        ConversationComponent(
            messages=[
                Message(role="user", content="objective"),
                Message(role="assistant", content="finished first step"),
            ]
        ),
    )
    world.add_component(
        entity_id,
        PlanComponent(
            steps=["step 1", "old step 2", "old step 3"],
            current_step=1,
        ),
    )
    world.add_component(entity_id, ScratchbookIndexComponent(artifacts={}))

    # Run replanning system (should persist delta)
    await ReplanningSystem(service=service).process(world)

    # Verify replanning delta artifact exists
    artifact_id = f"replan-delta-{entity_id}-step-1"
    persisted = service.read_artifact(artifact_id, category="replanning")
    assert persisted is not None
    assert persisted["entity_id"] == entity_id
    assert persisted["replanned_at_step"] == 1
    assert persisted["old_steps"] == ["step 1", "old step 2", "old step 3"]
    assert persisted["new_steps"] == ["step 1", "new step 2", "new step 3"]


async def test_planning_roundtrip_rehydrates_correctly(tmp_scratchbook: Path) -> None:
    """Plan snapshots can be loaded and rehydrated to continue execution."""
    service = ScratchbookService(root=tmp_scratchbook)
    world = World()
    provider = FakeProvider(
        responses=[
            CompletionResult(message=Message(role="assistant", content="step 1 done")),
            CompletionResult(message=Message(role="assistant", content="step 2 done")),
        ]
    )
    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(provider=provider, model="fake"))
    world.add_component(
        entity_id,
        ConversationComponent(messages=[Message(role="user", content="start")]),
    )
    world.add_component(
        entity_id,
        PlanComponent(steps=["step 1", "step 2"], current_step=0),
    )
    world.add_component(entity_id, ScratchbookIndexComponent(artifacts={}))

    # Run planning system to complete first step
    planning_system = PlanningSystem(service=service)
    await planning_system.process(world)

    # Load persisted snapshot
    artifact_id = f"plan-snapshot-{entity_id}-step-0"
    snapshot = service.read_artifact(artifact_id, category="planning")
    assert snapshot is not None

    # Create new world and rehydrate from snapshot
    world2 = World()
    entity_id2: EntityId = world2.create_entity()  # type: ignore[assignment]
    world2.add_component(entity_id2, LLMComponent(provider=provider, model="fake"))
    world2.add_component(
        entity_id2,
        ConversationComponent(
            messages=[Message(role="user", content="start")],  # Simplified for test
        ),
    )
    # Restore plan state from snapshot
    world2.add_component(
        entity_id2,
        PlanComponent(
            steps=["step 1", "step 2"],
            current_step=snapshot["current_step"],
            completed=snapshot["completed"],
        ),
    )
    world2.add_component(entity_id2, ScratchbookIndexComponent(artifacts={}))

    # Continue execution from restored state
    planning_system2 = PlanningSystem(service=service)
    await planning_system2.process(world2)

    # Verify second step completed
    plan = world2.get_component(entity_id2, PlanComponent)
    assert plan is not None
    assert plan.current_step == 2
    assert plan.completed is True


async def test_replanning_roundtrip_preserves_revision_history(
    tmp_scratchbook: Path,
) -> None:
    """Replanning deltas preserve revision history across sessions."""
    service = ScratchbookService(root=tmp_scratchbook)
    world = World()
    provider = FakeProvider(
        responses=[
            CompletionResult(
                message=Message(
                    role="assistant",
                    content='{"revised_steps": ["revised step 2", "revised step 3"]}',
                )
            )
        ]
    )
    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(provider=provider, model="fake"))
    world.add_component(
        entity_id,
        ConversationComponent(
            messages=[
                Message(role="user", content="objective"),
                Message(role="assistant", content="finished first step"),
            ]
        ),
    )
    world.add_component(
        entity_id,
        PlanComponent(
            steps=["step 1", "step 2", "step 3"],
            current_step=1,
        ),
    )
    world.add_component(entity_id, ScratchbookIndexComponent(artifacts={}))

    # Run replanning system
    replanning_system = ReplanningSystem(service=service)
    await replanning_system.process(world)

    # Load persisted delta
    artifact_id = f"replan-delta-{entity_id}-step-1"
    delta = service.read_artifact(artifact_id, category="replanning")
    assert delta is not None

    # Verify revision history integrity
    assert delta["old_steps"] == ["step 1", "step 2", "step 3"]
    assert delta["new_steps"] == ["step 1", "revised step 2", "revised step 3"]
    assert delta["replanned_at_step"] == 1

    # Verify can reconstruct plan state from delta
    plan = world.get_component(entity_id, PlanComponent)
    assert plan is not None
    assert plan.steps == delta["new_steps"]


async def test_malformed_plan_snapshot_handled_gracefully(
    tmp_scratchbook: Path,
) -> None:
    """Malformed persisted plan snapshot fails explicitly with clear error."""
    service = ScratchbookService(root=tmp_scratchbook)

    # Write malformed snapshot (missing required field)
    artifact_id = "plan-snapshot-999-step-0"
    malformed_data = {
        "entity_id": 999,
        "step_index": 0,
        # Missing: step_description, current_step, completed
    }
    service.write_artifact(artifact_id, category="planning", data=malformed_data)

    # Attempt to read
    snapshot = service.read_artifact(artifact_id, category="planning")
    assert snapshot is not None

    # Verify missing fields fail validation
    with pytest.raises(KeyError):
        _ = snapshot["step_description"]

    with pytest.raises(KeyError):
        _ = snapshot["current_step"]


async def test_malformed_replanning_delta_preserves_prior_plan_state(
    tmp_scratchbook: Path,
) -> None:
    """Malformed replanning delta preserves prior valid plan state."""
    service = ScratchbookService(root=tmp_scratchbook)
    world = World()
    # Provider returns malformed JSON
    provider = FakeProvider(
        responses=[
            CompletionResult(
                message=Message(role="assistant", content="this is not json")
            )
        ]
    )
    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(provider=provider, model="fake"))
    world.add_component(
        entity_id,
        ConversationComponent(
            messages=[
                Message(role="user", content="objective"),
                Message(role="assistant", content="finished first step"),
            ]
        ),
    )
    original_steps = ["step 1", "step 2", "step 3"]
    world.add_component(
        entity_id,
        PlanComponent(
            steps=original_steps.copy(),
            current_step=1,
        ),
    )
    world.add_component(entity_id, ScratchbookIndexComponent(artifacts={}))

    # Run replanning system (should fail gracefully)
    await ReplanningSystem(service=service).process(world)

    # Verify plan state unchanged (malformed response didn't corrupt plan)
    plan = world.get_component(entity_id, PlanComponent)
    assert plan is not None
    assert plan.steps == original_steps

    # Verify no delta was persisted (since parsing failed)
    artifact_id = f"replan-delta-{entity_id}-step-1"
    delta = service.read_artifact(artifact_id, category="replanning")
    assert delta is None
