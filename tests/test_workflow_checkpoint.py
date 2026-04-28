"""Tests for workflow checkpoint/resume round-trip and exact-once semantics."""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pytest

from ecs_agent.components import (
    RunnerStateComponent,
    TerminalComponent,
    WorkflowBindingComponent,
    WorkflowDefinitionComponent,
    WorkflowGateSnapshotComponent,
    WorkflowLastTransitionComponent,
    WorkflowRuntimeComponent,
)
from ecs_agent.core import Runner, World
from ecs_agent.core.entity import EntityId
from ecs_agent.serialization import WorldSerializer
from ecs_agent.systems.workflow_state import WorkflowStateSystem
from ecs_agent.workflows import install_workflow, workflow
from ecs_agent.workflows.contracts import PromptProfileSpec, has


@dataclass(slots=True)
class ReadyMarker:
    pass


@dataclass(slots=True)
class DoneMarker:
    pass


def _make_two_state_spec() -> object:
    return workflow(
        "checkpoint-test",
        initial="PENDING",
        profiles={
            "agent": {
                "p": PromptProfileSpec(profile_id="p", prompt="Checkpoint agent."),
            },
        },
        states={
            "PENDING": {
                "bind": {"agent": "p"},
                "go": {"ACTIVE": has(ReadyMarker)},
            },
            "ACTIVE": {
                "bind": {"agent": "p"},
                "go": {"DONE": has(DoneMarker)},
            },
            "DONE": {
                "bind": {"agent": "p"},
                "go": {},
            },
        },
    )


def _install_workflow_world() -> tuple[World, EntityId]:
    spec = _make_two_state_spec()
    world = World()
    eid = world.create_entity()
    install_workflow(world, eid, spec, agent_key="agent")  # type: ignore[arg-type]
    return world, eid


async def test_checkpoint_round_trip_preserves_current_state() -> None:
    world, eid = _install_workflow_world()

    world.add_component(eid, ReadyMarker())
    await WorkflowStateSystem().process(world)

    runtime = world.get_component(eid, WorkflowRuntimeComponent)
    assert runtime is not None
    assert runtime.current_state_id == "ACTIVE"
    assert len(runtime.transition_history) == 1

    data = WorldSerializer.to_dict(world)
    restored = WorldSerializer.from_dict(data, providers={}, tool_handlers={})

    restored_runtime = restored.get_component(eid, WorkflowRuntimeComponent)
    assert restored_runtime is not None
    assert restored_runtime.current_state_id == "ACTIVE"
    assert restored_runtime.transition_history == ["PENDING_to_ACTIVE"]


async def test_checkpoint_skips_workflow_definition_component() -> None:
    world, eid = _install_workflow_world()

    data = WorldSerializer.to_dict(world)
    entity_data = data["entities"][str(int(eid))]

    assert WorkflowDefinitionComponent.__name__ not in entity_data
    assert WorkflowRuntimeComponent.__name__ in entity_data
    assert WorkflowBindingComponent.__name__ in entity_data


async def test_checkpoint_round_trip_preserves_binding() -> None:
    world, eid = _install_workflow_world()

    data = WorldSerializer.to_dict(world)
    restored = WorldSerializer.from_dict(data, providers={}, tool_handlers={})

    binding = restored.get_component(eid, WorkflowBindingComponent)
    assert binding is not None
    assert binding.agent_key == "agent"


async def test_checkpoint_round_trip_preserves_last_transition() -> None:
    world, eid = _install_workflow_world()

    world.add_component(eid, ReadyMarker())
    await WorkflowStateSystem().process(world)

    last = world.get_component(eid, WorkflowLastTransitionComponent)
    assert last is not None
    assert last.from_state_id == "PENDING"
    assert last.to_state_id == "ACTIVE"

    data = WorldSerializer.to_dict(world)
    restored = WorldSerializer.from_dict(data, providers={}, tool_handlers={})

    restored_last = restored.get_component(eid, WorkflowLastTransitionComponent)
    assert restored_last is not None
    assert restored_last.from_state_id == "PENDING"
    assert restored_last.to_state_id == "ACTIVE"
    assert restored_last.transition_id == "PENDING_to_ACTIVE"


async def test_checkpoint_resume_no_duplicate_transition() -> None:
    world, eid = _install_workflow_world()

    world.add_component(eid, ReadyMarker())
    await WorkflowStateSystem().process(world)

    data = WorldSerializer.to_dict(world)
    restored = WorldSerializer.from_dict(data, providers={}, tool_handlers={})

    restored_runtime = restored.get_component(eid, WorkflowRuntimeComponent)
    assert restored_runtime is not None
    assert restored_runtime.current_state_id == "ACTIVE"
    history_before = list(restored_runtime.transition_history)

    await WorkflowStateSystem().process(restored)

    restored_runtime2 = restored.get_component(eid, WorkflowRuntimeComponent)
    assert restored_runtime2 is not None
    assert restored_runtime2.current_state_id == "ACTIVE", (
        "Without DoneMarker, no transition should fire on resume"
    )
    assert restored_runtime2.transition_history == history_before, (
        "Resume must not duplicate already-committed transitions"
    )


async def test_checkpoint_resume_with_already_satisfied_gate() -> None:
    spec = _make_two_state_spec()
    world, eid = _install_workflow_world()

    world.add_component(eid, ReadyMarker())
    await WorkflowStateSystem().process(world)

    runtime = world.get_component(eid, WorkflowRuntimeComponent)
    assert runtime is not None
    assert runtime.current_state_id == "ACTIVE"

    data = WorldSerializer.to_dict(world)
    restored = WorldSerializer.from_dict(data, providers={}, tool_handlers={})

    from ecs_agent.workflows.compiler import compile_workflow
    from ecs_agent.components import WorkflowDefinitionComponent

    compiled = compile_workflow(spec)  # type: ignore[arg-type]
    restored.add_component(eid, WorkflowDefinitionComponent(compiled=compiled))

    restored.add_component(eid, DoneMarker())

    await WorkflowStateSystem().process(restored)

    restored_runtime = restored.get_component(eid, WorkflowRuntimeComponent)
    assert restored_runtime is not None
    assert restored_runtime.current_state_id == "DONE"
    assert len(restored_runtime.transition_history) == 2


async def test_save_checkpoint_removes_terminal_component(tmp_path: Path) -> None:
    world, eid = _install_workflow_world()
    world.add_component(eid, TerminalComponent(reason="reasoning_complete"))
    world.add_component(eid, RunnerStateComponent(current_tick=5))

    checkpoint_path = tmp_path / "test.json"
    runner = Runner()
    runner.save_checkpoint(world, checkpoint_path)

    data = json.loads(checkpoint_path.read_text())
    entity_data = data["entities"][str(int(eid))]
    assert TerminalComponent.__name__ not in entity_data


async def test_save_and_load_checkpoint_roundtrip(tmp_path: Path) -> None:
    world, eid = _install_workflow_world()
    world.add_component(eid, ReadyMarker())
    world.add_component(eid, RunnerStateComponent(current_tick=3))

    await WorkflowStateSystem().process(world)
    runtime = world.get_component(eid, WorkflowRuntimeComponent)
    assert runtime is not None
    assert runtime.current_state_id == "ACTIVE"

    checkpoint_path = tmp_path / "ckpt.json"
    runner = Runner()
    runner.save_checkpoint(world, checkpoint_path)

    restored, tick = Runner.load_checkpoint(
        checkpoint_path, providers={}, tool_handlers={}
    )
    assert tick == 3

    restored_runtime = restored.get_component(eid, WorkflowRuntimeComponent)
    assert restored_runtime is not None
    assert restored_runtime.current_state_id == "ACTIVE"
    assert "PENDING_to_ACTIVE" in restored_runtime.transition_history

    restored_binding = restored.get_component(eid, WorkflowBindingComponent)
    assert restored_binding is not None
    assert restored_binding.agent_key == "agent"


async def test_checkpoint_shared_profile_state_transition_no_churn() -> None:
    world, eid = _install_workflow_world()
    world.add_component(eid, ReadyMarker())
    await WorkflowStateSystem().process(world)

    data = WorldSerializer.to_dict(world)
    restored = WorldSerializer.from_dict(data, providers={}, tool_handlers={})

    restored_runtime = restored.get_component(eid, WorkflowRuntimeComponent)
    assert restored_runtime is not None
    assert restored_runtime.current_state_id == "ACTIVE"

    history_len_after_restore = len(restored_runtime.transition_history)
    await WorkflowStateSystem().process(restored)
    assert len(restored_runtime.transition_history) == history_len_after_restore


async def test_transition_history_exact_once_across_multiple_ticks() -> None:
    world, eid = _install_workflow_world()

    world.add_component(eid, ReadyMarker())

    for _ in range(3):
        await WorkflowStateSystem().process(world)

    runtime = world.get_component(eid, WorkflowRuntimeComponent)
    assert runtime is not None
    assert runtime.current_state_id == "ACTIVE"
    assert runtime.transition_history.count("PENDING_to_ACTIVE") == 1, (
        "Transition must be committed exactly once even when gate stays satisfied"
    )
