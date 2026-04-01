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
    ToolRegistryComponent,
    UserPromptConfigComponent,
)
from ecs_agent.core import World
from ecs_agent.prompts.contracts import TriggerSpec
from ecs_agent.providers import FakeProvider
from ecs_agent.scratchbook import ArtifactRegistry
from ecs_agent.systems.planning import PlanningSystem
from ecs_agent.systems.replanning import ReplanningSystem
from ecs_agent.systems.tool_execution import ToolExecutionSystem
from ecs_agent.systems.user_prompt_normalization_system import (
    UserPromptNormalizationSystem,
)
from ecs_agent.types import CompletionResult, EntityId, Message, ToolCall, ToolSchema


pytestmark = pytest.mark.asyncio


@pytest.fixture
def registry(tmp_path: Path) -> ArtifactRegistry:
    return ArtifactRegistry(root=tmp_path)


async def test_plan_persists_to_canonical_plan_md_path(
    tmp_path: Path,
    registry: ArtifactRegistry,
) -> None:
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

    await PlanningSystem(registry=registry).process(world)

    expected_path = registry.plan_path(plan_name="start")
    plan_file = tmp_path / expected_path
    assert plan_file.exists()
    content = plan_file.read_text(encoding="utf-8")
    assert "# Plan:" in content
    assert "## Steps" in content
    assert "[DONE]" in content
    assert "[CURRENT]" in content


async def test_replanning_updates_same_plan_md_without_legacy_category_writes(
    tmp_path: Path,
    registry: ArtifactRegistry,
) -> None:
    world = World()
    provider = FakeProvider(
        responses=[
            CompletionResult(message=Message(role="assistant", content="step 1 done")),
            CompletionResult(
                message=Message(
                    role="assistant",
                    content='{"revised_steps": ["new step 2", "new step 3", "new step 4"]}',
                )
            ),
        ]
    )
    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(provider=provider, model="fake"))
    world.add_component(
        entity_id,
        ConversationComponent(
            messages=[Message(role="user", content="Objective / Q2")]
        ),
    )
    world.add_component(
        entity_id,
        PlanComponent(steps=["step 1", "old step 2", "old step 3"], current_step=0),
    )
    world.add_component(entity_id, ScratchbookIndexComponent(artifacts={}))

    await PlanningSystem(registry=registry).process(world)
    canonical_path = registry.plan_path(plan_name="Objective / Q2")
    plan_file = tmp_path / canonical_path
    assert plan_file.exists()
    before = plan_file.read_text(encoding="utf-8")

    await ReplanningSystem(registry=registry).process(world)

    after = plan_file.read_text(encoding="utf-8")
    assert plan_file == (tmp_path / registry.plan_path(plan_name="Objective / Q2"))
    assert after != before
    assert "new step 2" in after
    assert "new step 4" in after
    assert not (tmp_path / "planning").exists()
    assert not (tmp_path / "replanning").exists()


async def test_trigger_spec_creates_initial_boulder_file(
    tmp_path: Path,
    registry: ArtifactRegistry,
) -> None:
    async def plan_handler(
        world: World,
        entity_id: EntityId,
        user_text: str,
    ) -> str | None:
        return f"Planning started for: {user_text}"

    world = World()
    entity_id = world.create_entity()
    world.add_component(
        entity_id,
        ConversationComponent(
            messages=[Message(role="user", content="@plan Launch Roadmap")]
        ),
    )
    world.add_component(
        entity_id,
        UserPromptConfigComponent(
            triggers=[
                TriggerSpec(
                    pattern="@plan",
                    match_mode="keyword",
                    action="script",
                    content="plan_handler",
                )
            ],
            script_handlers={"plan_handler": plan_handler},
        ),
    )

    await UserPromptNormalizationSystem(registry=registry).process(world)

    expected_record_path = registry.boulder_path(plan_name="@plan Launch Roadmap")
    boulder_file = tmp_path / expected_record_path
    assert boulder_file.exists()


async def test_plan_and_tool_transitions_update_same_boulder_file(
    tmp_path: Path,
    registry: ArtifactRegistry,
) -> None:
    world = World()
    provider = FakeProvider(
        responses=[
            CompletionResult(
                message=Message(
                    role="assistant",
                    content="Calling tool",
                    tool_calls=[
                        ToolCall(
                            id="tool-1",
                            name="lookup",
                            arguments={"topic": "status"},
                        )
                    ],
                )
            )
        ]
    )

    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(provider=provider, model="fake"))
    world.add_component(
        entity_id,
        ConversationComponent(messages=[Message(role="user", content="Roadmap Q4")]),
    )
    world.add_component(entity_id, PlanComponent(steps=["run tool"], current_step=0))
    world.add_component(entity_id, ScratchbookIndexComponent(artifacts={}))

    async def lookup(topic: str) -> str:
        return f"ok:{topic}"

    world.add_component(
        entity_id,
        ToolRegistryComponent(
            tools={
                "lookup": ToolSchema(
                    name="lookup",
                    description="Lookup status",
                    parameters={"type": "object"},
                )
            },
            handlers={"lookup": lookup},
        ),
    )

    boulder_path = registry.create_boulder(
        plan_name="Roadmap Q4",
        initial_data={"trigger_pattern": "@plan"},
    )
    boulder_file = tmp_path / boulder_path
    before = json.loads(boulder_file.read_text(encoding="utf-8"))

    await PlanningSystem(registry=registry).process(world)
    await ToolExecutionSystem(registry=registry).process(world)

    after = json.loads(boulder_file.read_text(encoding="utf-8"))
    assert str(boulder_file).endswith("scratchbook/roadmap-q4/executes/boulder.json")
    assert after["schema_version"] == before["schema_version"]
    assert after["active_plan"] == before["active_plan"]
    assert after["started_at"] == before["started_at"]
    assert after["status"] == "running"
    assert after["current_step"] == 1
    assert after["last_step_description"] == "run tool"
    assert after["last_tool_call_id"] == "tool-1"
    assert after["last_tool_record_path"].startswith("scratchbook/records/tool/tool_")
    assert isinstance(after["last_updated_at"], str)


async def test_replanning_updates_existing_boulder_without_recreation(
    tmp_path: Path,
    registry: ArtifactRegistry,
) -> None:
    world = World()
    provider = FakeProvider(
        responses=[
            CompletionResult(message=Message(role="assistant", content="step 1 done")),
            CompletionResult(
                message=Message(
                    role="assistant",
                    content='{"revised_steps": ["new step 2", "new step 3"]}',
                )
            ),
        ]
    )
    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(provider=provider, model="fake"))
    world.add_component(
        entity_id,
        ConversationComponent(messages=[Message(role="user", content="Replan Goal")]),
    )
    world.add_component(
        entity_id,
        PlanComponent(steps=["step 1", "old step 2", "old step 3"], current_step=0),
    )

    boulder_path = registry.create_boulder(
        plan_name="Replan Goal",
        initial_data={"trigger_pattern": "@plan"},
    )
    boulder_file = tmp_path / boulder_path
    before = json.loads(boulder_file.read_text(encoding="utf-8"))

    await PlanningSystem(registry=registry).process(world)
    await ReplanningSystem(registry=registry).process(world)

    after = json.loads(boulder_file.read_text(encoding="utf-8"))
    assert after["active_plan"] == before["active_plan"]
    assert after["started_at"] == before["started_at"]
    assert after["schema_version"] == before["schema_version"]
    assert after["plan_name"] == before["plan_name"]
    assert after["status"] == "replanned"
    assert after["revised_steps_count"] == 3
    assert after["last_step_description"] == "step 1"
    assert isinstance(after["last_updated_at"], str)
