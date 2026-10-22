"""Live scenarios using explicit model profiles and shared resource ownership."""

from __future__ import annotations

from pathlib import Path

import pytest

from ecs_agent.components import (
    ConversationComponent,
    LLMComponent,
    PendingToolCallsComponent,
    PlanComponent,
    ToolRegistryComponent,
    UserPromptConfigComponent,
)
from ecs_agent.core import World
from ecs_agent.prompts.contracts import TriggerSpec
from ecs_agent.providers.config import ApiFormat
from ecs_agent.scratchbook import ArtifactRegistry
from ecs_agent.systems.planning import PlanningSystem
from ecs_agent.systems.tool_execution import ToolExecutionSystem
from ecs_agent.systems.user_prompt_normalization_system import (
    UserPromptNormalizationSystem,
)
from ecs_agent.types import EntityId, Message, ToolCall, ToolSchema

from tests.support.models import LiveModelFactory


_ENDPOINT_PARAMS = [
    pytest.param(ApiFormat.OPENAI_CHAT_COMPLETIONS, id="chat-completions"),
    pytest.param(ApiFormat.OPENAI_RESPONSES, id="responses"),
]


@pytest.mark.live(
    formats=["openai_chat_completions", "openai_responses"],
    capabilities=["text", "tools"],
)
@pytest.mark.tools
@pytest.mark.asyncio
@pytest.mark.parametrize("api_format", _ENDPOINT_PARAMS)
async def test_live_planning_writes_canonical_plan_md(
    tmp_path: Path, api_format: ApiFormat, *, live_model_factory: LiveModelFactory
) -> None:
    model = live_model_factory.openai()

    registry = ArtifactRegistry(root=tmp_path)
    world = World()
    entity = world.create_entity()

    world.add_component(entity, LLMComponent(model=model))
    world.add_component(
        entity,
        ConversationComponent(
            messages=[
                Message(role="user", content="Live plan canonical markdown example")
            ]
        ),
    )
    world.add_component(entity, PlanComponent(steps=["Write a concise status update"]))

    await PlanningSystem(registry=registry).process(world)

    plan_slug = registry.normalize_plan_name("Live plan canonical markdown example")
    plan_file = tmp_path / "scratchbook" / plan_slug / "plan.md"
    assert plan_file.exists()
    content = plan_file.read_text(encoding="utf-8")
    assert "# Plan:" in content
    assert "## Steps" in content
    assert "Write a concise status update" in content
    assert not content.lstrip().startswith("{")


@pytest.mark.live(
    formats=["openai_chat_completions", "openai_responses"],
    capabilities=["text", "tools"],
)
@pytest.mark.tools
@pytest.mark.asyncio
@pytest.mark.parametrize("api_format", _ENDPOINT_PARAMS)
async def test_live_no_legacy_category_paths_written(
    tmp_path: Path, api_format: ApiFormat, *, live_model_factory: LiveModelFactory
) -> None:
    model = live_model_factory.openai()

    registry = ArtifactRegistry(root=tmp_path)
    world = World()
    entity = world.create_entity()

    async def plan_handler(
        world: World,
        entity_id: EntityId,
        user_text: str,
    ) -> str | None:
        _ = world
        _ = entity_id
        return user_text

    async def live_tool(payload: str) -> str:
        return f"ok:{payload}"

    world.add_component(entity, LLMComponent(model=model))
    world.add_component(
        entity,
        ConversationComponent(
            messages=[Message(role="user", content="@plan Live canonical flow")]
        ),
    )
    world.add_component(entity, PlanComponent(steps=["Do one planning step"]))
    world.add_component(
        entity,
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
    await PlanningSystem(registry=registry).process(world)

    world.add_component(
        entity,
        ToolRegistryComponent(
            tools={
                "live_tool": ToolSchema(
                    name="live_tool",
                    description="Run live tool",
                    parameters={
                        "type": "object",
                        "properties": {
                            "payload": {"type": "string"},
                        },
                        "required": ["payload"],
                        "additionalProperties": False,
                    },
                )
            },
            handlers={"live_tool": live_tool},
        ),
    )

    world.add_component(
        entity,
        PendingToolCallsComponent(
            tool_calls=[
                ToolCall(
                    id="legacy-guard-1",
                    name="live_tool",
                    arguments={"payload": "guard"},
                )
            ]
        ),
    )
    await ToolExecutionSystem(registry=registry).process(world)

    plan_slug = registry.normalize_plan_name("@plan Live canonical flow")
    assert (tmp_path / "scratchbook" / plan_slug / "plan.md").exists()
    assert (tmp_path / "scratchbook" / plan_slug / "executes" / "boulder.json").exists()

    tool_files = list((tmp_path / "scratchbook" / "records" / "tool").glob("tool_*"))
    assert tool_files

    assert not (tmp_path / "scratchbook" / "tool_results").exists()
    assert not (tmp_path / "scratchbook" / "planning").exists()
    assert not (tmp_path / "scratchbook" / "replanning").exists()
