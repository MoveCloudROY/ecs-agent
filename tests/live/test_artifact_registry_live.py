from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from ecs_agent.components import (
    ConversationComponent,
    LLMComponent,
    PendingToolCallsComponent,
    PlanComponent,
    ToolRegistryComponent,
    ToolResultsComponent,
    UserPromptConfigComponent,
)
from ecs_agent.components.definitions import RenderedUserPromptComponent
from ecs_agent.core import World
from ecs_agent.prompts.contracts import TriggerSpec
from ecs_agent.providers import OpenAIModel
from ecs_agent.providers.config import ApiFormat, ProviderConfig
from ecs_agent.scratchbook import ArtifactRegistry
from ecs_agent.systems.planning import PlanningSystem
from ecs_agent.systems.tool_execution import ToolExecutionSystem
from ecs_agent.systems.user_prompt_normalization_system import (
    UserPromptNormalizationSystem,
)
from ecs_agent.types import EntityId, Message, ToolCall, ToolSchema

_ENDPOINT_PARAMS = [
    pytest.param(
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
        id="chat-completions",
    ),
    pytest.param(
        "https://dashscope.aliyuncs.com/api/v2/apps/protocols/compatible-mode/v1",
        id="responses",
    ),
]


def _api_format_for_base_url(base_url: str) -> ApiFormat:
    if "protocols/compatible-mode" in base_url:
        return ApiFormat.OPENAI_RESPONSES
    return ApiFormat.OPENAI_CHAT_COMPLETIONS


def _make_provider(api_key: str, base_url: str, model: str) -> OpenAIModel:
    return OpenAIModel(
        config=ProviderConfig(
            provider_id="aliyun",
            base_url=base_url,
            api_key=api_key,
            api_format=_api_format_for_base_url(base_url),
        ),
        model=model,
    )


@pytest.mark.asyncio
async def test_live_tool_execution_persists_canonical_artifact(
    live_api_key: str, tmp_path: Path
) -> None:
    _ = live_api_key
    registry = ArtifactRegistry(root=tmp_path)
    world = World()
    entity = world.create_entity()

    async def live_tool(payload: str) -> str:
        return f"tool-ok:{payload}"

    world.add_component(
        entity,
        ConversationComponent(messages=[Message(role="user", content="run live tool")]),
    )
    world.add_component(
        entity,
        ToolRegistryComponent(
            tools={
                "live_tool": ToolSchema(
                    name="live_tool",
                    description="Live tool execution",
                    parameters={"type": "object"},
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
                    id="live-tool-1",
                    name="live_tool",
                    arguments={"payload": "artifact"},
                )
            ]
        ),
    )

    await ToolExecutionSystem(registry=registry).process(world)

    tool_files = list((tmp_path / "scratchbook" / "records" / "tool").glob("tool_*"))
    assert len(tool_files) == 1

    results = world.get_component(entity, ToolResultsComponent)
    assert results is not None
    record_path = results.results["live-tool-1"]
    assert record_path.startswith("scratchbook/records/tool/tool_")
    assert (tmp_path / record_path).exists()

    payload = json.loads((tmp_path / record_path).read_text(encoding="utf-8"))
    assert payload["tool_call_id"] == "live-tool-1"
    assert payload["result"] == "tool-ok:artifact"


@pytest.mark.asyncio
@pytest.mark.parametrize("base_url", _ENDPOINT_PARAMS)
async def test_live_planning_writes_canonical_plan_md(
    live_api_key: str,
    tmp_path: Path,
    base_url: str,
) -> None:
    model = os.getenv("LLM_MODEL", "qwen3.5-flash")
    model = _make_provider(api_key=live_api_key, base_url=base_url, model=model)

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


@pytest.mark.asyncio
async def test_live_trigger_creates_boulder_on_plan_script(
    live_api_key: str, tmp_path: Path
) -> None:
    _ = live_api_key
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

    world.add_component(
        entity,
        ConversationComponent(
            messages=[Message(role="user", content="@plan Live Boulder state")]
        ),
    )
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

    rendered = world.get_component(entity, RenderedUserPromptComponent)
    assert rendered is not None
    assert rendered.text == "@plan Live Boulder state"

    plan_slug = registry.normalize_plan_name("@plan Live Boulder state")
    boulder_file = tmp_path / "scratchbook" / plan_slug / "executes" / "boulder.json"
    assert boulder_file.exists()

    payload = json.loads(boulder_file.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "1"
    assert payload["plan_name"] == plan_slug
    assert payload["status"] == "created"
    assert isinstance(payload["started_at"], str)
    assert payload["active_plan"] == "@plan Live Boulder state"


@pytest.mark.asyncio
@pytest.mark.parametrize("base_url", _ENDPOINT_PARAMS)
async def test_live_no_legacy_category_paths_written(
    live_api_key: str,
    tmp_path: Path,
    base_url: str,
) -> None:
    model = os.getenv("LLM_MODEL", "qwen3.5-flash")
    model = _make_provider(api_key=live_api_key, base_url=base_url, model=model)

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
