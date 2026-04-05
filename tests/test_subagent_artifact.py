from __future__ import annotations

import asyncio
import json
from pathlib import Path

from ecs_agent.components import ToolRegistryComponent
from ecs_agent.components.definitions import (
    SubagentRegistryComponent,
    SubagentSessionTableComponent,
)
from ecs_agent.core.world import World
from ecs_agent.providers.fake_provider import FakeProvider
from ecs_agent.scratchbook.artifact_registry import ArtifactRegistry
from ecs_agent.systems.subagent import SubagentSystem
from ecs_agent.types import CompletionResult, Message, SubagentConfig


async def test_runtime_session_id_remains_distinct_from_subagent_artifact_id(
    tmp_path: Path,
) -> None:
    world = World()
    parent = world.create_entity()

    world.add_component(parent, ToolRegistryComponent(tools={}, handlers={}))
    world.add_component(parent, SubagentSessionTableComponent())
    world.add_component(
        parent,
        SubagentRegistryComponent(
            subagents={
                "artifact-agent": SubagentConfig(
                    name="artifact-agent",
                    provider=FakeProvider(
                        responses=[
                            CompletionResult(
                                message=Message(
                                    role="assistant",
                                    content="artifact background result",
                                )
                            )
                        ]
                    ),
                    model="fake",
                )
            }
        ),
    )

    system = SubagentSystem(registry=ArtifactRegistry(tmp_path))
    system.install_subagent_tool(world, parent)
    system.install_subagent_control_tools(world, parent)

    tools = world.get_component(parent, ToolRegistryComponent)
    assert tools is not None

    launch_raw = await tools.handlers["subagent"](
        category="artifact-agent",
        prompt="run in background",
        load_skills=[],
        background=True,
        timeout=None,
    )
    launch_payload = json.loads(launch_raw)
    session_id = launch_payload["session_id"]

    await asyncio.sleep(0.1)

    result_raw = await tools.handlers["subagent_result"](
        session_id=session_id,
        timeout=None,
    )
    result_payload = json.loads(result_raw)

    assert result_payload["session_id"] == session_id
    assert result_payload["artifact_id"].startswith("subagent_")
    assert result_payload["session_id"] != result_payload["artifact_id"]


async def test_foreground_subagent_artifact_record_path_points_to_records_subagent(
    tmp_path: Path,
) -> None:
    world = World()
    parent = world.create_entity()

    world.add_component(parent, ToolRegistryComponent(tools={}, handlers={}))
    world.add_component(
        parent,
        SubagentRegistryComponent(
            subagents={
                "artifact-agent": SubagentConfig(
                    name="artifact-agent",
                    provider=FakeProvider(
                        responses=[
                            CompletionResult(
                                message=Message(
                                    role="assistant",
                                    content="foreground persisted result",
                                )
                            )
                        ]
                    ),
                    model="fake",
                )
            }
        ),
    )

    system = SubagentSystem(registry=ArtifactRegistry(tmp_path))
    system.install_subagent_tool(world, parent)

    tools = world.get_component(parent, ToolRegistryComponent)
    assert tools is not None

    result = await tools.handlers["subagent"](
        category="artifact-agent",
        prompt="run in foreground",
        load_skills=[],
        background=False,
        timeout=None,
    )
    assert result == "foreground persisted result"

    record_dir = tmp_path / "scratchbook/records/subagent"
    files = list(record_dir.iterdir())
    assert len(files) == 1
    assert files[0].name.startswith("subagent_")
    assert files[0].read_text(encoding="utf-8") == "foreground persisted result"


async def test_background_subagent_completion_persists_full_output_to_records_subagent(
    tmp_path: Path,
) -> None:
    full_result = "x" * 9000

    world = World()
    parent = world.create_entity()

    world.add_component(parent, ToolRegistryComponent(tools={}, handlers={}))
    world.add_component(parent, SubagentSessionTableComponent())
    world.add_component(
        parent,
        SubagentRegistryComponent(
            subagents={
                "artifact-agent": SubagentConfig(
                    name="artifact-agent",
                    provider=FakeProvider(
                        responses=[
                            CompletionResult(
                                message=Message(role="assistant", content=full_result)
                            )
                        ]
                    ),
                    model="fake",
                )
            }
        ),
    )

    system = SubagentSystem(registry=ArtifactRegistry(tmp_path))
    system.install_subagent_tool(world, parent)
    system.install_subagent_control_tools(world, parent)

    tools = world.get_component(parent, ToolRegistryComponent)
    assert tools is not None

    launch_raw = await tools.handlers["subagent"](
        category="artifact-agent",
        prompt="run in background",
        load_skills=[],
        background=True,
        timeout=None,
    )
    session_id = json.loads(launch_raw)["session_id"]

    await asyncio.sleep(0.1)

    result_raw = await tools.handlers["subagent_result"](
        session_id=session_id,
        timeout=None,
    )
    result_payload = json.loads(result_raw)
    record_path = result_payload["record_path"]

    assert record_path.startswith("scratchbook/records/subagent/subagent_")
    assert result_payload["inline_content"] is not None
    assert record_path in result_payload["inline_content"]

    persisted_file = tmp_path / record_path
    assert persisted_file.exists()
    assert persisted_file.read_text(encoding="utf-8") == full_result

    table = world.get_component(parent, SubagentSessionTableComponent)
    assert table is not None
    assert table.sessions[session_id].artifact_record_path == record_path
