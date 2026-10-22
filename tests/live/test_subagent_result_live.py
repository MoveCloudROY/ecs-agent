"""Explicit live integration coverage migrated from mixed/offline directories."""

from __future__ import annotations
import asyncio
import json
import pytest
from ecs_agent.components import (
    LLMComponent,
    SubagentRegistryComponent,
    SubagentSessionTableComponent,
    ToolRegistryComponent,
)
from ecs_agent.core import World
from ecs_agent.systems.error_handling import ErrorHandlingSystem
from ecs_agent.systems.reasoning import ReasoningSystem
from ecs_agent.systems.subagent import SubagentSystem
from ecs_agent.systems.subagent_wait import SubagentWaitSystem
from ecs_agent.types import SubagentConfig

from tests.support.models import LiveModelFactory


@pytest.mark.live(
    formats=["openai_chat_completions", "openai_responses"],
    capabilities=["text", "tools"],
)
@pytest.mark.workflow
async def test_subagent_result_background_real_llm(
    *, live_model_factory: LiveModelFactory
) -> None:
    model = live_model_factory.openai()

    world = World(name="parent")
    parent = world.create_entity()
    world.add_component(
        parent, LLMComponent(model=model, system_prompt="You coordinate.")
    )
    world.add_component(parent, ToolRegistryComponent(tools={}, handlers={}))
    world.add_component(parent, SubagentSessionTableComponent())
    world.add_component(
        parent,
        SubagentRegistryComponent(
            subagents={
                "worker": SubagentConfig(
                    name="worker",
                    model=model,
                    system_prompt="Answer concisely.",
                    max_ticks=3,
                )
            }
        ),
    )

    system = SubagentSystem()
    world.register_system(system, priority=-1)
    world.register_system(SubagentWaitSystem(), priority=-5)
    world.register_system(ReasoningSystem(priority=0), priority=0)
    world.register_system(ErrorHandlingSystem(priority=99), priority=99)
    await world.process()  # install tools
    system.install_subagent_control_tools(world, parent)

    subagent = world.get_component(parent, ToolRegistryComponent).handlers["subagent"]
    result_handler = world.get_component(parent, ToolRegistryComponent).handlers[
        "subagent_result"
    ]

    launch = json.loads(
        await subagent(
            category="worker",
            prompt="What is 2+2? Reply with just the number.",
            background=True,
        )
    )
    session_id = launch["session_id"]

    result = json.loads(
        await asyncio.wait_for(
            result_handler(session_id=session_id, timeout=120), timeout=150
        )
    )
    assert result["lifecycle_status"] == "succeeded", result
