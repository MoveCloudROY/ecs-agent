"""Live TodoSkill behavior test: a real model plans and completes a checklist."""

import os

import pytest

from ecs_agent.components import (
    ConversationComponent,
    ErrorComponent,
    LLMComponent,
    TodoListComponent,
)
from ecs_agent.core import Runner, World
from ecs_agent.providers import ApiFormat, OpenAIModel, ProviderConfig
from ecs_agent.skills import SkillManager
from ecs_agent.skills.todo import TodoSkill
from ecs_agent.systems.reasoning import ReasoningSystem
from ecs_agent.systems.tool_execution import ToolExecutionSystem
from ecs_agent.types import Message, TodoListUpdatedEvent

_NETWORK_ERROR_MARKERS = ("timeout", "timed out", "connect", "unreachable")


def _build_model(api_key: str) -> OpenAIModel:
    model = os.getenv("LLM_MODEL") or "qwen3.5-flash"
    return OpenAIModel(
        config=ProviderConfig(
            provider_id="aliyun",
            base_url=os.getenv(
                "LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
            ),
            api_key=api_key,
            api_format=ApiFormat.OPENAI_CHAT_COMPLETIONS,
        ),
        model=model,
    )


@pytest.mark.asyncio
async def test_live_todo_agent_plans_and_completes_checklist(
    live_api_key: str,
) -> None:
    world = World(name="todo-live")
    agent = world.create_entity()

    world.add_component(
        agent,
        LLMComponent(
            model=_build_model(live_api_key),
            system_prompt=(
                "You are a careful assistant. You MUST track this task with the "
                "todo_write tool: first call todo_write with a checklist of the "
                "three steps (exactly one item in_progress), then answer the "
                "steps one at a time, calling todo_write after each step to mark "
                "it completed and move the next item to in_progress. Every reply "
                "MUST include a todo_write call until all items are completed — "
                "never stop while any item is pending or in_progress. Only after "
                "the final todo_write marks everything completed may you reply "
                "with a plain-text summary and no tool calls."
            ),
        ),
    )
    world.add_component(
        agent,
        ConversationComponent(
            messages=[
                Message(
                    role="user",
                    content=(
                        "Complete these three steps in order, tracking progress "
                        "with your todo list: "
                        "1) Name the capital of France. "
                        "2) Compute 17 * 23. "
                        "3) Give a one-sentence definition of ECS architecture."
                    ),
                )
            ]
        ),
    )
    SkillManager().install(world, agent, TodoSkill())

    events: list[TodoListUpdatedEvent] = []

    async def collect(event: TodoListUpdatedEvent) -> None:
        events.append(event)

    world.event_bus.subscribe(TodoListUpdatedEvent, collect)

    world.register_system(ReasoningSystem(priority=0), priority=0)
    world.register_system(ToolExecutionSystem(priority=5), priority=5)

    await Runner().run(world, max_ticks=20)

    error = world.get_component(agent, ErrorComponent)
    if error is not None:
        if any(marker in error.error.lower() for marker in _NETWORK_ERROR_MARKERS):
            pytest.skip(f"Live endpoint network failure: {error.error}")
        pytest.fail(f"Agent errored during live run: {error.error}")

    assert len(events) >= 2, "expected the model to update its todo list at least twice"

    for event in events:
        in_progress = sum(
            1 for item in event.items if item.status == "in_progress"
        )
        assert in_progress <= 1, "invariant violated: multiple in_progress items"

    component = world.get_component(agent, TodoListComponent)
    assert component is not None
    assert component.items, "expected a non-empty final todo list"
    assert all(item.status == "completed" for item in component.items), (
        "expected every item completed at the end, got: "
        + ", ".join(f"{item.status}:{item.content}" for item in component.items)
    )
