"""
Demonstrates the TodoSkill: an agent that plans and tracks a multi-step task
via the todo_write tool, including a mid-task replan when new work is
discovered.

This example shows how to:
1. Install the TodoSkill tool bundle with SkillManager.
2. Observe progress from host code via TodoListUpdatedEvent.
3. Read the final TodoListComponent state after the run.

Runs against a real provider when LLM_API_KEY is set, otherwise against a
scripted FakeModel.
"""

import asyncio
import os

from ecs_agent import SkillManager, TodoSkill
from ecs_agent.components import ConversationComponent, LLMComponent, TodoListComponent
from ecs_agent.core import Runner, World
from ecs_agent.providers import FakeModel, Model
from ecs_agent.providers.config import ApiFormat
from ecs_agent.systems.error_handling import ErrorHandlingSystem
from ecs_agent.systems.reasoning import ReasoningSystem
from ecs_agent.systems.tool_execution import ToolExecutionSystem
from ecs_agent.types import (
    CompletionResult,
    Message,
    TodoListUpdatedEvent,
    ToolCall,
)

STATUS_MARKERS = {"completed": "[x]", "in_progress": "[→]", "pending": "[ ]"}


def make_fake_model() -> FakeModel:
    """Scripted plan → mid-task discovery → completion, mirroring real usage."""
    return FakeModel(
        responses=[
            CompletionResult(
                message=Message(
                    role="assistant",
                    content="Let me plan this task first.",
                    tool_calls=[
                        ToolCall(
                            id="call_1",
                            name="todo_write",
                            arguments={
                                "todos": [
                                    {
                                        "content": "Inspect the current config",
                                        "status": "in_progress",
                                    },
                                    {"content": "Apply the fix", "status": "pending"},
                                    {
                                        "content": "Verify with tests",
                                        "status": "pending",
                                    },
                                ]
                            },
                        )
                    ],
                )
            ),
            CompletionResult(
                message=Message(
                    role="assistant",
                    content="The config revealed an extra problem — adding it to the plan.",
                    tool_calls=[
                        ToolCall(
                            id="call_2",
                            name="todo_write",
                            arguments={
                                "todos": [
                                    {
                                        "content": "Inspect the current config",
                                        "status": "completed",
                                    },
                                    {
                                        "content": "Apply the fix",
                                        "status": "in_progress",
                                    },
                                    {
                                        "content": "Verify with tests",
                                        "status": "pending",
                                    },
                                    {
                                        "content": "Fix stale default discovered in config",
                                        "status": "pending",
                                    },
                                ]
                            },
                        )
                    ],
                )
            ),
            CompletionResult(
                message=Message(
                    role="assistant",
                    content="Wrapping up the checklist.",
                    tool_calls=[
                        ToolCall(
                            id="call_3",
                            name="todo_write",
                            arguments={
                                "todos": [
                                    {
                                        "content": "Inspect the current config",
                                        "status": "completed",
                                    },
                                    {"content": "Apply the fix", "status": "completed"},
                                    {
                                        "content": "Verify with tests",
                                        "status": "completed",
                                    },
                                    {
                                        "content": "Fix stale default discovered in config",
                                        "status": "completed",
                                    },
                                ]
                            },
                        )
                    ],
                )
            ),
            CompletionResult(
                message=Message(role="assistant", content="Task complete."),
            ),
        ]
    )


async def main() -> None:
    world = World()
    agent = world.create_entity()

    api_key = os.getenv("LLM_API_KEY")
    if api_key:
        model = Model(
            os.getenv("LLM_MODEL", "qwen3.5-flash"),
            base_url=os.getenv(
                "LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
            ),
            api_key=api_key,
            api_format=ApiFormat.OPENAI_CHAT_COMPLETIONS,
        )
    else:
        model = make_fake_model()

    world.add_component(
        agent,
        LLMComponent(
            model=model,
            system_prompt=(
                "You are a careful engineer. For multi-step tasks, maintain a "
                "checklist with the todo_write tool: plan first, keep exactly one "
                "item in_progress, mark items completed as you finish them."
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
                        "Walk through fixing a config bug end to end: inspect the "
                        "config, apply the fix, and verify with tests. Track your "
                        "progress with your todo list."
                    ),
                )
            ]
        ),
    )

    # Install the todo_write tool bundle
    SkillManager().install(world, agent, TodoSkill())

    # Host-side progress observation
    async def on_todo_updated(event: TodoListUpdatedEvent) -> None:
        print(f"  [event] progress {event.completed_count}/{event.total_count}")

    world.event_bus.subscribe(TodoListUpdatedEvent, on_todo_updated)

    world.register_system(ReasoningSystem(priority=0), priority=0)
    world.register_system(ToolExecutionSystem(priority=5), priority=5)
    world.register_system(ErrorHandlingSystem(priority=99), priority=99)

    print("Running todo agent...")
    await Runner().run(world, max_ticks=15)

    component = world.get_component(agent, TodoListComponent)
    print("\nFinal todo list:")
    if component is None or not component.items:
        print("  (empty)")
    else:
        for number, item in enumerate(component.items, start=1):
            print(f"  {number}. {STATUS_MARKERS[item.status]} {item.content}")


if __name__ == "__main__":
    asyncio.run(main())
