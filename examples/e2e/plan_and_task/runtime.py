"""Interactive runtime adapter for the plan-and-task example."""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from ecs_agent.components import UserInputComponent
from ecs_agent.components.definitions import ConversationComponent, TerminalComponent
from ecs_agent.logging import get_logger
from ecs_agent.systems import TerminalCleanupSystem
from ecs_agent.systems.user_input import UserInputSystem
from ecs_agent.types import ReasoningCompleteEvent, UserInputRequestedEvent

if TYPE_CHECKING:
    from ecs_agent.core import World
    from ecs_agent.types import EntityId

logger = get_logger(__name__)

_DEFAULT_WORKFLOW_ID = "plan-task-workflow"


@dataclass(slots=True)
class RuntimeConfig:
    """Configuration resolved from environment variables for the interactive runtime."""

    workflow_id: str


def resolve_workflow_id(workflow_id: str | None = None) -> str:
    if workflow_id:
        return workflow_id

    env_workflow_id = os.environ.get("PLAN_TASK_WORKFLOW_ID", "").strip()
    if env_workflow_id:
        return env_workflow_id

    return _DEFAULT_WORKFLOW_ID


def build_runtime_config(workflow_id: str | None = None) -> RuntimeConfig:
    return RuntimeConfig(workflow_id=resolve_workflow_id(workflow_id))


async def setup_interactive_input(
    world: World,
    agent_id: EntityId,
    workflow_id: str | None = None,
    command_handler: Callable[[str], bool] | None = None,
) -> RuntimeConfig:
    last_printed_index: list[int] = [0]
    runtime_config = build_runtime_config(workflow_id)

    async def provide_input(event: UserInputRequestedEvent) -> None:
        loop = asyncio.get_running_loop()

        conv = world.get_component(event.entity_id, ConversationComponent)
        if conv is not None:
            for msg in conv.messages[last_printed_index[0] :]:
                if msg.role == "assistant" and msg.content:
                    print(f"\nAssistant: {msg.content}\n")
            last_printed_index[0] = len(conv.messages)

        while True:
            try:
                user_text = await loop.run_in_executor(None, input, event.prompt)
            except EOFError:
                user_text = "exit"

            normalized = user_text.lower().strip()
            if normalized in ("exit", "quit"):
                logger.info(
                    "plan_task_user_exit",
                    entity_id=int(event.entity_id),
                    workflow_id=runtime_config.workflow_id,
                )
                world.add_component(
                    event.entity_id,
                    TerminalComponent(reason="user_exit_command"),
                )
                if not event.input_future.done():
                    event.input_future.set_result(user_text)
                return

            if user_text.startswith("/") and command_handler is not None:
                if command_handler(user_text):
                    continue

            if not event.input_future.done():
                event.input_future.set_result(user_text)
            return

    async def on_reasoning_complete(event: ReasoningCompleteEvent) -> None:
        if event.entity_id != agent_id:
            return

        logger.info(
            "plan_task_reasoning_complete",
            entity_id=int(agent_id),
            workflow_id=runtime_config.workflow_id,
        )
        world.add_component(agent_id, UserInputComponent(prompt="You> "))

    world.event_bus.subscribe(UserInputRequestedEvent, provide_input)
    world.event_bus.subscribe(ReasoningCompleteEvent, on_reasoning_complete)
    world.register_system(
        TerminalCleanupSystem(priority=1, clear_reasons=("reasoning_complete",)),
        priority=1,
    )
    world.register_system(UserInputSystem(priority=-5), priority=-5)

    if world.get_component(agent_id, UserInputComponent) is None:
        world.add_component(agent_id, UserInputComponent(prompt="You> "))

    return runtime_config
