"""Interactive input adapter for UI Design Flow example."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from ecs_agent.components import UserInputComponent
from ecs_agent.components.definitions import TerminalComponent
from ecs_agent.systems.user_input import UserInputSystem
from ecs_agent.types import UserInputRequestedEvent

if TYPE_CHECKING:
    from ecs_agent.core import World
    from ecs_agent.types import EntityId


async def setup_interactive_input(world: World, agent_id: EntityId) -> None:
    """Set up event-driven async stdin handling for the agent.

    Args:
        world: World instance
        agent_id: Agent entity ID to attach input component to

    """

    async def provide_input(event: UserInputRequestedEvent) -> None:
        loop = asyncio.get_running_loop()

        try:
            user_text = await loop.run_in_executor(None, input, event.prompt)
        except EOFError:
            user_text = "exit"

        normalized = user_text.lower().strip()
        if normalized in ("exit", "quit"):
            world.add_component(
                event.entity_id,
                TerminalComponent(reason="user_exit_command"),
            )

        if not event.input_future.done():
            event.input_future.set_result(user_text)

        if normalized not in ("exit", "quit"):
            world.add_component(
                event.entity_id, UserInputComponent(prompt=event.prompt)
            )

    world.event_bus.subscribe(UserInputRequestedEvent, provide_input)
    world.register_system(UserInputSystem(priority=-5), priority=-5)

    if world.get_component(agent_id, UserInputComponent) is None:
        world.add_component(agent_id, UserInputComponent(prompt="You> "))
