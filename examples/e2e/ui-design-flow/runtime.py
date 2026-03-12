"""Interactive input adapter for UI Design Flow example."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from ecs_agent.components import UserInputComponent
from ecs_agent.components.definitions import ConversationComponent, TerminalComponent
from ecs_agent.systems.user_input import UserInputSystem
from ecs_agent.types import UserInputRequestedEvent

if TYPE_CHECKING:
    from ecs_agent.core import World
    from ecs_agent.types import EntityId


class ClearTerminalForInputSystem:
    """Remove stale reasoning_complete TerminalComponent after each LLM turn.

    ReasoningSystem (priority=0) adds TerminalComponent(reason='reasoning_complete')
    when the LLM responds with no tool calls.  In interactive mode the entity
    re-attaches UserInputComponent to request another turn.  This system runs
    AFTER ReasoningSystem (priority=1) to clear the stale terminal before the
    Runner inspects it at the end of the tick.
    """

    def __init__(self, priority: int = 1) -> None:
        self.priority = priority

    async def process(self, world: World) -> None:
        for entity_id, _ in list(world.query(UserInputComponent)):
            terminal = world.get_component(entity_id, TerminalComponent)
            if terminal is not None and terminal.reason == "reasoning_complete":
                world.remove_component(entity_id, TerminalComponent)


async def setup_interactive_input(world: World, agent_id: EntityId) -> None:
    """Set up event-driven async stdin handling for the agent.

    Args:
        world: World instance
        agent_id: Agent entity ID to attach input component to

    """
    # Track how many messages were in the conversation the last time we printed,
    # so we can print only the new assistant messages before each new prompt.
    last_printed_index: list[int] = [0]

    async def provide_input(event: UserInputRequestedEvent) -> None:
        loop = asyncio.get_running_loop()

        # Print any new assistant messages that appeared since last prompt
        conv = world.get_component(event.entity_id, ConversationComponent)
        if conv is not None:
            for msg in conv.messages[last_printed_index[0] :]:
                if msg.role == "assistant" and msg.content:
                    print(f"\nAssistant: {msg.content}\n")
            last_printed_index[0] = len(conv.messages)

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
    world.register_system(ClearTerminalForInputSystem(priority=1), priority=1)
    world.register_system(UserInputSystem(priority=-5), priority=-5)

    if world.get_component(agent_id, UserInputComponent) is None:
        world.add_component(agent_id, UserInputComponent(prompt="You> "))
