"""Interactive input adapter for UI Design Flow example."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from ecs_agent.components import (
    CompactionConfigComponent,
    ConversationArchiveComponent,
    UserInputComponent,
)
from ecs_agent.components.definitions import ConversationComponent, TerminalComponent
from ecs_agent.systems import TerminalCleanupSystem
from ecs_agent.systems.compaction import CompactionSystem
from ecs_agent.systems.user_input import UserInputSystem
from ecs_agent.types import (
    CompactionMethod,
    ReasoningCompleteEvent,
    UserInputRequestedEvent,
)

if TYPE_CHECKING:
    from ecs_agent.core import World
    from ecs_agent.types import EntityId


# Auto-compaction defaults (ISSUE-2: without compaction, conversation history
# grows unbounded and is resent in full every turn). A generous threshold keeps
# short sessions untouched while capping long interactive runs.
DEFAULT_COMPACTION_THRESHOLD_TOKENS = 300_000
DEFAULT_COMPACTION_METHOD: CompactionMethod = "predrop_then_compact"
# Runs before SystemPromptRenderSystem (-20) so a fresh summary is rendered into
# the (volatile) system-prompt tail on the same tick it is produced.
COMPACTION_SYSTEM_PRIORITY = -30


def install_auto_compaction(
    world: World,
    agent_id: EntityId,
    *,
    threshold_tokens: int = DEFAULT_COMPACTION_THRESHOLD_TOKENS,
    compaction_method: CompactionMethod = DEFAULT_COMPACTION_METHOD,
) -> None:
    """Enable automatic conversation compaction for an agent (ISSUE-2).

    Attaches the per-entity config/archive components and registers
    ``CompactionSystem`` on the world so history is summarized once it exceeds
    ``threshold_tokens`` instead of growing without bound.

    Call once per world for the single-agent example.
    """
    world.add_component(
        agent_id,
        CompactionConfigComponent(
            threshold_tokens=threshold_tokens,
            compaction_method=compaction_method,
        ),
    )
    world.add_component(agent_id, ConversationArchiveComponent())
    world.register_system(
        CompactionSystem(), priority=COMPACTION_SYSTEM_PRIORITY
    )


async def setup_interactive_input(world: World, agent_id: EntityId) -> None:
    """Set up event-driven async stdin handling for the agent.

    The agent follows a two-phase loop per turn:

    1. ``UserInputSystem`` fires ``UserInputRequestedEvent``.
    2. ``provide_input`` reads stdin, appends the user message, then
       **removes** ``UserInputComponent`` so the agent can autonomously
       complete its full reasoning-tool cycle without being interrupted.
    3. Once ``ReasoningSystem`` produces a final text response (no tool
       calls), it publishes ``ReasoningCompleteEvent``.
    4. ``on_reasoning_complete`` re-arms ``UserInputComponent`` so the
       next turn begins only after the cycle is fully done.

    This prevents ``UserInputSystem`` (priority -5) from blocking the
    follow-up LLM call that processes tool results on the same or next tick.

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

        # Do NOT re-arm UserInputComponent here.  The agent may need to make
        # one or more autonomous tool-call turns before it is ready for the
        # next user message.  Re-arming is deferred to on_reasoning_complete,
        # which fires only after the LLM produces a final text response.

    async def on_reasoning_complete(event: ReasoningCompleteEvent) -> None:
        """Re-arm UserInputComponent after a full reasoning-tool cycle ends."""
        if event.entity_id != agent_id:
            return
        # Always replace the component with a fresh one (result=None, future=None).
        # UserInputComponent.result may be set (consumed) but the component is never
        # removed by UserInputSystem, so we cannot use None-check as a guard.
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
