"""Interactive input adapter for UI Design Flow example.

Placeholder module for interactive user input handling via UserInputSystem
and UserInputRequestedEvent. Will be fully implemented in Task 3.

Functions:
- setup_interactive_input: Configure event handlers for async user input
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ecs_agent.core import World
    from ecs_agent.types import EntityId


async def setup_interactive_input(world: World, agent_id: EntityId) -> None:
    """Setup interactive input handling for the agent.

    Args:
        world: World instance
        agent_id: Agent entity ID to attach input component to

    TODO: Implement in Task 3
    - Subscribe to UserInputRequestedEvent
    - Handle stdin/async input via run_in_executor
    - Resolve futures with user text
    - Handle exit commands
    """
    pass
