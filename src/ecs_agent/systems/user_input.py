"""System for awaiting external user input via asyncio Futures."""

from __future__ import annotations

import asyncio
import time

from ecs_agent.components.definitions import (
    ConversationComponent,
    ErrorComponent,
    TerminalComponent,
    UserInputComponent,
)
from ecs_agent.core.world import World
from ecs_agent.logging import get_logger
from ecs_agent.types import EntityId, Message, UserInputRequestedEvent

logger = get_logger(__name__)


class UserInputSystem:
    """Awaits user input for entities that have a UserInputComponent.

    Lifecycle per entity per tick:

    1. If ``component.future`` is ``None`` — create a new Future, publish
       a ``UserInputRequestedEvent`` via the EventBus, then ``await`` it
       (with optional timeout).
    2. If the Future resolves — store the result in ``component.result``,
       append it as a ``user`` message to ``ConversationComponent``, and
       clear the future so the entity can proceed.
    3. If the Future times out — attach ``ErrorComponent`` +
       ``TerminalComponent`` and log the timeout.

    External code (CLI, web handler, test harness) subscribes to
    ``UserInputRequestedEvent`` and calls
    ``event.input_future.set_result(text)`` to provide the input.
    """

    def __init__(self, priority: int = -10) -> None:
        self.priority = priority

    async def process(self, world: World) -> None:
        """Process all entities that are waiting for user input."""
        for entity_id, components in world.query(UserInputComponent):
            user_input = components[0]
            assert isinstance(user_input, UserInputComponent)

            # Already resolved on a previous tick — skip
            if user_input.result is not None:
                continue

            try:
                await self._handle_input(entity_id, user_input, world)
            except Exception as exc:
                logger.error(
                    "user_input_error",
                    entity_id=entity_id,
                    exception=str(exc),
                )
                world.add_component(
                    entity_id,
                    ErrorComponent(
                        error=str(exc),
                        system_name="UserInputSystem",
                        timestamp=time.time(),
                    ),
                )
                world.add_component(
                    entity_id,
                    TerminalComponent(reason="user_input_error"),
                )

    async def _handle_input(
        self,
        entity_id: EntityId,
        component: UserInputComponent,
        world: World,
    ) -> None:
        # Create a fresh Future if none exists yet
        if component.future is None:
            component.future = asyncio.get_running_loop().create_future()
            event = UserInputRequestedEvent(
                entity_id=entity_id,
                prompt=component.prompt,
                input_future=component.future,
            )
            await world.event_bus.publish(event)

        # Wait for the external code to resolve the Future
        try:
            result = await asyncio.wait_for(
                asyncio.shield(component.future),
                timeout=component.timeout,
            )
        except asyncio.TimeoutError:
            logger.error(
                "user_input_timeout",
                entity_id=entity_id,
                timeout=component.timeout,
            )
            world.add_component(
                entity_id,
                ErrorComponent(
                    error=f"User input timeout after {component.timeout}s",
                    system_name="UserInputSystem",
                    timestamp=time.time(),
                ),
            )
            world.add_component(
                entity_id,
                TerminalComponent(reason="user_input_timeout"),
            )
            return

        component.result = result
        component.future = None  # Clear for potential reuse

        logger.info(
            "user_input_received",
            entity_id=entity_id,
            input_length=len(result),
        )

        # Append user message to conversation if component exists
        conversation = world.get_component(entity_id, ConversationComponent)
        if conversation is not None:
            conversation.messages.append(
                Message(role="user", content=result),
            )


__all__ = ["UserInputSystem"]
