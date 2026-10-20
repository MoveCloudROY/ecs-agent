"""Scoped stream adapters that own and close their agent run."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager
from copy import deepcopy
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from ecs_agent.components import AgentTurnComponent, StreamingComponent
from ecs_agent.observability.context import current_run_id
from ecs_agent.types import (
    EntityId, Message, RunCompletedEvent, StreamContentDeltaEvent,
    StreamContentStartEvent, StreamEndEvent, StreamReasoningDeltaEvent,
    StreamReasoningEndEvent, StreamStartEvent,
)

if TYPE_CHECKING:
    from ecs_agent.core.agent import AgentHandle


@dataclass(slots=True, frozen=True)
class AgentStreamEvent:
    """A stable event snapshot attributed to one entity, turn, and run."""

    entity_id: EntityId
    turn_id: str
    run_id: str
    event: object


@asynccontextmanager
async def stream_run(
    handle: AgentHandle, message: str | Message | None, *, max_steps: int | None,
) -> AsyncIterator[AsyncIterator[AgentStreamEvent]]:
    """Share AgentHandle's driver and guarantee cleanup on context exit."""
    handle._check_idle(max_steps)
    world, entity = handle.world, handle.entity
    previous = world.get_component(entity, StreamingComponent)
    # A scoped stream's final result must follow its deltas. Legacy background
    # publishers can outlive a run and are restored only after this context exits.
    world.add_component(entity, replace(previous, enabled=True, non_blocking_delta_publish=False) if previous else StreamingComponent(enabled=True))
    queue: asyncio.Queue[AgentStreamEvent | None] = asyncio.Queue()
    run_id: str | None = None
    turn_id: str | None = None

    async def receive(event: object) -> None:
        event_entity = event.result.entity_id if isinstance(event, RunCompletedEvent) and event.result else getattr(event, "entity_id", None)
        if event_entity == entity and run_id is not None and turn_id is not None and current_run_id() == run_id:
            queue.put_nowait(AgentStreamEvent(entity, turn_id, run_id, deepcopy(event)))

    event_types = (
        StreamStartEvent, StreamContentStartEvent, StreamContentDeltaEvent,
        StreamReasoningDeltaEvent, StreamReasoningEndEvent, StreamEndEvent, RunCompletedEvent,
    )
    for event_type in event_types:
        world.event_bus.subscribe(event_type, receive)
    task = None
    iterator = None
    try:
        task = handle._start(message, max_steps=max_steps)
        active = world._agent_run
        turn = world.get_component(entity, AgentTurnComponent)
        assert active is not None and turn is not None
        run_id, turn_id = active.run_id, turn.turn_id
        task.add_done_callback(lambda _: queue.put_nowait(None))

        async def events() -> AsyncGenerator[AgentStreamEvent, None]:
            while (event := await queue.get()) is not None:
                yield event
            await task

        iterator = events()
        yield iterator
    finally:
        if task is not None:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        if iterator is not None:
            await iterator.aclose()
        for event_type in event_types:
            world.event_bus.unsubscribe(event_type, receive)
        if previous is None:
            world.remove_component(entity, StreamingComponent)
        else:
            world.add_component(entity, previous)
