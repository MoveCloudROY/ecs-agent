from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import time

from ecs_agent.components import (
    ErrorComponent,
    SubagentNotificationQueueComponent,
    SubagentWaitComponent,
    TerminalComponent,
)
from ecs_agent.core.world import World
from ecs_agent.logging import get_logger
from ecs_agent.types import EntityId, SubagentNotificationRecord

logger = get_logger(__name__)


def notification_matches_wait(
    notification: SubagentNotificationRecord,
    component: SubagentWaitComponent,
) -> bool:
    if notification.delivered_at is not None:
        return False
    if component.session_ids is None:
        return True
    return notification.session_id in component.session_ids


class SubagentWaitSystem:
    def __init__(self, priority: int = -5) -> None:
        self.priority = priority

    async def process(self, world: World) -> None:
        for entity_id, components in world.query(SubagentWaitComponent):
            wait_component = components[0]
            assert isinstance(wait_component, SubagentWaitComponent)

            try:
                if self._has_matching_notification(world, entity_id, wait_component):
                    self._resolve_wait(world, entity_id, wait_component)
                    continue

                await self._handle_wait(entity_id, wait_component, world)
            except Exception as exc:
                logger.error(
                    "subagent_wait_error",
                    entity_id=entity_id,
                    exception=str(exc),
                )
                world.add_component(
                    entity_id,
                    ErrorComponent(
                        error=str(exc),
                        system_name="SubagentWaitSystem",
                        timestamp=time.time(),
                    ),
                )
                world.add_component(
                    entity_id,
                    TerminalComponent(reason="subagent_wait_error"),
                )

    async def _handle_wait(
        self,
        entity_id: EntityId,
        component: SubagentWaitComponent,
        world: World,
    ) -> None:
        if component.future is None:
            component.future = asyncio.get_running_loop().create_future()

        timeout = self._remaining_timeout(component)
        if timeout is not None and timeout <= 0:
            self._mark_timeout(world, entity_id, component.timeout)
            return

        try:
            await asyncio.wait_for(asyncio.shield(component.future), timeout=timeout)
        except asyncio.TimeoutError:
            self._mark_timeout(world, entity_id, component.timeout)
            return

        self._resolve_wait(world, entity_id, component)

    def _has_matching_notification(
        self,
        world: World,
        entity_id: EntityId,
        component: SubagentWaitComponent,
    ) -> bool:
        queue = world.get_component(entity_id, SubagentNotificationQueueComponent)
        if queue is None:
            return False
        return any(
            notification_matches_wait(notification, component)
            for notification in queue.notifications
        )

    def _resolve_wait(
        self,
        world: World,
        entity_id: EntityId,
        component: SubagentWaitComponent,
    ) -> None:
        future = component.future
        if isinstance(future, asyncio.Future) and not future.done():
            future.set_result(None)
        world.remove_component(entity_id, SubagentWaitComponent)

    def _mark_timeout(
        self,
        world: World,
        entity_id: EntityId,
        timeout: float | None,
    ) -> None:
        logger.error(
            "subagent_wait_timeout",
            entity_id=entity_id,
            timeout=timeout,
        )
        world.add_component(
            entity_id,
            ErrorComponent(
                error=f"Subagent wait timeout after {timeout}s",
                system_name="SubagentWaitSystem",
                timestamp=time.time(),
            ),
        )
        world.add_component(
            entity_id,
            TerminalComponent(reason="subagent_wait_timeout"),
        )

    def _remaining_timeout(self, component: SubagentWaitComponent) -> float | None:
        if component.timeout is None:
            return None
        if component.started_at is None:
            return component.timeout

        started_at = datetime.fromisoformat(component.started_at.replace("Z", "+00:00"))
        elapsed = (datetime.now(timezone.utc) - started_at).total_seconds()
        return component.timeout - elapsed


__all__ = ["SubagentWaitSystem", "notification_matches_wait"]
