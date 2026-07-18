"""Terminal-notification coordination for background subagent sessions.

``NotificationCoordinator`` (Task 6 of the subagent package refactor) is the single
owner of "a session reached a terminal status -> enqueue a parent notification and,
if the parent is waiting on a matching scope, resolve its wait future". Previously
this logic was split between ``SubagentSystem`` and ``SubagentWaitSystem``; the
terminal check now routes through the one shared ``wait_scope_is_terminal`` helper.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Literal

from ecs_agent.components import (
    SubagentNotificationQueueComponent,
    SubagentWaitComponent,
)
from ecs_agent.components.definitions import SubagentSessionTableComponent
from ecs_agent.core.world import World
from ecs_agent.systems.subagent_wait import (
    notification_matches_wait,
    wait_scope_is_terminal,
)
from ecs_agent.types import (
    EntityId,
    SubagentNotificationRecord,
    SubagentSessionRecord,
    is_wake_worthy,
)


class NotificationCoordinator:
    """Enqueues terminal notifications and resolves matching wait futures."""

    def notification_summary(self, metadata: SubagentSessionRecord) -> str | None:
        return metadata.result_summary

    def get_or_create_notification_queue(
        self,
        world: World,
        parent_entity_id: EntityId,
    ) -> SubagentNotificationQueueComponent:
        queue = world.get_component(
            parent_entity_id, SubagentNotificationQueueComponent
        )
        if queue is not None:
            return queue

        queue = SubagentNotificationQueueComponent()
        world.add_component(parent_entity_id, queue)
        return queue

    def all_waited_sessions_terminal(
        self,
        world: World,
        parent_entity_id: EntityId,
        wait_component: SubagentWaitComponent,
    ) -> bool:
        """Return True when every session in the wait scope is terminal."""
        table = world.get_component(parent_entity_id, SubagentSessionTableComponent)
        effective_ids = (
            wait_component.resolved_session_ids or wait_component.session_ids
        )
        return wait_scope_is_terminal(table, effective_ids).all_terminal

    def enqueue_parent_notification(
        self,
        world: World,
        metadata: SubagentSessionRecord,
    ) -> None:
        if not metadata.background or not is_wake_worthy(metadata.status):
            return

        terminal_status: Literal["succeeded", "failed", "timed_out", "cancelled"]
        if metadata.status == "succeeded":
            terminal_status = "succeeded"
        elif metadata.status == "failed":
            terminal_status = "failed"
        elif metadata.status == "cancelled":
            terminal_status = "cancelled"
        else:
            terminal_status = "timed_out"

        notification_id = f"{metadata.session_id}:{terminal_status}"
        queue = self.get_or_create_notification_queue(world, metadata.parent_entity_id)
        if any(
            notification.notification_id == notification_id
            for notification in queue.notifications
        ):
            return

        notification = SubagentNotificationRecord(
            notification_id=notification_id,
            session_id=metadata.session_id,
            parent_entity_id=metadata.parent_entity_id,
            terminal_status=terminal_status,
            summary=self.notification_summary(metadata),
            error=metadata.error,
            created_at=datetime.now(tz=timezone.utc).isoformat(),
            delivered_at=None,
        )
        queue.notifications.append(notification)

        wait_component = world.get_component(
            metadata.parent_entity_id, SubagentWaitComponent
        )
        if wait_component is None:
            return

        future = wait_component.future
        if not notification_matches_wait(notification, wait_component):
            return
        if isinstance(future, asyncio.Future) and not future.done():
            if self.all_waited_sessions_terminal(
                world, metadata.parent_entity_id, wait_component
            ):
                future.set_result(None)


__all__ = ["NotificationCoordinator"]
