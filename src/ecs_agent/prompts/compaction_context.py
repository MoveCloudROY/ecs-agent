"""Compaction context provider seam.

Systems that hold state which must survive conversation compaction contribute
it to the summarization request through this neutral protocol, so
``CompactionSystem`` needs no knowledge of any contributor's domain.
Contributors are constructor-injected into ``CompactionSystem``;
:data:`DEFAULT_COMPACTION_CONTEXT_PROVIDERS` is the zero-config default set.

Unlike placeholder providers this protocol has no fingerprint: compaction has
no cache to invalidate — every trigger reads fresh state.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from ecs_agent.components import (
    SubagentNotificationQueueComponent,
    SubagentSessionTableComponent,
)
from ecs_agent.core.world import World
from ecs_agent.types import (
    COMPLETED_SUBAGENT_STATUSES,
    PENDING_SUBAGENT_STATUSES,
    EntityId,
)


class CompactionContextProvider(Protocol):
    provider_id: str

    def render_compaction_context(
        self, world: World, entity_id: EntityId
    ) -> str | None:
        """Return a text block for the summarization request, or None when
        there is nothing to contribute this round."""
        ...


class SubagentCompactionContextProvider:
    """Digest of subagent session state (pending/completed/notifications)."""

    provider_id = "subagent_sessions"

    def render_compaction_context(
        self, world: World, entity_id: EntityId
    ) -> str | None:
        table = world.get_component(entity_id, SubagentSessionTableComponent)
        queue = world.get_component(entity_id, SubagentNotificationQueueComponent)

        pending: list[str] = []
        completed: list[tuple[str, str]] = []
        if table is not None:
            for session_id, record in sorted(table.sessions.items()):
                if record.status in PENDING_SUBAGENT_STATUSES:
                    pending.append(session_id)
                    continue
                if record.status in COMPLETED_SUBAGENT_STATUSES:
                    completed.append((session_id, record.status))

        notifications: list[str] = []
        if queue is not None:
            for notification in sorted(
                queue.notifications,
                key=lambda item: (item.session_id, item.created_at),
            ):
                delivered = "yes" if notification.delivered_at is not None else "no"
                parts = [
                    f"{notification.session_id}: notification "
                    f"status={notification.terminal_status}",
                    f"delivered={delivered}",
                ]
                if notification.summary is not None:
                    parts.append(f'summary="{notification.summary}"')
                if notification.error is not None:
                    parts.append(f'error="{notification.error}"')
                notifications.append(" ".join(parts))

        if not pending and not completed and not notifications:
            return None

        lines = ["Subagent session state:"]
        for session_id in pending:
            lines.append(f"Pending: {session_id}")
        for session_id, status in completed:
            lines.append(f"Completed ({status}): {session_id}")
        lines.extend(notifications)
        return "\n".join(lines)


DEFAULT_COMPACTION_CONTEXT_PROVIDERS: tuple[CompactionContextProvider, ...] = (
    SubagentCompactionContextProvider(),
)


def render_compaction_context_blocks(
    world: World,
    entity_id: EntityId,
    providers: Sequence[CompactionContextProvider],
) -> str | None:
    """Join every provider's non-empty block with a blank line, in provider
    order. Returns None when no provider contributes."""
    blocks = [
        block
        for provider in providers
        if (block := provider.render_compaction_context(world, entity_id))
    ]
    if not blocks:
        return None
    return "\n\n".join(blocks)


__all__ = [
    "CompactionContextProvider",
    "DEFAULT_COMPACTION_CONTEXT_PROVIDERS",
    "SubagentCompactionContextProvider",
    "render_compaction_context_blocks",
]
