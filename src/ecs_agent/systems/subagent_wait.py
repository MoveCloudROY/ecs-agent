"""Subagent wait notification system."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timezone
import time

from ecs_agent.components import (
    ConversationComponent,
    ErrorComponent,
    SubagentNotificationQueueComponent,
    SubagentSessionTableComponent,
    SubagentWaitComponent,
    TerminalComponent,
)
from ecs_agent.core.world import World
from ecs_agent.logging import get_logger
from ecs_agent.types import EntityId, Message, SubagentNotificationRecord, SubagentSessionRecord

logger = get_logger(__name__)

PENDING_SUBAGENT_STATUSES = {"queued", "running"}
COMPLETED_SUBAGENT_STATUSES = {"succeeded", "failed", "timed_out", "cancelled"}
FAILED_SUBAGENT_STATUSES = {"failed", "timed_out", "cancelled"}

ResumeCallback = Callable[[str, EntityId, World], Awaitable[str]]


@dataclass(slots=True)
class SubagentCompactionState:
    pending: list[str]
    completed: list[tuple[str, str]]
    notifications: list[str]


def build_subagent_compaction_state(
    table: SubagentSessionTableComponent | None,
    queue: SubagentNotificationQueueComponent | None,
) -> SubagentCompactionState:
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
                f"{notification.session_id}: notification status={notification.terminal_status}",
                f"delivered={delivered}",
            ]
            if notification.summary is not None:
                parts.append(f'summary="{notification.summary}"')
            if notification.error is not None:
                parts.append(f'error="{notification.error}"')
            notifications.append(" ".join(parts))

    return SubagentCompactionState(
        pending=pending,
        completed=completed,
        notifications=notifications,
    )


def _effective_session_ids(component: SubagentWaitComponent) -> list[str] | None:
    """Return the effective wait scope: resolved_session_ids if set, else session_ids."""
    if component.resolved_session_ids is not None:
        return component.resolved_session_ids
    return component.session_ids


@dataclass(slots=True)
class WaitScopeStatus:
    """Result of evaluating whether a wait scope is fully terminal.

    Attributes:
        all_terminal: True only when every session_id in the scope exists
            and is in a terminal status.
        missing_session_ids: Session IDs that were not found in the table.
            Non-empty list means all_terminal is False.
    """

    all_terminal: bool
    missing_session_ids: list[str]


def wait_scope_is_terminal(
    table: SubagentSessionTableComponent | None,
    session_ids: list[str] | None,
) -> WaitScopeStatus:
    """Evaluate whether every session in the wait scope is terminal.

    A missing session is treated as non-terminal (not silent success) so the
    caller can surface the error instead of resolving the wait prematurely.
    """
    if session_ids is None or not session_ids:
        return WaitScopeStatus(all_terminal=True, missing_session_ids=[])

    if table is None:
        return WaitScopeStatus(
            all_terminal=False,
            missing_session_ids=list(session_ids),
        )

    missing: list[str] = []
    for session_id in session_ids:
        session = table.sessions.get(session_id)
        if session is None:
            missing.append(session_id)
            continue
        if session.status in PENDING_SUBAGENT_STATUSES:
            return WaitScopeStatus(
                all_terminal=False,
                missing_session_ids=missing,
            )

    if missing:
        return WaitScopeStatus(all_terminal=False, missing_session_ids=missing)
    return WaitScopeStatus(all_terminal=True, missing_session_ids=[])


def notification_matches_wait(
    notification: SubagentNotificationRecord,
    component: SubagentWaitComponent,
) -> bool:
    if notification.delivered_at is not None:
        return False
    effective_ids = _effective_session_ids(component)
    if effective_ids is None:
        return True
    return notification.session_id in effective_ids


class SubagentWaitSystem:
    """ECS system that waits for all scoped background subagent sessions to complete.

    Wait-all semantics: the future is only resolved when every session in the wait
    scope reaches a terminal status. On timeout, sessions still running cause the
    deadline to be extended; failed sessions are surfaced to the LLM via a
    ``role="user"`` message so it can call ``subagent_resume`` to restart them.
    """

    def __init__(
        self,
        priority: int = -5,
        resume_callback: ResumeCallback | None = None,
    ) -> None:
        self.priority = priority
        self._resume_callback = resume_callback

    async def process(self, world: World) -> None:
        for entity_id, components in world.query(SubagentWaitComponent):
            wait_component = components[0]
            assert isinstance(wait_component, SubagentWaitComponent)

            try:
                self._maybe_snapshot_session_ids(
                    world, entity_id, wait_component
                )

                if self._all_waited_sessions_terminal(
                    world, entity_id, wait_component
                ):
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
            await self._handle_wait_timeout(world, entity_id, component)
            return

        try:
            await asyncio.wait_for(asyncio.shield(component.future), timeout=timeout)
        except asyncio.TimeoutError:
            await self._handle_wait_timeout(world, entity_id, component)
            return

        self._resolve_wait(world, entity_id, component)

    def _maybe_snapshot_session_ids(
        self,
        world: World,
        entity_id: EntityId,
        component: SubagentWaitComponent,
    ) -> None:
        """Snapshot the wait scope on first processing.

        When ``session_ids`` is ``None``, the effective scope is all currently
        active (queued/running) sessions. When ``session_ids`` is explicitly
        provided, it is copied into ``resolved_session_ids`` for uniform
        downstream logic.
        """
        if component.resolved_session_ids is not None:
            return

        if component.session_ids is not None:
            component.resolved_session_ids = list(component.session_ids)
            return

        table = world.get_component(entity_id, SubagentSessionTableComponent)
        if table is None:
            component.resolved_session_ids = []
            return

        active_ids = [
            sid
            for sid, session in table.sessions.items()
            if session.status in PENDING_SUBAGENT_STATUSES
        ]
        component.resolved_session_ids = active_ids
        logger.info(
            "subagent_wait_session_snapshot",
            entity_id=entity_id,
            snapshotted_sessions=active_ids,
        )

    def _all_waited_sessions_terminal(
        self,
        world: World,
        entity_id: EntityId,
        component: SubagentWaitComponent,
    ) -> bool:
        """Return True when every session in the wait scope is terminal."""
        table = world.get_component(entity_id, SubagentSessionTableComponent)
        effective_ids = _effective_session_ids(component)
        return wait_scope_is_terminal(table, effective_ids).all_terminal

    def _resolve_wait(
        self,
        world: World,
        entity_id: EntityId,
        component: SubagentWaitComponent,
    ) -> None:
        future = component.future
        if isinstance(future, asyncio.Future) and not future.done():
            future.set_result(None)
        self._deliver_unread_notifications(world, entity_id, component)
        world.remove_component(entity_id, SubagentWaitComponent)

    def _deliver_unread_notifications(
        self,
        world: World,
        entity_id: EntityId,
        component: SubagentWaitComponent,
    ) -> None:
        queue = world.get_component(entity_id, SubagentNotificationQueueComponent)
        if queue is None:
            return

        effective_ids = _effective_session_ids(component)
        unread_notifications = [
            notification
            for notification in queue.notifications
            if notification.delivered_at is None
            and (
                effective_ids is None
                or notification.session_id in effective_ids
            )
        ]
        if not unread_notifications:
            return

        conversation = world.get_component(entity_id, ConversationComponent)
        if conversation is None:
            return

        conversation.messages.append(
            Message(
                role="user",
                content=self._format_notification_message(unread_notifications),
            )
        )

        delivered_at = datetime.now(tz=timezone.utc).isoformat()
        for notification in unread_notifications:
            notification.delivered_at = delivered_at

    def _format_notification_message(
        self,
        notifications: list[SubagentNotificationRecord],
    ) -> str:
        lines = ["Background subagent updates:"]
        for notification in notifications:
            lines.append(self._format_notification_line(notification))
        return "\n".join(lines)

    def _format_notification_line(
        self,
        notification: SubagentNotificationRecord,
    ) -> str:
        if notification.terminal_status == "succeeded":
            base = (
                f"- {notification.session_id} succeeded. "
                f'Call subagent_result(session_id="{notification.session_id}") '
                "for the full result"
            )
            if notification.summary is not None:
                base += (
                    f' or subagent_result(session_id="{notification.session_id}", '
                    'read_method="summary") for the cached summary'
                )
            return base + "."

        error_text = notification.error or "Unknown error"
        return (
            f"- {notification.session_id} {notification.terminal_status}: {error_text}. "
            f'Call subagent_result(session_id="{notification.session_id}") '
            "for details."
        )

    async def _handle_wait_timeout(
        self,
        world: World,
        entity_id: EntityId,
        component: SubagentWaitComponent,
    ) -> None:
        """Evaluate session statuses on timeout instead of terminating.

        - All sessions still running -> extend deadline, keep waiting.
        - Any session failed -> surface failures via role="user" message,
          resolve the wait so the LLM can call subagent_resume.
        - All sessions terminal -> resolve the wait normally.
        """
        effective_ids = _effective_session_ids(component)
        if effective_ids is None or not effective_ids:
            self._resolve_wait(world, entity_id, component)
            return

        table = world.get_component(entity_id, SubagentSessionTableComponent)
        has_running = False
        failed_sessions: list[tuple[str, SubagentSessionRecord]] = []

        scope_status = wait_scope_is_terminal(table, effective_ids)
        if scope_status.missing_session_ids:
            self._inject_missing_session_notification(
                world, entity_id, scope_status.missing_session_ids
            )
            self._resolve_wait(world, entity_id, component)
            return

        if table is not None:
            for session_id in effective_ids:
                session = table.sessions.get(session_id)
                if session is None:
                    continue
                if session.status in PENDING_SUBAGENT_STATUSES:
                    has_running = True
                elif session.status in FAILED_SUBAGENT_STATUSES:
                    failed_sessions.append((session_id, session))

        if failed_sessions:
            if (
                component.auto_restart_budget > 0
                and self._resume_callback is not None
            ):
                restarted, exhausted = await self._auto_restart_sessions(
                    world, entity_id, component, failed_sessions
                )
                if restarted and not exhausted:
                    component.started_at = self._utc_now_iso()
                    logger.info(
                        "subagent_wait_auto_restarted",
                        entity_id=entity_id,
                        restarted_count=len(restarted),
                    )
                    return
                if exhausted:
                    failed_sessions = exhausted

            self._inject_timeout_failure_notification(
                world, entity_id, failed_sessions, has_running
            )
            self._resolve_wait(world, entity_id, component)
            return

        if has_running:
            component.started_at = self._utc_now_iso()
            logger.info(
                "subagent_wait_extending",
                entity_id=entity_id,
                timeout=component.timeout,
            )
            return

        self._resolve_wait(world, entity_id, component)

    async def _auto_restart_sessions(
        self,
        world: World,
        entity_id: EntityId,
        component: SubagentWaitComponent,
        failed_sessions: list[tuple[str, SubagentSessionRecord]],
    ) -> tuple[
        list[tuple[str, str]],
        list[tuple[str, SubagentSessionRecord]],
    ]:
        """Try to auto-restart failed sessions within the configured budget.

        Returns ``(restarted, exhausted)`` where *restarted* is a list of
        ``(original_session_id, new_session_id)`` pairs and *exhausted* is the
        list of sessions that could not be restarted.
        """
        restarted: list[tuple[str, str]] = []
        exhausted: list[tuple[str, SubagentSessionRecord]] = []

        for session_id, session in failed_sessions:
            count = component.restart_counts.get(session_id, 0)
            if count >= component.auto_restart_budget:
                exhausted.append((session_id, session))
                continue

            assert self._resume_callback is not None
            try:
                new_session_id = await self._resume_callback(
                    session_id, entity_id, world
                )
            except Exception as exc:
                logger.error(
                    "subagent_auto_restart_failed",
                    entity_id=entity_id,
                    session_id=session_id,
                    exception=str(exc),
                )
                exhausted.append((session_id, session))
                continue

            new_count = count + 1
            component.restart_counts[session_id] = new_count
            component.restart_counts[new_session_id] = new_count
            if component.resolved_session_ids is not None:
                component.resolved_session_ids = [
                    new_session_id if sid == session_id else sid
                    for sid in component.resolved_session_ids
                ]
            restarted.append((session_id, new_session_id))
            logger.info(
                "subagent_auto_restart",
                entity_id=entity_id,
                original_session_id=session_id,
                new_session_id=new_session_id,
                restart_count=new_count,
            )

        return restarted, exhausted

    def _inject_missing_session_notification(
        self,
        world: World,
        entity_id: EntityId,
        missing_session_ids: list[str],
    ) -> None:
        """Inject a role="user" message for sessions not found in the table."""
        conversation = world.get_component(entity_id, ConversationComponent)
        if conversation is None:
            return

        lines = [
            "Subagent wait could not find the following session(s):"
        ]
        for session_id in missing_session_ids:
            lines.append(f"- {session_id}")
        lines.append("")
        lines.append(
            "Call subagent_status() to list all known sessions, "
            "then retry with the correct session ID."
        )

        conversation.messages.append(
            Message(role="user", content="\n".join(lines))
        )
        logger.warning(
            "subagent_wait_missing_sessions",
            entity_id=entity_id,
            missing_count=len(missing_session_ids),
        )

    def _inject_timeout_failure_notification(
        self,
        world: World,
        entity_id: EntityId,
        failed_sessions: list[tuple[str, SubagentSessionRecord]],
        has_running: bool,
    ) -> None:
        """Inject a role="user" message describing failed sessions for LLM action."""
        conversation = world.get_component(entity_id, ConversationComponent)
        if conversation is None:
            return

        lines = ["Subagent wait timed out. The following sessions need attention:"]
        for session_id, session in failed_sessions:
            error_text = session.error or "no error details"
            lines.append(
                f"- {session_id} ({session.category}): "
                f"{session.status} - {error_text}"
            )
            lines.append(
                f'  Call subagent_resume(session_id="{session_id}") '
                "to restart this subagent."
            )

        if has_running:
            lines.append("")
            lines.append(
                "Some sessions are still running. "
                "Call subagent_wait() again to wait for them "
                "after handling failures."
            )

        conversation.messages.append(
            Message(role="user", content="\n".join(lines))
        )
        logger.info(
            "subagent_wait_timeout_failures_injected",
            entity_id=entity_id,
            failed_count=len(failed_sessions),
            has_running=has_running,
        )

    def _remaining_timeout(self, component: SubagentWaitComponent) -> float | None:
        if component.timeout is None:
            return None
        if component.started_at is None:
            return component.timeout

        started_at = datetime.fromisoformat(component.started_at.replace("Z", "+00:00"))
        elapsed = (datetime.now(timezone.utc) - started_at).total_seconds()
        return component.timeout - elapsed

    def _utc_now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


__all__ = [
    "ResumeCallback",
    "SubagentCompactionState",
    "SubagentWaitSystem",
    "WaitScopeStatus",
    "build_subagent_compaction_state",
    "notification_matches_wait",
    "wait_scope_is_terminal",
]
