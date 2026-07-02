"""Process-global runtime support for background subagent sessions."""

from __future__ import annotations

import asyncio
import uuid
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from ecs_agent.logging import get_logger
from ecs_agent.types import (
    SubagentLifecycleStatus,
    SubagentSessionRecord,
    render_subagent_session_reminder_table,
    validate_subagent_lifecycle_transition,
)

if TYPE_CHECKING:
    from ecs_agent.core.world import World
    from ecs_agent.types import EntityId

logger = get_logger(__name__)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(slots=True)
class QueuedSession:
    session_id: str
    coroutine_factory: Callable[[], Awaitable[None]]


class BackgroundScheduler:
    def __init__(self, max_concurrency: int = 5) -> None:
        if max_concurrency <= 0:
            raise ValueError("max_concurrency must be greater than 0")

        self.max_concurrency = max_concurrency
        self.pending_queue: deque[QueuedSession] = deque()
        self.active_count = 0
        self._running_session_ids: set[str] = set()
        self._lock = asyncio.Lock()

    async def enqueue(
        self,
        session_id: str,
        coroutine_factory: Callable[[], Awaitable[None]],
    ) -> None:
        async with self._lock:
            self.pending_queue.append(
                QueuedSession(
                    session_id=session_id,
                    coroutine_factory=coroutine_factory,
                )
            )

    async def reserve_admissions(self) -> list[QueuedSession]:
        async with self._lock:
            admitted: list[QueuedSession] = []
            while self.pending_queue and self.active_count < self.max_concurrency:
                queued_session = self.pending_queue.popleft()
                self._running_session_ids.add(queued_session.session_id)
                self.active_count += 1
                admitted.append(queued_session)

            return admitted

    async def reserve_running_session(self, session_id: str) -> None:
        async with self._lock:
            if session_id in self._running_session_ids:
                return

            if self.active_count >= self.max_concurrency:
                raise RuntimeError(
                    "background scheduler capacity exceeded while registering running task"
                )

            self._running_session_ids.add(session_id)
            self.active_count += 1

    async def remove_queued(self, session_id: str) -> bool:
        async with self._lock:
            original_length = len(self.pending_queue)
            self.pending_queue = deque(
                item for item in self.pending_queue if item.session_id != session_id
            )
            return len(self.pending_queue) != original_length

    async def get_queue_position(self, session_id: str) -> int | None:
        async with self._lock:
            for index, queued_session in enumerate(self.pending_queue):
                if queued_session.session_id == session_id:
                    return index

            return None

    async def release_slot(self, session_id: str) -> bool:
        async with self._lock:
            if session_id not in self._running_session_ids:
                return False

            self._running_session_ids.remove(session_id)
            self.active_count -= 1
            return True


_GLOBAL_SCHEDULER: BackgroundScheduler | None = None


def reset_global_scheduler() -> None:
    """Reset the global scheduler singleton. Use in test teardown to avoid state leakage."""
    global _GLOBAL_SCHEDULER
    _GLOBAL_SCHEDULER = None


def get_global_scheduler(max_background_concurrency: int = 5) -> BackgroundScheduler:
    global _GLOBAL_SCHEDULER

    if _GLOBAL_SCHEDULER is None:
        _GLOBAL_SCHEDULER = BackgroundScheduler(
            max_concurrency=max_background_concurrency
        )
        return _GLOBAL_SCHEDULER

    if _GLOBAL_SCHEDULER.max_concurrency != max_background_concurrency:
        raise ValueError(
            "Conflicting max_background_concurrency for global scheduler: "
            f"existing={_GLOBAL_SCHEDULER.max_concurrency}, "
            f"requested={max_background_concurrency}"
        )

    return _GLOBAL_SCHEDULER


_TERMINAL_STATUSES: frozenset[SubagentLifecycleStatus] = frozenset(
    {"succeeded", "failed", "timed_out", "cancelled"}
)


class SubagentRuntimeManager:
    def __init__(self, max_background_concurrency: int = 5) -> None:
        self._sessions: dict[str, SubagentSessionRecord] = {}
        self._tasks: dict[str, asyncio.Task[Any]] = {}
        # Per-session sticky terminal signal. subagent_result awaits this instead of
        # polling; set on any terminal transition. Sticky (asyncio.Event), so a
        # transition that lands before a waiter arrives still wakes it.
        self._session_events: dict[str, asyncio.Event] = {}
        self._lock = asyncio.Lock()
        self._scheduler = get_global_scheduler(max_background_concurrency)

    def create_session(self) -> str:
        return uuid.uuid4().hex[:16]

    def get_or_create_session_event(self, session_id: str) -> asyncio.Event:
        """Return the per-session terminal Event, creating an unset one if needed."""
        return self._session_events.setdefault(session_id, asyncio.Event())

    def _signal_session_terminal(self, session_id: str) -> None:
        """Mark a session's terminal Event as set (create-and-set if absent)."""
        self._session_events.setdefault(session_id, asyncio.Event()).set()

    async def enqueue_session(
        self,
        session_id: str,
        metadata: SubagentSessionRecord,
        coroutine_factory: Callable[[], Awaitable[None]],
    ) -> None:
        async with self._lock:
            self._sessions[session_id] = metadata

        await self._scheduler.enqueue(session_id, coroutine_factory)
        logger.info(
            "session_enqueued",
            session_id=session_id,
            category=metadata.category,
            status=metadata.status,
        )
        await self._try_admit()

    async def restore_session_metadata(self, metadata: SubagentSessionRecord) -> None:
        async with self._lock:
            self._sessions[metadata.session_id] = metadata

        if metadata.status in _TERMINAL_STATUSES:
            self._signal_session_terminal(metadata.session_id)

        logger.info(
            "session_metadata_restored",
            session_id=metadata.session_id,
            category=metadata.category,
            status=metadata.status,
        )

    async def _run_session(
        self,
        session_id: str,
        coroutine_factory: Callable[[], Awaitable[None]],
    ) -> None:
        try:
            await coroutine_factory()
        finally:
            await self.release_slot(session_id)

    async def _try_admit(self) -> None:
        admitted = await self._scheduler.reserve_admissions()
        for queued_session in admitted:
            task = asyncio.create_task(
                self._run_session(
                    queued_session.session_id,
                    queued_session.coroutine_factory,
                )
            )
            async with self._lock:
                self._tasks[queued_session.session_id] = task

            logger.info(
                "session_admitted",
                session_id=queued_session.session_id,
                active_count=self._scheduler.active_count,
                max_concurrency=self._scheduler.max_concurrency,
            )

    async def release_slot(self, session_id: str) -> None:
        released = await self._scheduler.release_slot(session_id)
        if not released:
            return

        logger.info(
            "session_slot_released",
            session_id=session_id,
            active_count=self._scheduler.active_count,
            max_concurrency=self._scheduler.max_concurrency,
        )
        await self._try_admit()

    async def register_task(
        self,
        session_id: str,
        task: asyncio.Task[Any],
        metadata: SubagentSessionRecord,
    ) -> None:
        async with self._lock:
            self._sessions[session_id] = metadata
            self._tasks[session_id] = task

        await self._scheduler.reserve_running_session(session_id)
        logger.info(
            "session_registered",
            session_id=session_id,
            category=metadata.category,
            status=metadata.status,
        )

        if task.done():
            await self.release_slot(session_id)
            return

        def _release_when_done(completed_task: asyncio.Task[Any]) -> None:
            del completed_task
            asyncio.create_task(self.release_slot(session_id))

        task.add_done_callback(_release_when_done)

    async def get_session(self, session_id: str) -> SubagentSessionRecord | None:
        async with self._lock:
            return self._sessions.get(session_id)

    async def update_status(
        self,
        session_id: str,
        status: SubagentLifecycleStatus,
    ) -> None:
        async with self._lock:
            metadata = self._sessions.get(session_id)
            if metadata is None:
                raise ValueError(f"Session not found: {session_id}")

            old_status = metadata.status
            metadata.status = status

        if status in _TERMINAL_STATUSES:
            self._signal_session_terminal(session_id)

        logger.info(
            "session_status_updated",
            session_id=session_id,
            old_status=old_status,
            new_status=status,
        )

    async def cancel_session(self, session_id: str) -> None:
        removed_from_queue = await self._scheduler.remove_queued(session_id)
        if removed_from_queue:
            async with self._lock:
                metadata = self._sessions.get(session_id)
                if metadata is not None:
                    if metadata.status != "queued":
                        raise ValueError(
                            "Queued cancellation requires queued session status"
                        )
                    validate_subagent_lifecycle_transition(
                        metadata.status,
                        "cancelled",
                    )
                    metadata.status = "cancelled"
                    metadata.updated_at = _utc_now_iso()
                    metadata.finished_at = metadata.updated_at

            self._signal_session_terminal(session_id)
            logger.info("session_cancelled_while_queued", session_id=session_id)
            return

        task: asyncio.Task[Any] | None
        async with self._lock:
            metadata = self._sessions.get(session_id)
            task = self._tasks.get(session_id)

            if metadata is not None and metadata.status in ("queued", "running"):
                validate_subagent_lifecycle_transition(metadata.status, "cancelled")
                metadata.status = "cancelled"
                metadata.updated_at = _utc_now_iso()
                if metadata.finished_at is None:
                    metadata.finished_at = metadata.updated_at

        self._signal_session_terminal(session_id)

        if task is not None and not task.done():
            task.cancel()
            logger.info("session_task_cancelled", session_id=session_id)

        await self.release_slot(session_id)

    async def cleanup(self, session_id: str) -> None:
        await self.release_slot(session_id)

        async with self._lock:
            self._tasks.pop(session_id, None)
            self._sessions.pop(session_id, None)
            self._session_events.pop(session_id, None)

        logger.info("session_cleaned_up", session_id=session_id)

    async def get_all_sessions(self) -> dict[str, SubagentSessionRecord]:
        async with self._lock:
            return dict(self._sessions)

    async def get_task(self, session_id: str) -> asyncio.Task[Any] | None:
        async with self._lock:
            return self._tasks.get(session_id)

    async def get_queue_position(self, session_id: str) -> int | None:
        return await self._scheduler.get_queue_position(session_id)

    async def update_timeout(self, session_id: str, error: str) -> None:
        async with self._lock:
            metadata = self._sessions.get(session_id)
            if metadata is None:
                logger.warning("timeout_update_missing_session", session_id=session_id)
                return

            validate_subagent_lifecycle_transition(metadata.status, "timed_out")
            old_status = metadata.status
            metadata.status = "timed_out"
            metadata.error = error
            metadata.updated_at = _utc_now_iso()

        self._signal_session_terminal(session_id)

        logger.info(
            "session_timeout_updated",
            session_id=session_id,
            old_status=old_status,
            error=error,
        )

    async def sync_to_component(self, world: "World", entity_id: "EntityId") -> None:
        from ecs_agent.components.definitions import SubagentSessionTableComponent

        async with self._lock:
            table = world.get_component(entity_id, SubagentSessionTableComponent)
            if table is None:
                logger.warning(
                    "sync_missing_table",
                    entity_id=entity_id,
                    message="SubagentSessionTableComponent not found",
                )
                return

            table.sessions = {
                sid: replace(metadata) for sid, metadata in self._sessions.items()
            }

        logger.info(
            "sessions_synced_to_component",
            entity_id=entity_id,
            session_count=len(table.sessions),
        )


# render_subagent_session_reminder_table is imported from ecs_agent.types at the top
# and re-exported below to preserve the historical import path (Task 11 — the divergent
# duplicate that lived here was removed in favor of the canonical implementation).

__all__ = [
    "BackgroundScheduler",
    "QueuedSession",
    "SubagentRuntimeManager",
    "get_global_scheduler",
    "reset_global_scheduler",
    "render_subagent_session_reminder_table",
]
