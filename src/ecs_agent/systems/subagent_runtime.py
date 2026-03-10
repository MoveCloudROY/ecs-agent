"""Runtime session manager for subagent background execution."""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from ecs_agent.core.world import World
from ecs_agent.logging import get_logger
from ecs_agent.types import EntityId

logger = get_logger(__name__)


class SubagentLifecycleStatus(str, Enum):
    """Lifecycle status for delegated subagent sessions."""

    IDLE = "Idle"
    WORKING = "Working"
    DEAD = "Dead"
    TIMEOUT = "Timeout"
    CANCELLED = "Cancelled"


@dataclass(slots=True)
class SubagentSessionRecord:
    """Persistent metadata for one runtime subagent session."""

    session_id: str
    parent_entity_id: EntityId
    child_entity_id: EntityId | None
    subagent_name: str
    task: str
    status: SubagentLifecycleStatus
    created_at: str
    updated_at: str
    deadline_at: str | None
    correlation_id: str
    traceparent: str
    timeout_seconds: float | None = None
    result: str | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SubagentSessionTableComponent:
    """Serializable table of active and historical subagent sessions."""

    sessions: dict[str, SubagentSessionRecord] = field(default_factory=dict)


class SubagentRuntimeSessionManager:
    """Concurrency-safe runtime manager for async subagent task handles."""

    def __init__(self, world: World, table_entity_id: EntityId | None = None) -> None:
        self._world = world
        self._lock = asyncio.Lock()
        self._table_entity_id = table_entity_id
        self._task_handles: dict[str, asyncio.Task[Any]] = {}

    def create_session(self) -> str:
        return uuid.uuid4().hex[:16]

    async def register_task(
        self,
        session_id: str,
        task: asyncio.Task[Any],
        metadata: SubagentSessionRecord,
    ) -> None:
        async with self._lock:
            table = self._get_or_create_table_component()
            normalized = replace(
                metadata,
                session_id=session_id,
                updated_at=self._now_iso(),
            )
            table.sessions[session_id] = normalized
            self._task_handles[session_id] = task

    async def update_status(
        self,
        session_id: str,
        status: SubagentLifecycleStatus,
    ) -> None:
        async with self._lock:
            table = self._get_or_create_table_component()
            existing = table.sessions.get(session_id)
            if existing is None:
                raise KeyError(f"Unknown session_id: {session_id}")

            if not self._is_valid_transition(existing.status, status):
                raise ValueError(
                    f"Invalid lifecycle transition {existing.status.value} -> {status.value}"
                )

            table.sessions[session_id] = replace(
                existing,
                status=status,
                updated_at=self._now_iso(),
            )

    async def cancel_session(self, session_id: str) -> None:
        task: asyncio.Task[Any] | None
        async with self._lock:
            table = self._get_or_create_table_component()
            existing = table.sessions.get(session_id)
            if existing is None:
                return

            table.sessions[session_id] = replace(
                existing,
                status=SubagentLifecycleStatus.CANCELLED,
                updated_at=self._now_iso(),
            )
            task = self._task_handles.pop(session_id, None)

        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                logger.debug("subagent_runtime_task_cancelled", session_id=session_id)

    async def get_session(self, session_id: str) -> SubagentSessionRecord | None:
        async with self._lock:
            table = self._get_or_create_table_component()
            record = table.sessions.get(session_id)
            if record is None:
                return None
            return replace(record)

    async def cleanup(self, session_id: str) -> None:
        async with self._lock:
            table = self._get_or_create_table_component()
            table.sessions.pop(session_id, None)
            self._task_handles.pop(session_id, None)

    def _get_or_create_table_component(self) -> SubagentSessionTableComponent:
        if self._table_entity_id is not None:
            existing = self._world.get_component(
                self._table_entity_id,
                SubagentSessionTableComponent,
            )
            if existing is not None:
                assert isinstance(existing, SubagentSessionTableComponent)
                return existing

        session_table_entities = list(self._world.query(SubagentSessionTableComponent))
        if session_table_entities:
            entity_id, (table,) = session_table_entities[0]
            assert isinstance(table, SubagentSessionTableComponent)
            self._table_entity_id = entity_id
            return table

        entity_id = self._world.create_entity()
        table = SubagentSessionTableComponent()
        self._world.add_component(entity_id, table)
        self._table_entity_id = entity_id
        return table

    def _is_valid_transition(
        self,
        current: SubagentLifecycleStatus,
        next_status: SubagentLifecycleStatus,
    ) -> bool:
        if current in {
            SubagentLifecycleStatus.DEAD,
            SubagentLifecycleStatus.TIMEOUT,
            SubagentLifecycleStatus.CANCELLED,
        }:
            return False

        if current is SubagentLifecycleStatus.IDLE:
            return next_status is SubagentLifecycleStatus.WORKING

        if current is SubagentLifecycleStatus.WORKING:
            return next_status in {
                SubagentLifecycleStatus.IDLE,
                SubagentLifecycleStatus.DEAD,
                SubagentLifecycleStatus.TIMEOUT,
                SubagentLifecycleStatus.CANCELLED,
            }

        return False

    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()


__all__ = [
    "SubagentLifecycleStatus",
    "SubagentRuntimeSessionManager",
    "SubagentSessionRecord",
    "SubagentSessionTableComponent",
]
