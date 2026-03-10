"""Runtime session manager for async subagent handles.

This module provides in-memory management of async task handles and lifecycle
state transitions for background subagent sessions. It keeps asyncio.Task
handles separate from serializable ECS components.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any, Dict, Optional

from ecs_agent.logging import get_logger
from ecs_agent.types import SubagentLifecycleStatus, SubagentSessionRecord

logger = get_logger(__name__)


class SubagentRuntimeManager:
    """Manages async task handles and lifecycle transitions for subagent sessions.

    This manager maintains the runtime state (asyncio.Task handles) separately from
    the serializable metadata (SubagentSessionRecord in SubagentSessionTableComponent).

    Thread-safe for concurrent access via asyncio.Lock.
    """

    def __init__(self) -> None:
        """Initialize the runtime manager with empty state."""
        self._sessions: Dict[str, SubagentSessionRecord] = {}
        self._tasks: Dict[str, asyncio.Task[Any]] = {}
        self._lock = asyncio.Lock()

    def create_session(self) -> str:
        """Generate a unique session ID.

        Returns:
            A 16-character hexadecimal session ID.
        """
        return uuid.uuid4().hex[:16]

    async def register_task(
        self,
        session_id: str,
        task: asyncio.Task[Any],
        metadata: SubagentSessionRecord,
    ) -> None:
        """Register an async task with its session metadata.

        Args:
            session_id: Unique session identifier
            task: The asyncio.Task handle for the background execution
            metadata: Serializable session metadata
        """
        async with self._lock:
            self._sessions[session_id] = metadata
            self._tasks[session_id] = task
            logger.info(
                "session_registered",
                session_id=session_id,
                category=metadata.category,
                status=metadata.status,
            )

    async def get_session(self, session_id: str) -> Optional[SubagentSessionRecord]:
        """Retrieve session metadata by ID.

        Args:
            session_id: Session identifier to query

        Returns:
            SubagentSessionRecord if found, None otherwise
        """
        async with self._lock:
            return self._sessions.get(session_id)

    async def update_status(
        self,
        session_id: str,
        status: SubagentLifecycleStatus,
    ) -> None:
        """Update the lifecycle status of a session.

        Args:
            session_id: Session identifier
            status: New lifecycle status

        Raises:
            ValueError: If session not found
        """
        async with self._lock:
            metadata = self._sessions.get(session_id)
            if metadata is None:
                raise ValueError(f"Session not found: {session_id}")

            old_status = metadata.status
            metadata.status = status

            logger.info(
                "session_status_updated",
                session_id=session_id,
                old_status=old_status,
                new_status=status,
            )

    async def cancel_session(self, session_id: str) -> None:
        """Cancel a running session and mark it as Cancelled.

        Args:
            session_id: Session identifier to cancel
        """
        async with self._lock:
            task = self._tasks.get(session_id)
            if task and not task.done():
                task.cancel()
                logger.info("session_task_cancelled", session_id=session_id)

            metadata = self._sessions.get(session_id)
            if metadata:
                metadata.status = "Cancelled"
                logger.info("session_status_cancelled", session_id=session_id)

    async def cleanup(self, session_id: str) -> None:
        """Remove a completed or cancelled session from the runtime map.

        Args:
            session_id: Session identifier to clean up
        """
        async with self._lock:
            if session_id in self._tasks:
                del self._tasks[session_id]
            if session_id in self._sessions:
                del self._sessions[session_id]

            logger.info("session_cleaned_up", session_id=session_id)

    async def get_all_sessions(self) -> Dict[str, SubagentSessionRecord]:
        """Get all active session metadata.

        Returns:
            Dictionary mapping session_id to SubagentSessionRecord
        """
        async with self._lock:
            return dict(self._sessions)
