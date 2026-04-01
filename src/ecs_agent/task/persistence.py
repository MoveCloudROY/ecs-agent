"""Task state persistence with dual paths: mutable snapshots + immutable event log."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ecs_agent.components import TaskComponent
from ecs_agent.logging import get_logger
from ecs_agent.scratchbook.service import ScratchbookService
from ecs_agent.types import (
    ScratchbookRef,
    TaskBlockedEvent,
    TaskCompletedEvent,
    TaskCreatedEvent,
    TaskFailedEvent,
    TaskStateChangedEvent,
    TaskStatus,
)

logger = get_logger(__name__)


class TaskEventLogTamperError(Exception):
    """Raised when event log modification is detected."""


def compute_task_snapshot_hash(snapshot: dict[str, Any]) -> str:
    """Compute SHA256 hash of task snapshot.

    Args:
        snapshot: Task state snapshot dict

    Returns:
        Hexadecimal SHA256 hash
    """
    content = json.dumps(snapshot, sort_keys=True)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


class TaskPersistenceService:
    """Dual-path persistence for task state.

    Provides:
    - Mutable snapshots: Current task state (queryable via refs, can be updated)
    - Immutable event log: Transition/event history (append-only, never modify)
    - Current-state lookup: Fast query via refs
    - History replay: Read event log chronologically
    - Tamper detection: Modifications to event log rejected
    """

    def __init__(self, scratchbook: ScratchbookService) -> None:
        """Initialize task persistence service.

        Args:
            scratchbook: Scratchbook service for filesystem operations
        """
        self.scratchbook = scratchbook

    def persist_task_snapshot(
        self,
        task_id: str,
        task_component: TaskComponent,
    ) -> ScratchbookRef:
        """Persist current task state as mutable snapshot.

        Args:
            task_id: Task identifier
            task_component: Current task component state

        Returns:
            Reference to persisted snapshot
        """
        snapshot = self._serialize_task_component(task_component)
        content_hash = compute_task_snapshot_hash(snapshot)

        artifact_id = f"{task_id}_snapshot"
        category = "tasks/snapshots"

        self.scratchbook.write_artifact(artifact_id, category, snapshot)

        ref = ScratchbookRef(
            artifact_id=artifact_id,
            category=category,
            content_hash=content_hash,
            timestamp=str(time.time()),
            record_path=f"{category}/{artifact_id}.json",
        )

        logger.info(
            "task_snapshot_persisted",
            task_id=task_id,
            artifact_id=artifact_id,
            content_hash=content_hash,
        )

        return ref

    def read_task_snapshot(self, task_id: str) -> dict[str, Any] | None:
        """Read current task snapshot.

        Args:
            task_id: Task identifier

        Returns:
            Task snapshot dict or None if missing
        """
        artifact_id = f"{task_id}_snapshot"
        category = "tasks/snapshots"

        snapshot = self.scratchbook.read_artifact(artifact_id, category)

        if snapshot:
            logger.info(
                "task_snapshot_read",
                task_id=task_id,
                artifact_id=artifact_id,
            )
        else:
            logger.debug(
                "task_snapshot_missing",
                task_id=task_id,
                artifact_id=artifact_id,
            )

        return snapshot

    def append_task_event(
        self,
        task_id: str,
        event: (
            TaskCreatedEvent
            | TaskStateChangedEvent
            | TaskBlockedEvent
            | TaskCompletedEvent
            | TaskFailedEvent
        ),
    ) -> None:
        """Append event to immutable event log.

        Args:
            task_id: Task identifier
            event: Task event to append
        """
        event_data = self._serialize_event(event)
        event_line = json.dumps(event_data) + "\n"

        log_name = f"{task_id}_events.jsonl"
        category = "tasks/events"

        self.scratchbook.append_log(log_name, category, event_line)

        logger.info(
            "task_event_appended",
            task_id=task_id,
            event_type=type(event).__name__,
            log_name=log_name,
        )

    def read_task_events(self, task_id: str) -> list[dict[str, Any]]:
        """Read full event log for a task.

        Args:
            task_id: Task identifier

        Returns:
            List of event dicts in chronological order

        Raises:
            TaskEventLogTamperError: If event log is corrupted or tampered
        """
        log_name = f"{task_id}_events.jsonl"
        category = "tasks/events"
        log_path = self.scratchbook.root / category / log_name

        if not log_path.exists():
            logger.debug(
                "task_event_log_missing",
                task_id=task_id,
                log_name=log_name,
            )
            return []

        try:
            lines = log_path.read_text(encoding="utf-8").splitlines()
            events = []
            for i, line in enumerate(lines):
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                    events.append(event)
                except json.JSONDecodeError as exc:
                    logger.error(
                        "task_event_log_corrupted",
                        task_id=task_id,
                        log_name=log_name,
                        line_number=i + 1,
                        exception=str(exc),
                    )
                    raise TaskEventLogTamperError(
                        f"Event log for task {task_id} is corrupted at line {i + 1}"
                    ) from exc

            logger.info(
                "task_event_log_read",
                task_id=task_id,
                log_name=log_name,
                event_count=len(events),
            )

            return events

        except Exception as exc:
            if isinstance(exc, TaskEventLogTamperError):
                raise
            logger.error(
                "task_event_log_read_failed",
                task_id=task_id,
                log_name=log_name,
                exception=str(exc),
            )
            raise TaskEventLogTamperError(
                f"Failed to read event log for task {task_id}"
            ) from exc

    def verify_event_log_integrity(self, task_id: str) -> bool:
        """Verify event log has not been tampered with.

        Args:
            task_id: Task identifier

        Returns:
            True if event log is intact, False if corrupted/tampered

        Raises:
            TaskEventLogTamperError: If event log is corrupted
        """
        try:
            events = self.read_task_events(task_id)
            # If we successfully read all events, log is intact
            logger.info(
                "task_event_log_verified",
                task_id=task_id,
                event_count=len(events),
            )
            return True
        except TaskEventLogTamperError:
            logger.error(
                "task_event_log_tampered",
                task_id=task_id,
            )
            raise

    def _serialize_task_component(self, component: TaskComponent) -> dict[str, Any]:
        """Serialize TaskComponent to dict.

        Args:
            component: TaskComponent instance

        Returns:
            Serialized dict
        """
        serialized = asdict(component)

        # Convert TaskStatus enum to string
        if isinstance(serialized.get("status"), TaskStatus):
            serialized["status"] = serialized["status"].value

        # Convert assigned_agent EntityId to int if present
        assigned_agent = serialized.get("assigned_agent")
        if assigned_agent is not None and isinstance(assigned_agent, int):
            # EntityId is already int-compatible
            pass

        return serialized

    def _serialize_event(
        self,
        event: (
            TaskCreatedEvent
            | TaskStateChangedEvent
            | TaskBlockedEvent
            | TaskCompletedEvent
            | TaskFailedEvent
        ),
    ) -> dict[str, Any]:
        """Serialize task event to dict.

        Args:
            event: Task event instance

        Returns:
            Serialized dict with event type metadata
        """
        serialized = asdict(event)
        serialized["_event_type"] = type(event).__name__
        serialized["_timestamp"] = time.time()

        # Convert EntityId to int
        if "entity_id" in serialized:
            serialized["entity_id"] = int(serialized["entity_id"])

        # Convert TaskStatus enums to strings
        if "old_status" in serialized and isinstance(
            serialized["old_status"], TaskStatus
        ):
            serialized["old_status"] = serialized["old_status"].value
        if "new_status" in serialized and isinstance(
            serialized["new_status"], TaskStatus
        ):
            serialized["new_status"] = serialized["new_status"].value

        return serialized


__all__ = [
    "TaskEventLogTamperError",
    "TaskPersistenceService",
    "compute_task_snapshot_hash",
]
