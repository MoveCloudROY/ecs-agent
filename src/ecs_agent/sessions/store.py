"""Session storage protocol and in-memory implementation."""

from collections.abc import Iterator
from copy import deepcopy
from dataclasses import asdict, replace
from threading import RLock
from typing import Any, Protocol

from ecs_agent.sessions.types import RecordDraft, SessionInfo, SessionRecord


class SessionStore(Protocol):
    """Synchronous commits leave no async cancellation gap before memory updates.

    Writers must claim a session before appending. An explicit branch operation
    can select an older parent; ordinary appends must extend the current head.
    Returned values must never alias authoritative mutable data.
    """

    def create_session(self, session: SessionInfo) -> None: ...
    def get_session(self, session_id: str) -> SessionInfo: ...
    def sessions(self) -> list[SessionInfo]: ...
    def claim_writer(self, session_id: str, writer_id: str) -> None: ...
    def release_writer(self, session_id: str, writer_id: str) -> None: ...
    def append(self, draft: RecordDraft, *, writer_id: str, branch: bool = False, select_head: bool = True) -> SessionRecord: ...
    def get_record(self, record_id: str) -> SessionRecord: ...
    def records(self, session_id: str, *, after_seq: int = 0) -> Iterator[SessionRecord]: ...
    def path(self, session_id: str, head_record_id: str) -> Iterator[SessionRecord]: ...
    def save_snapshot(self, snapshot_id: str, payload: dict[str, Any]) -> None: ...
    def load_snapshot(self, snapshot_id: str) -> dict[str, Any]: ...
    def close(self) -> None: ...


def validate_session(session: SessionInfo, parent: SessionInfo | None) -> None:
    """Validate host-provided task-tree relationships before insertion."""
    if not session.session_id or not session.workspace_id or not session.agent_name:
        raise ValueError("Session identity, workspace and agent name must be nonempty")
    if session.head_record_id is not None:
        raise ValueError("New session must have an empty head")
    if parent is None:
        if session.root_session_id != session.session_id:
            raise ValueError("A root session must reference its own root identity")
    elif session.workspace_id != parent.workspace_id:
        raise ValueError("Child session must share its parent's workspace")
    elif session.root_session_id != parent.root_session_id:
        raise ValueError("Child session must share its parent's root")


def validate_append(
    session: SessionInfo, draft: RecordDraft, parent: SessionRecord | None, *, branch: bool,
) -> None:
    """Validate parent membership and expected-head concurrency control."""
    if parent is not None and parent.session_id != draft.session_id:
        raise ValueError("Parent record belongs to a different session")
    if not branch and draft.parent_record_id != session.head_record_id:
        raise ValueError("Session head changed; explicit branch required for an older parent")


def updated_session(session: SessionInfo, record: SessionRecord) -> SessionInfo:
    """Project durable record lifecycle fields into the catalog."""
    status = session.status
    if record.kind in {"run_started", "run_finished", "queued"}:
        status = str(record.payload.get("status", status))
    summary = session.task_summary
    if not summary and record.payload.get("role") == "user":
        summary = str(record.payload.get("content", ""))[:300]
    return replace(
        session, head_record_id=record.record_id, updated_at=record.timestamp,
        status=status, task_summary=summary,
    )


def is_same_draft(record: SessionRecord, draft: RecordDraft) -> bool:
    """Compare an idempotent retry without the store-assigned sequence."""
    payload = asdict(record)
    payload.pop("seq")
    return payload == asdict(draft)


class InMemorySessionStore:
    """Thread-safe ephemeral backend implementing the durable-store contract."""

    def __init__(self) -> None:
        self._sessions: dict[str, SessionInfo] = {}
        self._records: dict[str, SessionRecord] = {}
        self._order: dict[str, list[str]] = {}
        self._writers: dict[str, str] = {}
        self._snapshots: dict[str, dict[str, Any]] = {}
        self._lock = RLock()

    def create_session(self, session: SessionInfo) -> None:
        with self._lock:
            parent = self.get_session(session.parent_session_id) if session.parent_session_id else None
            validate_session(session, parent)
            if session.session_id in self._sessions:
                if self._sessions[session.session_id] == session:
                    return
                raise ValueError("session_id already exists")
            self._sessions[session.session_id] = session
            self._order[session.session_id] = []

    def get_session(self, session_id: str) -> SessionInfo:
        with self._lock:
            if session_id not in self._sessions:
                raise FileNotFoundError("Session is unavailable")
            return self._sessions[session_id]

    def sessions(self) -> list[SessionInfo]:
        with self._lock:
            return sorted(self._sessions.values(), key=lambda s: (s.created_at, s.session_id))

    def claim_writer(self, session_id: str, writer_id: str) -> None:
        with self._lock:
            self.get_session(session_id)
            owner = self._writers.get(session_id)
            if owner is not None and owner != writer_id:
                raise RuntimeError("Session already has an active writer")
            self._writers[session_id] = writer_id

    def release_writer(self, session_id: str, writer_id: str) -> None:
        with self._lock:
            if self._writers.get(session_id) == writer_id:
                del self._writers[session_id]

    def append(self, draft: RecordDraft, *, writer_id: str, branch: bool = False, select_head: bool = True) -> SessionRecord:
        with self._lock:
            if self._writers.get(draft.session_id) != writer_id:
                raise RuntimeError("Session writer is not owned by this caller")
            existing = self._records.get(draft.record_id)
            if existing is not None:
                if not is_same_draft(existing, draft):
                    raise ValueError("record_id reused with different content")
                return deepcopy(existing)
            session = self.get_session(draft.session_id)
            parent = self.get_record(draft.parent_record_id) if draft.parent_record_id else None
            validate_append(session, draft, parent, branch=branch)
            record = SessionRecord(**asdict(draft), seq=len(self._order[draft.session_id]) + 1)
            self._records[record.record_id] = record
            self._order[record.session_id].append(record.record_id)
            if select_head:
                self._sessions[session.session_id] = updated_session(session, record)
            return deepcopy(record)

    def get_record(self, record_id: str) -> SessionRecord:
        with self._lock:
            if record_id not in self._records:
                raise FileNotFoundError("Record is unavailable")
            return deepcopy(self._records[record_id])

    def records(self, session_id: str, *, after_seq: int = 0) -> Iterator[SessionRecord]:
        with self._lock:
            self.get_session(session_id)
            ids = list(self._order[session_id][max(after_seq, 0):])
        for record_id in ids:
            yield self.get_record(record_id)

    def path(self, session_id: str, head_record_id: str) -> Iterator[SessionRecord]:
        self.get_session(session_id)
        ids = []
        current: str | None = head_record_id
        while current is not None:
            record = self.get_record(current)
            if record.session_id != session_id:
                raise ValueError("Record belongs to a different session")
            ids.append(current)
            current = record.parent_record_id
        for record_id in reversed(ids):
            yield self.get_record(record_id)

    def save_snapshot(self, snapshot_id: str, payload: dict[str, Any]) -> None:
        with self._lock:
            if snapshot_id in self._snapshots:
                raise ValueError("Snapshot already exists")
            self._snapshots[snapshot_id] = deepcopy(payload)

    def load_snapshot(self, snapshot_id: str) -> dict[str, Any]:
        with self._lock:
            if snapshot_id not in self._snapshots:
                raise FileNotFoundError("Cursor snapshot unavailable; repeat discovery")
            return deepcopy(self._snapshots[snapshot_id])

    def close(self) -> None:
        with self._lock:
            self._writers.clear()
