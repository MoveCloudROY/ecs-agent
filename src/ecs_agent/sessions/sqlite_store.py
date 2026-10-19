"""Transactional SQLite session store, independent of World checkpoints."""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict
import json
import os
from pathlib import Path
import sqlite3
from threading import RLock
from typing import Any

from ecs_agent.logging import get_logger
from ecs_agent.sessions.store import is_same_draft, updated_session, validate_append, validate_session
from ecs_agent.sessions.types import RecordDraft, SessionInfo, SessionRecord

logger = get_logger(__name__)


class SQLiteSessionStore:
    """Append-only store with per-session writer ownership across connections.

    Closing releases this connection's claims. A claim from a dead process can
    be reclaimed; a live process is never displaced automatically. Hosts must
    explicitly release a live writer before checkpoint branching.
    """

    def __init__(self, path: str | Path) -> None:
        self.database_path = Path(path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(self.database_path), check_same_thread=False, isolation_level=None)
        self._db.execute("PRAGMA foreign_keys=ON")
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=FULL")
        self._db.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY, data TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS records (
                record_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL REFERENCES sessions(session_id),
                seq INTEGER NOT NULL,
                parent_record_id TEXT REFERENCES records(record_id),
                data TEXT NOT NULL,
                UNIQUE(session_id, seq)
            );
            CREATE TABLE IF NOT EXISTS writers (
                session_id TEXT PRIMARY KEY REFERENCES sessions(session_id),
                writer_id TEXT NOT NULL, pid INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS query_snapshots (
                snapshot_id TEXT PRIMARY KEY, data TEXT NOT NULL
            );
        """)
        self._lock = RLock()
        self._claims: set[tuple[str, str]] = set()
        self._closed = False

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                yield self._db
                self._db.commit()
            except BaseException as exc:
                self._db.rollback()
                logger.error("session_transaction_failed", exception=str(exc))
                raise

    def create_session(self, session: SessionInfo) -> None:
        with self._transaction() as db:
            parent = self.get_session(session.parent_session_id) if session.parent_session_id else None
            validate_session(session, parent)
            existing = db.execute("SELECT data FROM sessions WHERE session_id=?", (session.session_id,)).fetchone()
            if existing:
                if SessionInfo(**json.loads(existing[0])) == session:
                    return
                raise ValueError("session_id already exists")
            db.execute("INSERT INTO sessions VALUES (?, ?)", (session.session_id, json.dumps(asdict(session), ensure_ascii=False)))

    def get_session(self, session_id: str) -> SessionInfo:
        with self._lock:
            row = self._db.execute("SELECT data FROM sessions WHERE session_id=?", (session_id,)).fetchone()
            if row is None:
                raise FileNotFoundError("Session is unavailable")
            return SessionInfo(**json.loads(row[0]))

    def sessions(self) -> list[SessionInfo]:
        with self._lock:
            sessions = [SessionInfo(**json.loads(row[0])) for row in self._db.execute("SELECT data FROM sessions")]
        return sorted(sessions, key=lambda s: (s.created_at, s.session_id))

    def claim_writer(self, session_id: str, writer_id: str) -> None:
        with self._lock:
            attempted = False
            try:
                with self._transaction() as db:
                    self.get_session(session_id)
                    row = db.execute("SELECT writer_id, pid FROM writers WHERE session_id=?", (session_id,)).fetchone()
                    if row is not None and (row[0] != writer_id or (session_id, writer_id) not in self._claims):
                        try:
                            os.kill(row[1], 0)
                        except ProcessLookupError:
                            db.execute("DELETE FROM writers WHERE session_id=?", (session_id,))
                        else:
                            raise RuntimeError("Session already has an active writer")
                    db.execute("INSERT OR REPLACE INTO writers VALUES (?, ?, ?)", (session_id, writer_id, os.getpid()))
                    attempted = True
            except Exception:
                # Keep ownership only when a failed commit acknowledgement can
                # be confirmed. A rolled-back claim cannot override a new owner.
                if attempted:
                    row = self._db.execute("SELECT writer_id, pid FROM writers WHERE session_id=?", (session_id,)).fetchone()
                    if row == (writer_id, os.getpid()):
                        self._claims.add((session_id, writer_id))
                raise
            self._claims.add((session_id, writer_id))

    def release_writer(self, session_id: str, writer_id: str) -> None:
        with self._lock:
            if (session_id, writer_id) in self._claims:
                with self._transaction() as db:
                    db.execute("DELETE FROM writers WHERE session_id=? AND writer_id=?", (session_id, writer_id))
                self._claims.discard((session_id, writer_id))

    def append(self, draft: RecordDraft, *, writer_id: str, branch: bool = False, select_head: bool = True) -> SessionRecord:
        with self._transaction() as db:
            owner = db.execute("SELECT writer_id FROM writers WHERE session_id=?", (draft.session_id,)).fetchone()
            if owner is None or owner[0] != writer_id or (draft.session_id, writer_id) not in self._claims:
                raise RuntimeError("Session writer is not owned by this caller")
            row = db.execute("SELECT data FROM records WHERE record_id=?", (draft.record_id,)).fetchone()
            if row is not None:
                existing = SessionRecord(**json.loads(row[0]))
                if not is_same_draft(existing, draft):
                    raise ValueError("record_id reused with different content")
                return existing
            session = self.get_session(draft.session_id)
            parent = self.get_record(draft.parent_record_id) if draft.parent_record_id else None
            validate_append(session, draft, parent, branch=branch)
            seq = db.execute("SELECT COALESCE(MAX(seq), 0) + 1 FROM records WHERE session_id=?", (draft.session_id,)).fetchone()[0]
            record = SessionRecord(**asdict(draft), seq=seq)
            db.execute("INSERT INTO records VALUES (?, ?, ?, ?, ?)", (
                record.record_id, record.session_id, record.seq, record.parent_record_id,
                json.dumps(asdict(record), ensure_ascii=False),
            ))
            if select_head:
                db.execute("UPDATE sessions SET data=? WHERE session_id=?", (
                    json.dumps(asdict(updated_session(session, record)), ensure_ascii=False), session.session_id,
                ))
            return record

    def get_record(self, record_id: str) -> SessionRecord:
        with self._lock:
            row = self._db.execute("SELECT data FROM records WHERE record_id=?", (record_id,)).fetchone()
            if row is None:
                raise FileNotFoundError("Record is unavailable")
            return SessionRecord(**json.loads(row[0]))

    def records(self, session_id: str, *, after_seq: int = 0) -> Iterator[SessionRecord]:
        self.get_session(session_id)
        with self._lock:
            cursor = self._db.execute("SELECT data FROM records WHERE session_id=? AND seq>? ORDER BY seq", (session_id, after_seq))
        try:
            while True:
                with self._lock:
                    row = cursor.fetchone()
                if row is None:
                    break
                yield SessionRecord(**json.loads(row[0]))
        finally:
            cursor.close()

    def path(self, session_id: str, head_record_id: str) -> Iterator[SessionRecord]:
        self.get_session(session_id)
        if self.get_record(head_record_id).session_id != session_id:
            raise ValueError("Record belongs to a different session")
        with self._lock:
            cursor = self._db.execute("""
                WITH RECURSIVE path(record_id, parent_record_id, seq) AS (
                    SELECT record_id, parent_record_id, seq FROM records WHERE record_id=? AND session_id=?
                    UNION ALL
                    SELECT r.record_id, r.parent_record_id, r.seq FROM records r
                    JOIN path p ON r.record_id=p.parent_record_id WHERE r.session_id=?
                )
                SELECT r.data FROM path p JOIN records r ON r.record_id=p.record_id ORDER BY p.seq
            """, (head_record_id, session_id, session_id))
        try:
            while True:
                with self._lock:
                    row = cursor.fetchone()
                if row is None:
                    break
                yield SessionRecord(**json.loads(row[0]))
        finally:
            cursor.close()

    def save_snapshot(self, snapshot_id: str, payload: dict[str, Any]) -> None:
        with self._transaction() as db:
            db.execute("INSERT INTO query_snapshots VALUES (?, ?)", (snapshot_id, json.dumps(payload, ensure_ascii=False)))

    def load_snapshot(self, snapshot_id: str) -> dict[str, Any]:
        with self._lock:
            row = self._db.execute("SELECT data FROM query_snapshots WHERE snapshot_id=?", (snapshot_id,)).fetchone()
            if row is None:
                raise FileNotFoundError("Cursor snapshot unavailable; repeat discovery")
            payload: dict[str, Any] = json.loads(row[0])
            return payload

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            for session_id, writer_id in list(self._claims):
                self.release_writer(session_id, writer_id)
            self._db.close()
            self._closed = True
