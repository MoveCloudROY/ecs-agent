"""Durable session catalog, append-only records, and query pages."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


def timestamp_now() -> str:
    """Return a sortable UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def new_session_id() -> str:
    """Return a persistent transcript identity, distinct from background job IDs."""
    return f"sess_{uuid4().hex}"


def new_record_id() -> str:
    """Allocate an identity before committing, so retries can reuse it."""
    return f"rec_{uuid4().hex}"


@dataclass(slots=True, frozen=True)
class SessionInfo:
    """Catalog metadata; current head and lifecycle are derived during append."""

    session_id: str
    parent_session_id: str | None
    root_session_id: str
    agent_name: str
    workspace_id: str
    created_at: str = field(default_factory=timestamp_now)
    head_record_id: str | None = None
    updated_at: str | None = None
    status: str = "created"
    task_summary: str = ""
    source_session_id: str | None = None


@dataclass(slots=True, frozen=True)
class RecordDraft:
    """Stable append input; reuse the entire draft after an uncertain commit."""

    record_id: str
    session_id: str
    parent_record_id: str | None
    kind: str
    payload: dict[str, Any]
    history_id: str = ""
    revision: int = 0
    run_id: str | None = None
    timestamp: str = field(default_factory=timestamp_now)
    artifact_ref: str | None = None


@dataclass(slots=True, frozen=True)
class SessionRecord(RecordDraft):
    """Committed record with a per-session physical sequence number."""

    seq: int = 0


@dataclass(slots=True)
class SessionPage:
    """Bounded catalog or search page with an opaque continuation cursor."""

    items: list[dict[str, Any]]
    next_cursor: str | None
