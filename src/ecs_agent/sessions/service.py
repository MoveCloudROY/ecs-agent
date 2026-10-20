"""Session writes, policy checks, immutable query snapshots, and text projections."""

from collections.abc import Iterator
from dataclasses import asdict
import hashlib
import io
import json
from pathlib import Path
import re
from typing import Any, TextIO
from uuid import uuid4

from ecs_agent.components import SessionPolicyComponent, SessionRefComponent
from ecs_agent.logging import get_logger
from ecs_agent.scratchbook.artifact_registry import ArtifactKind, ArtifactRegistry
from ecs_agent.sessions.store import SessionStore, is_same_draft
from ecs_agent.sessions.types import (
    RecordDraft, SessionInfo, SessionPage, SessionRecord, new_record_id, new_session_id,
)
from ecs_agent.types import Message

logger = get_logger(__name__)

_PRIVATE_KEYS = {
    "provider_output", "reasoning_signature", "parts", "api_key", "apikey",
    "authorization", "password", "secret", "access_token", "refresh_token",
}
_CREDENTIAL = re.compile(
    r"(?i)(\b(?:api[_-]?key|authorization|password|access_token|refresh_token)\b[\"']?\s*[:=]\s*[\"']?)(?:Bearer\s+)?[^\s\"';,}]+"
)
_TOKEN = re.compile(r"\bsk-[A-Za-z0-9_-]{12,}|(?i:Bearer)\s+[A-Za-z0-9_.~+/-]{8,}=*")


def redact_text(text: str) -> str:
    """Remove recognizable credentials from model-visible text projections."""
    return _TOKEN.sub("[redacted]", _CREDENTIAL.sub(r"\1[redacted]", text))


def project_value(value: Any) -> Any:
    """Exclude provider-private replay, attachments, and named credentials."""
    if isinstance(value, dict):
        result = {key: project_value(item) for key, item in value.items() if key.lower() not in _PRIVATE_KEYS}
        if value.get("role") == "system":
            result["content"] = "[system configuration omitted]"
            result.pop("reasoning_content", None)
        return result
    if isinstance(value, list):
        return [project_value(item) for item in value]
    return redact_text(value) if isinstance(value, str) else value


class SessionService:
    """Shared runtime dependency for independent main and child Worlds."""

    def __init__(self, store: SessionStore, *, artifact_registry: ArtifactRegistry | None = None) -> None:
        self.store = store
        self.artifact_registry = artifact_registry
        self._verified_artifacts: dict[str, tuple[int, int, int, int]] = {}

    def create_session(
        self, *, agent_name: str, workspace_id: str, parent_session_id: str | None = None,
        source_session_id: str | None = None,
    ) -> SessionInfo:
        """Create a host-owned identity, inheriting task-tree identity from its parent."""
        session_id = new_session_id()
        root_id = self.store.get_session(parent_session_id).root_session_id if parent_session_id else session_id
        info = SessionInfo(
            session_id=session_id, parent_session_id=parent_session_id, root_session_id=root_id,
            agent_name=agent_name, workspace_id=workspace_id, source_session_id=source_session_id,
        )
        self.store.create_session(info)
        return info

    def claim_writer(self, session_id: str) -> str:
        """Claim exclusive writing; caller must release before another World resumes."""
        token = uuid4().hex
        self.store.claim_writer(session_id, token)
        return token

    def finish_external_session(self, session_id: str, *, status: str, reason: str) -> None:
        """Persist a host terminal outcome when no child runner owns the session.

        This covers queued cancellation, failed construction, lost live handles,
        and post-run finalization errors. A live writer cannot be displaced.
        """
        info = self.store.get_session(session_id)
        if info.status == status:
            return
        if status not in {"completed", "error", "cancelled", "interrupted", "timeout", "max_ticks", "budget_exhausted", "incomplete", "unknown"}:
            raise ValueError("Expected a terminal session status")
        writer = self.claim_writer(session_id)
        try:
            self.append(SessionRefComponent(session_id, info.head_record_id), "run_finished", {"status": status, "reason": reason, "source": "background_runtime"}, writer_id=writer)
        finally:
            self.store.release_writer(session_id, writer)

    def append(
        self, ref: SessionRefComponent, kind: str, payload: dict[str, Any], *, writer_id: str,
        history_id: str = "", revision: int = 0, run_id: str | None = None,
        record_id: str | None = None, artifact_ref: str | None = None, branch: bool = False,
        select_head: bool = True,
    ) -> SessionRecord:
        """Commit first, then advance the in-memory head without an async suspension."""
        draft = RecordDraft(
            record_id=record_id or new_record_id(), session_id=ref.session_id,
            parent_record_id=ref.head_record_id, kind=kind, payload=payload,
            history_id=history_id, revision=revision, run_id=run_id, artifact_ref=artifact_ref,
        )
        try:
            record = self.store.append(draft, writer_id=writer_id, branch=branch, select_head=select_head)
        except Exception as exc:
            logger.error("session_append_failed", session_id=ref.session_id, record_id=draft.record_id, exception=str(exc))
            # A transport/commit acknowledgement failure does not prove rollback.
            # Confirm the same stable ID before choosing whether memory advances.
            try:
                committed = self.store.get_record(draft.record_id)
            except Exception as probe_exc:
                logger.error("session_commit_unconfirmed", session_id=ref.session_id, record_id=draft.record_id, exception=str(probe_exc))
                raise exc from probe_exc
            if not is_same_draft(committed, draft):
                raise
            logger.warning("session_commit_confirmed_after_error", session_id=ref.session_id, record_id=draft.record_id)
            record = committed
        if select_head:
            ref.head_record_id = record.record_id
        return record

    def append_message(
        self, ref: SessionRefComponent, message: Message, *, writer_id: str,
        history_id: str = "", revision: int = 0, run_id: str | None = None,
        record_id: str | None = None, metadata: dict[str, Any] | None = None,
        branch: bool = False, select_head: bool = True,
    ) -> SessionRecord:
        """Preserve actual messages and publish large tool bodies before their references."""
        from ecs_agent.serialization import WorldSerializer

        payload = WorldSerializer._message_to_dict(message)
        if metadata:
            payload["source"] = metadata
        artifact_ref = None
        if message.role == "tool" and len(message.content.encode("utf-8")) > 2048 and self.artifact_registry:
            artifact = self.artifact_registry.persist(kind=ArtifactKind.TOOL, content=message.content)
            artifact_ref = artifact.record_path
            payload["content"] = message.content[:256]
            payload["artifact_sha256"] = hashlib.sha256(message.content.encode("utf-8")).hexdigest()
            payload["artifact_owner_session_id"] = ref.session_id
        return self.append(
            ref, "message", payload, writer_id=writer_id, history_id=history_id,
            revision=revision, run_id=run_id, record_id=record_id, artifact_ref=artifact_ref,
            branch=branch, select_head=select_head,
        )

    def can_read(self, caller_id: str, policy: SessionPolicyComponent, target: SessionInfo) -> bool:
        """Apply workspace, explicit grants, and self/tree/descendant scope uniformly."""
        if policy.scope not in {"self", "tree", "descendants"}:
            raise ValueError("Unsupported session policy scope")
        caller = self.store.get_session(caller_id)
        if caller.workspace_id != target.workspace_id:
            return False
        if target.session_id == caller_id or target.session_id in policy.shared_session_ids:
            return True
        if policy.scope == "tree":
            return caller.root_session_id == target.root_session_id
        if policy.scope == "descendants":
            parent = target.parent_session_id
            while parent is not None:
                if parent == caller_id:
                    return True
                parent = self.store.get_session(parent).parent_session_id
        return False

    def authorize(self, caller_id: str, policy: SessionPolicyComponent, target_id: str) -> SessionInfo:
        """Authorize on every request, including old URIs and cursor continuations."""
        target = self.store.get_session(target_id)
        if not self.can_read(caller_id, policy, target):
            raise PermissionError("Session is outside the caller's permitted scope")
        return target

    def transcript_uri(self, session: SessionInfo) -> str | None:
        """Return a fixed-head resource, or None for an empty session."""
        if session.head_record_id is None:
            return None
        return f"session://{session.session_id}/transcript?head={session.head_record_id}&v=1"

    def record_uri(self, record: SessionRecord) -> str:
        """Return an immutable single-record locator."""
        return f"session://{record.session_id}/records/{record.record_id}?v=1"

    def tool_result_uri(self, record: SessionRecord) -> str:
        """Return the full tool body locator without exposing a filesystem path."""
        return f"session://{record.session_id}/records/{record.record_id}/tool-result?v=1"

    def list_sessions(
        self, caller_id: str, policy: SessionPolicyComponent, *, parent_session_id: str | None = None,
        agent_name: str | None = None, cursor: str | None = None, limit: int = 20,
    ) -> SessionPage:
        """Discover authorized sessions with pinned metadata and transcript heads."""
        filters = {"operation": "list", "parent_session_id": parent_session_id, "agent_name": agent_name}
        rows = []
        if cursor is None:
            for session in self.store.sessions():
                if not self.can_read(caller_id, policy, session):
                    continue
                if parent_session_id is not None and session.parent_session_id != parent_session_id:
                    continue
                if agent_name is not None and session.agent_name != agent_name:
                    continue
                item = asdict(session)
                item["task_summary"] = redact_text(session.task_summary)
                item["read_uri"] = self.transcript_uri(session)
                rows.append(item)
        return self._page(caller_id, policy, filters, rows, cursor, limit)

    def search(
        self, caller_id: str, policy: SessionPolicyComponent, query: str, *, session_id: str | None = None,
        agent_name: str | None = None, cursor: str | None = None, limit: int = 20,
    ) -> SessionPage:
        """Search literal text on currently selected paths, including full tool artifacts."""
        if not query or len(query) > 2000:
            raise ValueError("Search query must contain 1 to 2000 characters")
        filters = {"operation": "search", "query": query, "session_id": session_id, "agent_name": agent_name}
        rows = []
        if cursor is None:
            for session in self.store.sessions():
                if not self.can_read(caller_id, policy, session):
                    continue
                if session_id is not None and session.session_id != session_id:
                    continue
                if agent_name is not None and session.agent_name != agent_name:
                    continue
                if session.head_record_id is None:
                    continue
                for record in self.store.path(session.session_id, session.head_record_id):
                    snippet = self._find_snippet(record, query)
                    if snippet is not None:
                        rows.append({
                            "session_id": session.session_id, "record_id": record.record_id,
                            "seq": record.seq, "snippet": snippet, "read_uri": self.record_uri(record),
                        })
        return self._page(caller_id, policy, filters, rows, cursor, limit)

    def _page(
        self, caller_id: str, policy: SessionPolicyComponent, filters: dict[str, Any],
        rows: list[dict[str, Any]], cursor: str | None, limit: int,
    ) -> SessionPage:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("Page limit must be a positive integer")
        limit = min(limit, 100)
        self.store.get_session(caller_id)
        policy_key = {"scope": policy.scope, "shared_session_ids": sorted(set(policy.shared_session_ids))}
        if cursor:
            try:
                snapshot_id, position = cursor.split(":")
                offset = int(position)
                if not re.fullmatch(r"cur_[a-f0-9]{32}", snapshot_id) or offset < 0:
                    raise ValueError("Invalid cursor")
            except (ValueError, TypeError) as exc:
                raise ValueError("Invalid session cursor") from exc
            snapshot = self.store.load_snapshot(snapshot_id)
            if snapshot["caller_id"] != caller_id or snapshot["policy"] != policy_key:
                raise PermissionError("Cursor policy changed; repeat discovery")
            if snapshot["filters"] != filters:
                raise ValueError("Cursor filters changed; repeat discovery")
            rows = snapshot["rows"]
            if offset > len(rows):
                raise ValueError("Cursor offset is outside its snapshot")
        else:
            offset = 0
            snapshot_id = f"cur_{uuid4().hex}"
            if len(rows) > limit:
                self.store.save_snapshot(snapshot_id, {"caller_id": caller_id, "policy": policy_key, "filters": filters, "rows": rows})
        # Recheck all selected rows before exposing any fields or snippets.
        items = rows[offset:offset + limit]
        for item in items:
            self.authorize(caller_id, policy, item["session_id"])
        next_offset = offset + len(items)
        return SessionPage(items, f"{snapshot_id}:{next_offset}" if next_offset < len(rows) else None)

    def _find_snippet(self, record: SessionRecord, query: str) -> str | None:
        metadata = project_value(record.payload)
        metadata.pop("content", None)
        for source in (iter([json.dumps(metadata, ensure_ascii=False)]), self.content_chunks(record)):
            tail = ""
            for chunk in source:
                text = tail + chunk
                position = text.find(query)
                if position >= 0:
                    return text[max(0, position - 80):position + len(query) + 80]
                tail = text[-(len(query) + 80):]
        return None

    def _artifact_path(self, record: SessionRecord) -> Path:
        if self.artifact_registry is None or record.artifact_ref is None:
            raise FileNotFoundError("Session artifact storage is unavailable")
        artifact_id = record.artifact_ref.rsplit("/", 1)[-1]
        if not self.artifact_registry.is_valid_artifact_id(artifact_id):
            raise ValueError("Invalid session artifact identity")
        canonical = self.artifact_registry.record_path_for(kind=ArtifactKind.TOOL, artifact_id=artifact_id)
        if canonical != record.artifact_ref or record.payload.get("artifact_owner_session_id") != record.session_id:
            raise ValueError("Session artifact ownership mismatch")
        root = self.artifact_registry.root.resolve()
        path = (root / canonical).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Session artifact is outside its storage root")
        try:
            stat = path.stat()
        except FileNotFoundError as exc:
            raise FileNotFoundError("Session artifact is unavailable") from exc
        signature = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
        if self._verified_artifacts.get(record.record_id) != signature:
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest != record.payload.get("artifact_sha256"):
                raise ValueError("Session artifact changed; immutable resource is unavailable")
            self._verified_artifacts[record.record_id] = signature
        return path

    def content_chunks(self, record: SessionRecord) -> Iterator[str]:
        """Yield stable projected text chunks without loading a whole transcript."""
        if record.payload.get("role") == "system":
            yield "[system configuration omitted]"
            return
        stream: TextIO
        if record.artifact_ref:
            stream = self._artifact_path(record).open("r", encoding="utf-8", newline="")
        else:
            stream = io.StringIO(str(record.payload.get("content", "")))
        with stream:
            for line in stream:
                safe_line = redact_text(line)
                for start in range(0, len(safe_line), 512):
                    yield safe_line[start:start + 512]

    def close(self) -> None:
        """Release writers and close the shared store; called by the root application."""
        self.store.close()

    def replay_history(self, session_id: str, *, head_record_id: str | None = None) -> list[Message]:
        """Reconstruct a committed model-facing history for host recovery.

        This host API preserves provider-private fields. Agent-visible session
        tools continue to use authorized, redacted transcript projections.
        Legacy delta-only replacement records cannot reconstruct a full view.
        """
        from ecs_agent.serialization import WorldSerializer

        info = self.store.get_session(session_id)
        head = head_record_id or info.head_record_id
        if head is None:
            return []
        messages = []
        complete = True
        for record in self.store.path(session_id, head):
            if record.kind == "message":
                payload = dict(record.payload)
                if record.artifact_ref is not None:
                    payload["content"] = self._artifact_path(record).read_text(encoding="utf-8")
                messages.append(WorldSerializer._message_from_dict(payload))
            elif record.kind == "history_replaced":
                if record.payload.get("snapshot_format") != "complete":
                    complete = False
                    messages = []
                else:
                    complete = True
                    messages = [WorldSerializer._message_from_dict(data) for data in record.payload["messages"]]
        if not complete:
            raise ValueError("Legacy history replacement lacks an ordered snapshot; restore its checkpoint instead")
        return messages
