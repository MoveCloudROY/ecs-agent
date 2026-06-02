"""Internal file read snapshots for safe LLM-visible file edits."""

from __future__ import annotations

import hashlib
import time
from contextlib import contextmanager
from contextvars import ContextVar
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from ecs_agent.components import ToolRuntimeStateComponent, ToolStateNamespace
from ecs_agent.tools.context import current_tool_runtime_state


def normalize_snapshot_line(content: str) -> str:
    """Normalize a line before anchor hashing."""
    return content.rstrip()


def compute_snapshot_line_hash(line_number: int, content: str) -> str:
    """Compute the same short line hash used by hash-anchored editing."""
    normalized = normalize_snapshot_line(content)
    payload = f"{line_number}:{normalized}"
    return hashlib.md5(payload.encode("utf-8")).hexdigest()[:4]


def compute_content_digest(content: str) -> str:
    """Compute a stable digest for a full file snapshot."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


@dataclass(slots=True, frozen=True)
class LineAnchor:
    """Internal anchor for one clean file line."""

    line_number: int
    content: str
    hash_id: str


@dataclass(slots=True, frozen=True)
class FileReadSnapshot:
    """Internal record of the latest clean read for one workspace file."""

    file_path: str
    absolute_path: str
    content: str
    content_digest: str
    offset: int
    limit: int
    anchors: tuple[LineAnchor, ...]
    created_at: float


@dataclass(slots=True)
class FileSnapshotHistory:
    """Bounded read history for one workspace file."""

    snapshots: list[FileReadSnapshot] = field(default_factory=list)


class FileSnapshotState:
    """File snapshot state for built-in file tools."""

    namespace_name = "file"
    store_key = "snapshots"
    max_snapshots_per_file = 20

    def __init__(self, component: ToolRuntimeStateComponent | None = None) -> None:
        self._histories: dict[tuple[str, str], FileSnapshotHistory] = {}
        if component is None:
            return

        namespace = component.namespaces.setdefault(
            self.namespace_name,
            ToolStateNamespace(),
        )
        state = namespace.values.get(self.store_key)
        if isinstance(state, FileSnapshotState):
            self._histories = state._histories
        else:
            namespace.values[self.store_key] = self
            namespace.version += 1

    @classmethod
    def from_component(
        cls,
        component: ToolRuntimeStateComponent,
    ) -> FileSnapshotState:
        """Return the snapshot state stored in a tool runtime component."""
        namespace = component.namespaces.setdefault(
            cls.namespace_name,
            ToolStateNamespace(),
        )
        state = namespace.values.get(cls.store_key)
        if isinstance(state, cls):
            return state

        state = cls()
        namespace.values[cls.store_key] = state
        namespace.version += 1
        return state

    def record_read(
        self,
        file_path: str,
        target: Path,
        content: str,
        offset: int,
        limit: int,
    ) -> None:
        """Record the latest clean read for a workspace file."""
        lines = content.splitlines()
        start = max(0, offset - 1)
        selected = lines[start:] if limit <= 0 else lines[start : start + limit]
        clean_content = "\n".join(selected)
        anchors = tuple(
            LineAnchor(
                line_number=line_number,
                content=line_content,
                hash_id=compute_snapshot_line_hash(line_number, line_content),
            )
            for line_number, line_content in enumerate(selected, start=start + 1)
        )
        snapshot = FileReadSnapshot(
            file_path=file_path,
            absolute_path=str(target.resolve()),
            content=clean_content,
            content_digest=compute_content_digest(content),
            offset=offset,
            limit=limit,
            anchors=anchors,
            created_at=time.monotonic(),
        )
        self._append_snapshot(self._target_key(target), snapshot)

    def latest_for(self, target: Path) -> FileReadSnapshot | None:
        """Return the latest snapshot for a target file, if any."""
        snapshots = self.snapshots_for(target)
        if not snapshots:
            return None
        return snapshots[-1]

    def snapshots_for(self, target: Path) -> tuple[FileReadSnapshot, ...]:
        """Return recent snapshots for a path in creation order."""
        history = self._histories.get(self._target_key(target))
        if history is None:
            return ()
        return tuple(history.snapshots)

    def find_anchor(
        self,
        target: Path,
        line_number: int,
        content_digest: str | None = None,
    ) -> LineAnchor:
        """Find a unique line anchor across recent snapshots for a file."""
        anchors: list[tuple[FileReadSnapshot, LineAnchor]] = []
        for snapshot in self.snapshots_for(target):
            if content_digest is not None and snapshot.content_digest != content_digest:
                continue
            for anchor in snapshot.anchors:
                if anchor.line_number == line_number:
                    anchors.append((snapshot, anchor))
        return _resolve_unique_anchor(anchors)

    def clear(self) -> None:
        """Clear all file snapshots."""
        self._histories.clear()

    def _append_snapshot(
        self,
        key: tuple[str, str],
        snapshot: FileReadSnapshot,
    ) -> None:
        history = self._histories.setdefault(key, FileSnapshotHistory())
        history.snapshots.append(snapshot)
        if len(history.snapshots) > self.max_snapshots_per_file:
            del history.snapshots[: len(history.snapshots) - self.max_snapshots_per_file]

    def _target_key(self, target: Path) -> tuple[str, str]:
        resolved = target.resolve()
        return (str(resolved.parent), resolved.name)

def _resolve_unique_anchor(
    anchors: list[tuple[FileReadSnapshot, LineAnchor]],
) -> LineAnchor:
    if not anchors:
        raise ValueError("line not found in the last read_file result")

    unique: dict[tuple[str, str, int, str], LineAnchor] = {}
    for snapshot, anchor in anchors:
        unique[
            (
                snapshot.absolute_path,
                snapshot.content_digest,
                anchor.line_number,
                anchor.hash_id,
            )
        ] = anchor

    deduplicated = list(unique.values())
    if len(deduplicated) > 1:
        raise ValueError("line is not unique in multiple read_file snapshots")
    return deduplicated[0]


SNAPSHOT_STATE = FileSnapshotState()
_CURRENT_SNAPSHOT_STATE: ContextVar[FileSnapshotState] = ContextVar(
    "ecs_agent_current_file_snapshot_state",
    default=SNAPSHOT_STATE,
)


def current_file_snapshot_state() -> FileSnapshotState:
    """Return ECS-backed file snapshots, falling back for direct tool calls."""
    try:
        return FileSnapshotState.from_component(current_tool_runtime_state())
    except RuntimeError:
        return _CURRENT_SNAPSHOT_STATE.get()


@contextmanager
def use_snapshot_state(snapshot_state: FileSnapshotState) -> Iterator[None]:
    """Temporarily bind snapshot state for one tool call context."""
    token = _CURRENT_SNAPSHOT_STATE.set(snapshot_state)
    try:
        yield
    finally:
        _CURRENT_SNAPSHOT_STATE.reset(token)


__all__ = [
    "FileReadSnapshot",
    "FileSnapshotHistory",
    "FileSnapshotState",
    "LineAnchor",
    "SNAPSHOT_STATE",
    "compute_content_digest",
    "compute_snapshot_line_hash",
    "current_file_snapshot_state",
    "normalize_snapshot_line",
    "use_snapshot_state",
]
