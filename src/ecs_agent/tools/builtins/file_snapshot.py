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


class FileSnapshotStore:
    """Read snapshot store for built-in file tools."""

    max_snapshots_per_file = 20

    def __init__(self) -> None:
        self._histories: dict[tuple[str, str], FileSnapshotHistory] = {}

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
        history = self._histories.setdefault(self._key(target), FileSnapshotHistory())
        history.snapshots.append(snapshot)
        if len(history.snapshots) > self.max_snapshots_per_file:
            del history.snapshots[: len(history.snapshots) - self.max_snapshots_per_file]

    def record_file_read(
        self,
        file_path: str,
        target: Path,
        content: str,
        offset: int,
        limit: int,
    ) -> None:
        """Record a read_file snapshot for a workspace file."""
        self.record_read(file_path, target, content, offset, limit)

    def latest_for(self, target: Path) -> FileReadSnapshot | None:
        """Return the latest snapshot for a target file, if any."""
        snapshots = self.snapshots_for(target)
        if not snapshots:
            return None
        return snapshots[-1]

    def snapshots_for(self, target: Path) -> tuple[FileReadSnapshot, ...]:
        """Return recent snapshots for a target file in creation order."""
        history = self._histories.get(self._key(target))
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
        """Clear all snapshots. Intended for tests."""
        self._histories.clear()

    def _key(self, target: Path) -> tuple[str, str]:
        resolved = target.resolve()
        return (str(resolved.parent), resolved.name)


class FileSnapshotState:
    """Typed accessor for file snapshots in an entity's tool runtime state."""

    namespace_name = "file"
    store_key = "snapshots"

    def __init__(self, component: ToolRuntimeStateComponent) -> None:
        namespace = component.namespaces.setdefault(
            self.namespace_name,
            ToolStateNamespace(),
        )
        store = namespace.values.get(self.store_key)
        if not isinstance(store, FileSnapshotStore):
            store = FileSnapshotStore()
            namespace.values[self.store_key] = store
            namespace.version += 1
        self._store = store

    def record_read(
        self,
        path: str,
        content: str,
        digest: str,
        offset: int,
        line_count: int,
    ) -> None:
        """Record a direct snapshot for tests or custom callers."""
        target = Path(path)
        selected = content.splitlines()
        anchors = tuple(
            LineAnchor(
                line_number=line_number,
                content=line_content,
                hash_id=compute_snapshot_line_hash(line_number, line_content),
            )
            for line_number, line_content in enumerate(selected, start=offset)
        )
        snapshot = FileReadSnapshot(
            file_path=path,
            absolute_path=str(target),
            content=content,
            content_digest=digest,
            offset=offset,
            limit=line_count,
            anchors=anchors,
            created_at=time.monotonic(),
        )
        key = self._path_key(path)
        history = self._store._histories.setdefault(key, FileSnapshotHistory())
        history.snapshots.append(snapshot)
        if len(history.snapshots) > self._store.max_snapshots_per_file:
            del history.snapshots[
                : len(history.snapshots) - self._store.max_snapshots_per_file
            ]

    def record_file_read(
        self,
        file_path: str,
        target: Path,
        content: str,
        offset: int,
        limit: int,
    ) -> None:
        """Record a read_file snapshot for a workspace file."""
        self._store.record_read(file_path, target, content, offset, limit)

    def latest_for(self, target: Path) -> FileReadSnapshot | None:
        """Return the latest snapshot for a target file, if any."""
        return self._store.latest_for(target)

    def snapshots_for(self, path: str | Path) -> tuple[FileReadSnapshot, ...]:
        """Return recent snapshots for a path in creation order."""
        if isinstance(path, Path):
            return self._store.snapshots_for(path)
        history = self._store._histories.get(self._path_key(path))
        if history is None:
            return ()
        return tuple(history.snapshots)

    def find_anchor(
        self,
        path: str | Path,
        line_number: int,
        content_digest: str | None = None,
    ) -> LineAnchor:
        """Find a unique line anchor across recent snapshots for a file."""
        if isinstance(path, Path):
            return self._store.find_anchor(path, line_number, content_digest)

        anchors: list[tuple[FileReadSnapshot, LineAnchor]] = []
        for snapshot in self.snapshots_for(path):
            if content_digest is not None and snapshot.content_digest != content_digest:
                continue
            for anchor in snapshot.anchors:
                if anchor.line_number == line_number:
                    anchors.append((snapshot, anchor))
        return _resolve_unique_anchor(anchors)

    def clear(self) -> None:
        """Clear all file snapshots."""
        self._store.clear()

    def _path_key(self, path: str) -> tuple[str, str]:
        target = Path(path)
        if target.parent == Path("."):
            return ("", target.name)
        return (str(target.parent), target.name)


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


SNAPSHOT_STORE = FileSnapshotStore()
_CURRENT_SNAPSHOT_STORE: ContextVar[FileSnapshotStore] = ContextVar(
    "ecs_agent_current_file_snapshot_store",
    default=SNAPSHOT_STORE,
)


def current_snapshot_store() -> FileSnapshotStore:
    """Return the snapshot store bound to the current tool call context."""
    return _CURRENT_SNAPSHOT_STORE.get()


def current_file_snapshot_state() -> FileSnapshotState | FileSnapshotStore:
    """Return ECS-backed file snapshots, falling back for direct tool calls."""
    try:
        return FileSnapshotState(current_tool_runtime_state())
    except RuntimeError:
        return _CURRENT_SNAPSHOT_STORE.get()


@contextmanager
def use_snapshot_store(snapshot_store: FileSnapshotStore) -> Iterator[None]:
    """Temporarily bind a snapshot store for one tool call context."""
    token = _CURRENT_SNAPSHOT_STORE.set(snapshot_store)
    try:
        yield
    finally:
        _CURRENT_SNAPSHOT_STORE.reset(token)


__all__ = [
    "FileReadSnapshot",
    "FileSnapshotHistory",
    "FileSnapshotState",
    "FileSnapshotStore",
    "LineAnchor",
    "SNAPSHOT_STORE",
    "compute_content_digest",
    "compute_snapshot_line_hash",
    "current_file_snapshot_state",
    "current_snapshot_store",
    "normalize_snapshot_line",
    "use_snapshot_store",
]
