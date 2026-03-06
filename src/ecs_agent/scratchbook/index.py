"""Scratchbook index schema and reference resolution."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from datetime import datetime, timezone

from ecs_agent.logging import get_logger

logger = get_logger(__name__)


class CorruptedIndexEntryError(Exception):
    """Raised when an index entry is malformed or missing required fields."""

    pass


@dataclass(slots=True)
class IndexEntry:
    """Schema for an index entry with stable ID, artifact metadata, and hash."""

    stable_id: str
    artifact_id: str
    artifact_type: str
    category: str
    content_hash: str
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class ScratchbookIndexer:
    """Index schema and resolver for deterministic artifact lookup.

    Provides fast lookup by task ID, artifact type, and category path.
    Includes stable IDs and content hash metadata.
    Handles corrupted index entries gracefully.
    """

    def __init__(self, root: Path | str) -> None:
        """Initialize scratchbook indexer.

        Args:
            root: Root directory for scratchbook storage
        """
        self.root = Path(root)
        self.index_dir = self.root / "index"
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.index_dir / "index.json"

    def create_index_entry(
        self,
        stable_id: str,
        artifact_id: str,
        artifact_type: str,
        category: str,
        content_hash: str,
    ) -> dict[str, Any]:
        """Create a new index entry with all required metadata.

        Args:
            stable_id: Stable identifier for the task/artifact
            artifact_id: Unique identifier for the artifact file
            artifact_type: Type of artifact (e.g., "planning", "execution")
            category: Category subfolder path
            content_hash: SHA256 hash of content for integrity

        Returns:
            Dictionary representing the index entry
        """
        entry = IndexEntry(
            stable_id=stable_id,
            artifact_id=artifact_id,
            artifact_type=artifact_type,
            category=category,
            content_hash=content_hash,
        )
        return {
            "stable_id": entry.stable_id,
            "artifact_id": entry.artifact_id,
            "artifact_type": entry.artifact_type,
            "category": entry.category,
            "content_hash": entry.content_hash,
            "timestamp": entry.timestamp,
        }

    def add_entry(
        self,
        stable_id: str,
        artifact_id: str,
        artifact_type: str,
        category: str,
        content_hash: str,
    ) -> None:
        """Add an entry to the index and persist atomically.

        Args:
            stable_id: Stable identifier for the task/artifact
            artifact_id: Unique identifier for the artifact file
            artifact_type: Type of artifact
            category: Category subfolder path
            content_hash: SHA256 hash of content
        """
        # Load current index
        current_index = self._load_index_data()
        entries = current_index.get("entries", [])

        # Create and add new entry
        new_entry = self.create_index_entry(
            stable_id=stable_id,
            artifact_id=artifact_id,
            artifact_type=artifact_type,
            category=category,
            content_hash=content_hash,
        )
        entries.append(new_entry)

        # Persist atomically
        self._write_index_atomic({"entries": entries})

        logger.info(
            "index_entry_added",
            stable_id=stable_id,
            artifact_id=artifact_id,
            artifact_type=artifact_type,
            category=category,
        )

    def lookup_by_task_id(self, task_id: str) -> list[dict[str, Any]]:
        """Lookup all artifacts for a given task ID.

        Args:
            task_id: Stable task identifier

        Returns:
            List of index entries for the task, sorted deterministically

        Raises:
            CorruptedIndexEntryError: If the requested task_id has corrupted entries
        """
        entries, corrupted_task_ids = self._load_entries_with_corruption_tracking()
        if task_id in corrupted_task_ids:
            raise CorruptedIndexEntryError(
                f"Index entries for task_id {task_id} are corrupted and cannot be retrieved"
            )
        matching = [e for e in entries if e.get("stable_id") == task_id]
        # Sort by artifact_id for deterministic ordering
        return sorted(matching, key=lambda x: x.get("artifact_id", ""))

    def lookup_by_artifact_type(self, artifact_type: str) -> list[dict[str, Any]]:
        """Lookup all artifacts of a given type.

        Args:
            artifact_type: Type of artifact (e.g., "planning", "execution")

        Returns:
            List of index entries, sorted deterministically
        """
        entries, _ = self._load_entries_with_corruption_tracking()
        matching = [e for e in entries if e.get("artifact_type") == artifact_type]
        # Sort by stable_id, then artifact_id for deterministic ordering
        return sorted(
            matching, key=lambda x: (x.get("stable_id", ""), x.get("artifact_id", ""))
        )

    def lookup_by_category(self, category: str) -> list[dict[str, Any]]:
        """Lookup all artifacts in a given category.

        Args:
            category: Category subfolder path

        Returns:
            List of index entries, sorted deterministically
        """
        entries, _ = self._load_entries_with_corruption_tracking()
        matching = [e for e in entries if e.get("category") == category]
        # Sort by stable_id, then artifact_id for deterministic ordering
        return sorted(
            matching, key=lambda x: (x.get("stable_id", ""), x.get("artifact_id", ""))
        )

    def _load_entries_with_corruption_tracking(
        self,
    ) -> tuple[list[dict[str, Any]], set[str]]:
        """Load index and track corrupted entries by task_id.

        Returns valid entries and filters out corrupted ones without raising.
        Only tracks which task_ids have corruption issues.

        Returns:
            Tuple of (valid_entries, corrupted_task_ids)
        """
        index_data = self._load_index_data()
        entries = index_data.get("entries", [])

        required_fields = {
            "stable_id",
            "artifact_id",
            "artifact_type",
            "category",
            "content_hash",
        }

        valid_entries: list[dict[str, Any]] = []
        corrupted_task_ids: set[str] = set()

        for i, entry in enumerate(entries):
            missing = required_fields - set(entry.keys())
            if missing:
                task_id = entry.get("stable_id")
                logger.error(
                    "index_entry_corrupted",
                    entry_index=i,
                    task_id=task_id,
                    missing_fields=list(missing),
                )
                if task_id:
                    corrupted_task_ids.add(task_id)
            else:
                valid_entries.append(entry)

        return valid_entries, corrupted_task_ids

    def _load_index_data(self) -> dict[str, Any]:
        """Load raw index data from file.

        Returns:
            Parsed index data or empty structure if file missing

        Raises:
            Does NOT raise on corrupted JSON; returns empty structure
        """
        if not self.index_file.exists():
            return {"entries": []}

        try:
            content = self.index_file.read_text(encoding="utf-8")
            data: dict[str, Any] = json.loads(content)
            return data
            content = self.index_file.read_text(encoding="utf-8")
            data = json.loads(content)
            return data
        except json.JSONDecodeError as exc:
            logger.error(
                "index_file_corrupted_json",
                path=str(self.index_file),
                exception=str(exc),
            )
            return {"entries": []}

    def _write_index_atomic(self, data: dict[str, Any]) -> None:
        """Write index atomically using temp file + os.replace pattern.

        Args:
            data: Index data to persist
        """
        import os

        temp_path = self.index_file.with_suffix(".json.tmp")
        temp_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        os.replace(temp_path, self.index_file)

        logger.info(
            "index_written_atomic",
            path=str(self.index_file),
            entry_count=len(data.get("entries", [])),
        )


def compute_content_hash(content: str | bytes) -> str:
    """Compute SHA256 hash of content.

    Args:
        content: Content to hash

    Returns:
        Hexadecimal SHA256 hash
    """
    if isinstance(content, str):
        content = content.encode("utf-8")
    return hashlib.sha256(content).hexdigest()
