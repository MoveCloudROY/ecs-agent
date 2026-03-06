"""Scratchbook filesystem service for categorized artifact storage."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from ecs_agent.logging import get_logger

logger = get_logger(__name__)


class ScratchbookService:
    """Filesystem service for scratchbook artifact storage.

    Provides:
    - Write/read artifacts to categorized subfolders
    - Append to log files
    - Atomic index updates using temp-file + os.replace pattern
    - UTF-8 encoding for all files
    """

    def __init__(self, root: Path | str) -> None:
        """Initialize scratchbook service.

        Args:
            root: Root directory for scratchbook storage (e.g., .scratchbook/)
        """
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def write_artifact(
        self, artifact_id: str, category: str, data: dict[str, Any]
    ) -> None:
        """Write artifact to categorized subfolder.

        Args:
            artifact_id: Unique identifier for the artifact
            category: Category subfolder (e.g., "planning", "execution")
            data: JSON-serializable data to write
        """
        category_path = self.root / category
        category_path.mkdir(parents=True, exist_ok=True)

        artifact_path = category_path / f"{artifact_id}.json"
        artifact_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

        logger.info(
            "write_artifact",
            artifact_id=artifact_id,
            category=category,
            path=str(artifact_path),
        )

    def read_artifact(self, artifact_id: str, category: str) -> dict[str, Any] | None:
        """Read artifact from categorized subfolder.

        Args:
            artifact_id: Unique identifier for the artifact
            category: Category subfolder

        Returns:
            Parsed JSON data or None if file missing/corrupted
        """
        artifact_path = self.root / category / f"{artifact_id}.json"

        if not artifact_path.exists():
            logger.debug(
                "read_artifact_missing",
                artifact_id=artifact_id,
                category=category,
            )
            return None

        try:
            content = artifact_path.read_text(encoding="utf-8")
            result: dict[str, Any] = json.loads(content)
            logger.info(
                "read_artifact",
                artifact_id=artifact_id,
                category=category,
                path=str(artifact_path),
            )
            return result
        except json.JSONDecodeError as exc:
            logger.error(
                "read_artifact_corrupted",
                artifact_id=artifact_id,
                category=category,
                path=str(artifact_path),
                exception=str(exc),
            )
            return None

    def append_log(self, log_name: str, category: str, line: str) -> None:
        """Append line to log file.

        Args:
            log_name: Log file name (e.g., "activity.log")
            category: Category subfolder
            line: Line to append (should include newline if desired)
        """
        category_path = self.root / category
        category_path.mkdir(parents=True, exist_ok=True)

        log_path = category_path / log_name

        with log_path.open("a", encoding="utf-8") as f:
            f.write(line)

        logger.info(
            "append_log",
            log_name=log_name,
            category=category,
            path=str(log_path),
            bytes_written=len(line),
        )

    def write_index(self, index_name: str, category: str, data: dict[str, Any]) -> None:
        """Atomically write index file using temp-file + os.replace pattern.

        This ensures that interrupted writes never produce partial/corrupted JSON.
        The index file is either fully written or the previous version remains intact.

        Args:
            index_name: Index file name (e.g., "task_index.json")
            category: Category subfolder
            data: JSON-serializable index data
        """
        category_path = self.root / category
        category_path.mkdir(parents=True, exist_ok=True)

        index_path = category_path / index_name
        temp_path = index_path.with_suffix(index_path.suffix + ".tmp")

        # Write to temp file first
        temp_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

        # Atomic replace: if interrupted here, temp file may exist but index is intact
        os.replace(temp_path, index_path)

        logger.info(
            "write_index_atomic",
            index_name=index_name,
            category=category,
            path=str(index_path),
        )

    def read_index(self, index_name: str, category: str) -> dict[str, Any] | None:
        """Read index file.

        Args:
            index_name: Index file name
            category: Category subfolder

        Returns:
            Parsed JSON data or None if file missing/corrupted
        """
        index_path = self.root / category / index_name

        if not index_path.exists():
            logger.debug(
                "read_index_missing",
                index_name=index_name,
                category=category,
            )
            return None

        try:
            data = json.loads(index_path.read_text(encoding="utf-8"))
            result: dict[str, Any] = data
            logger.info(
                "read_index",
                index_name=index_name,
                category=category,
                path=str(index_path),
            )
            return result
        except json.JSONDecodeError as exc:
            logger.error(
                "read_index_corrupted",
                index_name=index_name,
                category=category,
                path=str(index_path),
                exception=str(exc),
            )
            return None

    def list_artifacts(self, category: str) -> list[str]:
        """List all artifact IDs in category.

        Args:
            category: Category subfolder

        Returns:
            List of artifact IDs (without .json extension)
        """
        category_path = self.root / category

        if not category_path.exists():
            return []

        artifact_ids = [p.stem for p in category_path.glob("*.json") if p.is_file()]

        logger.info(
            "list_artifacts",
            category=category,
            count=len(artifact_ids),
        )

        return artifact_ids

    def delete_artifact(self, artifact_id: str, category: str) -> None:
        """Delete artifact file.

        Args:
            artifact_id: Unique identifier for the artifact
            category: Category subfolder
        """
        artifact_path = self.root / category / f"{artifact_id}.json"

        if artifact_path.exists():
            artifact_path.unlink()
            logger.info(
                "delete_artifact",
                artifact_id=artifact_id,
                category=category,
                path=str(artifact_path),
            )
        else:
            logger.debug(
                "delete_artifact_missing",
                artifact_id=artifact_id,
                category=category,
            )
