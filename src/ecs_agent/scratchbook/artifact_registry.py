"""Artifact registry for canonical scratchbook artifact IDs and paths."""

from __future__ import annotations

import asyncio
import os
import json
import re
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from ecs_agent.logging import get_logger

logger = get_logger(__name__)

_INLINE_THRESHOLD_BYTES = 2048
_ARTIFACT_ID_PATTERN = re.compile(r"^(tool|subagent)_[0-9a-f]{24}$")
_NON_ALNUM_RUN_PATTERN = re.compile(r"[^a-z0-9]+")
_boulder_locks: dict[str, asyncio.Lock] = {}
_boulder_locks_guard = asyncio.Lock()


class ArtifactKind(Enum):
    """Artifact kind for registry path routing."""

    TOOL = "tool"
    SUBAGENT = "subagent"
    PLAN = "plan"
    BOULDER = "boulder"


@dataclass(slots=True)
class ArtifactDescriptor:
    """Durable descriptor for a persisted artifact."""

    artifact_id: str
    kind: ArtifactKind
    record_path: str
    created_at: datetime
    inline_content: str | None


@dataclass(slots=True)
class ArtifactPersistResult:
    """Persist result envelope with descriptor and convenience fields."""

    descriptor: ArtifactDescriptor
    record_path: str
    inline_content: str | None


class ArtifactRegistry:
    """Canonical artifact ID and path registry over scratchbook storage."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def persist(self, *, kind: ArtifactKind, content: str) -> ArtifactPersistResult:
        artifact_id = self._generate_artifact_id(kind)
        record_path = self.record_path_for(kind=kind, artifact_id=artifact_id)
        self._write_text_atomic(record_path=record_path, content=content)

        inline_content = self._inline_content_for(content)
        descriptor = ArtifactDescriptor(
            artifact_id=artifact_id,
            kind=kind,
            record_path=record_path,
            created_at=datetime.now(timezone.utc),
            inline_content=inline_content,
        )
        return ArtifactPersistResult(
            descriptor=descriptor,
            record_path=record_path,
            inline_content=inline_content,
        )

    async def capture_stream(
        self,
        *,
        kind: ArtifactKind,
        source: AsyncIterator[str],
    ) -> ArtifactDescriptor:
        artifact_id = self._generate_artifact_id(kind)
        record_path = self.record_path_for(kind=kind, artifact_id=artifact_id)

        final_path = self.root / record_path
        final_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = final_path.with_name(f".{final_path.name}.{uuid.uuid4().hex}.tmp")

        try:
            with temp_path.open("w", encoding="utf-8", buffering=8192) as f:
                async for chunk in source:
                    f.write(chunk)
        except asyncio.CancelledError:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                logger.warning(
                    "capture_stream_cleanup_failed", temp_path=str(temp_path)
                )
            raise
        except Exception:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                logger.warning(
                    "capture_stream_cleanup_failed", temp_path=str(temp_path)
                )
            raise

        os.replace(temp_path, final_path)
        content = final_path.read_text(encoding="utf-8")
        inline_content = self._inline_content_for(content)
        return ArtifactDescriptor(
            artifact_id=artifact_id,
            kind=kind,
            record_path=record_path,
            created_at=datetime.now(timezone.utc),
            inline_content=inline_content,
        )

    def normalize_plan_name(self, raw_name: str) -> str:
        normalized = _NON_ALNUM_RUN_PATTERN.sub("-", raw_name.strip().lower())
        slug = normalized.strip("-")
        if slug:
            return slug
        return "plan"

    def resolve_plan_name_collision(
        self,
        *,
        normalized_plan_name: str,
        existing_plan_names: set[str],
    ) -> str:
        if normalized_plan_name not in existing_plan_names:
            return normalized_plan_name

        suffix = 2
        while True:
            candidate = f"{normalized_plan_name}-{suffix}"
            if candidate not in existing_plan_names:
                return candidate
            suffix += 1

    def plan_path(self, *, plan_name: str) -> str:
        slug = self.normalize_plan_name(plan_name)
        return f"scratchbook/{slug}/plan.md"

    def write_plan(self, *, plan_name: str, content: str) -> str:
        record_path = self.plan_path(plan_name=plan_name)
        self._write_text_atomic(record_path=record_path, content=content)
        return record_path

    def boulder_path(self, *, plan_name: str) -> str:
        slug = self.normalize_plan_name(plan_name)
        return f"scratchbook/{slug}/executes/boulder.json"

    def create_boulder(self, *, plan_name: str, initial_data: dict[str, Any]) -> str:
        record_path = self.boulder_path(plan_name=plan_name)
        target_path = self.root / record_path
        if target_path.exists():
            return record_path

        slug = self.normalize_plan_name(plan_name)
        payload: dict[str, Any] = dict(initial_data)
        payload["schema_version"] = "1"
        payload["plan_name"] = slug
        payload["active_plan"] = plan_name
        payload.setdefault(
            "started_at",
            datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        )
        payload.setdefault("status", "created")

        content = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
        self._write_text_atomic(record_path=record_path, content=content)
        return record_path

    async def update_boulder(self, *, plan_name: str, updates: dict[str, Any]) -> str:
        record_path = self.boulder_path(plan_name=plan_name)
        target_path = self.root / record_path
        lock_key = str(target_path)

        async with _boulder_locks_guard:
            if lock_key not in _boulder_locks:
                _boulder_locks[lock_key] = asyncio.Lock()
            lock = _boulder_locks[lock_key]

        async with lock:
            now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            slug = self.normalize_plan_name(plan_name)

            existing: dict[str, Any] = {}
            if target_path.exists():
                raw = target_path.read_text(encoding="utf-8")
                loaded = json.loads(raw)
                if isinstance(loaded, dict):
                    existing = loaded

            preserved_schema_version = str(existing.get("schema_version", "1"))
            preserved_plan_name = str(existing.get("plan_name", slug))
            preserved_active_plan = str(existing.get("active_plan", plan_name))
            preserved_started_at = str(existing.get("started_at", now))

            payload: dict[str, Any] = dict(existing)
            payload.update(updates)
            payload["schema_version"] = preserved_schema_version
            payload["plan_name"] = preserved_plan_name
            payload["active_plan"] = preserved_active_plan
            payload["started_at"] = preserved_started_at
            payload["last_updated_at"] = now
            payload.setdefault("status", "created")

            content = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
            self._write_text_atomic(record_path=record_path, content=content)
            return record_path

    def is_valid_artifact_id(self, artifact_id: str) -> bool:
        return _ARTIFACT_ID_PATTERN.fullmatch(artifact_id) is not None

    def record_path_for(self, *, kind: ArtifactKind, artifact_id: str) -> str:
        if kind is ArtifactKind.TOOL:
            return f"scratchbook/records/tool/{artifact_id}"
        if kind is ArtifactKind.SUBAGENT:
            return f"scratchbook/records/subagent/{artifact_id}"
        raise ValueError("record_path_for only supports TOOL and SUBAGENT artifacts")

    def _generate_artifact_id(self, kind: ArtifactKind) -> str:
        uuid24 = uuid.uuid4().hex[:24]
        if kind is ArtifactKind.TOOL:
            return f"tool_{uuid24}"
        if kind is ArtifactKind.SUBAGENT:
            return f"subagent_{uuid24}"
        raise ValueError("persist only supports TOOL and SUBAGENT artifacts")

    def _inline_content_for(self, content: str) -> str | None:
        size_bytes = len(content.encode("utf-8"))
        if size_bytes <= _INLINE_THRESHOLD_BYTES:
            return content
        return None

    def _write_text_atomic(self, *, record_path: str, content: str) -> None:
        target_path = self.root / record_path
        target_path.parent.mkdir(parents=True, exist_ok=True)

        temp_path = target_path.with_name(f".{target_path.name}.{uuid.uuid4().hex}.tmp")
        temp_path.write_text(content, encoding="utf-8")
        os.replace(temp_path, target_path)
