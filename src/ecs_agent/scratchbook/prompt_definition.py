"""Scratchbook prompt definition contracts and validation rules."""

from __future__ import annotations

import re
from dataclasses import dataclass

from ecs_agent.logging import get_logger

logger = get_logger(__name__)

_NON_ALNUM_RUN_PATTERN = re.compile(r"[^a-z0-9]+")


def normalize_artifact_type_id(raw_name: str) -> str:
    normalized = _NON_ALNUM_RUN_PATTERN.sub("_", raw_name.strip().lower())
    return normalized.strip("_")


def _validate_root_relative_posix_path(path: str, *, field_name: str) -> None:
    if not path:
        raise ValueError(f"{field_name} must not be empty")
    if path.startswith("/"):
        raise ValueError(f"{field_name} must be root-relative, not absolute")
    if "\\" in path:
        raise ValueError(f"{field_name} must use POSIX separators")


@dataclass(slots=True)
class ScratchbookArtifactPromptDef:
    artifact_type_id: str
    path: str
    purpose: str
    readonly: bool
    read_when: str
    default_template_override: str | None = None
    user_override_template: str | None = None

    def __post_init__(self) -> None:
        normalized_type_id = normalize_artifact_type_id(self.artifact_type_id)
        if not normalized_type_id:
            raise ValueError(
                "artifact_type_id must contain at least one alphanumeric character "
                "after normalization"
            )
        self.artifact_type_id = normalized_type_id

        _validate_root_relative_posix_path(self.path, field_name="path")


@dataclass(slots=True)
class ScratchbookPromptConfig:
    overview_default_template: str | None
    scratchbook_root_path: str
    artifacts: list[ScratchbookArtifactPromptDef]

    def __post_init__(self) -> None:
        _validate_root_relative_posix_path(
            self.scratchbook_root_path,
            field_name="scratchbook_root_path",
        )

        normalized_seen: dict[str, ScratchbookArtifactPromptDef] = {}
        for artifact in self.artifacts:
            normalized_id = normalize_artifact_type_id(artifact.artifact_type_id)
            if not normalized_id:
                raise ValueError(
                    "artifact_type_id must contain at least one alphanumeric "
                    "character after normalization"
                )

            existing = normalized_seen.get(normalized_id)
            if existing is not None:
                raise ValueError(
                    "artifact_type_id values must not normalize to the same ID: "
                    f"{existing.artifact_type_id!r} and {artifact.artifact_type_id!r} "
                    f"both normalize to {normalized_id!r}"
                )
            normalized_seen[normalized_id] = artifact


__all__ = [
    "ScratchbookArtifactPromptDef",
    "ScratchbookPromptConfig",
    "normalize_artifact_type_id",
]
