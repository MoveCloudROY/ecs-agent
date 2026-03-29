"""Immutable skill discovery catalog descriptors and process-level registry."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class SkillType(Enum):
    """Type discriminator for discovered skills."""

    MARKDOWN = "markdown"
    SCRIPT = "script"


@dataclass(frozen=True, slots=True)
class SkillDescriptor:
    """Immutable metadata-first descriptor for a discoverable skill."""

    name: str
    skill_type: SkillType
    source_path: Path
    _materializer: Callable[[], Any] = field(repr=False)
    metadata: dict[str, Any] = field(default_factory=dict)

    def materialize(self) -> Any:
        """Create a runtime skill instance from this descriptor."""

        return self._materializer()


# ---------------------------------------------------------------------------
# Process-level global catalog — stores immutable descriptors only.
# Never store workspace-bound or entity-bound state here.
# ---------------------------------------------------------------------------
_CATALOG: dict[str, SkillDescriptor] = {}


def register(descriptor: SkillDescriptor) -> None:
    """Register a descriptor in the process-level catalog (last write wins)."""
    _CATALOG[descriptor.name] = descriptor


def lookup(name: str) -> SkillDescriptor | None:
    """Return the catalog descriptor for *name*, or None if not registered."""
    return _CATALOG.get(name)


def all_descriptors() -> list[SkillDescriptor]:
    """Return a snapshot of all registered descriptors."""
    return list(_CATALOG.values())


def clear_catalog() -> None:
    """Remove all entries — intended for test isolation only."""
    _CATALOG.clear()
