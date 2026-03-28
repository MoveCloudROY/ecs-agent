"""Immutable skill discovery catalog descriptors."""

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
