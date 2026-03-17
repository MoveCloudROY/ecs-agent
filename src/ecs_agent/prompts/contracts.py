"""Prompt contract dataclasses for the prompt normalization pipeline."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class PromptTemplate:
    """A named prompt template with content and optional metadata."""

    template_id: str  # unique key, e.g. "coding-assistant"
    content: str  # the template text (may include {placeholders})
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PromptSectionSpec:
    """A named section with display title and content lines."""

    title: str
    lines: list[str] = field(default_factory=list)
    priority: int = 0  # higher = rendered earlier


@dataclass(slots=True)
class PromptRenderContext:
    """Variables injected into templates at render time."""

    variables: dict[str, str] = field(default_factory=dict)
    entity_id: int | None = None


@dataclass(slots=True)
class PromptInjectionArtifact:
    """A resolved injection block ready for prepending."""

    keyword: str  # the trigger keyword matched
    block: str  # fully rendered injection text
    source_template_id: str
