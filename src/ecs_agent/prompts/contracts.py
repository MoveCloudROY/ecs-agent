from dataclasses import dataclass, field
from typing import Any, Literal
from collections.abc import Callable
import re


_PLACEHOLDER_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True, slots=True)
class PromptTemplateSource:
    inline: str | None = None
    file_path: str | None = None

    def __post_init__(self) -> None:
        has_inline = self.inline is not None
        has_file_path = self.file_path is not None
        if has_inline == has_file_path:
            raise ValueError(
                "PromptTemplateSource requires exactly one of inline or file_path"
            )


@dataclass(frozen=True, slots=True)
class PlaceholderSpec:
    name: str
    value: str | Callable[[], str]

    def __post_init__(self) -> None:
        if not _PLACEHOLDER_IDENTIFIER_PATTERN.match(self.name):
            raise ValueError(
                "Invalid placeholder name "
                f"{self.name!r}: must match [A-Za-z_][A-Za-z0-9_]*"
            )
        if self.name.startswith("_"):
            raise ValueError(
                f"Invalid placeholder name {self.name!r}: names starting with '_' are reserved"
            )


@dataclass(frozen=True, slots=True)
class SystemPromptConfigSpec:
    template_source: PromptTemplateSource
    placeholders: list[PlaceholderSpec] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class TriggerSpec:
    pattern: str
    match_mode: Literal["keyword", "prefix", "contains"]
    action: Literal["replace", "skill", "script"]
    content: str
    priority: int = 0


@dataclass(slots=True)
class PromptTemplate:
    template_id: str
    content: str
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PromptSectionSpec:
    title: str
    lines: list[str] = field(default_factory=list)
    priority: int = 0


@dataclass(slots=True)
class PromptRenderContext:
    variables: dict[str, str] = field(default_factory=dict)
    entity_id: int | None = None


@dataclass(slots=True)
class PromptInjectionArtifact:
    keyword: str
    block: str
    source_template_id: str
