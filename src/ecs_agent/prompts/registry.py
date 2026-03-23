"""Prompt template registry and placeholder resolution utilities."""
from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from ecs_agent.prompts.contracts import PlaceholderSpec, PromptTemplate

_PLACEHOLDER_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_LEGACY_REQUIRED_TEMPLATE_KEYS: tuple[str, str, str] = (
    "toolSelection",
    "exploreSection",
    "librarianSection",
)


@dataclass(slots=True)
class PlaceholderContract:
    """Contract describing a placeholder key (core or extension)."""
    key: str
    is_core: bool


class PromptRegistry:
    def __init__(self) -> None:
        self._templates: dict[str, PromptTemplate] = {}
        self._keyword_map: dict[str, str] = {}

    def register(self, template: PromptTemplate) -> None:
        if template.template_id in self._templates:
            raise ValueError(f"Template already registered: {template.template_id!r}")
        self._templates[template.template_id] = template

    def get(self, template_id: str) -> PromptTemplate:
        if template_id not in self._templates:
            raise ValueError(f"Unknown template: {template_id!r}")
        return self._templates[template_id]

    def register_keyword(self, keyword: str, template_id: str) -> None:
        if template_id not in self._templates:
            raise ValueError(
                f"Cannot register keyword {keyword!r}: unknown template {template_id!r}"
            )
        self._keyword_map[keyword] = template_id

    def resolve_keyword(self, keyword: str) -> PromptTemplate | None:
        template_id = self._keyword_map.get(keyword)
        if template_id is None:
            return None
        return self._templates[template_id]

    def list_templates(self) -> list[PromptTemplate]:
        return list(self._templates.values())

    def list_keywords(self) -> list[str]:
        return list(self._keyword_map.keys())


class PlaceholderRegistry:
    def __init__(self) -> None:
        self._extensions: set[str] = set()

    def register_extension(self, key: str) -> None:
        _validate_placeholder_identifier(key)
        if key in _LEGACY_REQUIRED_TEMPLATE_KEYS:
            raise ValueError(f"Cannot override core placeholder: {key!r}")
        if key in self._extensions:
            raise ValueError(f"Extension placeholder already registered: {key!r}")
        self._extensions.add(key)

    def register_extensions(self, keys: list[str]) -> None:
        for key in keys:
            self.register_extension(key)

    def contains(self, key: str) -> bool:
        return key in _LEGACY_REQUIRED_TEMPLATE_KEYS or key in self._extensions

    def core_keys(self) -> tuple[str, ...]:
        return _LEGACY_REQUIRED_TEMPLATE_KEYS

    def extension_keys(self) -> tuple[str, ...]:
        return tuple(sorted(self._extensions))

    def ordered_keys(self) -> tuple[str, ...]:
        return (*_LEGACY_REQUIRED_TEMPLATE_KEYS, *self.extension_keys())

    def contracts(self) -> tuple[PlaceholderContract, ...]:
        """Return all placeholders (core and extensions) as contract objects."""
        contracts: list[PlaceholderContract] = []
        # Add core placeholders
        for key in _LEGACY_REQUIRED_TEMPLATE_KEYS:
            contracts.append(PlaceholderContract(key=key, is_core=True))
        # Add extensions (sorted)
        for key in self.extension_keys():
            contracts.append(PlaceholderContract(key=key, is_core=False))
        return tuple(contracts)
    def validate_core_placeholders(self, template: str) -> None:
        missing = [
            key
            for key in _LEGACY_REQUIRED_TEMPLATE_KEYS
            if not _template_contains_placeholder(template, key)
        ]
        if missing:
            missing_fields = "|".join(missing)
            raise ValueError(f"missing required core placeholders: {missing_fields}")


def resolve_placeholder_values(placeholders: list[PlaceholderSpec]) -> dict[str, str]:
    resolved: dict[str, str] = {}
    for spec in placeholders:
        if spec.name in resolved:
            raise ValueError(f"Duplicate placeholder name: {spec.name!r}")
        value = _resolve_placeholder_value(spec.name, spec.value)
        resolved[spec.name] = value
    return resolved


def _resolve_placeholder_value(name: str, value: str | Callable[[], str]) -> str:
    if isinstance(value, str):
        return value

    rendered = value()
    if not isinstance(rendered, str):
        raise ValueError(f"Placeholder callable for {name!r} must return str")
    return rendered


def _template_contains_placeholder(template: str, placeholder_key: str) -> bool:
    return f"${{{placeholder_key}}}" in template or f"${placeholder_key}" in template


def _validate_placeholder_identifier(key: str) -> None:
    if not key:
        raise ValueError("Placeholder key cannot be empty")
    if not _PLACEHOLDER_IDENTIFIER_PATTERN.match(key):
        raise ValueError(
            f"Invalid placeholder key {key!r}: must match [A-Za-z_][A-Za-z0-9_]*"
        )
