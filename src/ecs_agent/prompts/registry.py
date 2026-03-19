"""Centralized registries for prompt templates and placeholder contracts."""

from __future__ import annotations

import re

from ecs_agent.prompts.contracts import (
    CORE_PLACEHOLDER_KEYS,
    PlaceholderContract,
    PromptTemplate,
)

_PLACEHOLDER_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class PromptRegistry:
    """Central store for named prompt templates with keyword mapping."""

    def __init__(self) -> None:
        self._templates: dict[str, PromptTemplate] = {}
        self._keyword_map: dict[str, str] = {}  # keyword -> template_id

    def register(self, template: PromptTemplate) -> None:
        """Register a template. Raises ValueError if template_id already exists.

        Args:
            template: The PromptTemplate to register.
        """
        if template.template_id in self._templates:
            raise ValueError(f"Template already registered: {template.template_id!r}")
        self._templates[template.template_id] = template

    def get(self, template_id: str) -> PromptTemplate:
        """Return template by id. Raises ValueError with template_id context if unknown.

        Args:
            template_id: The unique identifier of the template.

        Returns:
            The matching PromptTemplate.
        """
        if template_id not in self._templates:
            raise ValueError(f"Unknown template: {template_id!r}")
        return self._templates[template_id]

    def register_keyword(self, keyword: str, template_id: str) -> None:
        """Map keyword to an already-registered template_id.

        Args:
            keyword: The keyword string to map (e.g. "@code").
            template_id: Must already be registered. Raises ValueError otherwise.
        """
        if template_id not in self._templates:
            raise ValueError(
                f"Cannot register keyword {keyword!r}: unknown template {template_id!r}"
            )
        self._keyword_map[keyword] = template_id

    def resolve_keyword(self, keyword: str) -> PromptTemplate | None:
        """Return template for keyword, or None if no mapping exists.

        Args:
            keyword: The keyword to resolve.

        Returns:
            The matching PromptTemplate, or None if keyword is unmapped.
        """
        template_id = self._keyword_map.get(keyword)
        if template_id is None:
            return None
        return self._templates[template_id]

    def list_templates(self) -> list[PromptTemplate]:
        """Return templates in stable insertion order (deterministic)."""
        return list(self._templates.values())

    def list_keywords(self) -> list[str]:
        """Return registered keywords in stable insertion order."""
        return list(self._keyword_map.keys())


class PlaceholderRegistry:
    """Registry for strict core placeholders and validated extension placeholders."""

    def __init__(self) -> None:
        self._extensions: dict[str, PlaceholderContract] = {}

    def register_extension(self, key: str) -> None:
        """Register a non-core placeholder key.

        Args:
            key: Placeholder key following string.Template identifier grammar.
        """
        _validate_placeholder_identifier(key)
        if key in CORE_PLACEHOLDER_KEYS:
            raise ValueError(f"Cannot override core placeholder: {key!r}")
        if key in self._extensions:
            raise ValueError(f"Extension placeholder already registered: {key!r}")
        self._extensions[key] = PlaceholderContract(key=key, is_core=False)

    def register_extensions(self, keys: list[str]) -> None:
        """Register multiple extension placeholder keys."""
        for key in keys:
            self.register_extension(key)

    def contains(self, key: str) -> bool:
        """Return whether key is known as core or extension placeholder."""
        return key in CORE_PLACEHOLDER_KEYS or key in self._extensions

    def core_keys(self) -> tuple[str, ...]:
        """Return strict core placeholder keys in canonical order."""
        return CORE_PLACEHOLDER_KEYS

    def extension_keys(self) -> tuple[str, ...]:
        """Return extension keys sorted lexicographically for deterministic ordering."""
        return tuple(sorted(self._extensions))

    def ordered_keys(self) -> tuple[str, ...]:
        """Return deterministic key order: core first, extensions sorted."""
        return (*CORE_PLACEHOLDER_KEYS, *self.extension_keys())

    def contracts(self) -> tuple[PlaceholderContract, ...]:
        """Return deterministic list of all placeholder contracts."""
        core_contracts = tuple(
            PlaceholderContract(key=key, is_core=True) for key in CORE_PLACEHOLDER_KEYS
        )
        extension_contracts = tuple(
            self._extensions[key] for key in self.extension_keys()
        )
        return (*core_contracts, *extension_contracts)

    def validate_core_placeholders(self, template: str) -> None:
        """Raise ValueError if any required core placeholder is missing from template."""
        missing = [
            key
            for key in CORE_PLACEHOLDER_KEYS
            if not _template_contains_placeholder(template, key)
        ]
        if missing:
            missing_fields = "|".join(missing)
            raise ValueError(f"missing required core placeholders: {missing_fields}")


def _template_contains_placeholder(template: str, placeholder_key: str) -> bool:
    return f"${{{placeholder_key}}}" in template or f"${placeholder_key}" in template


def _validate_placeholder_identifier(key: str) -> None:
    if not key:
        raise ValueError("Placeholder key cannot be empty")
    if not _PLACEHOLDER_IDENTIFIER_PATTERN.match(key):
        raise ValueError(
            f"Invalid placeholder key {key!r}: must match [A-Za-z_][A-Za-z0-9_]*"
        )
