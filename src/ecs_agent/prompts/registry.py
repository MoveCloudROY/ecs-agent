"""Centralized registry for named prompt templates with keyword mapping."""

from ecs_agent.prompts.contracts import PromptTemplate


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
