"""System that assembles opt-in system prompts from section contributions."""

from __future__ import annotations

from string import Template

from ecs_agent.components import (
    PromptConfigComponent,
    SystemPromptComponent,
)
from ecs_agent.core.world import World
from ecs_agent.placeholder import StrictPlaceholderRenderer
from ecs_agent.prompts.contracts import PromptSectionSpec
from ecs_agent.prompts.registry import PlaceholderRegistry
from ecs_agent.prompts.sections import render_sections

_PLACEHOLDER_REGISTRY = PlaceholderRegistry()


class SystemPromptAssemblySystem:
    """Compose SystemPromptComponent content for opt-in entities only."""

    def __init__(self, priority: int = 0) -> None:
        self.priority = priority
        self._renderer = StrictPlaceholderRenderer()

    async def process(self, world: World) -> None:
        for _, components in world.query(
            PromptConfigComponent,
            SystemPromptComponent,
        ):
            _, system_prompt = components
            if not system_prompt.template:
                system_prompt.content = render_sections(system_prompt.sections)
                continue

            placeholder_registry = _build_placeholder_registry(system_prompt)
            placeholder_registry.validate_core_placeholders(system_prompt.template)
            _validate_template_placeholders(system_prompt, placeholder_registry)
            snapshot = _build_placeholder_snapshot(system_prompt, placeholder_registry)
            system_prompt.content = self._renderer.substitute(
                system_prompt.template, snapshot
            )


def _build_placeholder_registry(
    system_prompt: SystemPromptComponent,
) -> PlaceholderRegistry:
    placeholder_registry = PlaceholderRegistry()
    template_keys = sorted(set(Template(system_prompt.template).get_identifiers()))
    section_titles = {section.title for section in system_prompt.sections}

    for key in template_keys:
        if key in _PLACEHOLDER_REGISTRY.core_keys():
            continue
        if key in section_titles:
            placeholder_registry.register_extension(key)

    return placeholder_registry


def _validate_template_placeholders(
    system_prompt: SystemPromptComponent,
    placeholder_registry: PlaceholderRegistry,
) -> None:
    template = Template(system_prompt.template)
    unknown_placeholders = sorted(
        {
            key
            for key in template.get_identifiers()
            if not placeholder_registry.contains(key)
        }
    )
    if unknown_placeholders:
        unknown_fields = "|".join(unknown_placeholders)
        raise ValueError(f"unknown placeholders in template: {unknown_fields}")


def _build_placeholder_snapshot(
    system_prompt: SystemPromptComponent,
    placeholder_registry: PlaceholderRegistry,
) -> dict[str, str]:
    snapshot: dict[str, str] = {}
    for key in placeholder_registry.ordered_keys():
        placeholder_sections = _sections_for_placeholder_key(
            system_prompt.sections, key
        )
        snapshot[key] = render_sections(placeholder_sections)
    return snapshot


def _sections_for_placeholder_key(
    sections: list[PromptSectionSpec], placeholder_key: str
) -> list[PromptSectionSpec]:
    return [section for section in sections if section.title == placeholder_key]


__all__ = ["SystemPromptAssemblySystem"]
