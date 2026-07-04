"""Placeholder-provider template rendering shared by prompt-producing systems.

Neutral layer below ``ecs_agent.systems``: both ``SystemPromptRenderSystem``
and ``CompactionSystem`` render templates through these helpers, so neither
system module has to import the other's code.
"""

from __future__ import annotations

import re
from string import Template

from ecs_agent.components import PhaseComponent
from ecs_agent.core.world import World
from ecs_agent.phases.prompt_provider import PhasePromptPlaceholderProvider
from ecs_agent.prompts.contracts import SystemPromptConfigSpec
from ecs_agent.prompts.provider import (
    BuiltinPlaceholderProvider,
    CompactionSummaryPlaceholderProvider,
    InventoryPlaceholderProvider,
)
from ecs_agent.prompts.registry import resolve_placeholder_values
from ecs_agent.scratchbook.prompt_definition import ScratchbookPromptConfig
from ecs_agent.scratchbook.prompt_provider import ScratchbookPromptPlaceholderProvider
from ecs_agent.types import EntityId

_BUILTIN_PLACEHOLDER_PROVIDERS: list[BuiltinPlaceholderProvider] = [
    InventoryPlaceholderProvider(),
    CompactionSummaryPlaceholderProvider(),
]

PlaceholderProviderRegistry = list[BuiltinPlaceholderProvider]

PLACEHOLDER_NAME_RE = re.compile(
    r"\$(?:\{(?P<braced>[_a-zA-Z][_a-zA-Z0-9]*)\}|(?P<named>[_a-zA-Z][_a-zA-Z0-9]*))"
)
_ESCAPED_DOLLAR_SENTINEL = "\u0000ecs_agent_escaped_template_dollar\u0000"


def render_prompt_template(
    template: str,
    world: World,
    entity: EntityId,
    placeholder_registry: PlaceholderProviderRegistry | None = None,
    user_values: dict[str, str] | None = None,
) -> tuple[str, dict[str, str]]:
    resolved_user_values = (
        dict(user_values)
        if user_values is not None
        else _resolve_entity_user_placeholders(world, entity)
    )
    builtins = _aggregate_provider_placeholders(
        world,
        entity,
        resolved_user_values,
        placeholder_registry=placeholder_registry,
    )
    snapshot = {**resolved_user_values, **builtins}
    return substitute_prompt_template(template, snapshot), snapshot


def render_compaction_prompt(template: str, world: World, entity: EntityId) -> str:
    rendered, _ = render_prompt_template(template=template, world=world, entity=entity)
    return rendered


def substitute_prompt_template(template_text: str, snapshot: dict[str, str]) -> str:
    current = _mask_escaped_template_dollars(template_text)
    for _ in range(5):
        template = Template(current)
        try:
            rendered = _mask_escaped_template_dollars(template.substitute(snapshot))
        except KeyError as exc:
            missing = str(exc).strip("'\"")
            raise ValueError(f"unknown placeholders in template: {missing}") from exc
        if rendered == current:
            if PLACEHOLDER_NAME_RE.search(rendered) is not None:
                raise ValueError(
                    "recursive placeholder expansion did not converge; "
                    "unresolved placeholders remain"
                )
            return _unmask_escaped_template_dollars(rendered)
        current = rendered
    if PLACEHOLDER_NAME_RE.search(current) is not None:
        raise ValueError(
            "recursive placeholder expansion exceeded limit; "
            "unresolved placeholders remain"
        )
    return _unmask_escaped_template_dollars(current)


def placeholder_provider_id(provider: BuiltinPlaceholderProvider) -> str:
    provider_id = getattr(provider, "provider_id", None)
    if not isinstance(provider_id, str):
        raise ValueError("provider missing provider_id")
    return provider_id


def iter_placeholder_providers(
    world: World,
    entity_id: EntityId,
) -> list[BuiltinPlaceholderProvider]:
    providers = list(_BUILTIN_PLACEHOLDER_PROVIDERS)
    scratchbook_provider = _resolve_entity_scratchbook_provider(world, entity_id)
    if scratchbook_provider is not None:
        providers.append(scratchbook_provider)
    phase_provider = _resolve_entity_phase_provider(world, entity_id)
    if phase_provider is not None:
        providers.append(phase_provider)
    return providers


def _resolve_user_placeholders(prompt_config: SystemPromptConfigSpec) -> dict[str, str]:
    return resolve_placeholder_values(prompt_config.placeholders)


def _resolve_entity_user_placeholders(
    world: World, entity_id: EntityId
) -> dict[str, str]:
    prompt_config = world.get_component(entity_id, SystemPromptConfigSpec)
    if prompt_config is None:
        return {}
    return _resolve_user_placeholders(prompt_config)


def _mask_escaped_template_dollars(template_text: str) -> str:
    return template_text.replace("$$", _ESCAPED_DOLLAR_SENTINEL)


def _unmask_escaped_template_dollars(template_text: str) -> str:
    return template_text.replace(_ESCAPED_DOLLAR_SENTINEL, "$")


def _aggregate_provider_placeholders(
    world: World,
    entity_id: EntityId,
    user_values: dict[str, str],
    placeholder_registry: PlaceholderProviderRegistry | None = None,
) -> dict[str, str]:
    aggregated: dict[str, str] = {}
    key_to_provider_id: dict[str, str] = {}

    providers = (
        list(placeholder_registry)
        if placeholder_registry is not None
        else iter_placeholder_providers(world, entity_id)
    )

    for provider in providers:
        provider_id = placeholder_provider_id(provider)
        values = provider.resolve_placeholders(world, entity_id)
        for key, value in values.items():
            first_provider_id = key_to_provider_id.get(key)
            if first_provider_id is not None:
                raise ValueError(
                    f"duplicate built-in key '{key}': emitted by both "
                    f"'{first_provider_id}' and '{provider_id}'"
                )
            key_to_provider_id[key] = provider_id
            aggregated[key] = value

    for key in aggregated:
        if key in user_values:
            raise ValueError(
                f"built-in placeholder key '{key}' collides with user placeholder"
            )

    return aggregated


def _resolve_entity_scratchbook_provider(
    world: World,
    entity_id: EntityId,
) -> BuiltinPlaceholderProvider | None:
    config = world.get_component(entity_id, ScratchbookPromptConfig)
    if config is None:
        return None
    return ScratchbookPromptPlaceholderProvider(config)


def _resolve_entity_phase_provider(
    world: World,
    entity_id: EntityId,
) -> BuiltinPlaceholderProvider | None:
    if world.get_component(entity_id, PhaseComponent) is None:
        return None
    return PhasePromptPlaceholderProvider()


__all__ = [
    "PLACEHOLDER_NAME_RE",
    "PlaceholderProviderRegistry",
    "iter_placeholder_providers",
    "placeholder_provider_id",
    "render_compaction_prompt",
    "render_prompt_template",
    "substitute_prompt_template",
]
