"""Render normalized system prompts from SystemPromptConfigSpec."""

from __future__ import annotations

from pathlib import Path
from string import Template

from ecs_agent.components import (
    LLMComponent,
    RenderedSystemPromptComponent,
    SystemPromptComponent,
)
from ecs_agent.core.world import World
from ecs_agent.logging import get_logger
from ecs_agent.prompts.contracts import SystemPromptConfigSpec, PromptTemplateSource
from ecs_agent.prompts.provider import (
    BuiltinPlaceholderProvider,
    InventoryPlaceholderProvider,
)
from ecs_agent.prompts.registry import resolve_placeholder_values
from ecs_agent.scratchbook.prompt_definition import ScratchbookPromptConfig
from ecs_agent.scratchbook.prompt_provider import ScratchbookPromptPlaceholderProvider
from ecs_agent.types import EntityId

_BUILTIN_PLACEHOLDER_PROVIDERS: list[BuiltinPlaceholderProvider] = [
    InventoryPlaceholderProvider(),
]

logger = get_logger(__name__)


class SystemPromptRenderSystem:
    def __init__(self, priority: int = 0) -> None:
        self.priority = priority

    async def process(self, world: World) -> None:
        for entity_id, (prompt_config,) in world.query(SystemPromptConfigSpec):
            cache_key = _render_cache_key(world, entity_id)
            rendered_component = world.get_component(
                entity_id, RenderedSystemPromptComponent
            )
            if rendered_component is not None:
                previous_cache_key = rendered_component.placeholder_snapshot.get(
                    "_cache_key"
                )
                if previous_cache_key == cache_key:
                    _bridge_rendered_prompt(world, entity_id, rendered_component.text)
                    continue
                world.remove_component(entity_id, RenderedSystemPromptComponent)

                logger.debug(
                    "system_prompt_render_invalidated",
                    entity_id=entity_id,
                    previous_cache_key=previous_cache_key,
                    current_cache_key=cache_key,
                )
            try:
                rendered, snapshot = _render_system_prompt(
                    world, entity_id, prompt_config
                )
                snapshot["_cache_key"] = cache_key
                world.add_component(
                    entity_id,
                    RenderedSystemPromptComponent(
                        text=rendered,
                        placeholder_snapshot=snapshot,
                    ),
                )

                logger.debug(
                    "system_prompt_rendered",
                    entity_id=entity_id,
                    prompt_length=len(rendered),
                    placeholder_count=len(snapshot),
                    prompt_text=rendered,
                )

                _bridge_rendered_prompt(world, entity_id, rendered)
            except Exception as exc:
                logger.error(
                    "system_prompt_render_failed",
                    entity_id=entity_id,
                    exception=str(exc),
                )
                raise


def _render_system_prompt(
    world: World,
    entity_id: EntityId,
    prompt_config: SystemPromptConfigSpec,
) -> tuple[str, dict[str, str]]:
    template_text = _read_template(prompt_config.template_source)
    template = Template(template_text)

    user_values = _resolve_user_placeholders(prompt_config)
    builtins = _aggregate_provider_placeholders(world, entity_id, user_values)
    snapshot = {**user_values, **builtins}

    try:
        rendered = template.substitute(snapshot)
    except KeyError as exc:
        missing = str(exc).strip("'\"")
        raise ValueError(f"unknown placeholders in template: {missing}") from exc
    return rendered, snapshot


def _read_template(template_source: PromptTemplateSource) -> str:
    inline = template_source.inline
    file_path = template_source.file_path
    if inline is not None:
        return inline

    assert file_path is not None
    path = Path(file_path)
    if not path.exists():
        raise ValueError(f"missing template file: {file_path}")
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"unreadable template file: {file_path}") from exc


def _resolve_user_placeholders(prompt_config: SystemPromptConfigSpec) -> dict[str, str]:
    return resolve_placeholder_values(prompt_config.placeholders)


def _provider_id(provider: BuiltinPlaceholderProvider) -> str:
    provider_id = getattr(provider, "provider_id", None)
    if not isinstance(provider_id, str):
        raise ValueError("provider missing provider_id")
    return provider_id


def _aggregate_provider_placeholders(
    world: World,
    entity_id: EntityId,
    user_values: dict[str, str],
) -> dict[str, str]:
    aggregated: dict[str, str] = {}
    key_to_provider_id: dict[str, str] = {}

    for provider in _iter_placeholder_providers(world, entity_id):
        provider_id = _provider_id(provider)
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


def _render_cache_key(world: World, entity_id: EntityId) -> str:
    fingerprints: list[str] = []
    for provider in _iter_placeholder_providers(world, entity_id):
        provider_id = _provider_id(provider)
        fingerprint = provider.provider_fingerprint(world, entity_id)
        fingerprints.append(f"{provider_id}:{fingerprint}")
    return "|".join(fingerprints)


def _iter_placeholder_providers(
    world: World,
    entity_id: EntityId,
) -> list[BuiltinPlaceholderProvider]:
    providers = list(_BUILTIN_PLACEHOLDER_PROVIDERS)
    scratchbook_provider = _resolve_entity_scratchbook_provider(world, entity_id)
    if scratchbook_provider is not None:
        providers.append(scratchbook_provider)
    return providers


def _resolve_entity_scratchbook_provider(
    world: World,
    entity_id: EntityId,
) -> BuiltinPlaceholderProvider | None:
    config = world.get_component(entity_id, ScratchbookPromptConfig)
    if config is None:
        return None
    return ScratchbookPromptPlaceholderProvider(config)


def _bridge_rendered_prompt(world: World, entity_id: EntityId, rendered: str) -> None:
    llm_component = world.get_component(entity_id, LLMComponent)
    if llm_component is not None:
        llm_component.system_prompt = rendered

    legacy_system_prompt = world.get_component(entity_id, SystemPromptComponent)
    if legacy_system_prompt is not None:
        legacy_system_prompt.content = rendered


__all__ = ["SystemPromptRenderSystem"]
