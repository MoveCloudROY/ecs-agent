"""Render normalized system prompts from SystemPromptConfigSpec."""

from __future__ import annotations

import re
from pathlib import Path
from string import Template

from ecs_agent.components import (
    CompactionConfigComponent,
    LLMComponent,
    RenderedSystemPromptComponent,
    SystemPromptComponent,
    WorkflowBindingComponent,
)
from ecs_agent.core.world import World
from ecs_agent.logging import get_logger
from ecs_agent.prompts.contracts import SystemPromptConfigSpec, PromptTemplateSource
from ecs_agent.prompts.provider import (
    BuiltinPlaceholderProvider,
    CompactionSummaryPlaceholderProvider,
    InventoryPlaceholderProvider,
)
from ecs_agent.prompts.registry import resolve_placeholder_values
from ecs_agent.scratchbook.prompt_definition import ScratchbookPromptConfig
from ecs_agent.scratchbook.prompt_provider import ScratchbookPromptPlaceholderProvider
from ecs_agent.types import EntityId, PromptReplacementEvent
from ecs_agent.workflows.prompt_provider import WorkflowPromptPlaceholderProvider

_BUILTIN_PLACEHOLDER_PROVIDERS: list[BuiltinPlaceholderProvider] = [
    InventoryPlaceholderProvider(),
    CompactionSummaryPlaceholderProvider(),
]

logger = get_logger(__name__)

PlaceholderProviderRegistry = list[BuiltinPlaceholderProvider]
_PLACEHOLDER_NAME_RE = re.compile(
    r"\$(?:\{(?P<braced>[_a-zA-Z][_a-zA-Z0-9]*)\}|(?P<named>[_a-zA-Z][_a-zA-Z0-9]*))"
)


class SystemPromptRenderSystem:
    def __init__(self, priority: int = 0) -> None:
        self.priority = priority

    async def process(self, world: World) -> None:
        for entity_id, prompt_config, template_metadata in _iter_render_targets(world):
            cache_key = (
                _render_cache_key(world, entity_id) + template_metadata.cache_suffix
            )
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
                template_text = _normalized_template_text(
                    world,
                    entity_id,
                    prompt_config,
                )
                rendered, snapshot = _render_system_prompt(
                    world, entity_id, prompt_config
                )
                snapshot["_cache_key"] = cache_key
                if template_metadata.legacy_template is not None:
                    snapshot["_legacy_template"] = template_metadata.legacy_template
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
                replacements = _prompt_replacements(template_text, snapshot)
                if rendered != template_text and replacements:
                    await world.event_bus.publish(
                        PromptReplacementEvent(
                            entity_id=entity_id,
                            prompt_kind="system",
                            source_text=template_text,
                            rendered_text=rendered,
                            replacements=replacements,
                            metadata={
                                "system_name": (
                                    "ecs_agent.systems.system_prompt_render_system."
                                    "SystemPromptRenderSystem"
                                ),
                            },
                        )
                    )
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
    template_text = _normalized_template_text(world, entity_id, prompt_config)
    rendered, snapshot = render_prompt_template(
        template=template_text,
        world=world,
        entity=entity_id,
    )
    return rendered, snapshot


def _normalized_template_text(
    world: World,
    entity_id: EntityId,
    prompt_config: SystemPromptConfigSpec,
) -> str:
    template_text = _read_template(prompt_config.template_source)
    return _normalize_compaction_summary_template(world, entity_id, template_text)


def _prompt_replacements(
    template_text: str,
    snapshot: dict[str, str],
) -> dict[str, str]:
    replacements: dict[str, str] = {}
    for match in _PLACEHOLDER_NAME_RE.finditer(template_text):
        if match.start() > 0 and template_text[match.start() - 1] == "$":
            continue
        name = match.group("braced") or match.group("named")
        if name is None:
            continue
        if name in snapshot:
            replacements[name] = snapshot[name]
    return replacements


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
    return _substitute_prompt_template(template, snapshot), snapshot


def render_compaction_prompt(template: str, world: World, entity: EntityId) -> str:
    rendered, _ = render_prompt_template(template=template, world=world, entity=entity)
    return rendered


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


class _RenderTemplateMetadata:
    def __init__(
        self,
        *,
        cache_suffix: str = "",
        legacy_template: str | None = None,
    ) -> None:
        self.cache_suffix = cache_suffix
        self.legacy_template = legacy_template


def _iter_render_targets(
    world: World,
) -> list[tuple[EntityId, SystemPromptConfigSpec, _RenderTemplateMetadata]]:
    targets: list[tuple[EntityId, SystemPromptConfigSpec, _RenderTemplateMetadata]] = []
    configured_entities: set[EntityId] = set()

    for entity_id, (prompt_config,) in world.query(SystemPromptConfigSpec):
        configured_entities.add(entity_id)
        targets.append((entity_id, prompt_config, _RenderTemplateMetadata()))

    for entity_id, _ in world.query(CompactionConfigComponent):
        if entity_id in configured_entities:
            continue
        legacy_template = _resolve_legacy_prompt_template(world, entity_id)
        if legacy_template is None:
            continue
        targets.append(
            (
                entity_id,
                SystemPromptConfigSpec(
                    template_source=PromptTemplateSource(inline=legacy_template)
                ),
                _RenderTemplateMetadata(
                    cache_suffix=f"|legacy:{legacy_template}",
                    legacy_template=legacy_template,
                ),
            )
        )

    return targets


_XML_TAIL_RE = re.compile(
    r"\n\$\{_chat_history_summary_xml\}$"
    r"|\n\$_chat_history_summary_xml$"
    r"|\n<chat_history_summary>[^<]*</chat_history_summary>$",
)


def _strip_compaction_xml_tail(template: str) -> str:
    return _XML_TAIL_RE.sub("", template)


def _resolve_legacy_prompt_template(world: World, entity_id: EntityId) -> str | None:
    rendered_component = world.get_component(entity_id, RenderedSystemPromptComponent)
    if rendered_component is not None:
        cached_template = rendered_component.placeholder_snapshot.get(
            "_legacy_template"
        )
        if cached_template is not None:
            return cached_template

    legacy_system_prompt = world.get_component(entity_id, SystemPromptComponent)
    if legacy_system_prompt is not None and legacy_system_prompt.content:
        return _strip_compaction_xml_tail(legacy_system_prompt.content)

    llm_component = world.get_component(entity_id, LLMComponent)
    if llm_component is not None and llm_component.system_prompt:
        return _strip_compaction_xml_tail(llm_component.system_prompt)

    return None


def _normalize_compaction_summary_template(
    world: World,
    entity_id: EntityId,
    template_text: str,
) -> str:
    if world.get_component(entity_id, CompactionConfigComponent) is None:
        return template_text
    placeholder = "${_chat_history_summary_xml}"
    if placeholder in template_text or "$_chat_history_summary_xml" in template_text:
        return template_text
    if not template_text:
        return placeholder
    return f"{template_text}\n{placeholder}"


def _resolve_user_placeholders(prompt_config: SystemPromptConfigSpec) -> dict[str, str]:
    return resolve_placeholder_values(prompt_config.placeholders)


def _resolve_entity_user_placeholders(
    world: World, entity_id: EntityId
) -> dict[str, str]:
    prompt_config = world.get_component(entity_id, SystemPromptConfigSpec)
    if prompt_config is None:
        return {}
    return _resolve_user_placeholders(prompt_config)


def _substitute_prompt_template(template_text: str, snapshot: dict[str, str]) -> str:
    template = Template(template_text)

    try:
        return template.substitute(snapshot)
    except KeyError as exc:
        missing = str(exc).strip("'\"")
        raise ValueError(f"unknown placeholders in template: {missing}") from exc


def _provider_id(provider: BuiltinPlaceholderProvider) -> str:
    provider_id = getattr(provider, "provider_id", None)
    if not isinstance(provider_id, str):
        raise ValueError("provider missing provider_id")
    return provider_id


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
        else _iter_placeholder_providers(world, entity_id)
    )

    for provider in providers:
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
        if not provider.resolve_placeholders(world, entity_id):
            continue
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
    workflow_provider = _resolve_entity_workflow_provider(world, entity_id)
    if workflow_provider is not None:
        providers.append(workflow_provider)
    return providers


def _resolve_entity_scratchbook_provider(
    world: World,
    entity_id: EntityId,
) -> BuiltinPlaceholderProvider | None:
    config = world.get_component(entity_id, ScratchbookPromptConfig)
    if config is None:
        return None
    return ScratchbookPromptPlaceholderProvider(config)


def _resolve_entity_workflow_provider(
    world: World,
    entity_id: EntityId,
) -> BuiltinPlaceholderProvider | None:
    binding = world.get_component(entity_id, WorkflowBindingComponent)
    if binding is None:
        return None
    return WorkflowPromptPlaceholderProvider()


def _bridge_rendered_prompt(world: World, entity_id: EntityId, rendered: str) -> None:
    llm_component = world.get_component(entity_id, LLMComponent)
    if llm_component is not None:
        llm_component.system_prompt = rendered

    legacy_system_prompt = world.get_component(entity_id, SystemPromptComponent)
    if legacy_system_prompt is not None:
        legacy_system_prompt.content = rendered


__all__ = [
    "SystemPromptRenderSystem",
    "render_compaction_prompt",
    "render_prompt_template",
]
