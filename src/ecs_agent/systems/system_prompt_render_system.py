"""Render normalized system prompts from SystemPromptConfigSpec."""

from __future__ import annotations

import re
from pathlib import Path

from ecs_agent.components import (
    CompactionConfigComponent,
    LLMComponent,
    RenderedSystemPromptComponent,
    SystemPromptComponent,
)
from ecs_agent.core.world import World
from ecs_agent.logging import get_logger
from ecs_agent.prompts.contracts import SystemPromptConfigSpec, PromptTemplateSource
from ecs_agent.prompts.template_render import (
    PLACEHOLDER_NAME_RE,
    PlaceholderProviderRegistry,
    iter_placeholder_providers,
    placeholder_provider_id,
    render_compaction_prompt,
    render_prompt_template,
    substitute_prompt_template,
)
from ecs_agent.types import EntityId, PromptReplacementEvent

logger = get_logger(__name__)

# Placeholders whose value changes *during* a conversation (compaction summary
# refreshes on compaction; phase prompt changes on phase transitions).
# They are emptied from the cache-stable prefix and relocated to a volatile tail
# so the prefix stays byte-stable for Anthropic prompt caching (ISSUE-6).
# Ordering here defines the deterministic tail order.
_VOLATILE_PLACEHOLDER_KEYS: tuple[str, ...] = (
    "_phase_prompt",
    "_chat_history_summary_xml",
)
VOLATILE_PLACEHOLDER_KEYS = frozenset(_VOLATILE_PLACEHOLDER_KEYS)


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
                rendered, stable_text, volatile_text, snapshot = _render_system_prompt(
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
                        stable_text=stable_text,
                        volatile_text=volatile_text,
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
) -> tuple[str, str, str, dict[str, str]]:
    """Render a system prompt split into ``(full, stable, volatile, snapshot)``.

    ``stable`` is the template with volatile placeholders emptied (byte-stable
    across turns); ``volatile`` is the relocated tail; ``full`` == stable + tail.
    """
    template_text = _normalized_template_text(world, entity_id, prompt_config)
    full_text, snapshot = render_prompt_template(
        template=template_text,
        world=world,
        entity=entity_id,
    )
    stable_text, volatile_text = _split_stable_volatile(template_text, snapshot)
    return full_text, stable_text, volatile_text, snapshot


def _split_stable_volatile(
    template_text: str, snapshot: dict[str, str]
) -> tuple[str, str]:
    """Split a rendered prompt into a cache-stable prefix and a volatile tail.

    The stable prefix is the template rendered with every
    :data:`VOLATILE_PLACEHOLDER_KEYS` value emptied, then ``rstrip``-ed to drop
    trailing whitespace left where a tail placeholder was removed. The volatile
    tail concatenates each volatile value in deterministic order, with nested
    (non-volatile) placeholders inside those values recursively expanded — a
    volatile value may itself be a whole prompt body (e.g. a workflow-state
    prompt referencing ``${_scratchbook_overview}``).

    ``full`` text is NOT reconstructed here: the caller stores the faithful
    render (``render_prompt_template`` output) as ``text``, so legacy/subagent
    consumers keep byte-for-byte behaviour.
    """
    stable_snapshot = {
        key: ("" if key in VOLATILE_PLACEHOLDER_KEYS else value)
        for key, value in snapshot.items()
    }
    stable_text = substitute_prompt_template(template_text, stable_snapshot).rstrip()

    volatile_parts: list[str] = []
    for key in _VOLATILE_PLACEHOLDER_KEYS:
        raw_value = snapshot.get(key)
        if not raw_value:
            continue
        # Expand nested placeholders using the volatile-emptied snapshot so a
        # volatile value that references other volatile keys cannot re-inject them.
        volatile_parts.append(
            substitute_prompt_template(raw_value, stable_snapshot)
        )
    volatile_text = "\n\n".join(volatile_parts)

    return stable_text, volatile_text


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
    for match in PLACEHOLDER_NAME_RE.finditer(template_text):
        if match.start() > 0 and template_text[match.start() - 1] == "$":
            continue
        name = match.group("braced") or match.group("named")
        if name is None:
            continue
        if name in snapshot:
            replacements[name] = snapshot[name]
    return replacements


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


def _render_cache_key(world: World, entity_id: EntityId) -> str:
    fingerprints: list[str] = []
    for provider in iter_placeholder_providers(world, entity_id):
        provider_id = placeholder_provider_id(provider)
        if not provider.resolve_placeholders(world, entity_id):
            continue
        fingerprint = provider.provider_fingerprint(world, entity_id)
        fingerprints.append(f"{provider_id}:{fingerprint}")
    return "|".join(fingerprints)


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
