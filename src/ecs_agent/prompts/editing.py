"""Prompt updates and side-effect-free diagnostics for entity-bound handles."""

from copy import deepcopy
from dataclasses import dataclass, replace

from ecs_agent.components import LLMComponent, RenderedSystemPromptComponent, SystemPromptComponent
from ecs_agent.core.world import World
from ecs_agent.history import invalidate_response_chain, record_session_event
from ecs_agent.prompts.bindings import prompt_config_to_dict
from ecs_agent.prompts.contracts import SystemPromptConfigSpec
from ecs_agent.types import EntityId


@dataclass(frozen=True, slots=True)
class PromptInspection:
    """Last rendered prompt and its provenance; reading never evaluates callbacks."""

    config_revision: int | None
    rendered_revision: int | None
    source: str
    stable_prefix: str | None
    volatile_suffix: str | None
    text: str | None
    placeholders: tuple[tuple[str, str], ...]
    legacy_sources: tuple[tuple[str, str], ...]
    cache_key: str | None = None
    placeholder_sources: tuple[tuple[str, str], ...] = ()


def inspect_prompt(world: World, entity: EntityId) -> PromptInspection:
    """Read configuration and the last rendered snapshot without consuming inputs."""
    config = world.get_component(entity, SystemPromptConfigSpec)
    rendered = world.get_component(entity, RenderedSystemPromptComponent)
    legacy = world.get_component(entity, SystemPromptComponent)
    llm = world.get_component(entity, LLMComponent)
    sources = []
    if legacy is not None:
        sources.append(("system", legacy.template or legacy.content))
    if llm is not None:
        sources.append(("llm", llm.system_prompt))
    rendered_revision = rendered.placeholder_snapshot.get("_config_revision") if rendered is not None else None
    return PromptInspection(
        config.revision if config is not None else None,
        int(rendered_revision) if rendered_revision is not None else None,
        (config.source_path or config.template_source.file_path or "inline") if config is not None else "legacy",
        rendered.stable_text if rendered is not None else None,
        rendered.volatile_text if rendered is not None else None,
        rendered.text if rendered is not None else None,
        tuple(sorted((key, value) for key, value in rendered.placeholder_snapshot.items() if key not in {"_cache_key", "_config_revision", "_legacy_template"})) if rendered is not None else (),
        tuple(sources),
        rendered.placeholder_snapshot.get("_cache_key") if rendered is not None else None,
        tuple(sorted((key, "built-in provider" if key.startswith("_") else "user placeholder") for key in rendered.placeholder_snapshot if key not in {"_cache_key", "_config_revision", "_legacy_template"})) if rendered is not None else (),
    )


def update_prompt(world: World, entity: EntityId, prompt: str | SystemPromptConfigSpec) -> None:
    """Commit a new canonical prompt; the owning handle enforces edit timing."""
    from ecs_agent.authoring.normalize import normalize_prompt
    from ecs_agent.systems.system_prompt_render_system import SystemPromptRenderSystem

    world.apply_pending_system_operations()
    if not any(isinstance(entry.system, SystemPromptRenderSystem) for entry in world._systems._systems):
        raise ValueError("Prompt editing requires SystemPromptRenderSystem; use build_agent or install prompt rendering first")
    previous = world.get_component(entity, SystemPromptConfigSpec)
    config = replace(normalize_prompt(prompt), revision=(previous.revision + 1) if previous is not None else 1)
    if world.session_service is not None:
        record_session_event(world, entity, "prompt_config_changed", {"config": prompt_config_to_dict(config)})
    world.add_component(entity, config)
    world.remove_component(entity, RenderedSystemPromptComponent)
    # Legacy readers receive the new template until the next render bridges
    # its accepted text. They are not independent configuration authorities.
    legacy = world.get_component(entity, SystemPromptComponent)
    if legacy is not None:
        legacy.content = config.template_source.inline or ""
    llm = world.get_component(entity, LLMComponent)
    if llm is not None:
        llm.system_prompt = config.template_source.inline or ""
    invalidate_response_chain(world, entity, reason="prompt_config_changed")
