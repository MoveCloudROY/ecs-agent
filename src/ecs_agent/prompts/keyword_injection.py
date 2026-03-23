"""Keyword-trigger stage-1 injection for prompt normalization."""

from __future__ import annotations

from ecs_agent.prompts.contracts import TriggerSpec
from ecs_agent.prompts.registry import PromptRegistry


def inject_keywords(text: str, registry: PromptRegistry) -> str:
    return inject_triggers(text, registry)


def inject_triggers(
    text: str,
    registry: PromptRegistry,
    *,
    trigger_specs: list[TriggerSpec] | None = None,
    active_events: set[str] | None = None,
) -> str:
    if "[PROMPT_INJECT:" in text:
        return text

    ordered_specs = _ordered_trigger_specs(
        registry=registry,
        trigger_specs=trigger_specs,
    )
    matched = _resolve_first_match(
        text=text,
        ordered_specs=ordered_specs,
        active_events=active_events,
    )
    if matched is None:
        return text

    template = registry.get(matched.content)
    marker = _trigger_marker(matched)
    return f"{marker}\n{template.content}\n\n{text}"


def _ordered_trigger_specs(
    *,
    registry: PromptRegistry,
    trigger_specs: list[TriggerSpec] | None,
) -> list[TriggerSpec]:
    if trigger_specs is None:
        synthesized: list[TriggerSpec] = []
        for keyword in registry.list_keywords():
            template = registry.resolve_keyword(keyword)
            if template is None:
                continue
            synthesized.append(
                TriggerSpec(
                    pattern=keyword,
                    match_mode="keyword",
                    action="skill",
                    content=template.template_id,
                    priority=0,
                )
            )
        trigger_specs = synthesized

    return sorted(trigger_specs, key=lambda spec: -spec.priority)


def _resolve_first_match(
    *,
    text: str,
    ordered_specs: list[TriggerSpec],
    active_events: set[str] | None,
) -> TriggerSpec | None:
    event_set = active_events or set()
    for spec in ordered_specs:
        if _matches(spec=spec, text=text, active_events=event_set):
            return spec
    return None


def _matches(*, spec: TriggerSpec, text: str, active_events: set[str]) -> bool:
    if spec.pattern.startswith("event:"):
        return spec.pattern.removeprefix("event:") in active_events
    if spec.match_mode == "prefix":
        return text.startswith(spec.pattern)
    return spec.pattern in text


def _trigger_marker(spec: TriggerSpec) -> str:
    if spec.pattern.startswith("event:"):
        return f"[PROMPT_INJECT:event:{spec.pattern.removeprefix('event:')}]"
    return f"[PROMPT_INJECT:{spec.pattern}]"


__all__ = ["inject_keywords", "inject_triggers"]
