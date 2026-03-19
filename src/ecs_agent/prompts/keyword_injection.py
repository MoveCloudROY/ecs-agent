"""Keyword-trigger stage-1 injection for prompt normalization."""

from __future__ import annotations

from ecs_agent.prompts.contracts import PromptTriggerSpec
from ecs_agent.prompts.registry import PromptRegistry


def inject_keywords(text: str, registry: PromptRegistry) -> str:
    return inject_triggers(text, registry)


def inject_triggers(
    text: str,
    registry: PromptRegistry,
    *,
    trigger_specs: list[PromptTriggerSpec] | None = None,
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

    template = registry.get(matched.template_id)
    marker = _trigger_marker(matched)
    return f"{marker}\n{template.content}\n\n{text}"


def _ordered_trigger_specs(
    *,
    registry: PromptRegistry,
    trigger_specs: list[PromptTriggerSpec] | None,
) -> list[PromptTriggerSpec]:
    if trigger_specs is None:
        synthesized: list[PromptTriggerSpec] = []
        for registration_order, keyword in enumerate(registry.list_keywords()):
            template = registry.resolve_keyword(keyword)
            if template is None:
                continue
            synthesized.append(
                PromptTriggerSpec(
                    kind="keyword",
                    trigger=keyword,
                    template_id=template.template_id,
                    priority=0,
                    registration_order=registration_order,
                )
            )
        trigger_specs = synthesized

    return sorted(
        trigger_specs, key=lambda spec: (-spec.priority, spec.registration_order)
    )


def _resolve_first_match(
    *,
    text: str,
    ordered_specs: list[PromptTriggerSpec],
    active_events: set[str] | None,
) -> PromptTriggerSpec | None:
    event_set = active_events or set()
    for spec in ordered_specs:
        if spec.kind == "keyword" and spec.trigger in text:
            return spec
        if spec.kind == "event" and spec.trigger in event_set:
            return spec
    return None


def _trigger_marker(spec: PromptTriggerSpec) -> str:
    if spec.kind == "keyword":
        return f"[PROMPT_INJECT:{spec.trigger}]"
    return f"[PROMPT_INJECT:{spec.kind}:{spec.trigger}]"


__all__ = ["inject_keywords", "inject_triggers"]
