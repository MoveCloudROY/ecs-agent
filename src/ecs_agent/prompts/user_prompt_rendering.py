"""Shared user-prompt rendering helper.

Pure function that applies trigger injection to user text.
Used by both ``UserPromptNormalizationSystem`` (pre-render at priority -10) and
``_with_transient_user_injection`` (inline fallback at call-time).

Context-pool rendering is intentionally excluded — it must remain at call-time
in ``prepare_outbound_messages`` because ``PromptContextCollectorSystem`` may add
entries after normalization.
"""

from __future__ import annotations

from ecs_agent.prompts.contracts import TriggerSpec


_PROMPT_INJECT_SENTINEL = "[PROMPT_INJECT:"


def render_user_prompt_text(
    user_text: str,
    *,
    trigger_specs: list[TriggerSpec] | None = None,
) -> str:
    """Apply trigger injection to *user_text* and return the transformed text.

    If the text already contains a ``[PROMPT_INJECT:`` sentinel, no further
    injection is performed (idempotency guard).

    Args:
        user_text: Raw user message text.
        trigger_specs: Trigger specs to apply (from ``UserPromptConfigComponent.triggers``).

    Returns:
        The (possibly transformed) user text with trigger content injected.
    """
    if trigger_specs:
        return _apply_trigger_specs(user_text, trigger_specs)

    return user_text


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def _apply_trigger_specs(user_text: str, trigger_specs: list[TriggerSpec]) -> str:
    """Apply a list of trigger specs, returning on the first match."""
    if not trigger_specs or _PROMPT_INJECT_SENTINEL in user_text:
        return user_text

    ordered_specs = sorted(trigger_specs, key=lambda spec: -spec.priority)
    for spec in ordered_specs:
        if not _matches(spec=spec, text=user_text):
            continue
        if spec.action == "replace":
            return spec.content
        marker = f"{_PROMPT_INJECT_SENTINEL}{spec.pattern}]"
        return f"{marker}\n{spec.content}\n\n{user_text}"

    return user_text


def _matches(*, spec: TriggerSpec, text: str) -> bool:
    if spec.match_mode == "keyword":
        return spec.pattern in text
    if spec.match_mode == "prefix":
        return text.startswith(spec.pattern)
    return spec.pattern in text


__all__ = ["render_user_prompt_text"]
