"""Keyword-trigger stage-1 injection for prompt normalization."""

from __future__ import annotations

from ecs_agent.prompts.registry import PromptRegistry


def inject_keywords(text: str, registry: PromptRegistry) -> str:
    """Inject first-match keyword template into user text.

    Args:
        text: The original user message text.
        registry: The prompt template registry.

    Returns:
        Modified text with marker and template block if a keyword matched.
    """
    # Duplicate prevention: check if any marker already exists
    if "[PROMPT_INJECT:" in text:
        return text

    keywords = registry.list_keywords()

    # Find first match
    matched_keyword = None
    for kw in keywords:
        if kw in text:
            matched_keyword = kw
            break

    if not matched_keyword:
        return text

    template = registry.resolve_keyword(matched_keyword)
    if not template:
        return text

    marker = f"[PROMPT_INJECT:{matched_keyword}]"

    # Injection order: marker -> keyword block -> original user text
    return f"{marker}\n{template.content}\n\n{text}"
