"""Section composition and rendering for prompts."""

from __future__ import annotations

from ecs_agent.prompts.contracts import PromptSectionSpec


def render_sections(sections: list[PromptSectionSpec]) -> str:
    """Render a list of sections in priority order.

    Higher priority sections are rendered first. For equal priority,
    sections are sorted by title to maintain determinism.

    Args:
        sections: List of PromptSectionSpec to render.

    Returns:
        A single string with all sections rendered.
    """
    if not sections:
        return ""

    # Sort by priority (descending) then title (ascending) for determinism
    sorted_sections = sorted(sections, key=lambda s: (-s.priority, s.title))

    rendered_blocks = []
    for section in sorted_sections:
        block = f"## {section.title}\n\n"
        block += "\n".join(section.lines)
        rendered_blocks.append(block)

    return "\n\n".join(rendered_blocks)
