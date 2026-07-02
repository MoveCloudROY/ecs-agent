"""Background-result envelope format and child prompt-template builders.

Pure, dependency-free helpers extracted from ``SubagentSystem`` (Task 2 of the
subagent package refactor). Owns the ``<subagent_background_result>`` wire format
and the child-world system-prompt templates.
"""

from __future__ import annotations

_BACKGROUND_RESULT_WRAPPER_START = "<subagent_background_result>"
_BACKGROUND_RESULT_WRAPPER_END = "</subagent_background_result>"
_BACKGROUND_RESULT_SUMMARY_START = "<summary>"
_BACKGROUND_RESULT_SUMMARY_END = "</summary>"
_BACKGROUND_RESULT_FULL_START = "<full_result>"
_BACKGROUND_RESULT_FULL_END = "</full_result>"
_BACKGROUND_RESULT_INSTRUCTION = (
    "\n\n## Background Result Format\n"
    "This is a background subagent run. Your final assistant message must be exactly:\n"
    "<subagent_background_result>\n"
    "<summary>brief cached summary for the parent</summary>\n"
    "<full_result>complete final result for the parent</full_result>\n"
    "</subagent_background_result>"
)


def _build_child_prompt_template(user_prompt: str) -> str:
    """Build the system-prompt template for a child world.

    Ensures the template always includes ${_installed_tools} and
    ${_installed_skills} placeholder sections so SystemPromptRenderSystem
    can expand them at runtime. If the caller's prompt already contains
    a placeholder, it is NOT duplicated.

    Args:
        user_prompt: Raw system prompt text from SubagentConfig.

    Returns:
        Template string ready for PromptTemplateSource(inline=...).
    """
    suffix_parts: list[str] = []
    if "${_installed_tools}" not in user_prompt:
        suffix_parts.append("\n\n## Available Tools\n${_installed_tools}")
    if "${_installed_skills}" not in user_prompt:
        suffix_parts.append("\n\n## Available Skills\n${_installed_skills}")
    return user_prompt + "".join(suffix_parts)


def _build_background_child_prompt_template(user_prompt: str) -> str:
    return _build_child_prompt_template(user_prompt) + _BACKGROUND_RESULT_INSTRUCTION


def parse_background_result_envelope(result: str) -> tuple[str, str] | None:
    """Parse a ``<subagent_background_result>`` envelope into (summary, full_result).

    Returns ``None`` when *result* is not a well-formed envelope: it must be
    wrapped exactly, the ``<summary>`` block must begin the body, and the
    ``<full_result>`` block must end it.
    """
    stripped = result.strip()
    if not (
        stripped.startswith(_BACKGROUND_RESULT_WRAPPER_START)
        and stripped.endswith(_BACKGROUND_RESULT_WRAPPER_END)
    ):
        return None

    body = stripped.removeprefix(_BACKGROUND_RESULT_WRAPPER_START).removesuffix(
        _BACKGROUND_RESULT_WRAPPER_END
    )
    body = body.strip()

    summary_start = body.find(_BACKGROUND_RESULT_SUMMARY_START)
    summary_end = body.find(_BACKGROUND_RESULT_SUMMARY_END)
    full_start = body.find(_BACKGROUND_RESULT_FULL_START)
    full_end = body.find(_BACKGROUND_RESULT_FULL_END)

    if min(summary_start, summary_end, full_start, full_end) < 0:
        return None
    if not (summary_start == 0 and summary_end > summary_start):
        return None

    summary_content_start = summary_start + len(_BACKGROUND_RESULT_SUMMARY_START)
    summary = body[summary_content_start:summary_end]

    full_section = body[summary_end + len(_BACKGROUND_RESULT_SUMMARY_END) :].strip()
    if not (
        full_section.startswith(_BACKGROUND_RESULT_FULL_START)
        and full_section.endswith(_BACKGROUND_RESULT_FULL_END)
    ):
        return None

    full_result = full_section.removeprefix(_BACKGROUND_RESULT_FULL_START).removesuffix(
        _BACKGROUND_RESULT_FULL_END
    )
    return (summary, full_result)


__all__ = [
    "parse_background_result_envelope",
    "_build_child_prompt_template",
    "_build_background_child_prompt_template",
    "_BACKGROUND_RESULT_INSTRUCTION",
    "_BACKGROUND_RESULT_WRAPPER_START",
    "_BACKGROUND_RESULT_WRAPPER_END",
    "_BACKGROUND_RESULT_SUMMARY_START",
    "_BACKGROUND_RESULT_SUMMARY_END",
    "_BACKGROUND_RESULT_FULL_START",
    "_BACKGROUND_RESULT_FULL_END",
]
