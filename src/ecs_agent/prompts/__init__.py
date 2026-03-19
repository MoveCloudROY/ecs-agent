"""Public API for the prompts package — re-exports all contract types."""

from ecs_agent.prompts.contracts import (
    CORE_PLACEHOLDER_KEYS,
    PlaceholderContract,
    PromptInjectionArtifact,
    PromptRenderContext,
    PromptSectionSpec,
    PromptTemplate,
    PromptTriggerKind,
    PromptTriggerSpec,
)
from ecs_agent.prompts.registry import PlaceholderRegistry, PromptRegistry
from ecs_agent.prompts.keyword_injection import inject_keywords, inject_triggers
from ecs_agent.prompts.message_assembly import assemble_messages, build_keyword_registry
from ecs_agent.prompts.renderers import render_string, render_table
from ecs_agent.prompts.sections import render_sections

__all__ = [
    "PromptInjectionArtifact",
    "PromptRenderContext",
    "PromptSectionSpec",
    "PromptTemplate",
    "PromptTriggerKind",
    "PromptTriggerSpec",
    "CORE_PLACEHOLDER_KEYS",
    "PlaceholderContract",
    "PlaceholderRegistry",
    "PromptRegistry",
    "inject_keywords",
    "inject_triggers",
    "assemble_messages",
    "build_keyword_registry",
    "render_string",
    "render_table",
    "render_sections",
]
