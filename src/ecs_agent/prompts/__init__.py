"""Public API for the ecs_agent prompts package."""

from ecs_agent.prompts.contracts import (
    PlaceholderSpec,
    SystemPromptConfigSpec,
    PromptTemplate,
    PromptTemplateSource,
    TriggerSpec,
)
from ecs_agent.prompts.registry import (
    PlaceholderRegistry,
    PromptRegistry,
    resolve_placeholder_values,
    _LEGACY_REQUIRED_TEMPLATE_KEYS as CORE_PLACEHOLDER_KEYS,
)
from ecs_agent.prompts.keyword_injection import inject_triggers
from ecs_agent.prompts.message_assembly import assemble_messages, build_keyword_registry
from ecs_agent.prompts.user_prompt_rendering import render_user_prompt_text

__all__ = [
    "CORE_PLACEHOLDER_KEYS",
    "PlaceholderSpec",
    "SystemPromptConfigSpec",
    "PromptTemplate",
    "PromptTemplateSource",
    "TriggerSpec",
    "PlaceholderRegistry",
    "PromptRegistry",
    "resolve_placeholder_values",
    "inject_triggers",
    "assemble_messages",
    "build_keyword_registry",
    "render_user_prompt_text",
]
