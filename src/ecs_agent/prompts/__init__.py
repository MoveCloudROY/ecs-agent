"""Public API for the prompts package — re-exports all contract types."""

from ecs_agent.prompts.contracts import (
    PromptInjectionArtifact,
    PromptRenderContext,
    PromptSectionSpec,
    PromptTemplate,
)
from ecs_agent.prompts.registry import PromptRegistry

__all__ = [
    "PromptInjectionArtifact",
    "PromptRenderContext",
    "PromptSectionSpec",
    "PromptTemplate",
    "PromptRegistry",
]
