"""Public API for the prompts package — re-exports all contract types."""

from ecs_agent.prompts.contracts import (
    PromptInjectionArtifact,
    PromptRenderContext,
    PromptSectionSpec,
    PromptTemplate,
)

__all__ = [
    "PromptInjectionArtifact",
    "PromptRenderContext",
    "PromptSectionSpec",
    "PromptTemplate",
]
