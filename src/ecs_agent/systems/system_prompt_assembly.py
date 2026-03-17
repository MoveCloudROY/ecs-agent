"""System that assembles opt-in system prompts from section contributions."""

from __future__ import annotations

from ecs_agent.components import (
    PromptConfigComponent,
    PromptContributionsComponent,
    SystemPromptComponent,
)
from ecs_agent.core.world import World
from ecs_agent.prompts.sections import render_sections


class SystemPromptAssemblySystem:
    """Compose SystemPromptComponent content for opt-in entities only."""

    def __init__(self, priority: int = 0) -> None:
        self.priority = priority

    async def process(self, world: World) -> None:
        for _, components in world.query(
            PromptConfigComponent,
            PromptContributionsComponent,
            SystemPromptComponent,
        ):
            _, contributions, system_prompt = components
            assembled = render_sections(contributions.sections)
            system_prompt.content = assembled


__all__ = ["SystemPromptAssemblySystem"]
