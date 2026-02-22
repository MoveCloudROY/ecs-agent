"""ECS-based LLM Agent framework."""

__version__ = "0.1.0"

from ecs_agent.types import (
    CompletionResult,
    EntityId,
    Message,
    ToolSchema,
)

__all__ = [
    "__version__",
    "Message",
    "CompletionResult",
    "ToolSchema",
    "EntityId",
]
