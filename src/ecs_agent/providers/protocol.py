"""LLM Provider Protocol definition."""

from typing import Protocol, runtime_checkable
from ecs_agent.types import Message, CompletionResult, ToolSchema


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol for LLM completion providers."""

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
    ) -> CompletionResult:
        """Get completion from LLM.

        Args:
            messages: Conversation messages.
            tools: Available tools for the LLM to call.

        Returns:
            Completion result with message and optional usage info.
        """
        ...
