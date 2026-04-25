"""LLM Model Protocol definition."""

from collections.abc import AsyncIterator
from typing import Any, Protocol, runtime_checkable
from ecs_agent.types import Message, CompletionResult, StreamDelta, ToolSchema


@runtime_checkable
class LLMModel(Protocol):
    """Protocol for LLM model implementations."""

    @property
    def model_id(self) -> str:
        """Identifier for the underlying model (e.g. 'gpt-4o', 'claude-3-5-haiku')."""
        ...

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        stream: bool = False,
        response_format: dict[str, Any] | None = None,
    ) -> CompletionResult | AsyncIterator[StreamDelta]:
        """Get completion from LLM.

        Args:
            messages: Conversation messages.
            tools: Available tools for the LLM to call.

        Returns:
            Completion result with message and optional usage info.
        """
        ...


# Backward-compat alias
LLMProvider = LLMModel
