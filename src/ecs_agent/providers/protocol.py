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
        thread_response_id: str | None = None,
    ) -> CompletionResult | AsyncIterator[StreamDelta]:
        """Get completion from LLM.

        Args:
            messages: Conversation messages.
            tools: Available tools for the LLM to call.
            thread_response_id: Previous response id for providers that chain
                stored responses (OpenAI Responses API). Implementations
                without response chaining accept and ignore it; wrapper models
                must forward it. Callers only pass it when a previous response
                id was actually recorded, so implementations predating this
                parameter keep working in non-chaining sessions.

        Returns:
            Completion result with message and optional usage info.
        """
        ...

