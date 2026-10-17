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
        covered_message_count: int | None = None,
    ) -> CompletionResult | AsyncIterator[StreamDelta]:
        """Get completion from LLM.

        Args:
            messages: Conversation messages.
            tools: Available tools for the LLM to call.
            thread_response_id: Previous response id for providers that chain
                stored responses (OpenAI Responses API). Implementations
                without response chaining accept and ignore it; wrapper models
                must forward it. Direct callers supply incremental input, or
                full history with covered_message_count. ReasoningSystem only
                passes an automatically selected ID after provenance, storage
                and protocol-prefix verification.
            covered_message_count: How many leading non-system messages the
                stored response already covers (its request items plus its
                assistant output). Providers that chain stored responses trim
                that many leading items from the wire ``input`` so the stored
                context is not replayed twice; other implementations accept
                and ignore it, and wrapper models must forward it. Only passed
                alongside ``thread_response_id``. Automatic callers recompute
                this boundary from a verified protocol-item prefix; persisted
                message counts alone never authorize trimming.

        Returns:
            Completion result with message and optional usage info.
        """
        ...

    async def aclose(self) -> None:
        """Release owned transport resources (HTTP connection pools).

        Implementations without owned resources no-op. There is no automatic
        shutdown hook yet — owners of long-lived processes should call this
        when a model instance is retired. Idempotent.
        """
        ...

