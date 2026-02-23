"""Fake LLM provider for testing."""

from ecs_agent.types import Message, CompletionResult, ToolSchema, StreamDelta
from typing import Any, AsyncIterator


class FakeProvider:
    """Fake provider that returns pre-defined responses sequentially."""

    def __init__(self, responses: list[CompletionResult]) -> None:
        """Initialize with list of responses to return.

        Args:
            responses: List of CompletionResult objects to return in order.
        """
        self._responses = responses
        self._index = 0
        self.last_response_format: dict[str, Any] | None = None


    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        stream: bool = False,
        response_format: dict[str, Any] | None = None,
    ) -> CompletionResult | AsyncIterator[StreamDelta]:
        """Return next response from the list or async generator of deltas.

        Args:
            messages: Ignored (not used by fake provider).
            tools: Ignored (not used by fake provider).

        Returns:
            CompletionResult if stream=False, else AsyncIterator[StreamDelta].

        Raises:
            IndexError: When all responses have been consumed.
        """
        if self._index >= len(self._responses):
            raise IndexError("No more responses available")
        self.last_response_format = response_format

        result = self._responses[self._index]
        self._index += 1
        if not stream:
            return result
        # For streaming, return an async generator that yields character-by-character deltas
        return self._stream_complete(result)

    async def _stream_complete(
        self, result: CompletionResult
    ) -> AsyncIterator[StreamDelta]:
        """Stream response as character-by-character deltas.

        Args:
            result: The CompletionResult to stream.

        Yields:
            StreamDelta objects with single characters, final one includes finish_reason and usage.
        """
        content = result.message.content or ""

        # Yield each character as a separate StreamDelta
        for char in content:
            yield StreamDelta(content=char)

        # Final chunk with finish_reason and usage info
        yield StreamDelta(finish_reason="stop", usage=result.usage)
