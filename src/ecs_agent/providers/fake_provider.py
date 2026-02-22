"""Fake LLM provider for testing."""

from ecs_agent.providers.protocol import LLMProvider
from ecs_agent.types import Message, CompletionResult, ToolSchema


class FakeProvider:
    """Fake provider that returns pre-defined responses sequentially."""

    def __init__(self, responses: list[CompletionResult]) -> None:
        """Initialize with list of responses to return.

        Args:
            responses: List of CompletionResult objects to return in order.
        """
        self._responses = responses
        self._index = 0

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
    ) -> CompletionResult:
        """Return next response from the list.

        Args:
            messages: Ignored (not used by fake provider).
            tools: Ignored (not used by fake provider).

        Returns:
            Next CompletionResult from the responses list.

        Raises:
            IndexError: When all responses have been consumed.
        """
        if self._index >= len(self._responses):
            raise IndexError("No more responses available")
        result = self._responses[self._index]
        self._index += 1
        return result
