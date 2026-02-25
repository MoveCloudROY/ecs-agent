"""LiteLLM provider wrapper for 100+ LLM providers.

LiteLLM normalizes 100+ LLM APIs to OpenAI format. This provider is a thin
adapter that delegates to litellm.acompletion().

Note: litellm is an optional dependency. If not installed, instantiation
will raise ImportError with installation instructions.
"""

from __future__ import annotations

import json
from typing import Any
from collections.abc import AsyncIterator

from ecs_agent.logging import get_logger
from ecs_agent.types import (
    Message,
    CompletionResult,
    StreamDelta,
    ToolSchema,
    ToolCall,
    Usage,
)

logger = get_logger(__name__)

# Optional import guard
try:
    import litellm  # type: ignore[import-not-found]

    HAS_LITELLM = True
except ImportError:
    HAS_LITELLM = False


class LiteLLMProvider:
    """LiteLLM provider wrapper for 100+ LLM providers.

    LiteLLM normalizes API calls to OpenAI format, supporting providers like:
    - OpenAI: "openai/gpt-4"
    - Anthropic: "anthropic/claude-3-opus-20240229"
    - Google: "gemini/gemini-pro"
    - Cohere, Mistral, and 100+ more

    Args:
        model: Model identifier in format "provider/model" (e.g., "anthropic/claude-3-opus-20240229")
        api_key: API key for the provider (optional, can use env vars)
        base_url: Base URL override (optional)

    Raises:
        ImportError: If litellm is not installed
    """

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        if not HAS_LITELLM:
            raise ImportError(
                "litellm is required but not installed. "
                "Install with: pip install litellm"
            )

        self._model = model
        self._api_key = api_key
        self._base_url = base_url

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        stream: bool = False,
        response_format: dict[str, Any] | None = None,
    ) -> CompletionResult | AsyncIterator[StreamDelta]:
        """Get completion from LLM via litellm.

        Args:
            messages: Conversation messages
            tools: Available tools for the LLM to call
            stream: Whether to stream response
            response_format: OpenAI response_format dict (e.g., {"type": "json_object"})

        Returns:
            CompletionResult or AsyncIterator[StreamDelta]
        """
        # Build kwargs for litellm.acompletion
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": self._convert_messages_to_openai(messages),
            "stream": stream,
        }

        if self._api_key is not None:
            kwargs["api_key"] = self._api_key

        if self._base_url is not None:
            kwargs["base_url"] = self._base_url

        if tools is not None:
            kwargs["tools"] = self._convert_tools_to_openai(tools)

        if response_format is not None:
            kwargs["response_format"] = response_format

        if stream:
            return self._stream_complete(kwargs)

        # Non-streaming path
        response = await litellm.acompletion(**kwargs)
        return self._parse_response(response)

    async def _stream_complete(
        self,
        kwargs: dict[str, Any],
    ) -> AsyncIterator[StreamDelta]:
        """Stream completion deltas.

        Args:
            kwargs: Arguments to pass to litellm.acompletion

        Yields:
            StreamDelta objects with content, tool_calls, finish_reason, usage
        """
        response = await litellm.acompletion(**kwargs)

        async for chunk in response:
            choice = chunk["choices"][0]
            delta = choice.get("delta", {})

            content = delta.get("content")
            finish_reason = choice.get("finish_reason")
            usage_data = chunk.get("usage")

            usage: Usage | None = None
            if usage_data:
                usage = Usage(
                    prompt_tokens=usage_data["prompt_tokens"],
                    completion_tokens=usage_data["completion_tokens"],
                    total_tokens=usage_data["total_tokens"],
                )

            # Tool calls handling (similar to OpenAIProvider)
            tool_calls: list[ToolCall] | None = None
            tool_calls_delta = delta.get("tool_calls")
            if tool_calls_delta:
                tool_calls = []
                for tc_delta in tool_calls_delta:
                    function_data = tc_delta.get("function", {})
                    tool_call = ToolCall(
                        id=tc_delta.get("id", ""),
                        name=function_data.get("name", ""),
                        arguments=self._parse_tool_arguments(
                            function_data.get("arguments", "")
                        ),
                    )
                    tool_calls.append(tool_call)

            # Skip empty deltas
            if (
                content is None
                and tool_calls is None
                and finish_reason is None
                and usage is None
            ):
                continue

            yield StreamDelta(
                content=content,
                tool_calls=tool_calls,
                finish_reason=finish_reason,
                usage=usage,
            )

    def _convert_messages_to_openai(
        self, messages: list[Message]
    ) -> list[dict[str, Any]]:
        """Convert Message objects to OpenAI format.

        Args:
            messages: List of Message objects

        Returns:
            List of dicts in OpenAI message format
        """
        openai_messages: list[dict[str, Any]] = []
        for msg in messages:
            openai_msg: dict[str, Any] = {
                "role": msg.role,
                "content": msg.content,
            }

            # Handle tool calls
            if msg.tool_calls and not msg.content:
                openai_msg["content"] = None
            if msg.tool_calls:
                openai_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": tc.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ]

            # Handle tool call responses
            if msg.tool_call_id:
                openai_msg["tool_call_id"] = msg.tool_call_id

            openai_messages.append(openai_msg)
        return openai_messages

    def _convert_tools_to_openai(self, tools: list[ToolSchema]) -> list[dict[str, Any]]:
        """Convert ToolSchema objects to OpenAI format.

        Args:
            tools: List of ToolSchema objects

        Returns:
            List of dicts in OpenAI tool format
        """
        openai_tools: list[dict[str, Any]] = []
        for tool in tools:
            openai_tool = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            openai_tools.append(openai_tool)
        return openai_tools

    def _parse_response(self, response: dict[str, Any]) -> CompletionResult:
        """Parse litellm response to CompletionResult.

        Args:
            response: Response dict from litellm (OpenAI format)

        Returns:
            CompletionResult with message and usage
        """
        message_data = response["choices"][0]["message"]

        role = message_data["role"]
        content = message_data.get("content") or ""

        tool_calls: list[ToolCall] | None = None
        if "tool_calls" in message_data and message_data["tool_calls"]:
            tool_calls = []
            for tc in message_data["tool_calls"]:
                tool_call = ToolCall(
                    id=tc["id"],
                    name=tc["function"]["name"],
                    arguments=json.loads(tc["function"]["arguments"]),
                )
                tool_calls.append(tool_call)

        message = Message(role=role, content=content, tool_calls=tool_calls)

        usage_data = response.get("usage")
        usage: Usage | None = None
        if usage_data:
            usage = Usage(
                prompt_tokens=usage_data["prompt_tokens"],
                completion_tokens=usage_data["completion_tokens"],
                total_tokens=usage_data["total_tokens"],
            )

        return CompletionResult(message=message, usage=usage)

    def _parse_tool_arguments(self, arguments_str: str) -> dict[str, Any]:
        """Parse tool call arguments string to dict.

        Args:
            arguments_str: JSON string of arguments

        Returns:
            Parsed dict, or {"_partial": arguments_str} if parsing fails
        """
        if not arguments_str:
            return {}

        try:
            parsed: dict[str, Any] = json.loads(arguments_str)
            return parsed
        except json.JSONDecodeError:
            return {"_partial": arguments_str}
