"""Anthropic Messages API adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ecs_agent.accounting.normalization import normalize_anthropic_usage
from ecs_agent.providers.config import ProviderConfig
from ecs_agent.types import (
    CompletionResult,
    FileRefPart,
    ImageUrlPart,
    Message,
    ToolCall,
    ToolSchema,
    Usage,
)


@dataclass(slots=True)
class AnthropicMessagesAdapterConfig:
    provider: ProviderConfig
    model: str
    max_tokens: int
    supports_vision: bool = False


@dataclass(slots=True)
class AnthropicMessagesRequest:
    messages: list[Message]
    tools: list[ToolSchema] | None = None
    response_format: dict[str, Any] | None = None
    stream: bool = False


class AnthropicMessagesAdapter:
    def __init__(self, config: AnthropicMessagesAdapterConfig) -> None:
        self._config = config

    def endpoint_url(self) -> str:
        return f"{self._config.provider.base_url}/v1/messages"

    def headers(self) -> dict[str, str]:
        base_headers = {
            "x-api-key": self._config.provider.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        return {**base_headers, **self._config.provider.extra_headers}

    def build_request_body(self, request: AnthropicMessagesRequest) -> dict[str, Any]:
        system_prompt, anthropic_messages = self.build_messages(request.messages)
        anthropic_tools = self.build_tools(request.tools)

        request_body: dict[str, Any] = {
            "model": self._config.model,
            "max_tokens": self._config.max_tokens,
            "messages": anthropic_messages,
        }
        if system_prompt is not None:
            request_body["system"] = system_prompt
        if anthropic_tools is not None:
            request_body["tools"] = anthropic_tools
        if request.response_format is not None:
            request_body["response_format"] = request.response_format
        if request.stream:
            request_body["stream"] = True

        return request_body

    def build_messages(
        self, messages: list[Message]
    ) -> tuple[str | list[dict[str, Any]] | None, list[dict[str, Any]]]:
        system_entries: list[tuple[str, bool]] = []
        anthropic_messages: list[dict[str, Any]] = []

        index = 0
        while index < len(messages):
            msg = messages[index]
            if msg.role == "system":
                system_entries.append((msg.content, msg.cache_control))
                index += 1
                continue

            if msg.role == "tool":
                tool_result_blocks: list[dict[str, Any]] = []
                while index < len(messages) and messages[index].role == "tool":
                    tool_message = messages[index]
                    if tool_message.tool_call_id is None:
                        raise ValueError("Tool message requires tool_call_id")
                    tool_result_blocks.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_message.tool_call_id,
                            "content": tool_message.content,
                        }
                    )
                    index += 1
                anthropic_messages.append(
                    {
                        "role": "user",
                        "content": tool_result_blocks,
                    }
                )
                continue

            content_blocks = self._build_content_blocks(msg)

            if msg.role == "assistant" and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    content_blocks.append(
                        {
                            "type": "tool_use",
                            "id": tool_call.id,
                            "name": tool_call.name,
                            "input": tool_call.arguments,
                        }
                    )

            anthropic_messages.append(
                {
                    "role": msg.role,
                    "content": content_blocks,
                }
            )
            index += 1

        caching = self._config.provider.enable_prompt_caching
        system_value = self._build_system_value(system_entries, caching=caching)
        if caching and anthropic_messages:
            flagged_system = sum(1 for _, cache_control in system_entries if cache_control)
            self._mark_message_cache_breakpoints(
                anthropic_messages,
                # Anthropic allows 4 breakpoints per request; one is reserved
                # for the tool block, the rest go to flagged system entries
                # and message markers.
                budget=max(1, 4 - 1 - flagged_system),
            )
        return system_value, anthropic_messages

    @staticmethod
    def _build_system_value(
        system_entries: list[tuple[str, bool]], *, caching: bool
    ) -> str | list[dict[str, Any]] | None:
        if not system_entries:
            return None
        if not caching:
            # Rollback / non-caching shape: single joined string.
            return "\n\n".join(content for content, _ in system_entries)
        blocks: list[dict[str, Any]] = []
        for content, cache_control in system_entries:
            block: dict[str, Any] = {"type": "text", "text": content}
            if cache_control:
                block["cache_control"] = {"type": "ephemeral"}
            blocks.append(block)
        return blocks

    # Anthropic checks ~20 content blocks behind each breakpoint for an
    # existing cache entry. Keeping the ladder marker at most this many blocks
    # ahead of the tail guarantees it can bridge back to the previous
    # request's tail entry for turns that append up to (window + ladder gap)
    # blocks — e.g. wide parallel tool batches.
    _LOOKBACK_WINDOW_BLOCKS = 20

    @classmethod
    def _mark_message_cache_breakpoints(
        cls, messages: list[dict[str, Any]], *, budget: int
    ) -> None:
        """Mark the trailing breakpoint plus, when the history is long enough
        and the budget allows, one intermediate "ladder" marker.

        A single trailing marker strands the previous request's cache entry
        whenever one turn appends more than the ~20-block lookback window
        (each tool_use / tool_result is its own block, so a wide parallel
        batch easily exceeds it). The ladder marker sits at the last message
        boundary within the window, extending the reachable span to roughly
        twice the window per request.
        """
        if not messages or budget <= 0:
            return
        cls._mark_cache_breakpoint(messages[-1])
        if budget < 2:
            return

        total_blocks = sum(cls._block_count(message) for message in messages)
        if total_blocks <= cls._LOOKBACK_WINDOW_BLOCKS:
            # The whole message list fits in one lookback window; a ladder
            # marker would only add cache-write cost.
            return

        blocks_behind_tail = 0
        ladder_candidate: dict[str, Any] | None = None
        for message in reversed(messages[:-1]):
            gap = blocks_behind_tail + cls._block_count(messages[-1])
            if gap > cls._LOOKBACK_WINDOW_BLOCKS:
                break
            ladder_candidate = message
            blocks_behind_tail += cls._block_count(message)
        if ladder_candidate is not None and ladder_candidate is not messages[-1]:
            cls._mark_cache_breakpoint(ladder_candidate)

    @staticmethod
    def _block_count(message: dict[str, Any]) -> int:
        content = message.get("content")
        return len(content) if isinstance(content, list) else 1

    @staticmethod
    def _mark_cache_breakpoint(message: dict[str, Any]) -> None:
        """Place a cache breakpoint on the last content block of a message."""
        content = message.get("content")
        if isinstance(content, list) and content:
            content[-1]["cache_control"] = {"type": "ephemeral"}

    def build_tools(
        self, tools: list[ToolSchema] | None
    ) -> list[dict[str, Any]] | None:
        if tools is None:
            return None

        anthropic_tools: list[dict[str, Any]] = []
        for tool in tools:
            anthropic_tools.append(
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.parameters,
                }
            )
        # Tools render first (tools -> system -> messages), so a breakpoint on the
        # last tool caches the entire — fully static — tool block.
        if self._config.provider.enable_prompt_caching and anthropic_tools:
            anthropic_tools[-1]["cache_control"] = {"type": "ephemeral"}
        return anthropic_tools

    def parse_response(self, response_data: dict[str, Any]) -> CompletionResult:
        content_blocks = response_data.get("content", [])
        message_content_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        reasoning_content: str | None = None
        reasoning_signature: str | None = None

        for block in content_blocks:
            block_type = block.get("type")
            if block_type == "text":
                text = block.get("text")
                if isinstance(text, str):
                    message_content_parts.append(text)
            elif block_type == "thinking":
                thinking_text = block.get("thinking")
                if isinstance(thinking_text, str):
                    reasoning_content = thinking_text
                signature = block.get("signature")
                if isinstance(signature, str):
                    reasoning_signature = signature
            elif block_type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block["id"],
                        name=block["name"],
                        arguments=block.get("input", {}),
                    )
                )

        usage_data = response_data.get("usage")
        usage: Usage | None = None
        if isinstance(usage_data, dict):
            usage = normalize_anthropic_usage(usage_data)
            usage.provider_id = self._config.provider.provider_id
            usage.model = self._config.model

        message = Message(
            role="assistant",
            content="".join(message_content_parts),
            tool_calls=tool_calls or None,
            reasoning_content=reasoning_content,
            reasoning_signature=reasoning_signature,
        )
        return CompletionResult(
            message=message,
            usage=usage,
            reasoning_content=reasoning_content,
        )

    def _build_content_blocks(self, msg: Message) -> list[dict[str, Any]]:
        content_blocks: list[dict[str, Any]] = []

        if msg.parts:
            for part in msg.parts:
                if isinstance(part, ImageUrlPart):
                    if not self._config.supports_vision:
                        raise ValueError(
                            "Unsupported multimodal part for Anthropic messages endpoint: ImageUrlPart"
                        )
                    content_blocks.append(
                        {
                            "type": "image",
                            "source": {"type": "url", "url": part.url},
                        }
                    )
                    continue

                if isinstance(part, FileRefPart):
                    if not self._config.supports_vision:
                        raise ValueError(
                            "Unsupported multimodal part for Anthropic messages endpoint: FileRefPart"
                        )
                    raise ValueError(
                        "Unsupported multimodal part for Anthropic messages endpoint: FileRefPart"
                    )

        if msg.reasoning_content is not None:
            thinking_block: dict[str, Any] = {
                "type": "thinking",
                "thinking": msg.reasoning_content,
            }
            if msg.reasoning_signature is not None:
                thinking_block["signature"] = msg.reasoning_signature
            content_blocks.append(thinking_block)

        if msg.content:
            content_blocks.append({"type": "text", "text": msg.content})

        return content_blocks
