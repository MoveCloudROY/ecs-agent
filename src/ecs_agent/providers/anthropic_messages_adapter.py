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
    ) -> tuple[str | None, list[dict[str, Any]]]:
        system_messages: list[str] = []
        anthropic_messages: list[dict[str, Any]] = []

        for msg in messages:
            if msg.role == "system":
                system_messages.append(msg.content)
                continue

            if msg.role == "tool":
                if msg.tool_call_id is None:
                    raise ValueError("Tool message requires tool_call_id")
                anthropic_messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": msg.tool_call_id,
                                "content": msg.content,
                            }
                        ],
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

        system_prompt = "\n\n".join(system_messages) if system_messages else None
        return system_prompt, anthropic_messages

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
        return anthropic_tools

    def parse_response(self, response_data: dict[str, Any]) -> CompletionResult:
        content_blocks = response_data.get("content", [])
        message_content_parts: list[str] = []
        tool_calls: list[ToolCall] = []

        for block in content_blocks:
            block_type = block.get("type")
            if block_type == "text":
                text = block.get("text")
                if isinstance(text, str):
                    message_content_parts.append(text)
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
        )
        return CompletionResult(message=message, usage=usage)

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

        if msg.content:
            content_blocks.append({"type": "text", "text": msg.content})

        return content_blocks
