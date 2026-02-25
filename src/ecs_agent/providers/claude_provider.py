from collections.abc import AsyncIterator
from typing import Any

import httpx
import structlog

from ecs_agent.types import (
    CompletionResult,
    Message,
    StreamDelta,
    ToolCall,
    ToolSchema,
    Usage,
)

logger = structlog.get_logger(__name__)


class ClaudeProvider:
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.anthropic.com",
        model: str = "claude-3-5-haiku-latest",
        max_tokens: int = 4096,
        connect_timeout: float = 10.0,
        read_timeout: float = 120.0,
        write_timeout: float = 10.0,
        pool_timeout: float = 10.0,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url
        self._model = model
        self._max_tokens = max_tokens
        self._timeout = httpx.Timeout(
            connect=connect_timeout,
            read=read_timeout,
            write=write_timeout,
            pool=pool_timeout,
        )
        self._client = httpx.AsyncClient(trust_env=False, timeout=self._timeout)

    def _build_messages(
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

            content_blocks: list[dict[str, Any]] = []
            if msg.content:
                content_blocks.append({"type": "text", "text": msg.content})

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

    def _build_tools(
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

    def _parse_response(self, response_data: dict[str, Any]) -> CompletionResult:
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
                tool_call = ToolCall(
                    id=block["id"],
                    name=block["name"],
                    arguments=block.get("input", {}),
                )
                tool_calls.append(tool_call)

        usage_data = response_data.get("usage")
        usage: Usage | None = None
        if usage_data:
            prompt_tokens = usage_data.get("input_tokens", 0)
            completion_tokens = usage_data.get("output_tokens", 0)
            usage = Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            )

        message = Message(
            role="assistant",
            content="".join(message_content_parts),
            tool_calls=tool_calls or None,
        )
        return CompletionResult(message=message, usage=usage)

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        stream: bool = False,
        response_format: dict[str, Any] | None = None,
    ) -> CompletionResult | AsyncIterator[StreamDelta]:
        if stream:
            raise NotImplementedError(
                "ClaudeProvider streaming path is not implemented"
            )

        system_prompt, anthropic_messages = self._build_messages(messages)
        anthropic_tools = self._build_tools(tools)

        request_body: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": anthropic_messages,
        }
        if system_prompt is not None:
            request_body["system"] = system_prompt
        if anthropic_tools is not None:
            request_body["tools"] = anthropic_tools
        if response_format is not None:
            request_body["response_format"] = response_format

        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        url = f"{self._base_url}/v1/messages"

        try:
            response = await self._client.post(url, json=request_body, headers=headers)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            logger.error(
                "llm_http_error",
                status_code=exc.response.status_code,
                response_body=exc.response.text,
                exception=str(exc),
            )
            raise
        except httpx.RequestError as exc:
            request_method: str | None = None
            request_url: str | None = None
            try:
                request_method = exc.request.method
                request_url = str(exc.request.url)
            except RuntimeError:
                pass
            logger.error(
                "llm_network_error",
                exception_type=type(exc).__name__,
                exception=str(exc),
                request_method=request_method,
                request_url=request_url,
            )
            raise

        return self._parse_response(response.json())
