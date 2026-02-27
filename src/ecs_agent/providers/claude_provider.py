import json
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
    """Native Anthropic Messages API provider with streaming and tool use support."""
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

    async def _stream_complete(
        self,
        url: str,
        headers: dict[str, str],
        request_body: dict[str, Any],
    ) -> AsyncIterator[StreamDelta]:
        stream_body = dict(request_body)
        stream_body["stream"] = True
        timeout = httpx.Timeout(
            connect=self._timeout.connect,
            read=None,
            write=self._timeout.write,
            pool=self._timeout.pool,
        )

        tool_use_state: dict[int, dict[str, str]] = {}

        async def handle_event(
            event_type: str | None,
            payload_data: str,
        ) -> AsyncIterator[StreamDelta]:
            if not payload_data or payload_data == "[DONE]":
                return

            event_data = json.loads(payload_data)
            resolved_event_type = event_type or event_data.get("type")

            if resolved_event_type == "content_block_start":
                index = event_data.get("index")
                block = event_data.get("content_block", {})
                if isinstance(index, int) and block.get("type") == "tool_use":
                    tool_use_state[index] = {
                        "id": block.get("id", ""),
                        "name": block.get("name", ""),
                        "arguments": "",
                    }
                return

            if resolved_event_type == "content_block_delta":
                index = event_data.get("index")
                delta = event_data.get("delta", {})
                delta_type = delta.get("type")

                if delta_type == "text_delta":
                    text = delta.get("text")
                    if isinstance(text, str):
                        yield StreamDelta(content=text)
                    return

                if delta_type == "input_json_delta" and isinstance(index, int):
                    partial_json = delta.get("partial_json")
                    if isinstance(partial_json, str):
                        accumulated = tool_use_state.setdefault(
                            index,
                            {
                                "id": f"index_{index}",
                                "name": "",
                                "arguments": "",
                            },
                        )
                        accumulated["arguments"] += partial_json
                return

            if resolved_event_type == "content_block_stop":
                index = event_data.get("index")
                if not isinstance(index, int):
                    return

                if index not in tool_use_state:
                    return
                accumulated = tool_use_state.pop(index)

                parsed_arguments: dict[str, Any]
                try:
                    parsed_arguments = json.loads(accumulated["arguments"])
                except json.JSONDecodeError:
                    parsed_arguments = {"_partial": accumulated["arguments"]}

                yield StreamDelta(
                    tool_calls=[
                        ToolCall(
                            id=accumulated["id"] or f"index_{index}",
                            name=accumulated["name"],
                            arguments=parsed_arguments,
                        )
                    ]
                )
                return

            if resolved_event_type == "message_delta":
                delta = event_data.get("delta", {})
                stop_reason = delta.get("stop_reason")
                if isinstance(stop_reason, str):
                    yield StreamDelta(finish_reason=stop_reason)
                return

        try:
            async with self._client.stream(
                "POST",
                url,
                json=stream_body,
                headers=headers,
                timeout=timeout,
            ) as response:
                response.raise_for_status()

                current_event: str | None = None
                current_data_lines: list[str] = []

                async for line in response.aiter_lines():
                    if not line:
                        if current_data_lines:
                            payload_data = "\n".join(current_data_lines).strip()
                            if payload_data == "[DONE]":
                                break
                            async for delta in handle_event(
                                current_event, payload_data
                            ):
                                yield delta
                        current_event = None
                        current_data_lines = []
                        continue

                    payload_line = line.strip()
                    if payload_line.startswith("event:"):
                        current_event = payload_line[6:].strip()
                        continue

                    if payload_line.startswith("data:"):
                        data_line = payload_line[5:].strip()
                        if data_line == "[DONE]":
                            break
                        current_data_lines.append(data_line)
                        continue

                    current_data_lines.append(payload_line)

                if current_data_lines:
                    payload_data = "\n".join(current_data_lines).strip()
                    if payload_data != "[DONE]":
                        async for delta in handle_event(current_event, payload_data):
                            yield delta
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

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        stream: bool = False,
        response_format: dict[str, Any] | None = None,
    ) -> CompletionResult | AsyncIterator[StreamDelta]:
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

        if stream:
            return self._stream_complete(url, headers, request_body)

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
