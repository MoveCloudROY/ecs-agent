"""OpenAI-compatible HTTP provider using httpx."""

import json

from typing import Any
from collections.abc import AsyncIterator
import httpx
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


class OpenAIProvider:
    """OpenAI-compatible LLM provider using httpx AsyncClient."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o-mini",
        connect_timeout: float = 10.0,
        read_timeout: float = 120.0,
        write_timeout: float = 10.0,
        pool_timeout: float = 10.0,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url
        self._model = model
        self._timeout = httpx.Timeout(
            connect=connect_timeout,
            read=read_timeout,
            write=write_timeout,
            pool=pool_timeout,
        )
        self._client = httpx.AsyncClient(trust_env=False, timeout=self._timeout)

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        stream: bool = False,
        response_format: dict[str, Any] | None = None,
    ) -> CompletionResult | AsyncIterator[StreamDelta]:
        url = f"{self._base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        request_body: dict[str, Any] = {
            "model": self._model,
            "messages": self._convert_messages_to_openai(messages),
        }

        if tools is not None:
            request_body["tools"] = self._convert_tools_to_openai(tools)

        if response_format is not None:
            request_body["response_format"] = response_format

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
        response_data = response.json()
        return self._parse_response(response_data)

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
        accumulated_tool_calls: dict[int, dict[str, str]] = {}

        try:
            async with self._client.stream(
                "POST",
                url,
                json=stream_body,
                headers=headers,
                timeout=timeout,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line:
                        continue

                    payload = line.strip()
                    if payload.startswith("data:"):
                        payload = payload[5:].strip()

                    if not payload:
                        continue

                    if payload == "[DONE]":
                        break

                    response_json = json.loads(payload)
                    choice = response_json["choices"][0]
                    delta = choice.get("delta", {})

                    content = delta.get("content")
                    finish_reason = choice.get("finish_reason")
                    usage_data = response_json.get("usage")
                    usage: Usage | None = None
                    if usage_data:
                        usage = Usage(
                            prompt_tokens=usage_data["prompt_tokens"],
                            completion_tokens=usage_data["completion_tokens"],
                            total_tokens=usage_data["total_tokens"],
                        )

                    tool_calls_delta = delta.get("tool_calls")
                    stream_tool_calls: list[ToolCall] | None = None
                    if tool_calls_delta:
                        for tool_call_delta in tool_calls_delta:
                            index = tool_call_delta.get("index", 0)
                            accumulated = accumulated_tool_calls.setdefault(
                                index,
                                {"id": "", "name": "", "arguments": ""},
                            )

                            if "id" in tool_call_delta and tool_call_delta["id"]:
                                accumulated["id"] = tool_call_delta["id"]

                            function_delta = tool_call_delta.get("function", {})
                            if "name" in function_delta and function_delta["name"]:
                                accumulated["name"] = function_delta["name"]
                            if (
                                "arguments" in function_delta
                                and function_delta["arguments"] is not None
                            ):
                                accumulated["arguments"] += function_delta["arguments"]

                        stream_tool_calls = []
                        for index in sorted(accumulated_tool_calls):
                            accumulated = accumulated_tool_calls[index]
                            parsed_arguments: dict[str, Any]
                            try:
                                parsed_arguments = json.loads(accumulated["arguments"])
                            except json.JSONDecodeError:
                                parsed_arguments = {
                                    "_partial": accumulated["arguments"]
                                }

                            stream_tool_calls.append(
                                ToolCall(
                                    id=accumulated["id"] or f"index_{index}",
                                    name=accumulated["name"] or "",
                                    arguments=parsed_arguments,
                                )
                            )

                    if (
                        content is None
                        and stream_tool_calls is None
                        and finish_reason is None
                        and usage is None
                    ):
                        continue

                    yield StreamDelta(
                        content=content,
                        tool_calls=stream_tool_calls,
                        finish_reason=finish_reason,
                        usage=usage,
                    )
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

    def _convert_messages_to_openai(
        self, messages: list[Message]
    ) -> list[dict[str, Any]]:
        openai_messages: list[dict[str, Any]] = []
        for msg in messages:
            openai_msg: dict[str, Any] = {
                "role": msg.role,
                "content": msg.content,
            }
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
            if msg.tool_call_id:
                openai_msg["tool_call_id"] = msg.tool_call_id
            openai_messages.append(openai_msg)
        return openai_messages

    def _convert_tools_to_openai(self, tools: list[ToolSchema]) -> list[dict[str, Any]]:
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

    def _parse_response(self, response_data: dict[str, Any]) -> CompletionResult:
        message_data = response_data["choices"][0]["message"]

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

        usage_data = response_data.get("usage")
        usage: Usage | None = None
        if usage_data:
            usage = Usage(
                prompt_tokens=usage_data["prompt_tokens"],
                completion_tokens=usage_data["completion_tokens"],
                total_tokens=usage_data["total_tokens"],
            )

        return CompletionResult(message=message, usage=usage)


def pydantic_to_response_format(model: type) -> dict[str, Any]:
    """Convert a Pydantic model to OpenAI response_format dict.

    Args:
        model: A Pydantic BaseModel class (not instance)

    Returns:
        Dictionary with type='json_schema' and json_schema containing:
        - name: model class name
        - schema: model_json_schema() output
        - strict: True

    Example:
        >>> from pydantic import BaseModel
        >>> class User(BaseModel):
        ...     name: str
        ...     age: int
        >>> response_format = pydantic_to_response_format(User)
        >>> response_format['type']
        'json_schema'
    """
    try:
        # Import here to avoid hard dependency on pydantic
        from pydantic import BaseModel

        if not isinstance(model, type) or not issubclass(model, BaseModel):
            raise TypeError(
                f"model must be a Pydantic BaseModel class, got {type(model)}"
            )

        schema = model.model_json_schema()
        return {
            "type": "json_schema",
            "json_schema": {
                "name": model.__name__,
                "schema": schema,
                "strict": True,
            },
        }
    except ImportError:
        raise ImportError(
            "pydantic must be installed to use pydantic_to_response_format"
        )
