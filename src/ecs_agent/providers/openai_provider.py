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
        use_responses_api: bool = False,
        connect_timeout: float = 10.0,
        read_timeout: float = 120.0,
        write_timeout: float = 10.0,
        pool_timeout: float = 10.0,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url
        self._model = model
        self.use_responses_api = use_responses_api
        self._responses_api_available: bool | None = None
        self.previous_response_id: str | None = None
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
        if self.use_responses_api and self._responses_api_available is not False:
            if stream:
                return self._stream_responses_api(messages, tools, response_format)

            try:
                result = await self._complete_responses_api(
                    messages, tools, response_format
                )
                self._responses_api_available = True
                return result
            except httpx.HTTPStatusError as exc:
                if not self._should_fallback_from_responses(exc):
                    raise
                self._responses_api_available = False
                logger.info(
                    "responses_api_fallback",
                    status_code=exc.response.status_code,
                    endpoint=f"{self._base_url}/responses",
                )

        url, headers, request_body = self._build_chat_completion_request(
            messages,
            tools,
            response_format,
        )

        if stream:
            return self._stream_complete(url, headers, request_body)

        return await self._complete_non_streaming(url, headers, request_body)

    def _build_chat_completion_request(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None,
        response_format: dict[str, Any] | None,
    ) -> tuple[str, dict[str, str], dict[str, Any]]:
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

        return url, headers, request_body

    def _should_fallback_from_responses(self, exc: httpx.HTTPStatusError) -> bool:
        return exc.response.status_code == 404

    async def _complete_non_streaming(
        self,
        url: str,
        headers: dict[str, str],
        request_body: dict[str, Any],
    ) -> CompletionResult:
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

    async def _complete_responses_api(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> CompletionResult:
        url = f"{self._base_url}/responses"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        instructions = self._extract_responses_instructions(messages)
        request_body: dict[str, Any] = {
            "model": self._model,
            "input": self._convert_messages_to_responses_input(messages),
            "store": False,
        }

        if instructions:
            request_body["instructions"] = instructions
        if tools is not None:
            request_body["tools"] = self._convert_tools_to_openai(tools)
        if response_format is not None:
            request_body["response_format"] = response_format
        if self.previous_response_id is not None:
            request_body["previous_response_id"] = self.previous_response_id

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
        response_id = response_data.get("id")
        if isinstance(response_id, str) and response_id:
            self.previous_response_id = response_id

        message = self._parse_responses_output(response_data.get("output", []))
        usage = self._parse_responses_usage(response_data.get("usage"))
        return CompletionResult(message=message, usage=usage)

    async def _stream_responses_api(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> AsyncIterator[StreamDelta]:
        """Stream completion using OpenAI Responses API with SSE events."""
        url = f"{self._base_url}/responses"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        instructions = self._extract_responses_instructions(messages)
        request_body: dict[str, Any] = {
            "model": self._model,
            "input": self._convert_messages_to_responses_input(messages),
            "store": False,
            "stream": True,
        }

        if instructions:
            request_body["instructions"] = instructions
        if tools is not None:
            request_body["tools"] = self._convert_tools_to_openai(tools)
        if response_format is not None:
            request_body["response_format"] = response_format
        if self.previous_response_id is not None:
            request_body["previous_response_id"] = self.previous_response_id

        timeout = httpx.Timeout(
            connect=self._timeout.connect,
            read=None,
            write=self._timeout.write,
            pool=self._timeout.pool,
        )

        # Track output items and tool call accumulation
        output_items: dict[int, dict[str, Any]] = {}
        current_response_id: str | None = None

        try:
            async with self._client.stream(
                "POST",
                url,
                json=request_body,
                headers=headers,
                timeout=timeout,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line:
                        continue

                    stripped = line.strip()
                    if not stripped:
                        continue

                    # Parse SSE format: 'event: <event_name>' or 'data: <json>'
                    if stripped.startswith("event:"):
                        continue

                    if stripped.startswith("data:"):
                        data_str = stripped[5:].strip()
                        if not data_str or data_str == "[DONE]":
                            break

                        try:
                            event_data = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue

                        event_type = event_data.get("type")

                        # Handle response.created event
                        if event_type == "response.created":
                            response_obj = event_data.get("response", {})
                            response_id = response_obj.get("id")
                            if isinstance(response_id, str):
                                current_response_id = response_id
                            continue

                        # Handle output_item.added event
                        if event_type == "response.output_item.added":
                            output_index = event_data.get("output_index", 0)
                            item = event_data.get("item", {})
                            output_items[output_index] = {
                                "type": item.get("type"),
                                "role": item.get("role"),
                                "id": item.get("id"),
                                "name": item.get("name"),
                                "content_parts": [],
                                "arguments": "",
                            }
                            continue

                        # Handle output_item.delta event
                        if event_type == "response.output_item.delta":
                            output_index = event_data.get("output_index", 0)
                            delta = event_data.get("delta", {})
                            delta_type = delta.get("type")

                            if output_index not in output_items:
                                continue

                            item_data = output_items[output_index]

                            # Content delta
                            if delta_type == "content_delta":
                                text = delta.get("text")
                                if isinstance(text, str) and text:
                                    item_data["content_parts"].append(text)
                                    yield StreamDelta(content=text)

                            # Arguments delta (for function calls)
                            elif delta_type == "arguments_delta":
                                arguments = delta.get("arguments")
                                if isinstance(arguments, str):
                                    item_data["arguments"] += arguments

                            continue

                        # Handle output_item.done event
                        if event_type == "response.output_item.done":
                            output_index = event_data.get("output_index", 0)
                            if output_index not in output_items:
                                continue

                            item_data = output_items[output_index]

                            # If it's a function_call, yield tool calls
                            if item_data.get("type") == "function_call":
                                tool_call_id = item_data.get("id", "")
                                name = item_data.get("name", "")
                                arguments_str = item_data.get("arguments", "{}")

                                try:
                                    parsed_args = json.loads(arguments_str)
                                except json.JSONDecodeError:
                                    parsed_args = {}

                                if isinstance(parsed_args, dict):
                                    tool_call = ToolCall(
                                        id=tool_call_id,
                                        name=name,
                                        arguments=parsed_args,
                                    )
                                    yield StreamDelta(tool_calls=[tool_call])

                            continue

                        # Handle response.done event
                        if event_type == "response.done":
                            response_obj = event_data.get("response", {})
                            response_id = response_obj.get("id")
                            usage_data = response_obj.get("usage")

                            if isinstance(response_id, str) and response_id:
                                current_response_id = response_id

                            usage = self._parse_responses_usage(usage_data)
                            yield StreamDelta(
                                finish_reason="stop",
                                usage=usage,
                            )
                            break

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

        # Update response_id after streaming completes
        if current_response_id:
            self.previous_response_id = current_response_id
        self._responses_api_available = True

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
                            "arguments": json.dumps(tc.arguments),
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

    def _extract_responses_instructions(self, messages: list[Message]) -> str | None:
        system_instructions = [
            msg.content.strip()
            for msg in messages
            if msg.role == "system" and msg.content.strip()
        ]
        if not system_instructions:
            return None
        return "\n\n".join(system_instructions)

    def _convert_messages_to_responses_input(
        self, messages: list[Message]
    ) -> list[dict[str, Any]]:
        input_items: list[dict[str, Any]] = []

        for msg in messages:
            if msg.role == "system":
                continue

            if msg.role == "tool":
                if msg.tool_call_id is None:
                    continue
                input_items.append(
                    {
                        "type": "function_call_output",
                        "call_id": msg.tool_call_id,
                        "output": msg.content,
                    }
                )
                continue

            if msg.role in ("user", "assistant") and msg.content:
                content_type = "input_text" if msg.role == "user" else "output_text"
                input_items.append(
                    {
                        "type": "message",
                        "role": msg.role,
                        "content": [{"type": content_type, "text": msg.content}],
                    }
                )

            if msg.role == "assistant" and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    input_items.append(
                        {
                            "type": "function_call",
                            "id": tool_call.id,
                            "name": tool_call.name,
                            "arguments": json.dumps(tool_call.arguments),
                        }
                    )

        return input_items

    def _parse_responses_output(self, output_items: Any) -> Message:
        if not isinstance(output_items, list):
            return Message(role="assistant", content="")

        role = "assistant"
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []

        for output_item in output_items:
            if not isinstance(output_item, dict):
                continue

            item_type = output_item.get("type")
            if item_type == "message":
                item_role = output_item.get("role")
                if isinstance(item_role, str):
                    role = item_role

                item_content = output_item.get("content")
                if isinstance(item_content, list):
                    for content_item in item_content:
                        if not isinstance(content_item, dict):
                            continue
                        content_type = content_item.get("type")
                        text = content_item.get("text")
                        if content_type in (
                            "output_text",
                            "input_text",
                            "text",
                        ) and isinstance(text, str):
                            text_parts.append(text)

            if item_type == "function_call":
                tool_call_id = output_item.get("id") or output_item.get("call_id")
                name = output_item.get("name")
                arguments = output_item.get("arguments", "{}")
                if not isinstance(tool_call_id, str) or not isinstance(name, str):
                    continue

                parsed_arguments: dict[str, Any]
                if isinstance(arguments, str):
                    try:
                        loaded_arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        loaded_arguments = {}
                elif isinstance(arguments, dict):
                    loaded_arguments = arguments
                else:
                    loaded_arguments = {}

                if isinstance(loaded_arguments, dict):
                    parsed_arguments = loaded_arguments
                else:
                    parsed_arguments = {}

                tool_calls.append(
                    ToolCall(
                        id=tool_call_id,
                        name=name,
                        arguments=parsed_arguments,
                    )
                )

        return Message(
            role=role,
            content="".join(text_parts),
            tool_calls=tool_calls or None,
        )

    def _parse_responses_usage(self, usage_data: Any) -> Usage | None:
        if not isinstance(usage_data, dict):
            return None

        prompt_tokens = usage_data.get("prompt_tokens", usage_data.get("input_tokens"))
        completion_tokens = usage_data.get(
            "completion_tokens", usage_data.get("output_tokens")
        )
        total_tokens = usage_data.get("total_tokens")

        if not isinstance(prompt_tokens, int) or not isinstance(completion_tokens, int):
            return None

        if not isinstance(total_tokens, int):
            total_tokens = prompt_tokens + completion_tokens

        return Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )

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
