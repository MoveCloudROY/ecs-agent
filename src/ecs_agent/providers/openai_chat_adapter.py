"""OpenAI Chat Completions adapter."""

from __future__ import annotations

import json

from collections.abc import AsyncIterator
from typing import Any, Protocol

import httpx

from ecs_agent.types import (
    CompletionResult,
    FileRefPart,
    ImageUrlPart,
    Message,
    MessagePart,
    StreamDelta,
    ToolCall,
    ToolSchema,
    Usage,
)


class _OpenAIModelFacade(Protocol):
    _api_key: str
    _base_url: str
    _model: str
    _client: httpx.AsyncClient
    _timeout: httpx.Timeout

    def _build_headers(self) -> dict[str, str]: ...

    def _handle_http_error(self, exc: httpx.HTTPStatusError) -> None: ...

    def _handle_request_error(self, exc: httpx.RequestError) -> None: ...

    def _convert_tools_to_openai(
        self, tools: list[ToolSchema]
    ) -> list[dict[str, Any]]: ...

    def _usage_from_raw(self, usage_data: Any) -> Usage | None: ...


class OpenAIChatAdapter:
    """Adapter for OpenAI-compatible Chat Completions requests."""

    def __init__(self, facade: _OpenAIModelFacade) -> None:
        self._facade = facade

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> CompletionResult:
        url = f"{self._facade._base_url}/chat/completions"
        request_body = self._build_request_body(messages, tools, response_format)

        try:
            response = await self._facade._client.post(
                url,
                json=request_body,
                headers=self._facade._build_headers(),
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            self._facade._handle_http_error(exc)
            raise
        except httpx.RequestError as exc:
            self._facade._handle_request_error(exc)
            raise

        response_data = response.json()
        return self._parse_response(response_data)

    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> AsyncIterator[StreamDelta]:
        url = f"{self._facade._base_url}/chat/completions"
        request_body = self._build_request_body(messages, tools, response_format)
        stream_body = dict(request_body)
        stream_body["stream"] = True
        # Spec-compliant providers only emit the final usage chunk (token
        # counts incl. cached_tokens) when explicitly requested.
        stream_body["stream_options"] = {"include_usage": True}

        timeout = httpx.Timeout(
            connect=self._facade._timeout.connect,
            read=None,
            write=self._facade._timeout.write,
            pool=self._facade._timeout.pool,
        )
        accumulated_tool_calls: dict[int, dict[str, str]] = {}

        try:
            async with self._facade._client.stream(
                "POST",
                url,
                json=stream_body,
                headers=self._facade._build_headers(),
                timeout=timeout,
            ) as response:
                if response.is_error:
                    # Buffer the error body inside the stream context so
                    # HTTPStatusError handlers can read response.text after
                    # the context closes.
                    await response.aread()
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

                    try:
                        response_json = json.loads(payload)
                    except json.JSONDecodeError:
                        continue

                    # The terminal usage chunk carries "choices": [] (sent by
                    # OpenAI under stream_options.include_usage, and by some
                    # gateways unconditionally).
                    choices = response_json.get("choices")
                    if isinstance(choices, list) and choices:
                        choice = choices[0]
                        delta = choice.get("delta", {})
                    else:
                        choice = {}
                        delta = {}

                    content = delta.get("content")
                    reasoning_content = delta.get("reasoning_content")
                    if content is not None and not isinstance(content, str):
                        content = None
                    if reasoning_content is not None and not isinstance(
                        reasoning_content, str
                    ):
                        reasoning_content = None

                    finish_reason = choice.get("finish_reason")
                    usage = self._facade._usage_from_raw(response_json.get("usage"))

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
                        and reasoning_content is None
                        and stream_tool_calls is None
                        and finish_reason is None
                        and usage is None
                    ):
                        continue

                    yield StreamDelta(
                        content=content,
                        reasoning_content=reasoning_content,
                        tool_calls=stream_tool_calls,
                        finish_reason=finish_reason,
                        usage=usage,
                    )
        except httpx.HTTPStatusError as exc:
            self._facade._handle_http_error(exc)
            raise
        except httpx.RequestError as exc:
            self._facade._handle_request_error(exc)
            raise

    def _build_request_body(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None,
        response_format: dict[str, Any] | None,
    ) -> dict[str, Any]:
        request_body: dict[str, Any] = {
            "model": self._facade._model,
            "messages": self._convert_messages_to_openai(messages),
        }

        if tools is not None:
            request_body["tools"] = self._facade._convert_tools_to_openai(tools)
        if response_format is not None:
            request_body["response_format"] = response_format
        return request_body

    def _convert_messages_to_openai(
        self, messages: list[Message]
    ) -> list[dict[str, Any]]:
        openai_messages: list[dict[str, Any]] = []
        for msg in messages:
            openai_msg: dict[str, Any] = {"role": msg.role}

            message_content = self._convert_message_content(msg)
            if message_content is not None:
                openai_msg["content"] = message_content
            else:
                openai_msg["content"] = msg.content

            if msg.tool_calls and not msg.content and not msg.parts:
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

    def _convert_message_content(
        self, message: Message
    ) -> str | list[dict[str, Any]] | None:
        if message.parts is None:
            return None

        content_parts: list[dict[str, Any]] = []
        if message.content:
            content_parts.append({"type": "text", "text": message.content})

        for part in message.parts:
            if isinstance(part, ImageUrlPart):
                image_url: dict[str, Any] = {"url": part.url}
                if part.detail is not None:
                    image_url["detail"] = part.detail
                content_parts.append({"type": "image_url", "image_url": image_url})
                continue

            if isinstance(part, FileRefPart):
                file_payload: dict[str, Any] = {"file_id": part.file_id}
                if part.filename is not None:
                    file_payload["filename"] = part.filename
                content_parts.append({"type": "file", "file": file_payload})

        return content_parts

    def _parse_response(self, response_data: dict[str, Any]) -> CompletionResult:
        message_data = response_data["choices"][0]["message"]
        message = self._parse_chat_message(message_data)
        usage = self._facade._usage_from_raw(response_data.get("usage"))
        return CompletionResult(message=message, usage=usage)

    def _parse_chat_message(self, message_data: dict[str, Any]) -> Message:
        role = message_data.get("role", "assistant")
        content = ""
        message_parts: list[MessagePart] = []

        raw_content = message_data.get("content")
        if isinstance(raw_content, str):
            content = raw_content
        elif isinstance(raw_content, list):
            for item in raw_content:
                if not isinstance(item, dict):
                    continue

                item_type = item.get("type")
                if item_type in {"text", "output_text", "input_text"}:
                    text = item.get("text")
                    if isinstance(text, str):
                        content += text
                    continue

                image_part = self._parse_image_part(item)
                if image_part is not None:
                    message_parts.append(image_part)
                    continue

                file_part = self._parse_file_part(item)
                if file_part is not None:
                    message_parts.append(file_part)

        tool_calls: list[ToolCall] | None = None
        raw_tool_calls = message_data.get("tool_calls")
        if isinstance(raw_tool_calls, list) and raw_tool_calls:
            tool_calls = []
            for tool_call_data in raw_tool_calls:
                function_data = tool_call_data.get("function", {})
                arguments_raw = function_data.get("arguments", "{}")
                parsed_arguments: dict[str, Any]
                if isinstance(arguments_raw, str):
                    try:
                        loaded = json.loads(arguments_raw)
                    except json.JSONDecodeError:
                        loaded = {}
                elif isinstance(arguments_raw, dict):
                    loaded = arguments_raw
                else:
                    loaded = {}
                parsed_arguments = loaded if isinstance(loaded, dict) else {}

                tool_calls.append(
                    ToolCall(
                        id=tool_call_data["id"],
                        name=function_data["name"],
                        arguments=parsed_arguments,
                    )
                )

        return Message(
            role=role,
            content=content,
            parts=message_parts or None,
            tool_calls=tool_calls,
        )

    def _parse_image_part(self, item: dict[str, Any]) -> ImageUrlPart | None:
        item_type = item.get("type")
        if item_type not in {"image_url", "input_image", "image"}:
            return None

        image_data = item.get("image_url")
        detail: str | None = None
        if isinstance(image_data, dict):
            url = image_data.get("url")
            raw_detail = image_data.get("detail")
            if isinstance(raw_detail, str):
                detail = raw_detail
        elif isinstance(item.get("url"), str):
            url = item["url"]
            raw_detail = item.get("detail")
            if isinstance(raw_detail, str):
                detail = raw_detail
        elif isinstance(item.get("image_url"), str):
            url = item["image_url"]
        else:
            return None

        if not isinstance(url, str):
            return None
        return ImageUrlPart(url=url, detail=detail)

    def _parse_file_part(self, item: dict[str, Any]) -> FileRefPart | None:
        item_type = item.get("type")
        if item_type not in {"file", "input_file"}:
            return None

        raw_file = item.get("file")
        file_data: dict[str, Any]
        if isinstance(raw_file, dict):
            file_data = raw_file
        else:
            file_data = item
        file_id = file_data.get("file_id")
        filename = file_data.get("filename")
        if not isinstance(file_id, str):
            return None
        return FileRefPart(
            file_id=file_id, filename=filename if isinstance(filename, str) else None
        )
