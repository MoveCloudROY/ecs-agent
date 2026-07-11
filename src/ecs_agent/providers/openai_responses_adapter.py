"""OpenAI Responses API adapter."""

from __future__ import annotations

import json

from collections.abc import AsyncIterator
from typing import Any, Protocol, cast

import httpx

from ecs_agent.logging import get_logger
from ecs_agent.providers.config import ProviderConfig
from ecs_agent.types import (
    CompletionResult,
    FileRefPart,
    ImageUrlPart,
    Message,
    MessagePart,
    MessageRole,
    StreamDelta,
    ToolCall,
    ToolSchema,
    Usage,
)

logger = get_logger(__name__)


class _OpenAIModelFacade(Protocol):
    _base_url: str
    _model: str
    _client: httpx.AsyncClient
    _timeout: httpx.Timeout
    _responses_api_available: bool | None
    _provider_config: ProviderConfig

    def _build_headers(self) -> dict[str, str]: ...

    def _handle_http_error(self, exc: httpx.HTTPStatusError) -> None: ...

    def _handle_request_error(self, exc: httpx.RequestError) -> None: ...

    def _convert_tools_to_openai(
        self, tools: list[ToolSchema]
    ) -> list[dict[str, Any]]: ...

    def _convert_tools_to_responses(
        self, tools: list[ToolSchema]
    ) -> list[dict[str, Any]]: ...

    def _usage_from_raw(self, usage_data: Any) -> Usage | None: ...

    def _extract_responses_instructions(
        self, messages: list[Message]
    ) -> str | None: ...


class OpenAIResponsesAdapter:
    """Adapter for OpenAI-compatible Responses requests."""

    def __init__(self, facade: _OpenAIModelFacade) -> None:
        self._facade = facade
        # Latched when a provider 400s a request because of
        # previous_response_id (e.g. gateways whose HTTP endpoint does not
        # implement stored-response chaining). Once set, the chain is
        # omitted from every subsequent request of this model instance.
        self._previous_response_id_rejected = False

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        response_format: dict[str, Any] | None = None,
        previous_response_id: str | None = None,
    ) -> CompletionResult:
        request_body = self._build_request_body(
            messages,
            tools,
            response_format,
            previous_response_id,
        )
        url = f"{self._facade._base_url}/responses"

        while True:
            try:
                response = await self._facade._client.post(
                    url,
                    json=request_body,
                    headers=self._facade._build_headers(),
                )
                response.raise_for_status()
                break
            except httpx.HTTPStatusError as exc:
                if self._should_retry_without_previous_response_id(
                    exc, request_body
                ):
                    request_body = self._request_body_without_previous_response_id(
                        request_body, exc
                    )
                    continue
                self._facade._handle_http_error(exc)
                raise
            except httpx.RequestError as exc:
                self._facade._handle_request_error(exc)
                raise

        response_data = response.json()
        response_id = response_data.get("id")
        resolved_response_id = response_id if isinstance(response_id, str) else None

        status = response_data.get("status")
        if status == "failed":
            error = response_data.get("error") or {}
            code = error.get("code", "unknown")
            msg = error.get("message", "unknown error")
            raise ValueError(f"Responses API returned failed status: [{code}] {msg}")

        message = self._parse_responses_output(response_data.get("output", []))
        usage = self._facade._usage_from_raw(response_data.get("usage"))
        return CompletionResult(
            message=message,
            usage=usage,
            response_id=resolved_response_id,
        )

    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        response_format: dict[str, Any] | None = None,
        previous_response_id: str | None = None,
    ) -> AsyncIterator[StreamDelta]:
        request_body = self._build_request_body(
            messages,
            tools,
            response_format,
            previous_response_id,
        )
        request_body["stream"] = True

        url = f"{self._facade._base_url}/responses"
        timeout = httpx.Timeout(
            connect=self._facade._timeout.connect,
            read=None,
            write=self._facade._timeout.write,
            pool=self._facade._timeout.pool,
        )

        while True:
            try:
                async for delta in self._stream_once(url, request_body, timeout):
                    yield delta
                break
            except httpx.HTTPStatusError as exc:
                # raise_for_status fires before any delta is yielded, so
                # retrying here never replays partial output.
                if self._should_retry_without_previous_response_id(
                    exc, request_body
                ):
                    request_body = self._request_body_without_previous_response_id(
                        request_body, exc
                    )
                    continue
                self._facade._handle_http_error(exc)
                raise
            except httpx.RequestError as exc:
                self._facade._handle_request_error(exc)
                raise

        self._facade._responses_api_available = True

    async def _stream_once(
        self,
        url: str,
        request_body: dict[str, Any],
        timeout: httpx.Timeout,
    ) -> AsyncIterator[StreamDelta]:
        """Run a single streaming request and yield its deltas."""
        output_items: dict[int, dict[str, Any]] = {}
        current_response_id: str | None = None

        async with self._facade._client.stream(
            "POST",
            url,
            json=request_body,
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

                stripped = line.strip()
                if not stripped or stripped.startswith("event:"):
                    continue
                if not stripped.startswith("data:"):
                    continue

                data_str = stripped[5:].strip()
                if not data_str or data_str == "[DONE]":
                    break

                try:
                    event_data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                event_type = event_data.get("type")
                if event_type == "response.created":
                    response_obj = event_data.get("response", {})
                    response_id = response_obj.get("id")
                    if isinstance(response_id, str):
                        current_response_id = response_id
                    continue

                if event_type == "response.output_item.added":
                    output_index = event_data.get("output_index", 0)
                    item = event_data.get("item", {})
                    output_items[output_index] = {
                        "type": item.get("type"),
                        "role": item.get("role"),
                        # call_id pairs with function_call_output on replay;
                        # fall back to the item id for providers that omit it.
                        "id": item.get("call_id") or item.get("id"),
                        "name": item.get("name"),
                        "arguments": "",
                    }
                    continue

                if event_type == "response.output_item.delta":
                    output_index = event_data.get("output_index", 0)
                    delta = event_data.get("delta", {})
                    delta_type = delta.get("type")

                    if output_index not in output_items:
                        continue

                    item_data = output_items[output_index]
                    if delta_type == "content_delta":
                        text = delta.get("text")
                        if isinstance(text, str) and text:
                            yield StreamDelta(content=text)
                    elif delta_type == "arguments_delta":
                        arguments = delta.get("arguments")
                        if isinstance(arguments, str):
                            item_data["arguments"] += arguments
                    continue

                # Standard OpenAI Responses dialect: "delta" is a plain string
                # (the block above handles the object-shaped legacy dialect).
                if event_type == "response.output_text.delta":
                    text_delta = event_data.get("delta")
                    if isinstance(text_delta, str) and text_delta:
                        yield StreamDelta(content=text_delta)
                    continue

                if event_type == "response.function_call_arguments.delta":
                    output_index = event_data.get("output_index", 0)
                    arguments_delta = event_data.get("delta")
                    if output_index in output_items and isinstance(
                        arguments_delta, str
                    ):
                        output_items[output_index]["arguments"] += arguments_delta
                    continue

                if event_type == "response.failed":
                    response_obj = event_data.get("response", {})
                    error = response_obj.get("error") or {}
                    code = error.get("code", "unknown")
                    msg = error.get("message", "unknown error")
                    raise ValueError(
                        f"Responses API returned failed status: [{code}] {msg}"
                    )

                if event_type == "response.output_item.done":
                    output_index = event_data.get("output_index", 0)
                    if output_index not in output_items:
                        continue

                    item_data = output_items[output_index]
                    # Standard dialect: the done item carries the complete
                    # arguments string, which is authoritative over whatever
                    # accumulated from delta events.
                    done_item = event_data.get("item")
                    if isinstance(done_item, dict):
                        done_arguments = done_item.get("arguments")
                        if isinstance(done_arguments, str) and done_arguments:
                            item_data["arguments"] = done_arguments

                    if item_data.get("type") == "function_call":
                        tool_call_id = item_data.get("id", "")
                        name = item_data.get("name", "")
                        arguments_str = item_data.get("arguments", "{}")

                        try:
                            parsed_args = json.loads(arguments_str)
                        except json.JSONDecodeError:
                            parsed_args = {}

                        if isinstance(parsed_args, dict):
                            yield StreamDelta(
                                tool_calls=[
                                    ToolCall(
                                        id=tool_call_id,
                                        name=name,
                                        arguments=parsed_args,
                                    )
                                ]
                            )
                    continue

                if event_type in ("response.done", "response.completed"):
                    response_obj = event_data.get("response", {})
                    response_id = response_obj.get("id")
                    usage = self._facade._usage_from_raw(
                        response_obj.get("usage")
                    )
                    if isinstance(response_id, str) and response_id:
                        current_response_id = response_id

                    yield StreamDelta(
                        finish_reason="stop",
                        usage=usage,
                        response_id=current_response_id,
                    )
                    break

    def _build_request_body(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None,
        response_format: dict[str, Any] | None,
        previous_response_id: str | None,
    ) -> dict[str, Any]:
        request_body: dict[str, Any] = {
            "model": self._facade._model,
            "input": self._convert_messages_to_responses_input(messages),
            "store": self._facade._provider_config.enable_store,
        }

        instructions = self._facade._extract_responses_instructions(messages)
        if instructions:
            request_body["instructions"] = instructions
        if tools is not None:
            request_body["tools"] = self._facade._convert_tools_to_responses(tools)
        if response_format is not None:
            request_body["response_format"] = response_format
        # Only chain to a prior response when storage is enabled: with
        # store=false the referenced response was never persisted server-side,
        # and providers reject previous_response_id for unstored responses.
        # The full message history is always sent in "input", so omitting the
        # chain loses no context.
        if (
            previous_response_id is not None
            and self._facade._provider_config.enable_store
            and not self._previous_response_id_rejected
        ):
            request_body["previous_response_id"] = previous_response_id
        return request_body

    def _should_retry_without_previous_response_id(
        self, exc: httpx.HTTPStatusError, request_body: dict[str, Any]
    ) -> bool:
        """True when a 400 blames the previous_response_id we sent.

        Seen on gateways whose HTTP Responses endpoint does not implement
        stored-response chaining (e.g. "previous_response_id is only
        supported on Responses WebSocket v2"). The full history is already
        in "input", so retrying without the chain loses no context.
        """
        if "previous_response_id" not in request_body:
            return False
        if exc.response.status_code != 400:
            return False
        return "previous_response_id" in exc.response.text

    def _request_body_without_previous_response_id(
        self, request_body: dict[str, Any], exc: httpx.HTTPStatusError
    ) -> dict[str, Any]:
        """Latch chaining off for this model instance and rebuild the body."""
        self._previous_response_id_rejected = True
        logger.warning(
            "responses_previous_response_id_rejected",
            status_code=exc.response.status_code,
            response_body=exc.response.text,
            endpoint=f"{self._facade._base_url}/responses",
        )
        return {
            key: value
            for key, value in request_body.items()
            if key != "previous_response_id"
        }

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

            if msg.role in ("user", "assistant"):
                content_items = self._responses_content_parts(msg)
                if content_items:
                    input_items.append(
                        {
                            "type": "message",
                            "role": msg.role,
                            "content": content_items,
                        }
                    )

            if msg.role == "assistant" and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    input_items.append(
                        {
                            "type": "function_call",
                            # call_id (not "id") is what providers match against
                            # the function_call_output items on stateless replay.
                            "call_id": tool_call.id,
                            "name": tool_call.name,
                            "arguments": json.dumps(tool_call.arguments),
                        }
                    )

        return input_items

    def _responses_content_parts(self, message: Message) -> list[dict[str, Any]]:
        parts: list[dict[str, Any]] = []
        text_type = "input_text" if message.role == "user" else "output_text"

        if message.content:
            parts.append({"type": text_type, "text": message.content})

        if message.parts is None:
            return parts

        for part in message.parts:
            if isinstance(part, ImageUrlPart):
                payload: dict[str, Any] = {
                    "type": "input_image",
                    "image_url": part.url,
                }
                if part.detail is not None:
                    payload["detail"] = part.detail
                parts.append(payload)
                continue

            if isinstance(part, FileRefPart):
                payload = {
                    "type": "input_file",
                    "file_id": part.file_id,
                }
                if part.filename is not None:
                    payload["filename"] = part.filename
                parts.append(payload)

        return parts

    def _parse_responses_output(self, output_items: Any) -> Message:
        if not isinstance(output_items, list):
            return Message(role="assistant", content="")

        role = "assistant"
        # (phase, text) pairs: some gateways split one response into several
        # message items tagged with a non-standard "phase" (e.g. "commentary"
        # preamble + "final_answer"), often repeating the same text in both.
        text_entries: list[tuple[str | None, str]] = []
        message_parts: list[MessagePart] = []
        tool_calls: list[ToolCall] = []

        for output_item in output_items:
            if not isinstance(output_item, dict):
                continue

            item_type = output_item.get("type")
            if item_type == "message":
                item_role = output_item.get("role")
                if isinstance(item_role, str):
                    role = item_role

                raw_phase = output_item.get("phase")
                item_phase = raw_phase if isinstance(raw_phase, str) else None
                item_content = output_item.get("content")
                if isinstance(item_content, list):
                    item_texts: list[str] = []
                    for content_item in item_content:
                        if not isinstance(content_item, dict):
                            continue
                        self._collect_message_part(
                            content_item, item_texts, message_parts
                        )
                    text_entries.extend((item_phase, text) for text in item_texts)

            if item_type == "function_call":
                # Prefer call_id (pairs with function_call_output.call_id on
                # replay); the item "id" is a distinct item identifier that
                # providers reject as a call reference.
                tool_call_id = output_item.get("call_id") or output_item.get("id")
                name = output_item.get("name")
                arguments = output_item.get("arguments", "{}")
                if not isinstance(tool_call_id, str) or not isinstance(name, str):
                    continue

                if isinstance(arguments, str):
                    try:
                        loaded_arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        loaded_arguments = {}
                elif isinstance(arguments, dict):
                    loaded_arguments = arguments
                else:
                    loaded_arguments = {}

                parsed_arguments = (
                    loaded_arguments if isinstance(loaded_arguments, dict) else {}
                )
                tool_calls.append(
                    ToolCall(
                        id=tool_call_id,
                        name=name,
                        arguments=parsed_arguments,
                    )
                )

        # When the provider marks a "final_answer" message, it is THE
        # user-facing text; any commentary items are preamble that usually
        # duplicates it. Without phase tags, keep every text (standard OpenAI).
        final_texts = [
            text for phase, text in text_entries if phase == "final_answer"
        ]
        text_parts = (
            final_texts if final_texts else [text for _, text in text_entries]
        )

        return Message(
            role=cast(MessageRole, role),
            content="".join(text_parts),
            parts=message_parts or None,
            tool_calls=tool_calls or None,
        )

    def _collect_message_part(
        self,
        content_item: dict[str, Any],
        text_parts: list[str],
        message_parts: list[MessagePart],
    ) -> None:
        content_type = content_item.get("type")

        if content_type in {"output_text", "input_text", "text"}:
            text = content_item.get("text")
            if isinstance(text, str):
                text_parts.append(text)

            return

        if content_type in {"input_image", "image_url", "image"}:
            image_url = content_item.get("image_url")
            detail = content_item.get("detail")
            if isinstance(image_url, dict):
                image_url = image_url.get("url")
            if isinstance(image_url, str):
                message_parts.append(
                    ImageUrlPart(
                        url=image_url,
                        detail=detail if isinstance(detail, str) else None,
                    )
                )
            return

        if content_type in {"input_file", "file"}:
            file_id = content_item.get("file_id")
            filename = content_item.get("filename")
            file_data = content_item.get("file")
            if isinstance(file_data, dict):
                file_id = file_data.get("file_id", file_id)
                filename = file_data.get("filename", filename)

            if isinstance(file_id, str):
                message_parts.append(
                    FileRefPart(
                        file_id=file_id,
                        filename=filename if isinstance(filename, str) else None,
                    )
                )
