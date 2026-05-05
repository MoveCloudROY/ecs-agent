import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import structlog

from ecs_agent.providers.anthropic_messages_adapter import (
    AnthropicMessagesAdapter,
    AnthropicMessagesAdapterConfig,
    AnthropicMessagesRequest,
)
from ecs_agent.providers.config import ApiFormat, ProviderConfig
from ecs_agent.types import (
    CompletionResult,
    Message,
    StreamDelta,
    ToolCall,
    ToolSchema,
)

logger = structlog.get_logger(__name__)


class ClaudeModel:
    """Native Anthropic Messages API provider with streaming and tool use support."""

    def __init__(
        self,
        config: ProviderConfig,
        model: str = "glm-5.1",
        max_tokens: int = 65535,
        connect_timeout: float = 10.0,
        read_timeout: float = 600.0,
        write_timeout: float = 60.0,
        pool_timeout: float = 60.0,
        supports_vision: bool = False,
    ) -> None:
        if config.api_format is not ApiFormat.ANTHROPIC_MESSAGES:
            raise ValueError("ProviderConfig.api_format must be ANTHROPIC_MESSAGES")

        timeout_override = config.timeout
        if timeout_override is not None:
            connect_timeout = timeout_override
            read_timeout = timeout_override
            write_timeout = timeout_override
            pool_timeout = timeout_override

        self._provider_config = config
        self._api_key = config.api_key
        self._base_url = config.base_url
        self._model = model
        self._max_tokens = max_tokens
        self._supports_vision = supports_vision
        self._messages_adapter = AnthropicMessagesAdapter(
            AnthropicMessagesAdapterConfig(
                provider=self._provider_config,
                model=self._model,
                max_tokens=self._max_tokens,
                supports_vision=self._supports_vision,
            )
        )
        self._timeout = httpx.Timeout(
            connect=connect_timeout,
            read=read_timeout,
            write=write_timeout,
            pool=pool_timeout,
        )
        self._client = httpx.AsyncClient(trust_env=False, timeout=self._timeout)

    @property
    def model_id(self) -> str:
        return self._model

    @property
    def provider_id(self) -> str:
        """Low-cardinality provider label used for accounting and metrics."""
        return self._provider_config.provider_id

    def _build_messages(
        self, messages: list[Message]
    ) -> tuple[str | None, list[dict[str, Any]]]:
        return self._messages_adapter.build_messages(messages)

    def _build_tools(
        self, tools: list[ToolSchema] | None
    ) -> list[dict[str, Any]] | None:
        return self._messages_adapter.build_tools(tools)

    def _parse_response(self, response_data: dict[str, Any]) -> CompletionResult:
        return self._messages_adapter.parse_response(response_data)

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
        request = AnthropicMessagesRequest(
            messages=messages,
            tools=tools,
            response_format=response_format,
        )
        request_body = self._messages_adapter.build_request_body(request)
        headers = self._messages_adapter.headers()
        url = self._messages_adapter.endpoint_url()

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
