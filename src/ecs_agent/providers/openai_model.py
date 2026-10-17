"""OpenAI-compatible provider facade with explicit API adapters."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import httpx

from ecs_agent.accounting.normalization import normalize_openai_usage
from ecs_agent.logging import get_logger
from ecs_agent.providers._openai_format import convert_tools_to_openai
from ecs_agent.providers.config import ApiFormat, ProviderConfig
from ecs_agent.providers.openai_chat_adapter import OpenAIChatAdapter
from ecs_agent.providers.openai_responses_adapter import OpenAIResponsesAdapter
from ecs_agent.providers.responses_state import ResponseContext
from ecs_agent.types import CompletionResult, Message, StreamDelta, ToolSchema, Usage

logger = get_logger(__name__)


async def _aclose_if_supported(stream: AsyncIterator[StreamDelta]) -> None:
    """Close a stream that is an async generator; no-op otherwise.

    The adapters yield (so they are async generators with aclose at runtime),
    but the declared AsyncIterator type does not expose it — getattr keeps the
    checker happy and tolerates plain iterators.
    """
    aclose = getattr(stream, "aclose", None)
    if aclose is not None:
        await aclose()


class OpenAIModel:
    """OpenAI-compatible LLM model facade over chat/responses adapters."""

    def __init__(
        self,
        config: ProviderConfig,
        model: str = "gpt-4o-mini",
        connect_timeout: float = 10.0,
        read_timeout: float = 120.0,
        write_timeout: float = 10.0,
        pool_timeout: float = 10.0,
        stream_read_timeout: float | None = None,
    ) -> None:
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
        self._responses_api_available: bool | None = None
        self._timeout = httpx.Timeout(
            connect=connect_timeout,
            read=read_timeout,
            write=write_timeout,
            pool=pool_timeout,
        )
        # Per-chunk read timeout for streaming only. None (the default) leaves
        # streams unbounded so a slow model is never cut off; a value turns it
        # into a stall detector — see the adapters' stream() timeout note. It is
        # deliberately independent of read_timeout/config.timeout so bounding
        # whole-request time never silently caps streaming reasoning.
        self._stream_read_timeout = stream_read_timeout
        self._client = httpx.AsyncClient(trust_env=False, timeout=self._timeout)
        self._chat_adapter = OpenAIChatAdapter(self)
        self._responses_adapter = OpenAIResponsesAdapter(self)

    async def aclose(self) -> None:
        """Close the HTTP pool shared by Chat and the Responses SDK client."""
        await self._client.aclose()

    @property
    def model_id(self) -> str:
        return self._model

    def get_response_context(self, messages: list[Message]) -> ResponseContext | None:
        """Describe the exact protocol prefix used for verified automatic chains."""
        if self._provider_config.api_format != ApiFormat.OPENAI_RESPONSES or self._responses_api_available is False:
            return None
        adapter = self._responses_adapter
        items: list[dict[str, Any]] = []
        ends: list[int] = []
        for message in messages:
            if message.role == "system":
                continue
            items.extend(adapter._convert_messages_to_responses_input([message]))
            ends.append(len(items))
        return ResponseContext(
            origin=adapter.origin, items=items, message_ends=ends,
            automatic_chaining=self._provider_config.enable_response_chaining and not adapter._previous_response_id_rejected,
        )

    @property
    def provider_id(self) -> str:
        """Low-cardinality provider label used for accounting and metrics."""
        return self._provider_config.provider_id

    async def _fallback_to_chat(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None,
        response_format: dict[str, Any] | None,
        stream: bool,
    ) -> CompletionResult | AsyncIterator[StreamDelta]:
        if stream:
            return self._chat_adapter.stream(messages, tools, response_format)
        return await self._chat_adapter.complete(messages, tools, response_format)

    async def _responses_stream_with_fallback(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None,
        response_format: dict[str, Any] | None,
        thread_response_id: str | None,
        covered_message_count: int | None = None,
    ) -> AsyncIterator[StreamDelta]:
        """Stream via the Responses API, falling back to Chat on a 404.

        Mirrors the non-streaming path: the adapter's ``raise_for_status``
        fires before any delta is yielded, so a 404 (endpoint does not
        implement Responses) latches the format off and replays the request
        against Chat Completions. The ``started`` guard refuses to fall back
        once real output has flowed, so partial responses are never duplicated.
        """
        # aclosing: closing THIS wrapper generator must close the inner
        # adapter stream too — PEP 525 gives `async for` no implicit aclose,
        # so the inner generator (holding the HTTP connection) would otherwise
        # stay open until garbage collection. The adapters return async
        # generators (declared AsyncIterator); guard aclose via getattr so the
        # type-checker stays happy and non-generator iterators are tolerated.
        stream = self._responses_adapter.stream(
            messages, tools, response_format, thread_response_id, covered_message_count
        )
        started = False
        try:
            try:
                async for delta in stream:
                    started = True
                    yield delta
            finally:
                await _aclose_if_supported(stream)
        except httpx.HTTPStatusError as exc:
            if started or not self._should_fallback_from_responses(exc):
                raise
            self._responses_api_available = False
            logger.info(
                "responses_api_fallback",
                status_code=exc.response.status_code,
                endpoint=f"{self._base_url}/responses",
            )
            fallback_stream = self._chat_adapter.stream(
                messages, tools, response_format
            )
            try:
                async for delta in fallback_stream:
                    yield delta
            finally:
                await _aclose_if_supported(fallback_stream)

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        stream: bool = False,
        response_format: dict[str, Any] | None = None,
        thread_response_id: str | None = None,
        covered_message_count: int | None = None,
    ) -> CompletionResult | AsyncIterator[StreamDelta]:
        api_format = self._provider_config.api_format

        if api_format == ApiFormat.OPENAI_RESPONSES:
            if self._responses_api_available is False:
                if self._provider_config.enable_store is not None:
                    logger.error("responses_storage_contract_unavailable", store=self._provider_config.enable_store)
                    raise ValueError("Responses unavailable; explicit store cannot be dropped during Chat fallback")
                return await self._fallback_to_chat(
                    messages, tools, response_format, stream
                )

            if stream:
                return self._responses_stream_with_fallback(
                    messages,
                    tools,
                    response_format,
                    thread_response_id,
                    covered_message_count,
                )

            try:
                result = await self._responses_adapter.complete(
                    messages,
                    tools,
                    response_format,
                    thread_response_id,
                    covered_message_count,
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
                return await self._fallback_to_chat(
                    messages, tools, response_format, stream
                )

        if api_format == ApiFormat.OPENAI_CHAT_COMPLETIONS:
            if stream:
                return self._chat_adapter.stream(messages, tools, response_format)
            return await self._chat_adapter.complete(messages, tools, response_format)

        raise ValueError(
            "Unsupported OpenAI provider api_format "
            f"'{api_format}'. Supported formats: "
            f"{ApiFormat.OPENAI_CHAT_COMPLETIONS.value}, "
            f"{ApiFormat.OPENAI_RESPONSES.value}."
        )

    def _build_headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        headers.update(self._provider_config.extra_headers)
        return headers

    def _should_fallback_from_responses(self, exc: httpx.HTTPStatusError) -> bool:
        return (
            exc.response.status_code == 404
            and not self._responses_adapter._blames_previous_response_id(exc.response.text)
            and self._provider_config.enable_store is None
        )

    def _usage_from_raw(self, usage_data: Any) -> Usage | None:
        if not isinstance(usage_data, dict):
            return None
        return normalize_openai_usage(usage_data)

    def _convert_tools_to_openai(self, tools: list[ToolSchema]) -> list[dict[str, Any]]:
        return convert_tools_to_openai(tools)

    def _convert_tools_to_responses(self, tools: list[ToolSchema]) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            }
            for tool in tools
        ]

    def _extract_responses_instructions(self, messages: list[Message]) -> str | None:
        system_instructions = [
            message.content.strip()
            for message in messages
            if message.role == "system" and message.content.strip()
        ]
        if not system_instructions:
            return None
        return "\n\n".join(system_instructions)

    def _handle_http_error(self, exc: httpx.HTTPStatusError) -> None:
        logger.error(
            "llm_http_error",
            status_code=exc.response.status_code,
            response_body=exc.response.text,
            exception=str(exc),
        )

    def _handle_request_error(self, exc: httpx.RequestError) -> None:
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


def pydantic_to_response_format(model: type) -> dict[str, Any]:
    """Convert a Pydantic model class to OpenAI response_format."""
    try:
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
