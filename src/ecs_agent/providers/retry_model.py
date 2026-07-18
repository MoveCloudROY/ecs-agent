"""Retry wrapper provider."""

from collections.abc import AsyncIterator
from typing import Any

import httpx
from tenacity import (
    AsyncRetrying,
    RetryCallState,
    retry_if_exception,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ecs_agent.accounting.models import LLMRetryEvent
from ecs_agent.core.event_bus import EventBus
from ecs_agent.logging import get_logger
from ecs_agent.providers.protocol import LLMModel
from ecs_agent.types import (
    CompletionResult,
    Message,
    RetryConfig,
    StreamDelta,
    ToolSchema,
)

logger = get_logger(__name__)


class RetryModel:
    def __init__(
        self,
        model: LLMModel,
        retry_config: RetryConfig | None = None,
    ) -> None:
        self._model = model
        self._retry_config = retry_config or RetryConfig()
        self._event_bus: EventBus | None = None

    @property
    def model_id(self) -> str:
        return self._model.model_id

    @property
    def provider_id(self) -> str:
        provider_id = getattr(self._model, "provider_id", None)
        if isinstance(provider_id, str) and provider_id:
            return provider_id
        return type(self._model).__name__

    def set_event_bus(self, event_bus: EventBus) -> None:
        """Attach the current world's EventBus for retry observation events."""
        self._event_bus = event_bus

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        stream: bool = False,
        response_format: dict[str, Any] | None = None,
        thread_response_id: str | None = None,
    ) -> CompletionResult | AsyncIterator[StreamDelta]:
        # Forward the thread id only when set, mirroring the caller-side
        # contract: wrapped models predating the parameter keep working in
        # non-chaining sessions.
        extra_kwargs: dict[str, Any] = {}
        if thread_response_id is not None:
            extra_kwargs["thread_response_id"] = thread_response_id

        if stream:
            return await self._model.complete(
                messages=messages,
                tools=tools,
                stream=True,
                response_format=response_format,
                **extra_kwargs,
            )

        retry_condition = retry_if_exception_type(
            (httpx.HTTPStatusError, httpx.RequestError)
        ) & retry_if_exception(self._should_retry_exception)

        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self._retry_config.max_attempts),
            wait=wait_exponential(
                multiplier=self._retry_config.multiplier,
                min=self._retry_config.min_wait,
                max=self._retry_config.max_wait,
            ),
            retry=retry_condition,
            before_sleep=self._record_retry_attempt,
            reraise=True,
        ):
            with attempt:
                return await self._model.complete(
                    messages=messages,
                    tools=tools,
                    stream=False,
                    response_format=response_format,
                    **extra_kwargs,
                )

        raise RuntimeError("Retry loop exited unexpectedly")

    def _should_retry_exception(self, exc: BaseException) -> bool:
        if isinstance(exc, httpx.HTTPStatusError):
            if exc.response is None:
                return False
            return exc.response.status_code in self._retry_config.retry_status_codes
        return isinstance(exc, httpx.RequestError)

    async def _record_retry_attempt(self, retry_state: RetryCallState) -> None:
        if retry_state.outcome is None or not retry_state.outcome.failed:
            return
        error = retry_state.outcome.exception()
        if error is None:
            return
        logger.warning(
            "retrying_llm_call",
            attempt=retry_state.attempt_number,
            error=str(error),
            wait_seconds=retry_state.upcoming_sleep,
        )
        if self._event_bus is None:
            return
        await self._event_bus.publish(
            LLMRetryEvent(
                provider_id=self.provider_id,
                model=self.model_id,
                reason=self._retry_reason(error),
                attempt=retry_state.attempt_number,
            )
        )

    def _retry_reason(self, exc: BaseException) -> str:
        if isinstance(exc, httpx.HTTPStatusError):
            if exc.response is not None:
                return f"http_{exc.response.status_code}"
            return "unknown"
        if isinstance(exc, httpx.TimeoutException):
            return "timeout"
        if isinstance(exc, httpx.RequestError):
            return "request_error"
        return "unknown"
