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

from ecs_agent.logging import get_logger
from ecs_agent.providers.protocol import LLMProvider
from ecs_agent.types import (
    CompletionResult,
    Message,
    RetryConfig,
    StreamDelta,
    ToolSchema,
)

logger = get_logger(__name__)


class RetryProvider:
    def __init__(
        self,
        provider: LLMProvider,
        retry_config: RetryConfig | None = None,
    ) -> None:
        self._provider = provider
        self._retry_config = retry_config or RetryConfig()

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        stream: bool = False,
        response_format: dict[str, Any] | None = None,
    ) -> CompletionResult | AsyncIterator[StreamDelta]:
        if stream:
            return await self._provider.complete(
                messages=messages,
                tools=tools,
                stream=True,
                response_format=response_format,
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
            before_sleep=self._log_retry_attempt,
            reraise=True,
        ):
            with attempt:
                return await self._provider.complete(
                    messages=messages,
                    tools=tools,
                    stream=False,
                    response_format=response_format,
                )

        raise RuntimeError("Retry loop exited unexpectedly")

    def _should_retry_exception(self, exc: BaseException) -> bool:
        if isinstance(exc, httpx.HTTPStatusError):
            if exc.response is None:
                return False
            return exc.response.status_code in self._retry_config.retry_status_codes
        return isinstance(exc, httpx.RequestError)

    def _log_retry_attempt(self, retry_state: RetryCallState) -> None:
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
