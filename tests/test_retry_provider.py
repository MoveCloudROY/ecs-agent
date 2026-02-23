from collections.abc import AsyncIterator

import asyncio
import httpx
import pytest

from ecs_agent.providers.retry_provider import RetryProvider
from ecs_agent.types import CompletionResult, Message, RetryConfig, StreamDelta


def _result(content: str = "ok") -> CompletionResult:
    return CompletionResult(message=Message(role="assistant", content=content))


def _http_status_error(status_code: int) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://example.test/v1/chat/completions")
    response = httpx.Response(status_code=status_code, request=request)
    return httpx.HTTPStatusError(
        f"HTTP {status_code}",
        request=request,
        response=response,
    )


async def _stream_delta_iter() -> AsyncIterator[StreamDelta]:
    yield StreamDelta(content="chunk")


class SequencedProvider:
    def __init__(
        self,
        outcomes: list[CompletionResult | Exception],
        stream_response: AsyncIterator[StreamDelta] | Exception | None = None,
    ) -> None:
        self._outcomes = outcomes
        self._index = 0
        self._stream_response = stream_response or _stream_delta_iter()
        self.call_count = 0
        self.stream_call_count = 0

    async def complete(
        self,
        messages: list[Message],
        tools=None,
        stream: bool = False,
        response_format=None,
    ):
        self.call_count += 1
        if stream:
            self.stream_call_count += 1
            if isinstance(self._stream_response, Exception):
                raise self._stream_response
            return self._stream_response

        if self._index >= len(self._outcomes):
            raise IndexError("No more outcomes")

        outcome = self._outcomes[self._index]
        self._index += 1

        if isinstance(outcome, Exception):
            raise outcome
        return outcome


@pytest.mark.asyncio
async def test_successful_call_no_retry() -> None:
    provider = SequencedProvider(outcomes=[_result("done")])
    retry_provider = RetryProvider(provider)

    result = await retry_provider.complete([Message(role="user", content="hello")])

    assert result.message.content == "done"
    assert provider.call_count == 1


@pytest.mark.asyncio
async def test_retry_on_429_with_eventual_success() -> None:
    provider = SequencedProvider(outcomes=[_http_status_error(429), _result("ok")])
    retry_provider = RetryProvider(
        provider,
        RetryConfig(max_attempts=3, min_wait=0.0, max_wait=0.0),
    )

    result = await retry_provider.complete([Message(role="user", content="rate limit")])

    assert result.message.content == "ok"
    assert provider.call_count == 2


@pytest.mark.asyncio
async def test_retry_on_500_with_eventual_success() -> None:
    provider = SequencedProvider(outcomes=[_http_status_error(500), _result("ok")])
    retry_provider = RetryProvider(
        provider,
        RetryConfig(max_attempts=3, min_wait=0.0, max_wait=0.0),
    )

    result = await retry_provider.complete(
        [Message(role="user", content="server error")]
    )

    assert result.message.content == "ok"
    assert provider.call_count == 2


@pytest.mark.asyncio
async def test_max_retries_exhausted_raises() -> None:
    request = httpx.Request("POST", "https://example.test/v1/chat/completions")
    provider = SequencedProvider(
        outcomes=[
            httpx.ConnectError("no route", request=request),
            httpx.ConnectError("no route", request=request),
            httpx.ConnectError("no route", request=request),
        ]
    )
    retry_provider = RetryProvider(
        provider,
        RetryConfig(max_attempts=3, min_wait=0.0, max_wait=0.0),
    )

    with pytest.raises(httpx.ConnectError):
        await retry_provider.complete([Message(role="user", content="hello")])

    assert provider.call_count == 3


@pytest.mark.asyncio
async def test_streaming_calls_bypass_retry() -> None:
    request = httpx.Request("POST", "https://example.test/v1/chat/completions")
    stream_error = httpx.ConnectError("stream dropped", request=request)
    provider = SequencedProvider(outcomes=[], stream_response=stream_error)
    retry_provider = RetryProvider(provider, RetryConfig(max_attempts=5))

    with pytest.raises(httpx.ConnectError):
        await retry_provider.complete(
            [Message(role="user", content="stream")],
            stream=True,
        )

    assert provider.call_count == 1
    assert provider.stream_call_count == 1


@pytest.mark.asyncio
async def test_retry_config_custom_values_applied(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    waits: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        waits.append(seconds)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    request = httpx.Request("POST", "https://example.test/v1/chat/completions")
    provider = SequencedProvider(
        outcomes=[httpx.ConnectError("flaky", request=request), _result("ok")]
    )
    config = RetryConfig(max_attempts=2, multiplier=3.0, min_wait=5.0, max_wait=5.0)
    retry_provider = RetryProvider(provider, config)

    result = await retry_provider.complete([Message(role="user", content="x")])

    assert result.message.content == "ok"
    assert provider.call_count == 2
    assert waits == [5.0]


@pytest.mark.asyncio
async def test_exponential_backoff_timing(monkeypatch: pytest.MonkeyPatch) -> None:
    waits: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        waits.append(seconds)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    request = httpx.Request("POST", "https://example.test/v1/chat/completions")
    provider = SequencedProvider(
        outcomes=[
            httpx.ConnectError("1", request=request),
            httpx.ConnectError("2", request=request),
            httpx.ConnectError("3", request=request),
            _result("ok"),
        ]
    )
    config = RetryConfig(max_attempts=4, multiplier=1.0, min_wait=0.0, max_wait=10.0)
    retry_provider = RetryProvider(provider, config)

    result = await retry_provider.complete([Message(role="user", content="x")])

    assert result.message.content == "ok"
    assert waits == [1.0, 2.0, 4.0]
