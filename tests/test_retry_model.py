from collections.abc import AsyncIterator

import asyncio
import httpx
import pytest

from ecs_agent.providers.retry_model import RetryModel
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


class ThreadRecordingProvider:
    """Records the chaining parameters received on each complete() call."""

    def __init__(self) -> None:
        self.received_thread_response_ids: list[str | None] = []
        self.received_covered_message_counts: list[int | None] = []

    @property
    def model_id(self) -> str:
        return "thread-recorder"

    async def complete(
        self,
        messages: list[Message],
        tools=None,
        stream: bool = False,
        response_format=None,
        thread_response_id: str | None = None,
        covered_message_count: int | None = None,
    ):
        self.received_thread_response_ids.append(thread_response_id)
        self.received_covered_message_counts.append(covered_message_count)
        if stream:
            return _stream_delta_iter()
        return _result("ok")


@pytest.mark.asyncio
async def test_retry_model_forwards_thread_response_id_non_streaming() -> None:
    """Wrapping a chaining-capable model must not drop the thread id."""
    inner = ThreadRecordingProvider()
    retry_model = RetryModel(inner)

    await retry_model.complete(
        [Message(role="user", content="hello")],
        thread_response_id="resp_prev_1",
    )

    assert inner.received_thread_response_ids == ["resp_prev_1"]


@pytest.mark.asyncio
async def test_retry_model_forwards_thread_response_id_streaming() -> None:
    inner = ThreadRecordingProvider()
    retry_model = RetryModel(inner)

    stream = await retry_model.complete(
        [Message(role="user", content="hello")],
        stream=True,
        thread_response_id="resp_prev_2",
    )
    _ = [delta async for delta in stream]

    assert inner.received_thread_response_ids == ["resp_prev_2"]


@pytest.mark.asyncio
async def test_retry_model_forwards_covered_message_count_non_streaming() -> None:
    """Wrapping a chaining-capable model must not drop the covered count."""
    inner = ThreadRecordingProvider()
    retry_model = RetryModel(inner)

    await retry_model.complete(
        [Message(role="user", content="hello")],
        thread_response_id="resp_prev_3",
        covered_message_count=4,
    )

    assert inner.received_thread_response_ids == ["resp_prev_3"]
    assert inner.received_covered_message_counts == [4]


@pytest.mark.asyncio
async def test_retry_model_forwards_covered_message_count_streaming() -> None:
    inner = ThreadRecordingProvider()
    retry_model = RetryModel(inner)

    stream = await retry_model.complete(
        [Message(role="user", content="hello")],
        stream=True,
        thread_response_id="resp_prev_4",
        covered_message_count=5,
    )
    _ = [delta async for delta in stream]

    assert inner.received_covered_message_counts == [5]


@pytest.mark.asyncio
async def test_successful_call_no_retry() -> None:
    model = SequencedProvider(outcomes=[_result("done")])
    retry_model = RetryModel(model)

    result = await retry_model.complete([Message(role="user", content="hello")])

    assert result.message.content == "done"
    assert model.call_count == 1


@pytest.mark.asyncio
async def test_retry_on_429_with_eventual_success() -> None:
    model = SequencedProvider(outcomes=[_http_status_error(429), _result("ok")])
    retry_model = RetryModel(
        model,
        RetryConfig(max_attempts=3, min_wait=0.0, max_wait=0.0),
    )

    result = await retry_model.complete([Message(role="user", content="rate limit")])

    assert result.message.content == "ok"
    assert model.call_count == 2


@pytest.mark.asyncio
async def test_retry_on_500_with_eventual_success() -> None:
    model = SequencedProvider(outcomes=[_http_status_error(500), _result("ok")])
    retry_model = RetryModel(
        model,
        RetryConfig(max_attempts=3, min_wait=0.0, max_wait=0.0),
    )

    result = await retry_model.complete(
        [Message(role="user", content="server error")]
    )

    assert result.message.content == "ok"
    assert model.call_count == 2


@pytest.mark.asyncio
async def test_max_retries_exhausted_raises() -> None:
    request = httpx.Request("POST", "https://example.test/v1/chat/completions")
    model = SequencedProvider(
        outcomes=[
            httpx.ConnectError("no route", request=request),
            httpx.ConnectError("no route", request=request),
            httpx.ConnectError("no route", request=request),
        ]
    )
    retry_model = RetryModel(
        model,
        RetryConfig(max_attempts=3, min_wait=0.0, max_wait=0.0),
    )

    with pytest.raises(httpx.ConnectError):
        await retry_model.complete([Message(role="user", content="hello")])

    assert model.call_count == 3


@pytest.mark.asyncio
async def test_streaming_calls_bypass_retry() -> None:
    request = httpx.Request("POST", "https://example.test/v1/chat/completions")
    stream_error = httpx.ConnectError("stream dropped", request=request)
    model = SequencedProvider(outcomes=[], stream_response=stream_error)
    retry_model = RetryModel(model, RetryConfig(max_attempts=5))

    with pytest.raises(httpx.ConnectError):
        await retry_model.complete(
            [Message(role="user", content="stream")],
            stream=True,
        )

    assert model.call_count == 1
    assert model.stream_call_count == 1


@pytest.mark.asyncio
async def test_retry_config_custom_values_applied(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    waits: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        waits.append(seconds)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    request = httpx.Request("POST", "https://example.test/v1/chat/completions")
    model = SequencedProvider(
        outcomes=[httpx.ConnectError("flaky", request=request), _result("ok")]
    )
    config = RetryConfig(max_attempts=2, multiplier=3.0, min_wait=5.0, max_wait=5.0)
    retry_model = RetryModel(model, config)

    result = await retry_model.complete([Message(role="user", content="x")])

    assert result.message.content == "ok"
    assert model.call_count == 2
    assert waits == [5.0]


@pytest.mark.asyncio
async def test_exponential_backoff_timing(monkeypatch: pytest.MonkeyPatch) -> None:
    waits: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        waits.append(seconds)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    request = httpx.Request("POST", "https://example.test/v1/chat/completions")
    model = SequencedProvider(
        outcomes=[
            httpx.ConnectError("1", request=request),
            httpx.ConnectError("2", request=request),
            httpx.ConnectError("3", request=request),
            _result("ok"),
        ]
    )
    config = RetryConfig(max_attempts=4, multiplier=1.0, min_wait=0.0, max_wait=10.0)
    retry_model = RetryModel(model, config)

    result = await retry_model.complete([Message(role="user", content="x")])

    assert result.message.content == "ok"
    assert waits == [1.0, 2.0, 4.0]


class _ClosableProvider(ThreadRecordingProvider):
    def __init__(self) -> None:
        super().__init__()
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_retry_model_aclose_delegates_to_wrapped_model() -> None:
    inner = _ClosableProvider()
    retry_model = RetryModel(inner)

    await retry_model.aclose()

    assert inner.closed is True


@pytest.mark.asyncio
async def test_retry_model_aclose_tolerates_models_without_aclose() -> None:
    inner = ThreadRecordingProvider()  # legacy: no aclose
    retry_model = RetryModel(inner)

    await retry_model.aclose()  # must not raise
