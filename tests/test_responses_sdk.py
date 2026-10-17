"""Responses SDK integration: transport, configuration, retries and cleanup."""

import asyncio
import subprocess
import sys
from collections.abc import AsyncIterator

import httpx
import pytest

from ecs_agent.providers.config import ApiFormat, ProviderConfig
from ecs_agent.providers.openai_model import OpenAIModel
from ecs_agent.providers.retry_model import RetryModel
from ecs_agent.types import CompletionResult, Message, RetryConfig, StreamDelta
from tests.http_mock import MockHTTP, SSEStream, sse_response


def _model(transport: MockHTTP) -> OpenAIModel:
    """Use the real SDK with an in-memory HTTP transport."""
    model = OpenAIModel(
        config=ProviderConfig(
            provider_id="test",
            api_key="test-key",
            base_url="https://gateway.example/custom/v1/",
            api_format=ApiFormat.OPENAI_RESPONSES,
        ),
    )
    model._client = transport.client()
    return model


def _success() -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "resp_test",
            "status": "completed",
            "output": [{
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "ok"}],
            }],
        },
    )


@pytest.mark.parametrize("streaming", [False, True])
async def test_sdk_preserves_configuration_and_closes_shared_pool(streaming: bool) -> None:
    transport = MockHTTP()
    if streaming:
        async def lines() -> AsyncIterator[str]:
            yield 'data: {"type":"response.completed","response":{"id":"resp_test"}}'
        transport.responses = [sse_response(lines())]
    else:
        transport.responses = [_success()]
    model = OpenAIModel(
        config=ProviderConfig(
            provider_id="test",
            api_key="test-key",
            base_url="https://gateway.example/custom/v1/",
            api_format=ApiFormat.OPENAI_RESPONSES,
            extra_headers={"X-Custom": "test-value", "Authorization": "Bearer custom"},
            timeout=19.0,
        ),
        stream_read_timeout=7.0,
    )
    model._client = transport.client()
    try:
        result = await model.complete([Message(role="user", content="hi")], stream=streaming)
        if not isinstance(result, CompletionResult):
            _ = [delta async for delta in result]
        request = transport.requests[0]
        assert str(request.url) == "https://gateway.example/custom/v1/responses"
        assert request.headers["authorization"] == "Bearer custom"
        assert request.headers["x-custom"] == "test-value"
        assert request.headers["user-agent"].startswith("AsyncOpenAI/Python")
        assert request.extensions["timeout"] == {
            "connect": 19.0, "read": 7.0 if streaming else 19.0,
            "write": 19.0, "pool": 19.0,
        }
        sdk = model._responses_adapter._get_client()
        assert sdk is model._responses_adapter._get_client()
        assert sdk.max_retries == 0
    finally:
        await model.aclose()
    assert model._client.is_closed
    assert sdk.is_closed()


@pytest.mark.parametrize("status", [401, 429, 500])
@pytest.mark.parametrize("streaming", [False, True])
async def test_sdk_status_errors_keep_httpx_contract_without_hidden_retries(
    status: int, streaming: bool,
) -> None:
    transport = MockHTTP()
    transport.responses = [httpx.Response(status, json={"error": {"message": "failure"}})]
    model = _model(transport)
    try:
        with pytest.raises(httpx.HTTPStatusError) as caught:
            result = await model.complete([Message(role="user", content="hi")], stream=streaming)
            if not isinstance(result, CompletionResult):
                _ = [delta async for delta in result]
        assert caught.value.response.status_code == status
        assert caught.value.response.json()["error"]["message"] == "failure"
        assert len(transport.requests) == 1
        assert model._responses_api_available is None
    finally:
        await model.aclose()


@pytest.mark.parametrize("error_type", [httpx.ConnectError, httpx.ReadTimeout])
async def test_sdk_preserves_transport_exception_subtypes(error_type: type[httpx.RequestError]) -> None:
    transport = MockHTTP()
    original = error_type("transport failed")
    transport.responses = [original]
    model = _model(transport)
    try:
        with pytest.raises(error_type) as caught:
            await model.complete([Message(role="user", content="hi")])
        assert caught.value is original
        assert len(transport.requests) == 1
    finally:
        await model.aclose()


async def test_sdk_retry_model_controls_attempt_count() -> None:
    transport = MockHTTP()
    transport.responses = [
        httpx.Response(429, json={"error": {"message": "rate limited"}}),
        _success(),
    ]
    model = _model(transport)
    retry_model = RetryModel(model, RetryConfig(max_attempts=2, min_wait=0, max_wait=0))
    try:
        result = await retry_model.complete([Message(role="user", content="hi")])
        assert isinstance(result, CompletionResult)
        assert result.message.content == "ok"
        assert len(transport.requests) == 2
    finally:
        await retry_model.aclose()


async def test_sdk_accepts_empty_key_with_gateway_authentication() -> None:
    transport = MockHTTP()
    transport.responses = [_success()]
    model = _model(transport)
    model._provider_config.api_key = ""
    model._api_key = ""
    model._provider_config.extra_headers = {"Authorization": "Basic gateway"}
    try:
        await model.complete([Message(role="user", content="hi")])
        assert transport.requests[0].headers["authorization"] == "Basic gateway"
    finally:
        await model.aclose()


async def test_sdk_decodes_multiline_sse_and_ignores_comments() -> None:
    transport = MockHTTP()
    transport.responses = [httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        content=(
            ': heartbeat\r\n\r\n'
            'event: response.output_text.delta\r\n'
            'data: {"type":"response.output_text.delta",\r\n'
            'data: "output_index":0,"delta":"你好"}\r\n\r\n'
            'data: {"type":"response.completed","response":{"id":"resp_sse"}}\r\n\r\n'
        ).encode(),
    )]
    model = _model(transport)
    try:
        stream = await model.complete([Message(role="user", content="hi")], stream=True)
        assert not isinstance(stream, CompletionResult)
        deltas = [delta async for delta in stream]
        assert deltas[0].content == "你好"
        assert deltas[-1].response_id == "resp_sse"
        assert transport.requests[0].extensions["timeout"]["read"] is None
        assert transport.returned[0].is_closed
    finally:
        await model.aclose()


async def test_sdk_cancelled_stream_releases_connection() -> None:
    waiting = asyncio.Event()

    async def lines() -> AsyncIterator[str]:
        yield 'data: {"type":"response.output_text.delta","delta":"partial"}'
        waiting.set()
        await asyncio.Event().wait()

    transport = MockHTTP()
    response = sse_response(lines())
    byte_stream = response.stream
    assert isinstance(byte_stream, SSEStream)
    transport.responses = [response]
    model = _model(transport)

    async def consume() -> list[StreamDelta]:
        stream = await model.complete([Message(role="user", content="hi")], stream=True)
        assert not isinstance(stream, CompletionResult)
        return [delta async for delta in stream]

    task = asyncio.create_task(consume())
    try:
        await asyncio.wait_for(waiting.wait(), timeout=5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert response.is_closed
        assert byte_stream.closed
        assert len(transport.requests) == 1
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await model.aclose()


async def test_sdk_stream_failure_does_not_replay_partial_output() -> None:
    async def lines() -> AsyncIterator[str]:
        yield 'data: {"type":"response.output_text.delta","delta":"partial"}'
        raise httpx.ReadError("connection lost")

    transport = MockHTTP()
    transport.responses = [sse_response(lines())]
    model = _model(transport)
    deltas: list[StreamDelta] = []
    try:
        stream = await model.complete([Message(role="user", content="hi")], stream=True)
        assert not isinstance(stream, CompletionResult)
        with pytest.raises(httpx.ReadError):
            async for delta in stream:
                deltas.append(delta)
        assert [delta.content for delta in deltas] == ["partial"]
        assert len(transport.requests) == 1
        assert transport.returned[0].is_closed
    finally:
        await model.aclose()


def test_sdk_import_is_lazy() -> None:
    result = subprocess.run(
        [sys.executable, "-c", (
            "import sys; import ecs_agent; "
            "from ecs_agent.providers.openai_model import OpenAIModel; "
            "assert 'openai' not in sys.modules"
        )],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("malformed_json", [False, True])
async def test_sdk_stream_errors_propagate_and_close_response(malformed_json: bool) -> None:
    import json

    from openai import APIError

    payload = (
        "data: not-json\n\n"
        if malformed_json
        else 'data: {"error":{"message":"stream failed"}}\n\n'
    )
    transport = MockHTTP()
    transport.responses = [httpx.Response(
        200, headers={"content-type": "text/event-stream"}, content=payload,
    )]
    model = _model(transport)
    try:
        stream = await model.complete([Message(role="user", content="hi")], stream=True)
        assert not isinstance(stream, CompletionResult)
        with pytest.raises(json.JSONDecodeError if malformed_json else APIError):
            _ = [delta async for delta in stream]
        assert transport.returned[0].is_closed
        assert len(transport.requests) == 1
    finally:
        await model.aclose()


@pytest.mark.parametrize("streaming", [False, True])
async def test_sdk_failed_response_is_not_successful_completion(streaming: bool) -> None:
    transport = MockHTTP()
    response_data = {
        "id": "resp_failed", "status": "failed", "output": [],
        "error": {"code": "server_error", "message": "generation failed"},
    }
    if streaming:
        import json

        transport.responses = [httpx.Response(
            200, headers={"content-type": "text/event-stream"},
            content="data: " + json.dumps({"type": "response.failed", "response": response_data}) + "\n\n",
        )]
    else:
        transport.responses = [httpx.Response(200, json=response_data)]
    model = _model(transport)
    try:
        result = await model.complete([Message(role="user", content="hi")], stream=streaming)
        if isinstance(result, CompletionResult):
            terminal = result
        else:
            terminal = [delta async for delta in result][-1]
        assert terminal.finish_reason == "error"
        assert terminal.error is not None and "generation failed" in terminal.error
        assert transport.returned[0].is_closed
        assert len(transport.requests) == 1
    finally:
        await model.aclose()
