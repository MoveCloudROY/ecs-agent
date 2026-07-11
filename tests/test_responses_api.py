"""Tests for OpenAI Responses API support."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, Mock

import httpx
import pytest

from ecs_agent.components import (
    ConversationComponent,
    InterruptionComponent,
    LLMComponent,
    ResponsesAPIStateComponent,
)
from ecs_agent.core import Runner, World
from ecs_agent.observability import RecordingTelemetrySink, install_observability
from ecs_agent.providers.config import ApiFormat, ProviderConfig
from ecs_agent.providers.openai_model import OpenAIModel
from ecs_agent.systems.reasoning import ReasoningSystem
from ecs_agent.types import ResponsesAPICallEvent, EntityId
from ecs_agent.types import InterruptionReason, Message, ToolSchema


def _openai_config(
    *,
    api_key: str = "test-key",
    base_url: str = "https://api.openai.com/v1",
    api_format: ApiFormat = ApiFormat.OPENAI_RESPONSES,
    enable_store: bool = False,
) -> ProviderConfig:
    return ProviderConfig(
        provider_id="openai",
        base_url=base_url,
        api_key=api_key,
        api_format=api_format,
        enable_store=enable_store,
    )


def test_responses_api_state_component_defaults() -> None:
    """Test ResponsesAPIStateComponent has correct default values."""
    component = ResponsesAPIStateComponent()
    assert component.previous_response_id is None


def test_responses_api_state_component_stores_response_id() -> None:
    """Test ResponsesAPIStateComponent stores and retrieves response_id."""
    component = ResponsesAPIStateComponent()
    component.previous_response_id = "resp_abc123"
    assert component.previous_response_id == "resp_abc123"


def test_responses_api_state_component_is_dataclass_with_slots() -> None:
    """Test ResponsesAPIStateComponent is a dataclass with slots."""
    component = ResponsesAPIStateComponent()

    # Verify it's a dataclass (has __dataclass_fields__)
    assert hasattr(ResponsesAPIStateComponent, "__dataclass_fields__")

    # Verify slots are enabled (no __dict__ attribute)
    assert not hasattr(component, "__dict__")


def test_responses_api_call_event_creation() -> None:
    """Test ResponsesAPICallEvent can be created with required fields."""
    entity_id = EntityId(1)
    event = ResponsesAPICallEvent(
        entity_id=entity_id,
        response_id="resp_xyz789",
        model="gpt-4o",
    )

    assert event.entity_id == entity_id
    assert event.response_id == "resp_xyz789"
    assert event.model == "gpt-4o"


def test_responses_api_call_event_is_dataclass_with_slots() -> None:
    """Test ResponsesAPICallEvent is a dataclass with slots."""
    event = ResponsesAPICallEvent(
        entity_id=EntityId(1),
        response_id="resp_test",
        model="gpt-4o",
    )

    # Verify it's a dataclass
    assert hasattr(ResponsesAPICallEvent, "__dataclass_fields__")

    # Verify slots are enabled
    assert not hasattr(event, "__dict__")


@pytest.mark.asyncio
async def test_responses_api_complete_non_streaming_sends_correct_request() -> None:
    mock_response = Mock(spec=httpx.Response)
    mock_response.json.return_value = {
        "id": "resp_001",
        "model": "gpt-4o-mini",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "done"}],
            }
        ],
    }
    mock_response.raise_for_status = Mock()

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.post.return_value = mock_response

    model = OpenAIModel(
        config=_openai_config(
            api_key="test-key",
            base_url="https://test.openai.com/v1",
            api_format=ApiFormat.OPENAI_RESPONSES,
        ),
        model="gpt-4o-mini",
    )
    model._client = mock_client

    messages = [
        Message(role="system", content="You are a helpful assistant."),
        Message(role="user", content="Hello"),
        Message(role="assistant", content="Hi there"),
    ]

    await model.complete(messages)

    assert mock_client.post.called
    call_args = mock_client.post.call_args
    assert call_args[0][0] == "https://test.openai.com/v1/responses"

    body = call_args[1]["json"]
    assert body["model"] == "gpt-4o-mini"
    assert body["instructions"] == "You are a helpful assistant."
    assert body["store"] is False
    assert "messages" not in body
    assert body["input"] == [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "Hello"}],
        },
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Hi there"}],
        },
    ]


@pytest.mark.asyncio
async def test_responses_api_complete_non_streaming_parses_response() -> None:
    mock_response = Mock(spec=httpx.Response)
    mock_response.json.return_value = {
        "id": "resp_abc123",
        "model": "gpt-4o-mini",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Response from API"}],
            }
        ],
        "usage": {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
    }
    mock_response.raise_for_status = Mock()

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.post.return_value = mock_response

    model = OpenAIModel(
        config=_openai_config(api_key="test-key", api_format=ApiFormat.OPENAI_RESPONSES)
    )
    model._client = mock_client

    result = await model.complete([Message(role="user", content="hello")])

    assert result.message.role == "assistant"
    assert result.message.content == "Response from API"
    assert result.message.tool_calls is None
    assert result.usage is not None
    assert result.usage.prompt_tokens == 10
    assert result.usage.completion_tokens == 20
    assert result.usage.total_tokens == 30


@pytest.mark.asyncio
async def test_responses_api_complete_non_streaming_with_tools() -> None:
    mock_response = Mock(spec=httpx.Response)
    mock_response.json.return_value = {
        "id": "resp_tools_1",
        "model": "gpt-4o-mini",
        "output": [
            {
                "type": "function_call",
                "id": "call_123",
                "name": "get_weather",
                "arguments": '{"city": "SF"}',
            }
        ],
    }
    mock_response.raise_for_status = Mock()

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.post.return_value = mock_response

    model = OpenAIModel(
        config=_openai_config(api_key="test-key", api_format=ApiFormat.OPENAI_RESPONSES)
    )
    model._client = mock_client

    result = await model.complete([Message(role="user", content="weather?")])

    assert result.message.role == "assistant"
    assert result.message.content == ""
    assert result.message.tool_calls is not None
    assert len(result.message.tool_calls) == 1
    assert result.message.tool_calls[0].id == "call_123"
    assert result.message.tool_calls[0].name == "get_weather"
    assert result.message.tool_calls[0].arguments == {"city": "SF"}


@pytest.mark.asyncio
async def test_responses_api_final_answer_phase_wins_over_commentary_duplicate() -> (
    None
):
    mock_response = Mock(spec=httpx.Response)
    mock_response.json.return_value = {
        "id": "resp_phase_1",
        "model": "gpt-5-mini",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "phase": "commentary",
                "content": [{"type": "output_text", "text": "What is in scope?"}],
            },
            {
                "type": "message",
                "role": "assistant",
                "phase": "final_answer",
                "content": [{"type": "output_text", "text": "What is in scope?"}],
            },
        ],
    }
    mock_response.raise_for_status = Mock()

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.post.return_value = mock_response

    model = OpenAIModel(
        config=_openai_config(api_key="test-key", api_format=ApiFormat.OPENAI_RESPONSES)
    )
    model._client = mock_client

    result = await model.complete([Message(role="user", content="begin")])

    assert result.message.content == "What is in scope?"


@pytest.mark.asyncio
async def test_responses_api_commentary_text_kept_when_no_final_answer() -> None:
    mock_response = Mock(spec=httpx.Response)
    mock_response.json.return_value = {
        "id": "resp_phase_2",
        "model": "gpt-5-mini",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "phase": "commentary",
                "content": [
                    {"type": "output_text", "text": "Reading the draft first."}
                ],
            },
            {
                "type": "function_call",
                "call_id": "call_r1",
                "name": "read_file",
                "arguments": '{"file_path": "draft.md"}',
            },
        ],
    }
    mock_response.raise_for_status = Mock()

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.post.return_value = mock_response

    model = OpenAIModel(
        config=_openai_config(api_key="test-key", api_format=ApiFormat.OPENAI_RESPONSES)
    )
    model._client = mock_client

    result = await model.complete([Message(role="user", content="begin")])

    assert result.message.content == "Reading the draft first."
    assert result.message.tool_calls is not None
    assert result.message.tool_calls[0].id == "call_r1"


@pytest.mark.asyncio
async def test_responses_api_parse_prefers_call_id_over_item_id() -> None:
    mock_response = Mock(spec=httpx.Response)
    mock_response.json.return_value = {
        "id": "resp_tools_2",
        "model": "gpt-4o-mini",
        "output": [
            {
                "type": "function_call",
                "id": "fc_item_1",
                "call_id": "call_1",
                "name": "get_weather",
                "arguments": '{"city": "SF"}',
            }
        ],
    }
    mock_response.raise_for_status = Mock()

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.post.return_value = mock_response

    model = OpenAIModel(
        config=_openai_config(api_key="test-key", api_format=ApiFormat.OPENAI_RESPONSES)
    )
    model._client = mock_client

    result = await model.complete([Message(role="user", content="weather?")])

    assert result.message.tool_calls is not None
    assert result.message.tool_calls[0].id == "call_1"


@pytest.mark.asyncio
async def test_responses_api_replays_function_call_with_matching_call_id() -> None:
    mock_response = Mock(spec=httpx.Response)
    mock_response.json.return_value = {
        "id": "resp_replay_1",
        "model": "gpt-4o-mini",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "72F"}],
            }
        ],
    }
    mock_response.raise_for_status = Mock()

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.post.return_value = mock_response

    model = OpenAIModel(
        config=_openai_config(api_key="test-key", api_format=ApiFormat.OPENAI_RESPONSES)
    )
    model._client = mock_client

    from ecs_agent.types import ToolCall

    messages = [
        Message(role="user", content="weather?"),
        Message(
            role="assistant",
            content="",
            tool_calls=[
                ToolCall(id="call_9", name="get_weather", arguments={"city": "SF"})
            ],
        ),
        Message(role="tool", content='{"temp": 72}', tool_call_id="call_9"),
    ]

    await model.complete(messages)

    body = mock_client.post.call_args[1]["json"]
    function_calls = [i for i in body["input"] if i["type"] == "function_call"]
    outputs = [i for i in body["input"] if i["type"] == "function_call_output"]
    assert function_calls == [
        {
            "type": "function_call",
            "call_id": "call_9",
            "name": "get_weather",
            "arguments": '{"city": "SF"}',
        }
    ]
    assert outputs == [
        {
            "type": "function_call_output",
            "call_id": "call_9",
            "output": '{"temp": 72}',
        }
    ]


@pytest.mark.asyncio
async def test_responses_api_complete_non_streaming_state_component_updates_on_success() -> (
    None
):
    mock_response = Mock(spec=httpx.Response)
    mock_response.json.return_value = {
        "id": "resp_state_123",
        "model": "gpt-4o-mini",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "ok"}],
            }
        ],
    }
    mock_response.raise_for_status = Mock()

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.post.return_value = mock_response

    world = World()
    entity = world.create_entity()
    model = OpenAIModel(
        config=_openai_config(api_key="test-key", api_format=ApiFormat.OPENAI_RESPONSES)
    )
    model._client = mock_client
    world.add_component(entity, LLMComponent(model=model))
    world.add_component(
        entity,
        ConversationComponent(messages=[Message(role="user", content="hello")]),
    )
    world.add_component(
        entity,
        ResponsesAPIStateComponent(previous_response_id="resp_state_old"),
    )

    await ReasoningSystem().process(world)

    state = world.get_component(entity, ResponsesAPIStateComponent)
    assert state is not None
    assert state.previous_response_id == "resp_state_123"


@pytest.mark.asyncio
async def test_responses_api_reasoning_emits_api_observation() -> None:
    """Real Responses API reasoning publishes api.responses telemetry."""
    mock_response = Mock(spec=httpx.Response)
    mock_response.json.return_value = {
        "id": "resp_observed_123",
        "model": "gpt-4o-mini",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "observed"}],
            }
        ],
    }
    mock_response.raise_for_status = Mock()

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.post.return_value = mock_response

    world = World()
    sink = RecordingTelemetrySink()
    install_observability(world, sink)
    entity = world.create_entity()
    model = OpenAIModel(
        config=_openai_config(api_key="test-key", api_format=ApiFormat.OPENAI_RESPONSES),
        model="gpt-4o-mini",
    )
    model._client = mock_client
    world.add_component(entity, LLMComponent(model=model))
    world.add_component(
        entity,
        ConversationComponent(messages=[Message(role="user", content="hello")]),
    )
    world.register_system(ReasoningSystem(), priority=0)

    await Runner().run(world, max_ticks=1)

    api_record = next(record for record in sink.records if record.name == "api.responses")
    trace = next(record for record in sink.records if record.kind == "trace")
    generation = next(record for record in sink.records if record.kind == "generation")
    assert api_record.kind == "span"
    assert api_record.trace_id == trace.trace_id
    assert api_record.parent_observation_id == generation.observation_id
    assert api_record.entity_id == int(entity)
    assert api_record.metadata == {
        "api": "responses",
        "response_id": "resp_observed_123",
        "model_id": "gpt-4o-mini",
    }


@pytest.mark.asyncio
async def test_responses_api_complete_non_streaming_previous_response_id_from_state_component() -> (
    None
):
    mock_response = Mock(spec=httpx.Response)
    mock_response.json.return_value = {
        "id": "resp_new_1",
        "model": "gpt-4o-mini",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "threaded"}],
            }
        ],
    }
    mock_response.raise_for_status = Mock()

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.post.return_value = mock_response

    world = World()
    entity = world.create_entity()
    model = OpenAIModel(
        config=_openai_config(
            api_key="test-key",
            api_format=ApiFormat.OPENAI_RESPONSES,
            enable_store=True,
        )
    )
    model._client = mock_client
    world.add_component(entity, LLMComponent(model=model))
    world.add_component(
        entity,
        ConversationComponent(messages=[Message(role="user", content="continue")]),
    )
    world.add_component(
        entity,
        ResponsesAPIStateComponent(previous_response_id="resp_old_789"),
    )

    await ReasoningSystem().process(world)

    body = mock_client.post.call_args[1]["json"]
    assert body["previous_response_id"] == "resp_old_789"


@pytest.mark.asyncio
async def test_responses_api_complete_omits_previous_response_id_when_store_disabled() -> (
    None
):
    mock_response = Mock(spec=httpx.Response)
    mock_response.json.return_value = {
        "id": "resp_new_2",
        "model": "gpt-4o-mini",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "stateless"}],
            }
        ],
    }
    mock_response.raise_for_status = Mock()

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.post.return_value = mock_response

    world = World()
    entity = world.create_entity()
    model = OpenAIModel(
        config=_openai_config(api_key="test-key", api_format=ApiFormat.OPENAI_RESPONSES)
    )
    model._client = mock_client
    world.add_component(entity, LLMComponent(model=model))
    world.add_component(
        entity,
        ConversationComponent(messages=[Message(role="user", content="continue")]),
    )
    world.add_component(
        entity,
        ResponsesAPIStateComponent(previous_response_id="resp_old_789"),
    )

    await ReasoningSystem().process(world)

    body = mock_client.post.call_args[1]["json"]
    assert body["store"] is False
    assert "previous_response_id" not in body


_PREV_ID_REJECTED_BODY = (
    '{"error":{"message":"previous_response_id is only supported on Responses '
    'WebSocket v2","type":"invalid_request_error","param":"","code":null}}'
)


def _prev_id_rejected_error() -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://test.openai.com/v1/responses")
    response = httpx.Response(400, request=request, text=_PREV_ID_REJECTED_BODY)
    return httpx.HTTPStatusError(
        "400 Bad Request", request=request, response=response
    )


def _responses_success(response_id: str, text: str) -> Mock:
    success = Mock(spec=httpx.Response)
    success.json.return_value = {
        "id": response_id,
        "model": "gpt-4o-mini",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text}],
            }
        ],
    }
    success.raise_for_status = Mock()
    return success


@pytest.mark.asyncio
async def test_responses_api_complete_retries_without_previous_response_id_on_400() -> (
    None
):
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.post.side_effect = [
        _prev_id_rejected_error(),
        _responses_success("resp_after_retry", "recovered"),
    ]

    model = OpenAIModel(
        config=_openai_config(
            api_key="test-key",
            base_url="https://test.openai.com/v1",
            api_format=ApiFormat.OPENAI_RESPONSES,
            enable_store=True,
        )
    )
    model._client = mock_client

    result = await model.complete(
        [Message(role="user", content="hello")],
        thread_response_id="resp_old_400",
    )

    assert result.message.content == "recovered"
    assert result.response_id == "resp_after_retry"
    assert mock_client.post.call_count == 2
    first_body = mock_client.post.call_args_list[0][1]["json"]
    retry_body = mock_client.post.call_args_list[1][1]["json"]
    assert first_body["previous_response_id"] == "resp_old_400"
    assert "previous_response_id" not in retry_body
    # The retry stays on the Responses endpoint with storage untouched.
    assert retry_body["store"] is True
    assert (
        mock_client.post.call_args_list[1][0][0]
        == "https://test.openai.com/v1/responses"
    )
    assert model._responses_api_available is True


@pytest.mark.asyncio
async def test_responses_api_complete_stops_chaining_after_rejection() -> None:
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.post.side_effect = [
        _prev_id_rejected_error(),
        _responses_success("resp_1", "first"),
        _responses_success("resp_2", "second"),
    ]

    model = OpenAIModel(
        config=_openai_config(
            api_key="test-key",
            base_url="https://test.openai.com/v1",
            api_format=ApiFormat.OPENAI_RESPONSES,
            enable_store=True,
        )
    )
    model._client = mock_client

    await model.complete(
        [Message(role="user", content="hello")],
        thread_response_id="resp_old_400",
    )
    result = await model.complete(
        [Message(role="user", content="again")],
        thread_response_id="resp_1",
    )

    assert result.message.content == "second"
    # Second complete() must be a single round-trip without the chain.
    assert mock_client.post.call_count == 3
    third_body = mock_client.post.call_args_list[2][1]["json"]
    assert "previous_response_id" not in third_body
    assert third_body["store"] is True


@pytest.mark.asyncio
async def test_responses_api_streaming_retries_without_previous_response_id_on_400() -> (
    None
):
    error_cm = MagicMock()
    error_cm.__aenter__.return_value = httpx.Response(
        400,
        request=httpx.Request("POST", "https://test.openai.com/v1/responses"),
        text=_PREV_ID_REJECTED_BODY,
    )

    ok_response = AsyncMock()
    ok_response.is_error = False
    ok_response.raise_for_status = Mock()

    async def ok_aiter_lines():
        yield 'data: {"type": "response.created", "response": {"id": "resp_stream_retry"}}'
        yield 'data: {"type": "response.output_item.added", "output_index": 0, "item": {"type": "message", "role": "assistant"}}'
        yield 'data: {"type": "response.output_item.delta", "output_index": 0, "delta": {"type": "content_delta", "text": "recovered"}}'
        yield 'data: {"type": "response.done", "response": {"id": "resp_stream_retry"}}'

    ok_response.aiter_lines = ok_aiter_lines
    ok_cm = MagicMock()
    ok_cm.__aenter__.return_value = ok_response

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.stream = MagicMock(side_effect=[error_cm, ok_cm])

    model = OpenAIModel(
        config=_openai_config(
            api_key="test-key",
            base_url="https://test.openai.com/v1",
            api_format=ApiFormat.OPENAI_RESPONSES,
            enable_store=True,
        )
    )
    model._client = mock_client

    deltas = []
    stream_result = await model.complete(
        [Message(role="user", content="hi")],
        stream=True,
        thread_response_id="resp_old_400",
    )
    async for delta in stream_result:
        deltas.append(delta)

    assert any(d.content == "recovered" for d in deltas)
    assert deltas[-1].response_id == "resp_stream_retry"
    assert mock_client.stream.call_count == 2
    first_body = mock_client.stream.call_args_list[0][1]["json"]
    retry_body = mock_client.stream.call_args_list[1][1]["json"]
    assert first_body["previous_response_id"] == "resp_old_400"
    assert "previous_response_id" not in retry_body
    assert retry_body["stream"] is True


def _streaming_model(lines: list[str]) -> tuple[OpenAIModel, AsyncMock]:
    mock_response = AsyncMock()
    mock_response.is_error = False
    mock_response.raise_for_status = Mock()

    async def mock_aiter_lines():
        for line in lines:
            yield line

    mock_response.aiter_lines = mock_aiter_lines

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.stream = MagicMock()
    mock_client.stream.return_value.__aenter__.return_value = mock_response

    model = OpenAIModel(
        config=_openai_config(api_key="test-key", api_format=ApiFormat.OPENAI_RESPONSES)
    )
    model._client = mock_client
    return model, mock_client


@pytest.mark.asyncio
async def test_responses_api_streaming_parses_standard_output_text_delta_dialect() -> (
    None
):
    """Standard OpenAI Responses SSE: string deltas + response.completed."""
    model, _ = _streaming_model(
        [
            'data: {"type": "response.created", "response": {"id": "resp_std_1"}}',
            'data: {"type": "response.output_item.added", "output_index": 0, "item": {"type": "message", "role": "assistant"}}',
            'data: {"type": "response.output_text.delta", "output_index": 0, "content_index": 0, "delta": "Hel"}',
            'data: {"type": "response.output_text.delta", "output_index": 0, "content_index": 0, "delta": "lo"}',
            'data: {"type": "response.output_text.done", "output_index": 0, "text": "Hello"}',
            'data: {"type": "response.output_item.done", "output_index": 0, "item": {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Hello"}]}}',
            'data: {"type": "response.completed", "response": {"id": "resp_std_1", "usage": {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5}}}',
        ]
    )

    deltas = []
    stream_result = await model.complete(
        [Message(role="user", content="Hi")], stream=True
    )
    async for delta in stream_result:
        deltas.append(delta)

    contents = [d.content for d in deltas if d.content]
    assert contents == ["Hel", "lo"]
    final = deltas[-1]
    assert final.finish_reason == "stop"
    assert final.response_id == "resp_std_1"
    assert final.usage is not None
    assert final.usage.total_tokens == 5


@pytest.mark.asyncio
async def test_responses_api_streaming_standard_function_call_arguments_delta() -> (
    None
):
    """Arguments accumulate from response.function_call_arguments.delta."""
    model, _ = _streaming_model(
        [
            'data: {"type": "response.output_item.added", "output_index": 0, "item": {"type": "function_call", "id": "fc_item_1", "call_id": "call_std_1", "name": "get_weather", "arguments": ""}}',
            'data: {"type": "response.function_call_arguments.delta", "output_index": 0, "delta": "{\\"city\\":"}',
            'data: {"type": "response.function_call_arguments.delta", "output_index": 0, "delta": " \\"Paris\\"}"}',
            'data: {"type": "response.output_item.done", "output_index": 0, "item": {"type": "function_call", "id": "fc_item_1", "call_id": "call_std_1", "name": "get_weather"}}',
            'data: {"type": "response.completed", "response": {"id": "resp_std_2"}}',
        ]
    )

    deltas = []
    stream_result = await model.complete(
        [Message(role="user", content="weather?")], stream=True
    )
    async for delta in stream_result:
        deltas.append(delta)

    tool_deltas = [d for d in deltas if d.tool_calls]
    assert len(tool_deltas) == 1
    tool_call = tool_deltas[0].tool_calls[0]
    assert tool_call.id == "call_std_1"
    assert tool_call.name == "get_weather"
    assert tool_call.arguments == {"city": "Paris"}


@pytest.mark.asyncio
async def test_responses_api_streaming_done_item_arguments_are_authoritative() -> (
    None
):
    """A done item carrying full arguments wins even without delta events."""
    model, _ = _streaming_model(
        [
            'data: {"type": "response.output_item.added", "output_index": 0, "item": {"type": "function_call", "id": "fc_item_2", "call_id": "call_std_2", "name": "get_weather", "arguments": ""}}',
            'data: {"type": "response.output_item.done", "output_index": 0, "item": {"type": "function_call", "id": "fc_item_2", "call_id": "call_std_2", "name": "get_weather", "arguments": "{\\"city\\": \\"Oslo\\"}"}}',
            'data: {"type": "response.completed", "response": {"id": "resp_std_3"}}',
        ]
    )

    deltas = []
    stream_result = await model.complete(
        [Message(role="user", content="weather?")], stream=True
    )
    async for delta in stream_result:
        deltas.append(delta)

    tool_deltas = [d for d in deltas if d.tool_calls]
    assert len(tool_deltas) == 1
    assert tool_deltas[0].tool_calls[0].arguments == {"city": "Oslo"}


@pytest.mark.asyncio
async def test_responses_api_streaming_failed_response_raises() -> None:
    model, _ = _streaming_model(
        [
            'data: {"type": "response.created", "response": {"id": "resp_fail_1"}}',
            'data: {"type": "response.failed", "response": {"id": "resp_fail_1", "error": {"code": "server_error", "message": "boom"}}}',
        ]
    )

    stream_result = await model.complete(
        [Message(role="user", content="Hi")], stream=True
    )
    with pytest.raises(ValueError, match=r"\[server_error\] boom"):
        async for _delta in stream_result:
            pass


@pytest.mark.asyncio
async def test_responses_api_complete_unrelated_400_raises_without_retry() -> None:
    request = httpx.Request("POST", "https://test.openai.com/v1/responses")
    response = httpx.Response(
        400,
        request=request,
        text='{"error":{"message":"invalid input"}}',
    )
    error = httpx.HTTPStatusError("400", request=request, response=response)

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.post.side_effect = error

    model = OpenAIModel(
        config=_openai_config(
            api_key="test-key",
            base_url="https://test.openai.com/v1",
            api_format=ApiFormat.OPENAI_RESPONSES,
            enable_store=True,
        )
    )
    model._client = mock_client

    with pytest.raises(httpx.HTTPStatusError):
        await model.complete(
            [Message(role="user", content="hello")],
            thread_response_id="resp_x",
        )

    assert mock_client.post.call_count == 1


@pytest.mark.asyncio
async def test_responses_api_complete_400_when_chain_not_sent_raises_without_retry() -> (
    None
):
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.post.side_effect = _prev_id_rejected_error()

    model = OpenAIModel(
        config=_openai_config(
            api_key="test-key",
            base_url="https://test.openai.com/v1",
            api_format=ApiFormat.OPENAI_RESPONSES,
            enable_store=True,
        )
    )
    model._client = mock_client

    with pytest.raises(httpx.HTTPStatusError):
        await model.complete([Message(role="user", content="hello")])

    assert mock_client.post.call_count == 1


@pytest.mark.asyncio
async def test_responses_api_complete_non_streaming_falls_back_when_disabled() -> None:
    mock_response = Mock(spec=httpx.Response)
    mock_response.json.return_value = {
        "choices": [{"message": {"role": "assistant", "content": "fallback"}}]
    }
    mock_response.raise_for_status = Mock()

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.post.return_value = mock_response

    model = OpenAIModel(
        config=_openai_config(
            api_key="test-key",
            base_url="https://test.openai.com/v1",
            api_format=ApiFormat.OPENAI_CHAT_COMPLETIONS,
        )
    )
    model._client = mock_client

    tools = [
        ToolSchema(
            name="get_weather",
            description="Get weather",
            parameters={"type": "object", "properties": {"city": {"type": "string"}}},
        )
    ]
    await model.complete([Message(role="user", content="hello")], tools=tools)

    call_args = mock_client.post.call_args
    assert call_args[0][0] == "https://test.openai.com/v1/chat/completions"


@pytest.mark.asyncio
async def test_responses_api_complete_non_streaming_falls_back_on_404() -> None:
    responses_request = httpx.Request("POST", "https://test.openai.com/v1/responses")
    responses_response = httpx.Response(
        404,
        request=responses_request,
        text="Not Found",
    )
    responses_error = httpx.HTTPStatusError(
        "404 Not Found",
        request=responses_request,
        response=responses_response,
    )

    fallback_response = Mock(spec=httpx.Response)
    fallback_response.json.return_value = {
        "choices": [{"message": {"role": "assistant", "content": "fallback"}}]
    }
    fallback_response.raise_for_status = Mock()

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.post.side_effect = [responses_error, fallback_response]

    model = OpenAIModel(
        config=_openai_config(
            api_key="test-key",
            base_url="https://test.openai.com/v1",
            api_format=ApiFormat.OPENAI_RESPONSES,
        )
    )
    model._client = mock_client

    result = await model.complete([Message(role="user", content="hello")])

    assert result.message.role == "assistant"
    assert result.message.content == "fallback"
    assert model._responses_api_available is False
    assert mock_client.post.call_count == 2
    assert (
        mock_client.post.call_args_list[0][0][0]
        == "https://test.openai.com/v1/responses"
    )
    assert (
        mock_client.post.call_args_list[1][0][0]
        == "https://test.openai.com/v1/chat/completions"
    )


@pytest.mark.asyncio
async def test_responses_api_streaming_yields_stream_deltas() -> None:
    """Test Responses API streaming yields StreamDelta objects."""
    mock_response = AsyncMock()
    mock_response.raise_for_status = Mock()

    async def mock_aiter_lines():
        yield "event: response.created"
        yield 'data: {"type": "response.created", "response": {"id": "resp_stream_1"}}'
        yield ""
        yield "event: response.output_item.added"
        yield 'data: {"type": "response.output_item.added", "output_index": 0, "item": {"type": "message", "role": "assistant"}}'
        yield ""
        yield "event: response.output_item.delta"
        yield 'data: {"type": "response.output_item.delta", "output_index": 0, "delta": {"type": "content_delta", "text": "Hello"}}'
        yield ""
        yield "event: response.output_item.delta"
        yield 'data: {"type": "response.output_item.delta", "output_index": 0, "delta": {"type": "content_delta", "text": " world"}}'
        yield ""
        yield "event: response.output_item.done"
        yield 'data: {"type": "response.output_item.done", "output_index": 0}'
        yield ""
        yield "event: response.done"
        yield 'data: {"type": "response.done", "response": {"id": "resp_stream_1", "usage": {"input_tokens": 5, "output_tokens": 10, "total_tokens": 15}}}'
        yield ""

    mock_response.aiter_lines = mock_aiter_lines

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.stream = MagicMock()
    mock_client.stream.return_value.__aenter__.return_value = mock_response

    model = OpenAIModel(
        config=_openai_config(api_key="test-key", api_format=ApiFormat.OPENAI_RESPONSES)
    )
    model._client = mock_client

    deltas = []
    stream_result = await model.complete(
        [Message(role="user", content="Hi")], stream=True
    )
    async for delta in stream_result:
        deltas.append(delta)

    assert len(deltas) >= 2
    assert any(d.content == "Hello" for d in deltas)
    assert any(d.content == " world" for d in deltas)


@pytest.mark.asyncio
async def test_responses_api_streaming_handles_tool_calls() -> None:
    """Test Responses API streaming accumulates tool calls correctly."""
    mock_response = AsyncMock()
    mock_response.raise_for_status = Mock()

    async def mock_aiter_lines():
        yield "event: response.output_item.added"
        yield 'data: {"type": "response.output_item.added", "output_index": 0, "item": {"type": "function_call", "id": "call_abc", "name": "get_weather"}}'
        yield ""
        yield "event: response.output_item.delta"
        yield 'data: {"type": "response.output_item.delta", "output_index": 0, "delta": {"type": "arguments_delta", "arguments": "{\\"city\\":"}}'
        yield ""
        yield "event: response.output_item.delta"
        yield 'data: {"type": "response.output_item.delta", "output_index": 0, "delta": {"type": "arguments_delta", "arguments": " \\"SF\\"}"}}'
        yield ""
        yield "event: response.output_item.done"
        yield 'data: {"type": "response.output_item.done", "output_index": 0}'
        yield ""
        yield "event: response.done"
        yield 'data: {"type": "response.done", "response": {"id": "resp_tool_1"}}'
        yield ""

    mock_response.aiter_lines = mock_aiter_lines

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.stream = MagicMock()
    mock_client.stream.return_value.__aenter__.return_value = mock_response

    model = OpenAIModel(
        config=_openai_config(api_key="test-key", api_format=ApiFormat.OPENAI_RESPONSES)
    )
    model._client = mock_client

    deltas = []
    stream_result = await model.complete(
        [Message(role="user", content="weather?")], stream=True
    )
    async for delta in stream_result:
        deltas.append(delta)

    # Find the delta with tool calls
    tool_delta = next((d for d in deltas if d.tool_calls), None)
    assert tool_delta is not None
    assert len(tool_delta.tool_calls) == 1
    assert tool_delta.tool_calls[0].name == "get_weather"
    assert tool_delta.tool_calls[0].arguments == {"city": "SF"}


@pytest.mark.asyncio
async def test_responses_api_streaming_emits_done() -> None:
    """Test Responses API streaming emits final delta with finish_reason and usage."""
    mock_response = AsyncMock()
    mock_response.raise_for_status = Mock()

    async def mock_aiter_lines():
        yield "event: response.output_item.added"
        yield 'data: {"type": "response.output_item.added", "output_index": 0, "item": {"type": "message"}}'
        yield ""
        yield "event: response.output_item.delta"
        yield 'data: {"type": "response.output_item.delta", "output_index": 0, "delta": {"type": "content_delta", "text": "ok"}}'
        yield ""
        yield "event: response.output_item.done"
        yield 'data: {"type": "response.output_item.done", "output_index": 0}'
        yield ""
        yield "event: response.done"
        yield 'data: {"type": "response.done", "response": {"id": "resp_done", "usage": {"input_tokens": 3, "output_tokens": 5, "total_tokens": 8}}}'
        yield ""

    mock_response.aiter_lines = mock_aiter_lines

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.stream = MagicMock()
    mock_client.stream.return_value.__aenter__.return_value = mock_response

    model = OpenAIModel(
        config=_openai_config(api_key="test-key", api_format=ApiFormat.OPENAI_RESPONSES)
    )
    model._client = mock_client

    deltas = []
    stream_result = await model.complete(
        [Message(role="user", content="test")], stream=True
    )
    async for delta in stream_result:
        deltas.append(delta)

    # Last delta should have finish_reason and usage
    last_delta = deltas[-1]
    assert last_delta.finish_reason == "stop"
    assert last_delta.usage is not None
    assert last_delta.usage.prompt_tokens == 3
    assert last_delta.usage.completion_tokens == 5
    assert last_delta.usage.total_tokens == 8


@pytest.mark.asyncio
async def test_responses_api_streaming_previous_response_id_updates_state_component() -> (
    None
):
    mock_response = AsyncMock()
    mock_response.raise_for_status = Mock()

    async def mock_aiter_lines():
        yield "event: response.created"
        yield 'data: {"type": "response.created", "response": {"id": "resp_streaming_123"}}'
        yield ""
        yield "event: response.output_item.added"
        yield 'data: {"type": "response.output_item.added", "output_index": 0, "item": {"type": "message"}}'
        yield ""
        yield "event: response.output_item.delta"
        yield 'data: {"type": "response.output_item.delta", "output_index": 0, "delta": {"type": "content_delta", "text": "done"}}'
        yield ""
        yield "event: response.done"
        yield 'data: {"type": "response.done", "response": {"id": "resp_streaming_123"}}'
        yield ""

    mock_response.aiter_lines = mock_aiter_lines

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.stream = MagicMock()
    mock_client.stream.return_value.__aenter__.return_value = mock_response

    world = World()
    entity = world.create_entity()
    model = OpenAIModel(
        config=_openai_config(api_key="test-key", api_format=ApiFormat.OPENAI_RESPONSES)
    )
    model._client = mock_client
    world.add_component(entity, LLMComponent(model=model))
    world.add_component(
        entity,
        ConversationComponent(messages=[Message(role="user", content="hi")]),
    )
    world.add_component(
        entity,
        ResponsesAPIStateComponent(previous_response_id="resp_before_stream"),
    )

    from ecs_agent.components import StreamingComponent

    world.add_component(entity, StreamingComponent(enabled=True))

    await ReasoningSystem().process(world)

    state = world.get_component(entity, ResponsesAPIStateComponent)
    assert state is not None
    assert state.previous_response_id == "resp_streaming_123"


@pytest.mark.asyncio
async def test_responses_api_non_streaming_previous_response_id_preserved_on_failure() -> (
    None
):
    request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    response = httpx.Response(500, request=request, text="server error")
    error = httpx.HTTPStatusError("boom", request=request, response=response)

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.post.side_effect = error

    world = World()
    entity = world.create_entity()
    model = OpenAIModel(
        config=_openai_config(api_key="test-key", api_format=ApiFormat.OPENAI_RESPONSES)
    )
    model._client = mock_client
    world.add_component(entity, LLMComponent(model=model))
    world.add_component(
        entity,
        ConversationComponent(messages=[Message(role="user", content="continue")]),
    )
    world.add_component(
        entity,
        ResponsesAPIStateComponent(previous_response_id="resp_stable_old"),
    )

    await ReasoningSystem().process(world)

    state = world.get_component(entity, ResponsesAPIStateComponent)
    assert state is not None
    assert state.previous_response_id == "resp_stable_old"


@pytest.mark.asyncio
async def test_responses_api_streaming_previous_response_id_preserved_on_interruption() -> (
    None
):
    world = World()
    entity = world.create_entity()

    mock_response = AsyncMock()
    mock_response.raise_for_status = Mock()

    async def mock_aiter_lines():
        yield "event: response.created"
        yield 'data: {"type": "response.created", "response": {"id": "resp_streaming_new"}}'
        yield ""
        yield "event: response.output_item.delta"
        yield 'data: {"type": "response.output_item.delta", "output_index": 0, "delta": {"type": "content_delta", "text": "partial"}}'
        yield ""
        world.add_component(
            entity,
            InterruptionComponent(
                reason=InterruptionReason.USER_REQUESTED,
                message="stop",
                metadata={},
            ),
        )
        yield "event: response.done"
        yield 'data: {"type": "response.done", "response": {"id": "resp_streaming_new"}}'
        yield ""

    mock_response.aiter_lines = mock_aiter_lines

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.stream = MagicMock()
    mock_client.stream.return_value.__aenter__.return_value = mock_response

    model = OpenAIModel(
        config=_openai_config(api_key="test-key", api_format=ApiFormat.OPENAI_RESPONSES)
    )
    model._client = mock_client
    world.add_component(entity, LLMComponent(model=model))
    world.add_component(
        entity,
        ConversationComponent(messages=[Message(role="user", content="interrupt")]),
    )
    world.add_component(
        entity, ResponsesAPIStateComponent(previous_response_id="resp_keep_old")
    )

    from ecs_agent.components import StreamingComponent

    world.add_component(entity, StreamingComponent(enabled=True))

    with pytest.raises(asyncio.CancelledError):
        await ReasoningSystem().process(world)

    state = world.get_component(entity, ResponsesAPIStateComponent)
    assert state is not None
    assert state.previous_response_id == "resp_keep_old"
