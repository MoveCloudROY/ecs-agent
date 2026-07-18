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
from ecs_agent.types import CompletionResult, InterruptionReason, Message, ToolSchema


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


class _ThreadRecordingModel:
    """Chaining-capable fake that records received thread_response_id values."""

    def __init__(self, response_id: str = "resp_new_1") -> None:
        self._response_id = response_id
        self.received_thread_response_ids: list[str | None] = []

    @property
    def model_id(self) -> str:
        return "thread-recorder"

    async def complete(
        self,
        messages,
        tools=None,
        stream=False,
        response_format=None,
        thread_response_id=None,
    ):
        self.received_thread_response_ids.append(thread_response_id)
        return CompletionResult(
            message=Message(role="assistant", content="ok"),
            response_id=self._response_id,
        )


@pytest.mark.asyncio
async def test_reasoning_passes_thread_response_id_through_wrapper_models() -> None:
    """Response chaining must survive wrapper models such as RetryModel.

    Regression: reasoning used to sniff ``isinstance(model, OpenAIModel)``, so
    any wrapper silently disabled previous_response_id chaining and the
    ResponsesAPICallEvent.
    """
    from ecs_agent.providers.retry_model import RetryModel

    inner = _ThreadRecordingModel(response_id="resp_new_1")
    model = RetryModel(inner)

    world = World()
    entity = world.create_entity()
    world.add_component(entity, LLMComponent(model=model))
    world.add_component(
        entity,
        ConversationComponent(messages=[Message(role="user", content="hello")]),
    )
    world.add_component(
        entity,
        ResponsesAPIStateComponent(previous_response_id="resp_state_old"),
    )

    api_events: list[ResponsesAPICallEvent] = []

    async def on_api_call(event: ResponsesAPICallEvent) -> None:
        api_events.append(event)

    world.event_bus.subscribe(ResponsesAPICallEvent, on_api_call)

    await ReasoningSystem().process(world)

    assert inner.received_thread_response_ids == ["resp_state_old"]
    state = world.get_component(entity, ResponsesAPIStateComponent)
    assert state is not None
    assert state.previous_response_id == "resp_new_1"
    assert [event.response_id for event in api_events] == ["resp_new_1"]


@pytest.mark.asyncio
async def test_reasoning_omits_thread_response_id_without_state_component() -> None:
    """Models keep receiving plain complete() calls when no thread is tracked.

    Custom LLMModel implementations predating the thread_response_id parameter
    must not break in the common non-chaining case.
    """

    class _LegacySignatureModel:
        def __init__(self) -> None:
            self.calls = 0

        @property
        def model_id(self) -> str:
            return "legacy"

        async def complete(self, messages, tools=None, stream=False, response_format=None):
            self.calls += 1
            return CompletionResult(message=Message(role="assistant", content="ok"))

    model = _LegacySignatureModel()
    world = World()
    entity = world.create_entity()
    world.add_component(entity, LLMComponent(model=model))
    world.add_component(
        entity,
        ConversationComponent(messages=[Message(role="user", content="hello")]),
    )

    await ReasoningSystem().process(world)

    assert model.calls == 1
    conversation = world.get_component(entity, ConversationComponent)
    assert conversation is not None
    assert conversation.messages[-1].content == "ok"


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


@pytest.mark.asyncio
async def test_responses_api_streaming_propagates_read_timeout_stall() -> None:
    """A stalled gateway surfaces as an error instead of hanging the turn.

    With stream_read_timeout set, httpx raises ReadTimeout once the connection
    goes silent for the whole window. The adapter must propagate it (it is a
    RequestError, not a 400 to de-chain) so ReasoningSystem records the failure
    and the interactive frontend can return control to the user — rather than
    the read=None behaviour where the turn hangs forever.
    """
    stalled_response = AsyncMock()
    stalled_response.is_error = False
    stalled_response.raise_for_status = Mock()

    async def stalling_aiter_lines():
        raise httpx.ReadTimeout("stream went silent")
        yield ""  # pragma: no cover — only marks this an async generator

    stalled_response.aiter_lines = stalling_aiter_lines
    cm = MagicMock()
    cm.__aenter__.return_value = stalled_response

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.stream = MagicMock(return_value=cm)

    model = OpenAIModel(
        config=_openai_config(
            api_key="test-key", api_format=ApiFormat.OPENAI_RESPONSES
        ),
        stream_read_timeout=30.0,
    )
    model._client = mock_client

    stream_result = await model.complete(
        [Message(role="user", content="hi")], stream=True
    )
    with pytest.raises(httpx.ReadTimeout):
        async for _ in stream_result:
            pass
    # The stall window was actually applied to the streaming request.
    assert mock_client.stream.call_args[1]["timeout"].read == 30.0


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


@pytest.mark.asyncio
async def test_responses_api_translates_json_schema_response_format_to_text() -> None:
    """Chat-shaped json_schema response_format becomes a flattened text.format.

    The Responses API configures structured output via text.format, not the
    top-level response_format parameter, and expects {name, schema, strict}
    alongside the type rather than nested under a json_schema key.
    """
    mock_response = Mock(spec=httpx.Response)
    mock_response.json.return_value = {
        "id": "resp_fmt_1",
        "model": "gpt-4o-mini",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "{}"}],
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

    schema = {"type": "object", "properties": {"city": {"type": "string"}}}
    response_format = {
        "type": "json_schema",
        "json_schema": {"name": "CityInfo", "schema": schema, "strict": True},
    }

    await model.complete(
        [Message(role="user", content="hello")], response_format=response_format
    )

    body = mock_client.post.call_args[1]["json"]
    assert "response_format" not in body
    assert body["text"] == {
        "format": {
            "type": "json_schema",
            "name": "CityInfo",
            "schema": schema,
            "strict": True,
        }
    }


@pytest.mark.asyncio
async def test_responses_api_translates_json_object_response_format_to_text() -> None:
    """A bare {"type": "json_object"} maps straight into text.format."""
    mock_response = Mock(spec=httpx.Response)
    mock_response.json.return_value = {
        "id": "resp_fmt_2",
        "model": "gpt-4o-mini",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "{}"}],
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

    await model.complete(
        [Message(role="user", content="hello")],
        response_format={"type": "json_object"},
    )

    body = mock_client.post.call_args[1]["json"]
    assert "response_format" not in body
    assert body["text"] == {"format": {"type": "json_object"}}


@pytest.mark.asyncio
async def test_responses_api_complete_extracts_reasoning_summary() -> None:
    """A reasoning output item's summary_text populates reasoning_content."""
    mock_response = Mock(spec=httpx.Response)
    mock_response.json.return_value = {
        "id": "resp_reason_1",
        "model": "gpt-5",
        "output": [
            {
                "type": "reasoning",
                "id": "rs_1",
                "summary": [
                    {"type": "summary_text", "text": "First I consider the goal."},
                    {"type": "summary_text", "text": "Then I pick an approach."},
                ],
            },
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "The answer is 42."}],
            },
        ],
    }
    mock_response.raise_for_status = Mock()

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.post.return_value = mock_response

    model = OpenAIModel(
        config=_openai_config(api_key="test-key", api_format=ApiFormat.OPENAI_RESPONSES),
        model="gpt-5",
    )
    model._client = mock_client

    result = await model.complete([Message(role="user", content="question")])

    assert result.message.content == "The answer is 42."
    assert result.reasoning_content == (
        "First I consider the goal.\nThen I pick an approach."
    )


@pytest.mark.asyncio
async def test_responses_api_streaming_yields_reasoning_summary_delta() -> None:
    """Streaming reasoning summary deltas surface as StreamDelta.reasoning_content."""
    model, _ = _streaming_model(
        [
            'data: {"type": "response.created", "response": {"id": "resp_reason_stream"}}',
            'data: {"type": "response.reasoning_summary_text.delta", "output_index": 0, "delta": "think "}',
            'data: {"type": "response.reasoning_summary_text.delta", "output_index": 0, "delta": "hard"}',
            'data: {"type": "response.output_item.added", "output_index": 1, "item": {"type": "message", "role": "assistant"}}',
            'data: {"type": "response.output_text.delta", "output_index": 1, "delta": "done"}',
            'data: {"type": "response.completed", "response": {"id": "resp_reason_stream"}}',
        ]
    )

    deltas = []
    stream_result = await model.complete(
        [Message(role="user", content="Hi")], stream=True
    )
    async for delta in stream_result:
        deltas.append(delta)

    reasoning = [d.reasoning_content for d in deltas if d.reasoning_content]
    assert reasoning == ["think ", "hard"]
    contents = [d.content for d in deltas if d.content]
    assert contents == ["done"]


@pytest.mark.asyncio
async def test_responses_api_streaming_falls_back_to_chat_on_404() -> None:
    """A 404 on the streaming Responses endpoint falls back to Chat Completions."""
    error_response = httpx.Response(
        404,
        request=httpx.Request("POST", "https://test.openai.com/v1/responses"),
        text="Not Found",
    )
    error_cm = MagicMock()
    error_cm.__aenter__.return_value = error_response

    chat_response = AsyncMock()
    chat_response.is_error = False
    chat_response.raise_for_status = Mock()

    async def chat_aiter_lines():
        yield 'data: {"choices": [{"delta": {"content": "fallback"}, "finish_reason": null}]}'
        yield 'data: {"choices": [{"delta": {}, "finish_reason": "stop"}]}'
        yield "data: [DONE]"

    chat_response.aiter_lines = chat_aiter_lines
    chat_cm = MagicMock()
    chat_cm.__aenter__.return_value = chat_response

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.stream = MagicMock(side_effect=[error_cm, chat_cm])

    model = OpenAIModel(
        config=_openai_config(
            api_key="test-key",
            base_url="https://test.openai.com/v1",
            api_format=ApiFormat.OPENAI_RESPONSES,
        )
    )
    model._client = mock_client

    deltas = []
    stream_result = await model.complete(
        [Message(role="user", content="hello")], stream=True
    )
    async for delta in stream_result:
        deltas.append(delta)

    assert any(d.content == "fallback" for d in deltas)
    assert model._responses_api_available is False
    assert mock_client.stream.call_count == 2
    responses_url = mock_client.stream.call_args_list[0][0][1]
    chat_url = mock_client.stream.call_args_list[1][0][1]
    assert responses_url == "https://test.openai.com/v1/responses"
    assert chat_url == "https://test.openai.com/v1/chat/completions"


@pytest.mark.asyncio
async def test_responses_api_streaming_emits_done_only_message_text() -> None:
    """A message delivered solely via output_item.done still yields its text.

    Some gateways skip content deltas and hand the whole message in the
    terminal output_item.done item; the parser must recover that text.
    """
    model, _ = _streaming_model(
        [
            'data: {"type": "response.created", "response": {"id": "resp_doneonly"}}',
            'data: {"type": "response.output_item.added", "output_index": 0, "item": {"type": "message", "role": "assistant"}}',
            'data: {"type": "response.output_item.done", "output_index": 0, "item": {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "full answer"}]}}',
            'data: {"type": "response.completed", "response": {"id": "resp_doneonly"}}',
        ]
    )

    deltas = []
    stream_result = await model.complete(
        [Message(role="user", content="Hi")], stream=True
    )
    async for delta in stream_result:
        deltas.append(delta)

    contents = [d.content for d in deltas if d.content]
    assert contents == ["full answer"]


@pytest.mark.asyncio
async def test_responses_api_streaming_done_text_not_duplicated_after_deltas() -> None:
    """When text already streamed via deltas, the done item must not re-emit it."""
    model, _ = _streaming_model(
        [
            'data: {"type": "response.created", "response": {"id": "resp_dup"}}',
            'data: {"type": "response.output_item.added", "output_index": 0, "item": {"type": "message", "role": "assistant"}}',
            'data: {"type": "response.output_text.delta", "output_index": 0, "delta": "Hel"}',
            'data: {"type": "response.output_text.delta", "output_index": 0, "delta": "lo"}',
            'data: {"type": "response.output_item.done", "output_index": 0, "item": {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Hello"}]}}',
            'data: {"type": "response.completed", "response": {"id": "resp_dup"}}',
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


@pytest.mark.asyncio
async def test_responses_api_retries_on_400_with_structured_param_error() -> None:
    """A 400 whose error.param names previous_response_id triggers the retry."""
    param_body = (
        '{"error":{"message":"Invalid value","type":"invalid_request_error",'
        '"param":"previous_response_id","code":null}}'
    )
    request = httpx.Request("POST", "https://test.openai.com/v1/responses")
    error = httpx.HTTPStatusError(
        "400 Bad Request",
        request=request,
        response=httpx.Response(400, request=request, text=param_body),
    )

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.post.side_effect = [error, _responses_success("resp_ok", "recovered")]

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
        [Message(role="user", content="hi")], thread_response_id="resp_old"
    )

    assert result.message.content == "recovered"
    assert mock_client.post.call_count == 2
    assert "previous_response_id" not in mock_client.post.call_args_list[1][1]["json"]


@pytest.mark.asyncio
async def test_responses_api_no_retry_when_token_outside_error_object() -> None:
    """A 400 that only echoes the id outside error.* must not trigger a retry.

    The full JSON body contains the token (in an echoed input field), but the
    structured error blames something else, so retrying would be wrong.
    """
    unrelated_body = (
        '{"error":{"message":"model is overloaded","type":"server_error",'
        '"param":null,"code":null},'
        '"echo":{"previous_response_id":"resp_old"}}'
    )
    request = httpx.Request("POST", "https://test.openai.com/v1/responses")
    error = httpx.HTTPStatusError(
        "400 Bad Request",
        request=request,
        response=httpx.Response(400, request=request, text=unrelated_body),
    )

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
            [Message(role="user", content="hi")], thread_response_id="resp_old"
        )

    assert mock_client.post.call_count == 1
