"""Tests for OpenAI-compatible provider."""

import json
import pytest
import httpx
from unittest.mock import AsyncMock, Mock
from ecs_agent.providers.openai_provider import OpenAIProvider
from ecs_agent.providers.config import ApiFormat, ProviderConfig
from ecs_agent.providers.protocol import LLMProvider
from ecs_agent.types import (
    FileRefPart,
    ImageUrlPart,
    Message,
    ToolCall,
    ToolSchema,
)

COMPACTION_SENTINEL = "[COMPACTION SUMMARY]\n"


def _openai_config(
    *,
    api_key: str = "test-key",
    base_url: str = "https://api.openai.com/v1",
    api_format: ApiFormat = ApiFormat.OPENAI_CHAT_COMPLETIONS,
) -> ProviderConfig:
    return ProviderConfig(
        provider_id="openai",
        base_url=base_url,
        api_key=api_key,
        api_format=api_format,
    )


@pytest.mark.asyncio
async def test_constructor_instantiation() -> None:
    """Test OpenAIProvider can be instantiated with required parameters."""
    provider = OpenAIProvider(
        config=_openai_config(
            api_key="test-key",
            base_url="https://test.openai.com/v1",
        ),
        model="gpt-4o-mini",
    )
    assert provider is not None


@pytest.mark.asyncio
async def test_request_format(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test HTTP request format matches OpenAI spec."""
    mock_response = Mock(spec=httpx.Response)
    mock_response.json.return_value = {
        "choices": [{"message": {"role": "assistant", "content": "test response"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
    mock_response.raise_for_status = Mock()

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.post.return_value = mock_response

    provider = OpenAIProvider(
        config=_openai_config(
            api_key="test-key",
            base_url="https://test.openai.com/v1",
        ),
        model="gpt-4o-mini",
    )
    provider._client = mock_client

    messages = [Message(role="user", content="test message")]
    tools = [
        ToolSchema(
            name="test_tool",
            description="test description",
            parameters={"type": "object", "properties": {}},
        )
    ]

    await provider.complete(messages, tools)

    # Verify POST was called
    assert mock_client.post.called
    call_args = mock_client.post.call_args

    # Verify URL
    assert call_args[0][0] == "https://test.openai.com/v1/chat/completions"

    # Verify headers
    assert call_args[1]["headers"]["Authorization"] == "Bearer test-key"
    assert call_args[1]["headers"]["Content-Type"] == "application/json"

    # Verify request body
    body = call_args[1]["json"]
    assert body["model"] == "gpt-4o-mini"
    assert body["messages"] == [{"role": "user", "content": "test message"}]
    assert body["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "test_tool",
                "description": "test description",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]


@pytest.mark.asyncio
async def test_response_parsing_content_and_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test response parsing extracts content and usage correctly."""
    mock_response = Mock(spec=httpx.Response)
    mock_response.json.return_value = {
        "choices": [{"message": {"role": "assistant", "content": "Hello world"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
    mock_response.raise_for_status = Mock()

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.post.return_value = mock_response

    provider = OpenAIProvider(config=_openai_config(api_key="test-key"))
    provider._client = mock_client

    messages = [Message(role="user", content="test")]
    result = await provider.complete(messages)

    assert result.message.role == "assistant"
    assert result.message.content == "Hello world"
    assert result.message.tool_calls is None
    assert result.usage is not None
    assert result.usage.prompt_tokens == 10
    assert result.usage.completion_tokens == 5
    assert result.usage.total_tokens == 15


@pytest.mark.asyncio
async def test_response_parsing_tool_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test response parsing handles tool calls correctly."""
    mock_response = Mock(spec=httpx.Response)
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_123",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"location": "NYC"}',
                            },
                        }
                    ],
                }
            }
        ],
        "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
    }
    mock_response.raise_for_status = Mock()

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.post.return_value = mock_response

    provider = OpenAIProvider(config=_openai_config(api_key="test-key"))
    provider._client = mock_client

    messages = [Message(role="user", content="What's the weather?")]
    result = await provider.complete(messages)

    assert result.message.role == "assistant"
    assert result.message.content == ""
    assert result.message.tool_calls is not None
    assert len(result.message.tool_calls) == 1
    assert result.message.tool_calls[0].id == "call_123"
    assert result.message.tool_calls[0].name == "get_weather"
    assert result.message.tool_calls[0].arguments == {"location": "NYC"}


@pytest.mark.asyncio
async def test_http_error_handling(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test HTTP 4xx/5xx errors are raised and full error content is printed."""
    mock_response = Mock(spec=httpx.Response)
    mock_response.status_code = 429
    mock_response.text = (
        '{"error": {"message": "Rate limit exceeded", "type": "rate_limit_error"}}'
    )
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "Too Many Requests", request=Mock(spec=httpx.Request), response=mock_response
    )

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.post.return_value = mock_response

    provider = OpenAIProvider(config=_openai_config(api_key="test-key"))
    provider._client = mock_client

    messages = [Message(role="user", content="test")]

    with pytest.raises(httpx.HTTPStatusError):
        await provider.complete(messages)


@pytest.mark.asyncio
async def test_http_error_prints_response_body(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test HTTP error prints status code and full response body."""
    mock_response = Mock(spec=httpx.Response)
    mock_response.status_code = 400
    mock_response.text = (
        '{"error": {"message": "Invalid model", "type": "invalid_request_error"}}'
    )
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "Bad Request", request=Mock(spec=httpx.Request), response=mock_response
    )

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.post.return_value = mock_response

    provider = OpenAIProvider(config=_openai_config(api_key="test-key"))
    provider._client = mock_client

    messages = [Message(role="user", content="test")]

    with pytest.raises(httpx.HTTPStatusError):
        await provider.complete(messages)

    captured = capsys.readouterr()
    assert "400" in captured.out
    assert "Invalid model" in captured.out


@pytest.mark.asyncio
async def test_network_error_prints_full_message(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test connection/timeout errors print full error details."""
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.post.side_effect = httpx.ConnectError("Connection refused")

    provider = OpenAIProvider(config=_openai_config(api_key="test-key"))
    provider._client = mock_client

    messages = [Message(role="user", content="test")]

    with pytest.raises(httpx.ConnectError):
        await provider.complete(messages)

    captured = capsys.readouterr()
    assert "Connection refused" in captured.out


@pytest.mark.asyncio
async def test_request_without_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test request format when tools parameter is None."""
    mock_response = Mock(spec=httpx.Response)
    mock_response.json.return_value = {
        "choices": [{"message": {"role": "assistant", "content": "response"}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
    }
    mock_response.raise_for_status = Mock()

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.post.return_value = mock_response

    provider = OpenAIProvider(config=_openai_config(api_key="test-key"))
    provider._client = mock_client

    messages = [Message(role="user", content="test")]
    await provider.complete(messages, tools=None)

    # Verify tools field is not included in request when None
    call_args = mock_client.post.call_args
    body = call_args[1]["json"]
    assert "tools" not in body


@pytest.mark.asyncio
async def test_protocol_compliance() -> None:
    """Test OpenAIProvider satisfies LLMProvider Protocol."""
    provider = OpenAIProvider(config=_openai_config(api_key="test-key"))
    assert isinstance(provider, LLMProvider)


@pytest.mark.asyncio
async def test_tool_call_messages_serialize_content_as_null() -> None:
    mock_response = Mock(spec=httpx.Response)
    mock_response.json.return_value = {
        "choices": [{"message": {"role": "assistant", "content": "final answer"}}],
        "usage": {"prompt_tokens": 50, "completion_tokens": 20, "total_tokens": 70},
    }
    mock_response.raise_for_status = Mock()

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.post.return_value = mock_response

    provider = OpenAIProvider(config=_openai_config(api_key="test-key"))
    provider._client = mock_client

    messages = [
        Message(role="user", content="compare cities"),
        Message(
            role="assistant",
            content="",
            tool_calls=[
                ToolCall(id="call_1", name="get_weather", arguments={"city": "Beijing"})
            ],
        ),
        Message(role="tool", content="Sunny 28C", tool_call_id="call_1"),
    ]
    await provider.complete(messages)

    body = mock_client.post.call_args[1]["json"]
    assistant_msg = body["messages"][1]
    assert assistant_msg["content"] is None
    assert assistant_msg["tool_calls"][0]["id"] == "call_1"

    tool_msg = body["messages"][2]
    assert tool_msg["content"] == "Sunny 28C"
    assert tool_msg["tool_call_id"] == "call_1"


@pytest.mark.asyncio
async def test_assistant_message_with_content_and_tool_calls_preserves_content() -> (
    None
):
    mock_response = Mock(spec=httpx.Response)
    mock_response.json.return_value = {
        "choices": [{"message": {"role": "assistant", "content": "ok"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
    mock_response.raise_for_status = Mock()

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.post.return_value = mock_response

    provider = OpenAIProvider(config=_openai_config(api_key="test-key"))
    provider._client = mock_client

    messages = [
        Message(
            role="assistant",
            content="Let me check the weather",
            tool_calls=[
                ToolCall(id="call_2", name="get_weather", arguments={"city": "NYC"})
            ],
        ),
    ]
    await provider.complete(messages)

    body = mock_client.post.call_args[1]["json"]
    assistant_msg = body["messages"][0]
    assert assistant_msg["content"] == "Let me check the weather"


@pytest.mark.asyncio
async def test_convert_messages_serializes_tool_call_arguments_as_json_string() -> None:
    """Test _convert_messages serializes tool call arguments as JSON string, not dict."""
    mock_response = Mock(spec=httpx.Response)
    mock_response.json.return_value = {
        "choices": [{"message": {"role": "assistant", "content": "done"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
    mock_response.raise_for_status = Mock()

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.post.return_value = mock_response

    provider = OpenAIProvider(config=_openai_config(api_key="test-key"))
    provider._client = mock_client

    # Create message with tool call containing dict arguments
    message = Message(
        role="assistant",
        content="",
        tool_calls=[
            ToolCall(
                id="call_123",
                name="test_function",
                arguments={"key": "value", "count": 42},
            )
        ],
    )

    await provider.complete([message])

    # Verify the request was made with JSON-serialized arguments
    body = mock_client.post.call_args[1]["json"]
    args = body["messages"][0]["tool_calls"][0]["function"]["arguments"]

    # CRITICAL: Must be JSON string, not dict
    assert isinstance(args, str), f"Expected string, got {type(args)}"

    # Verify round-trip: string should deserialize back to original dict
    assert json.loads(args) == {"key": "value", "count": 42}


@pytest.mark.asyncio
async def test_multimodal_chat_request_maps_parts_to_chat_content() -> None:
    mock_response = Mock(spec=httpx.Response)
    mock_response.json.return_value = {
        "choices": [{"message": {"role": "assistant", "content": "ok"}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }
    mock_response.raise_for_status = Mock()

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.post.return_value = mock_response

    provider = OpenAIProvider(config=_openai_config(api_key="test-key"))
    provider._client = mock_client

    await provider.complete(
        [
            Message(
                role="user",
                content="Describe this",
                parts=[
                    ImageUrlPart(url="https://example.com/cat.png", detail="high"),
                    FileRefPart(file_id="file_123", filename="notes.txt"),
                ],
            )
        ]
    )

    body = mock_client.post.call_args[1]["json"]
    content = body["messages"][0]["content"]
    assert isinstance(content, list)
    assert content[0] == {"type": "text", "text": "Describe this"}
    assert content[1] == {
        "type": "image_url",
        "image_url": {"url": "https://example.com/cat.png", "detail": "high"},
    }
    assert content[2] == {
        "type": "file",
        "file": {"file_id": "file_123", "filename": "notes.txt"},
    }


@pytest.mark.asyncio
async def test_multimodal_responses_request_maps_image_and_file_parts() -> None:
    mock_response = Mock(spec=httpx.Response)
    mock_response.json.return_value = {
        "id": "resp_mm_1",
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

    provider = OpenAIProvider(
        config=_openai_config(api_key="test-key", api_format=ApiFormat.OPENAI_RESPONSES)
    )
    provider._client = mock_client

    await provider.complete(
        [
            Message(
                role="user",
                content="Analyze",
                parts=[
                    ImageUrlPart(url="https://example.com/a.png", detail="low"),
                    FileRefPart(file_id="file_456", filename="report.pdf"),
                ],
            )
        ]
    )

    body = mock_client.post.call_args[1]["json"]
    content = body["input"][0]["content"]
    assert content[0] == {"type": "input_text", "text": "Analyze"}
    assert content[1] == {
        "type": "input_image",
        "image_url": "https://example.com/a.png",
        "detail": "low",
    }
    assert content[2] == {
        "type": "input_file",
        "file_id": "file_456",
        "filename": "report.pdf",
    }


@pytest.mark.asyncio
async def test_openai_chat_adapter_delivers_system_prompt_containing_summary_xml() -> (
    None
):
    """Test that OpenAI Chat Completions request contains XML summary in system message."""
    mock_response = Mock(spec=httpx.Response)
    mock_response.json.return_value = {
        "choices": [{"message": {"role": "assistant", "content": "ok"}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }
    mock_response.raise_for_status = Mock()

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.post.return_value = mock_response

    provider = OpenAIProvider(config=_openai_config(api_key="test-key"))
    provider._client = mock_client

    summary_xml = "<chat_history_summary>Summary: X happened</chat_history_summary>"
    await provider.complete(
        [
            Message(role="system", content=f"You are helpful.\n\n{summary_xml}"),
            Message(role="user", content="Hello"),
        ]
    )

    body = mock_client.post.call_args[1]["json"]
    messages = body["messages"]
    assert len(messages) >= 1
    assert messages[0]["role"] == "system"
    assert summary_xml in messages[0]["content"]


@pytest.mark.asyncio
async def test_openai_responses_adapter_delivers_summary_xml_in_instructions() -> None:
    """Test that OpenAI Responses API request contains XML summary in instructions field only."""
    mock_response = Mock(spec=httpx.Response)
    mock_response.json.return_value = {
        "id": "resp_compaction_1",
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

    provider = OpenAIProvider(
        config=_openai_config(api_key="test-key", api_format=ApiFormat.OPENAI_RESPONSES)
    )
    provider._client = mock_client

    summary_xml = "<chat_history_summary>Summary: X happened</chat_history_summary>"
    await provider.complete(
        [
            Message(role="system", content=f"You are helpful.\n\n{summary_xml}"),
            Message(role="user", content="Hello"),
        ]
    )

    body = mock_client.post.call_args[1]["json"]

    # Verify instructions field contains the XML summary
    assert "instructions" in body
    assert summary_xml in body["instructions"]

    # Verify no input item contains the summary XML
    assert "input" in body
    for item in body["input"]:
        if item.get("type") == "message" and item.get("role") == "user":
            content = item.get("content", [])
            for content_item in content:
                if "text" in content_item:
                    assert summary_xml not in content_item["text"]


@pytest.mark.asyncio
async def test_vision_responses_output_parses_parts_into_message_parts() -> None:
    mock_response = Mock(spec=httpx.Response)
    mock_response.json.return_value = {
        "id": "resp_vision_1",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": "Found details."},
                    {
                        "type": "input_image",
                        "image_url": "https://example.com/processed.png",
                        "detail": "auto",
                    },
                    {
                        "type": "input_file",
                        "file_id": "file_vision",
                        "filename": "vision.json",
                    },
                ],
            }
        ],
    }
    mock_response.raise_for_status = Mock()

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.post.return_value = mock_response

    provider = OpenAIProvider(
        config=_openai_config(api_key="test-key", api_format=ApiFormat.OPENAI_RESPONSES)
    )
    provider._client = mock_client

    result = await provider.complete([Message(role="user", content="vision")])
    assert result.message.content == "Found details."
    assert result.message.parts is not None
    assert isinstance(result.message.parts[0], ImageUrlPart)
    assert isinstance(result.message.parts[1], FileRefPart)


@pytest.mark.asyncio
async def test_invalid_api_format_string_raises_value_error() -> None:
    config = _openai_config(api_key="test-key")
    config.api_format = "invalid_api_format"
    provider = OpenAIProvider(config=config)

    with pytest.raises(ValueError, match="Unsupported OpenAI provider api_format"):
        await provider.complete([Message(role="user", content="hello")])


@pytest.mark.asyncio
async def test_unsupported_api_format_raises_clear_error() -> None:
    provider = OpenAIProvider(
        config=_openai_config(
            api_key="test-key",
            api_format=ApiFormat.OPENAI_EMBEDDINGS,
        )
    )

    with pytest.raises(ValueError, match="Unsupported OpenAI provider api_format"):
        await provider.complete([Message(role="user", content="hello")])
