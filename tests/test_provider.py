"""Tests for LLMModel Protocol compliance."""

import pytest
from ecs_agent.providers import LLMModel, OpenAIModel, FakeModel
from ecs_agent.types import Message, CompletionResult, ToolSchema, Usage


def test_openai_model_conforms_to_protocol() -> None:
    from ecs_agent.providers.config import ApiFormat, ProviderConfig

    instance = OpenAIModel(
        config=ProviderConfig(
            provider_id="openai",
            base_url="http://test",
            api_key="test",
            api_format=ApiFormat.OPENAI_CHAT_COMPLETIONS,
        ),
        model="gpt-4",
    )
    assert isinstance(instance, LLMModel)


def test_fake_model_conforms_to_protocol() -> None:
    instance = FakeModel(
        responses=[CompletionResult(message=Message(role="assistant", content="hi"))]
    )
    assert isinstance(instance, LLMModel)


@pytest.mark.asyncio
async def test_fake_model_complete_no_extra_params() -> None:
    result = CompletionResult(
        message=Message(role="assistant", content="test response"), usage=None
    )
    model = FakeModel([result])
    response = await model.complete([Message(role="user", content="test")])
    assert response.message.content == "test response"


@pytest.mark.asyncio
async def test_fake_model_complete_with_stream_and_format() -> None:
    result = CompletionResult(
        message=Message(role="assistant", content="test response"), usage=None
    )
    model = FakeModel([result])
    response = await model.complete(
        [Message(role="user", content="test")],
        stream=False,
        response_format=None,
    )
    assert response.message.content == "test response"
