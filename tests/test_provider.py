"""Tests for LLMModel Protocol."""

import pytest
from typing import Protocol, get_type_hints
from ecs_agent.providers import LLMModel, OpenAIModel, FakeModel
from ecs_agent.types import Message, CompletionResult, ToolSchema, Usage


def test_llm_model_is_protocol() -> None:
    assert isinstance(LLMModel, type)
    assert hasattr(LLMModel, "_is_protocol")


def test_llm_model_has_complete_method() -> None:
    assert hasattr(LLMModel, "complete")


def test_llm_model_complete_signature() -> None:
    complete_method = getattr(LLMModel, "complete")
    assert complete_method is not None
    assert callable(complete_method)

    hints = get_type_hints(complete_method)
    assert "messages" in hints
    assert "tools" in hints
    assert "return" in hints


def test_llm_model_complete_is_async() -> None:
    import inspect

    complete_method = getattr(LLMModel, "complete")
    assert inspect.iscoroutinefunction(complete_method)


def test_llm_model_has_stream_parameter() -> None:
    import inspect

    complete_method = getattr(LLMModel, "complete")
    sig = inspect.signature(complete_method)
    assert "stream" in sig.parameters
    assert sig.parameters["stream"].default is False


def test_llm_model_has_response_format_parameter() -> None:
    import inspect

    complete_method = getattr(LLMModel, "complete")
    sig = inspect.signature(complete_method)
    assert "response_format" in sig.parameters
    assert sig.parameters["response_format"].default is None


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
