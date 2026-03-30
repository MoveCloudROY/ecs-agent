import os

import pytest

from ecs_agent.types import CompletionResult, ImageUrlPart, Message, TextPart

OpenAIProvider = pytest.importorskip(
    "ecs_agent.providers.openai_provider"
).OpenAIProvider
ClaudeProvider = pytest.importorskip(
    "ecs_agent.providers.claude_provider"
).ClaudeProvider


@pytest.mark.live
@pytest.mark.asyncio
async def test_live_openai_chat_text_response(live_api_key: str) -> None:
    model = os.getenv("LLM_MODEL", "qwen3.5-flash")
    provider = OpenAIProvider(
        api_key=live_api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model=model,
    )

    result = await provider.complete(
        [Message(role="user", content="Say hello in 5 words.")]
    )

    assert isinstance(result, CompletionResult)
    assert len(result.message.content.strip()) > 0


@pytest.mark.live
@pytest.mark.asyncio
async def test_live_openai_responses_text_response(live_api_key: str) -> None:
    model = os.getenv("LLM_MODEL", "qwen3.5-flash")
    provider = OpenAIProvider(
        api_key=live_api_key,
        base_url="https://dashscope.aliyuncs.com/api/v2/apps/protocols/compatible-mode/v1",
        model=model,
        use_responses_api=True,
    )

    result = await provider.complete(
        [Message(role="user", content="Say hello in 5 words.")]
    )

    assert isinstance(result, CompletionResult)
    assert len(result.message.content.strip()) > 0


@pytest.mark.live
@pytest.mark.asyncio
async def test_live_openai_responses_vision_response(
    live_api_key: str,
    live_image_url: str,
) -> None:
    model = os.getenv("LLM_MODEL", "qwen3-vl-flash")
    provider = OpenAIProvider(
        api_key=live_api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model=model,
        use_responses_api=True,
    )

    result = await provider.complete(
        [
            Message(
                role="user",
                content="",
                parts=[
                    TextPart(text="Describe this image briefly."),
                    ImageUrlPart(url=live_image_url),
                ],
            )
        ]
    )

    assert isinstance(result, CompletionResult)
    assert len(result.message.content.strip()) > 0


@pytest.mark.live
@pytest.mark.asyncio
async def test_live_anthropic_messages_text_response(live_api_key: str) -> None:
    model = os.getenv("LLM_MODEL", "kimi-k2.5")
    provider = ClaudeProvider(
        api_key=live_api_key,
        base_url="https://dashscope.aliyuncs.com/apps/anthropic",
        model=model,
    )

    result = await provider.complete(
        [Message(role="user", content="Say hello in 5 words.")]
    )

    assert isinstance(result, CompletionResult)
    assert len(result.message.content.strip()) > 0
