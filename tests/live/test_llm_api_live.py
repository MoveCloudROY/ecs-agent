import os

import pytest

from ecs_agent.providers.config import ApiFormat
from ecs_agent.types import CompletionResult, ImageUrlPart, Message

from tests.live.api_format import live_openai_base_url, live_openai_model

_registry_module = pytest.importorskip("ecs_agent.providers.registry")
ProviderRegistry = _registry_module.ProviderRegistry
get_model = _registry_module.get_model


def _live_registry() -> ProviderRegistry:
    anthropic_base_url = (
        os.getenv("ANTHROPIC_LIVE_BASE_URL")
        or "https://dashscope.aliyuncs.com/apps/anthropic"
    )
    return ProviderRegistry.from_dict(
        {
            "aliyun": {
                "base_url": live_openai_base_url(ApiFormat.OPENAI_CHAT_COMPLETIONS),
                "api_format": "openai_chat_completions",
            },
            "aliyun-responses": {
                "base_url": live_openai_base_url(ApiFormat.OPENAI_RESPONSES),
                "api_format": "openai_responses",
            },
            "aliyun-vision": {
                "base_url": live_openai_base_url(ApiFormat.OPENAI_CHAT_COMPLETIONS),
                "api_format": "openai_responses",
            },
            "aliyun-anthropic": {
                "base_url": anthropic_base_url,
                "api_format": "anthropic_messages",
            },
        }
    )


@pytest.mark.asyncio
async def test_live_openai_chat_text_response(live_api_key: str) -> None:
    import httpx

    model = live_openai_model()
    model = get_model(
        f"aliyun/{model}",
        registry=_live_registry(),
        api_key=live_api_key,
    )

    try:
        result = await model.complete(
            [Message(role="user", content="Say hello in 5 words.")]
        )
    except httpx.ReadTimeout:
        pytest.skip("Aliyun chat completions endpoint timed out (flaky network)")

    assert isinstance(result, CompletionResult)
    assert len(result.message.content.strip()) > 0


@pytest.mark.asyncio
async def test_live_openai_responses_text_response(live_api_key: str) -> None:
    import httpx

    model = live_openai_model()
    model = get_model(
        f"aliyun-responses/{model}",
        registry=_live_registry(),
        api_key=live_api_key,
    )

    try:
        result = await model.complete(
            [Message(role="user", content="Say hello in 5 words.")]
        )
    except httpx.ReadTimeout:
        pytest.skip("Aliyun Responses API endpoint timed out (flaky network)")

    assert isinstance(result, CompletionResult)
    assert len(result.message.content.strip()) > 0


@pytest.mark.asyncio
async def test_live_openai_responses_vision_response(
    live_api_key: str, live_image_url: str
) -> None:
    model = live_openai_model("qwen3-vl-flash")
    model = get_model(
        f"aliyun-vision/{model}",
        registry=_live_registry(),
        api_key=live_api_key,
    )

    result = await model.complete(
        [
            Message(
                role="user",
                content="Describe this image briefly.",
                parts=[
                    ImageUrlPart(url=live_image_url),
                ],
            )
        ]
    )

    assert isinstance(result, CompletionResult)
    assert len(result.message.content.strip()) > 0


@pytest.mark.asyncio
async def test_live_anthropic_messages_text_response() -> None:
    api_key = os.getenv("ANTHROPIC_LIVE_API_KEY")
    if not api_key:
        pytest.skip("ANTHROPIC_LIVE_API_KEY is not set")
    model_name = os.getenv("ANTHROPIC_LIVE_MODEL") or "kimi-k2.5"
    model = get_model(
        f"aliyun-anthropic/{model_name}",
        registry=_live_registry(),
        api_key=api_key,
    )

    result = await model.complete(
        [Message(role="user", content="Say hello in 5 words.")]
    )

    assert isinstance(result, CompletionResult)
    assert len(result.message.content.strip()) > 0
