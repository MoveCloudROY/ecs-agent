import os

import pytest

from ecs_agent.types import CompletionResult, ImageUrlPart, Message

_registry_module = pytest.importorskip("ecs_agent.providers.registry")
ProviderRegistry = _registry_module.ProviderRegistry
get_llm_provider = _registry_module.get_llm_provider


def _live_registry() -> ProviderRegistry:
    return ProviderRegistry.from_dict(
        {
            "aliyun": {
                "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                "api_format": "openai_chat_completions",
            },
            "aliyun-responses": {
                "base_url": "https://dashscope.aliyuncs.com/api/v2/apps/protocols/compatible-mode/v1",
                "api_format": "openai_responses",
            },
            "aliyun-vision": {
                "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                "api_format": "openai_responses",
            },
            "aliyun-anthropic": {
                "base_url": "https://dashscope.aliyuncs.com/apps/anthropic",
                "api_format": "anthropic_messages",
            },
        }
    )


@pytest.mark.live
@pytest.mark.asyncio
async def test_live_openai_chat_text_response(live_api_key: str) -> None:
    model = os.getenv("LLM_MODEL", "qwen3.5-flash")
    provider = get_llm_provider(
        f"aliyun/{model}",
        registry=_live_registry(),
        api_key=live_api_key,
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
    provider = get_llm_provider(
        f"aliyun-responses/{model}",
        registry=_live_registry(),
        api_key=live_api_key,
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
    provider = get_llm_provider(
        f"aliyun-vision/{model}",
        registry=_live_registry(),
        api_key=live_api_key,
    )

    result = await provider.complete(
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


@pytest.mark.live
@pytest.mark.asyncio
async def test_live_anthropic_messages_text_response(live_api_key: str) -> None:
    model = os.getenv("LLM_MODEL", "kimi-k2.5")
    provider = get_llm_provider(
        f"aliyun-anthropic/{model}",
        registry=_live_registry(),
        api_key=live_api_key,
    )

    result = await provider.complete(
        [Message(role="user", content="Say hello in 5 words.")]
    )

    assert isinstance(result, CompletionResult)
    assert len(result.message.content.strip()) > 0
