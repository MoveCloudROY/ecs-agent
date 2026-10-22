"""Live registry construction and protocol-specific text/image completion.

Select a profile with --live; see docs/testing.md.
"""

import pytest

from ecs_agent.providers.registry import ProviderRegistry, get_model
from ecs_agent.providers.protocol import LLMModel
from ecs_agent.types import CompletionResult, ImageUrlPart, Message
from tests.support.models import LiveModelFactory

pytestmark = [pytest.mark.live(capabilities=["text"]), pytest.mark.smoke]


def _model(factory: LiveModelFactory) -> LLMModel:
    registry = ProviderRegistry.from_dict({"live": factory.registry_entry()})
    return factory.track(
        get_model(
            f"live/{factory.profile.model}", registry=registry, api_key=factory.key()
        )
    )


@pytest.mark.live(formats=["openai_chat_completions"])
async def test_live_openai_chat_text_response(
    live_model_factory: LiveModelFactory,
) -> None:
    result = await _model(live_model_factory).complete(
        [Message(role="user", content="Say hello in 5 words.")]
    )
    assert isinstance(result, CompletionResult)
    assert len(result.message.content.strip()) > 0


@pytest.mark.live(formats=["openai_responses"])
async def test_live_openai_responses_text_response(
    live_model_factory: LiveModelFactory,
) -> None:
    result = await _model(live_model_factory).complete(
        [Message(role="user", content="Say hello in 5 words.")]
    )
    assert isinstance(result, CompletionResult)
    assert len(result.message.content.strip()) > 0


@pytest.mark.vision
@pytest.mark.live(
    formats=["openai_responses"], capabilities=["text", "vision"], role="vision"
)
async def test_live_openai_responses_vision_response(
    live_model_factory: LiveModelFactory, live_image_url: str
) -> None:
    result = await _model(live_model_factory).complete(
        [
            Message(
                role="user",
                content="Describe this image briefly.",
                parts=[ImageUrlPart(url=live_image_url)],
            )
        ]
    )
    assert isinstance(result, CompletionResult)
    assert len(result.message.content.strip()) > 0


@pytest.mark.live(formats=["anthropic_messages"])
async def test_live_anthropic_messages_text_response(
    live_model_factory: LiveModelFactory,
) -> None:
    result = await _model(live_model_factory).complete(
        [Message(role="user", content="Say hello in 5 words.")]
    )
    assert isinstance(result, CompletionResult)
    assert len(result.message.content.strip()) > 0
