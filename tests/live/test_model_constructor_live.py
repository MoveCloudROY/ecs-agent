"""Live contract tests for direct Model constructor branches and identity.

Select a profile with --live; see docs/testing.md.
"""

import pytest

from ecs_agent.providers.config import ApiFormat
from ecs_agent.providers.model_constructor import Model
from ecs_agent.types import CompletionResult, Message
from tests.support.models import LiveModelFactory

pytestmark = [pytest.mark.live(capabilities=["text"]), pytest.mark.smoke]
_GREETING = "Say 'hello world' in exactly those two words and nothing else."


async def _complete(factory: LiveModelFactory, api_format: ApiFormat) -> str:
    """Exercise the explicit api_format constructor branch with owned cleanup."""
    model = factory.track(
        Model(factory.profile.model, api_format=api_format, **factory.connection())
    )
    result = await model.complete([Message(role="user", content=_GREETING)])
    assert isinstance(result, CompletionResult)
    assert result.message.content
    return result.message.content


@pytest.mark.live(formats=["anthropic_messages"])
async def test_model_anthropic_messages_live(
    live_model_factory: LiveModelFactory,
) -> None:
    content = await _complete(live_model_factory, ApiFormat.ANTHROPIC_MESSAGES)
    assert len(content.strip()) > 0


@pytest.mark.live(formats=["anthropic_messages"])
async def test_model_anthropic_messages_via_model_type_live(
    live_model_factory: LiveModelFactory,
) -> None:
    factory = live_model_factory
    model = factory.track(
        Model(factory.profile.model, model_type="claude", **factory.connection())
    )
    result = await model.complete([Message(role="user", content=_GREETING)])
    assert isinstance(result, CompletionResult)
    assert result.message.content


@pytest.mark.live(formats=["openai_chat_completions"])
async def test_model_openai_chat_completions_live(
    live_model_factory: LiveModelFactory,
) -> None:
    content = await _complete(live_model_factory, ApiFormat.OPENAI_CHAT_COMPLETIONS)
    assert len(content.strip()) > 0


@pytest.mark.live(formats=["openai_chat_completions"])
async def test_model_openai_chat_completions_via_model_type_live(
    live_model_factory: LiveModelFactory,
) -> None:
    factory = live_model_factory
    model = factory.track(
        Model(factory.profile.model, model_type="openai", **factory.connection())
    )
    result = await model.complete([Message(role="user", content=_GREETING)])
    assert isinstance(result, CompletionResult)
    assert result.message.content


@pytest.mark.live(formats=["openai_responses"])
async def test_model_openai_responses_api_live(
    live_model_factory: LiveModelFactory,
) -> None:
    content = await _complete(live_model_factory, ApiFormat.OPENAI_RESPONSES)
    assert len(content.strip()) > 0


@pytest.mark.live(formats=["openai_chat_completions"])
async def test_model_api_format_string_live(
    live_model_factory: LiveModelFactory,
) -> None:
    factory = live_model_factory
    model = factory.track(
        Model(
            factory.profile.model,
            api_format="openai_chat_completions",
            **factory.connection(),
        )
    )
    result = await model.complete([Message(role="user", content=_GREETING)])
    assert isinstance(result, CompletionResult)
    assert result.message.content


@pytest.mark.live(formats=["openai_chat_completions"])
async def test_model_model_id_preserved_after_live_call(
    live_model_factory: LiveModelFactory,
) -> None:
    factory = live_model_factory
    model_id = factory.profile.model
    model = factory.track(
        Model(
            model_id,
            api_format=ApiFormat.OPENAI_CHAT_COMPLETIONS,
            **factory.connection(),
        )
    )
    assert model.model_id == model_id
    result = await model.complete([Message(role="user", content="Hi")])
    assert isinstance(result, CompletionResult)
    assert model.model_id == model_id
