from __future__ import annotations

import importlib
import os
from typing import Any
from unittest.mock import AsyncMock, patch

from ecs_agent.components import ConversationComponent
from ecs_agent.components.definitions import (
    RenderedSystemPromptComponent,
    RenderedUserPromptComponent,
)
from ecs_agent.providers.fake_model import FakeModel
from ecs_agent.providers.openai_chat_adapter import OpenAIChatAdapter
from ecs_agent.types import CompletionResult, ImageUrlPart, Message


def _load_module() -> Any:
    return importlib.import_module("examples.vision_agent")


def _fake_completion(content: str) -> CompletionResult:
    return CompletionResult(message=Message(role="assistant", content=content))


async def test_vision_agent_builds_multimodal_message() -> None:
    module = _load_module()
    captured_conversation: ConversationComponent | None = None

    original_add_component = module.World.add_component

    def capture_add_component(self: Any, entity_id: Any, component: Any) -> None:
        nonlocal captured_conversation
        if isinstance(component, ConversationComponent):
            captured_conversation = component
        original_add_component(self, entity_id, component)

    with patch.dict(os.environ, {}, clear=True):
        with patch.object(module.World, "add_component", capture_add_component):
            with patch.object(module.Runner, "run", AsyncMock(return_value=None)):
                await module.main()

    assert captured_conversation is not None
    assert captured_conversation.messages
    first_message = captured_conversation.messages[0]
    assert first_message.role == "user"
    # content must be non-empty so UserPromptNormalizationSystem can process it
    assert first_message.content == "Describe this image in detail."
    assert first_message.parts is not None
    assert any(isinstance(part, ImageUrlPart) for part in first_message.parts)


async def test_vision_agent_uses_system_prompt_and_user_normalization_systems() -> None:
    """SystemPromptRenderSystem and UserPromptNormalizationSystem must be registered."""
    module = _load_module()
    from ecs_agent.systems.system_prompt_render_system import SystemPromptRenderSystem
    from ecs_agent.systems.user_prompt_normalization_system import (
        UserPromptNormalizationSystem,
    )

    registered_system_types: list[type] = []
    original_register = module.World.register_system

    def capture_register(self: Any, system: Any, priority: int = 0) -> None:
        registered_system_types.append(type(system))
        original_register(self, system, priority=priority)

    with patch.dict(os.environ, {}, clear=True):
        with patch.object(module.World, "register_system", capture_register):
            with patch.object(module.Runner, "run", AsyncMock(return_value=None)):
                await module.main()

    assert SystemPromptRenderSystem in registered_system_types, (
        "SystemPromptRenderSystem must be registered"
    )
    assert UserPromptNormalizationSystem in registered_system_types, (
        "UserPromptNormalizationSystem must be registered"
    )


async def test_vision_agent_receives_description() -> None:
    module = _load_module()
    expected = "A dog and a girl are in the image."
    captured_world: Any = None

    original_world_init = module.World.__init__

    def capture_world_init(self: Any, *args: Any, **kwargs: Any) -> None:
        nonlocal captured_world
        original_world_init(self, *args, **kwargs)
        captured_world = self

    with patch.dict(os.environ, {}, clear=True):
        with patch(
            "examples.vision_agent.FakeModel",
            return_value=FakeModel(responses=[_fake_completion(expected)]),
        ):
            with patch.object(module.World, "__init__", capture_world_init):
                await module.main()

    assert captured_world is not None
    assert captured_world._entity_ids

    first_entity_id = next(iter(captured_world._entity_ids))
    conversation = captured_world.get_component(first_entity_id, ConversationComponent)
    assert conversation is not None
    assert expected in conversation.messages[-1].content


async def test_vision_agent_fake_mode_no_api_key() -> None:
    module = _load_module()

    with patch.dict(os.environ, {}, clear=True):
        with patch(
            "examples.vision_agent.FakeModel",
            side_effect=FakeModel,
        ) as fake_ctor:
            with patch(
                "examples.vision_agent.OpenAIModel", create=True
            ) as openai_ctor:
                await module.main()

    fake_ctor.assert_called_once()
    openai_ctor.assert_not_called()


async def test_vision_agent_system_prompt_rendered() -> None:
    """SystemPromptRenderSystem must produce a RenderedSystemPromptComponent."""
    module = _load_module()
    captured_world: Any = None

    original_world_init = module.World.__init__

    def capture_world_init(self: Any, *args: Any, **kwargs: Any) -> None:
        nonlocal captured_world
        original_world_init(self, *args, **kwargs)
        captured_world = self

    with patch.dict(os.environ, {}, clear=True):
        with patch.object(module.World, "__init__", capture_world_init):
            await module.main()

    assert captured_world is not None
    first_entity_id = next(iter(captured_world._entity_ids))
    rendered_system = captured_world.get_component(
        first_entity_id, RenderedSystemPromptComponent
    )
    assert rendered_system is not None
    assert "vision assistant" in rendered_system.text


def test_vision_agent_image_url_part_serialization() -> None:
    class _ProviderStub:
        _api_key = "test"
        _base_url = "https://example.com"
        _model = "test-model"
        _client = None
        _timeout = None

        def _build_headers(self) -> dict[str, str]:
            return {}

        def _handle_http_error(self, exc: Exception) -> None:
            raise RuntimeError(str(exc))

        def _handle_request_error(self, exc: Exception) -> None:
            raise RuntimeError(str(exc))

        def _convert_tools_to_openai(self, tools: list[Any]) -> list[dict[str, Any]]:
            return []

        def _usage_from_raw(self, usage_data: Any) -> Any:
            return None

    adapter = OpenAIChatAdapter(_ProviderStub())
    msg = Message(
        role="user",
        content="Describe this image in detail.",
        parts=[
            ImageUrlPart(url="https://example.com/img.jpg"),
        ],
    )

    payload = adapter._convert_message_content(msg)
    assert isinstance(payload, list)
    assert {
        "type": "image_url",
        "image_url": {"url": "https://example.com/img.jpg"},
    } in payload
