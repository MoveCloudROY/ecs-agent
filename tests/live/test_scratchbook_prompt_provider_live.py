"""Capture real outbound prompts with HTTP hooks for both OpenAI protocols."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from ecs_agent.components import ConversationComponent, LLMComponent
from ecs_agent.components.definitions import RenderedSystemPromptComponent
from ecs_agent.core import Runner, World
from ecs_agent.prompts.contracts import PromptTemplateSource, SystemPromptConfigSpec
from ecs_agent.providers.config import ApiFormat
from ecs_agent.scratchbook.prompt_definition import (
    ScratchbookArtifactPromptDef,
    ScratchbookPromptConfig,
)
from ecs_agent.systems.error_handling import ErrorHandlingSystem
from ecs_agent.systems.reasoning import ReasoningSystem
from ecs_agent.systems.system_prompt_render_system import SystemPromptRenderSystem
from ecs_agent.types import Message

from tests.support.models import LiveModelFactory


_ENDPOINT_PARAMS = [
    pytest.param(ApiFormat.OPENAI_CHAT_COMPLETIONS, id="chat-completions"),
    pytest.param(ApiFormat.OPENAI_RESPONSES, id="responses"),
]


def _build_scratchbook_prompt_config() -> ScratchbookPromptConfig:
    return ScratchbookPromptConfig(
        overview_default_template=(
            "Scratchbook path: ${scratchbook_path}\n"
            "Artifact types:\n${artifact_types}\n"
            "Use builtin tools to inspect or update artifacts when allowed."
        ),
        scratchbook_root_path=".sisyphus/notepads/scratchbook-prompt-model",
        artifacts=[
            ScratchbookArtifactPromptDef(
                artifact_type_id="plan notes",
                path=".sisyphus/notepads/scratchbook-prompt-model/plan.md",
                purpose="Tracks active task intent and rationale.",
                readonly=False,
                read_when="Before changing task behavior.",
            ),
            ScratchbookArtifactPromptDef(
                artifact_type_id="run log",
                path=".sisyphus/notepads/scratchbook-prompt-model/live-run.log",
                purpose="Stores immutable live execution traces.",
                readonly=True,
                read_when="When validating endpoint-mode behavior.",
            ),
        ],
    )


@pytest.mark.live(formats=["openai_chat_completions"], capabilities=["text", "tools"])
@pytest.mark.prompts
@pytest.mark.asyncio
@pytest.mark.parametrize("api_format", _ENDPOINT_PARAMS)
async def test_live_scratchbook_rendered_prompt_reaches_aliyun_outbound_channel(
    api_format: ApiFormat, *, live_model_factory: LiveModelFactory
) -> None:
    model = live_model_factory.openai()
    captured: dict[str, Any] = {}

    async def capture(request: httpx.Request) -> None:
        captured["url"] = str(request.url)
        captured["payload"] = json.loads(request.content)

    model._client.event_hooks["request"].append(capture)

    world = World()
    entity_id = world.create_entity()
    world.add_component(
        entity_id,
        LLMComponent(
            model=model,
        ),
    )
    world.add_component(
        entity_id,
        ConversationComponent(
            messages=[Message(role="user", content="Respond with a short greeting.")]
        ),
    )
    world.add_component(
        entity_id,
        SystemPromptConfigSpec(
            template_source=PromptTemplateSource(
                inline=(
                    "SB-LIVE-OUTBOUND\n"
                    "${_scratchbook_overview}\n"
                    "${_scratchbook_artifact_plan_notes}\n"
                    "${_scratchbook_artifact_run_log}"
                )
            )
        ),
    )
    world.add_component(entity_id, _build_scratchbook_prompt_config())

    world.register_system(SystemPromptRenderSystem(), priority=-20)
    world.register_system(ReasoningSystem(priority=0), priority=0)
    world.register_system(ErrorHandlingSystem(priority=99), priority=99)

    await Runner().run(world, max_ticks=1)

    rendered = world.get_component(entity_id, RenderedSystemPromptComponent)
    assert rendered is not None
    assert "SB-LIVE-OUTBOUND" in rendered.text

    outbound_url = captured.get("url")
    outbound_payload = captured.get("payload")
    assert outbound_url is not None
    assert outbound_payload is not None

    if api_format == ApiFormat.OPENAI_CHAT_COMPLETIONS:
        assert outbound_url.endswith("/chat/completions")
        messages = outbound_payload.get("messages")
        assert isinstance(messages, list)
        assert len(messages) >= 1
        first_message = messages[0]
        assert isinstance(first_message, dict)
        assert first_message.get("role") == "system"
        system_content = first_message.get("content")
        assert isinstance(system_content, str)
        assert system_content == rendered.text
        assert "SB-LIVE-OUTBOUND" in system_content
        assert "plan_notes" in system_content.lower()
    else:
        assert outbound_url.endswith("/responses")
        instructions = outbound_payload.get("instructions")
        assert isinstance(instructions, str)
        assert instructions == rendered.text
        assert "SB-LIVE-OUTBOUND" in instructions
        assert "run_log" in instructions.lower()

        input_items = outbound_payload.get("input")
        assert isinstance(input_items, list)
        assert any(
            isinstance(item, dict)
            and item.get("type") == "message"
            and item.get("role") == "user"
            for item in input_items
        )

    conversation = world.get_component(entity_id, ConversationComponent)
    assert conversation is not None
    assistant_message = next(
        (
            message
            for message in reversed(conversation.messages)
            if message.role == "assistant"
        ),
        None,
    )
    assert assistant_message is not None
    assert len(assistant_message.content.strip()) > 0
