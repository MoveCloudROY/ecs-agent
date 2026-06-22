"""Live LLM tests for TriggerSpec script action.

Environment variables required (skip gracefully if absent):
    LLM_API_KEY   — Aliyun DashScope key
    LLM_BASE_URL  — https://dashscope.aliyuncs.com/compatible-mode/v1
    LLM_MODEL     — qwen3.5-flash

Run:
    LLM_API_KEY=sk-... LLM_BASE_URL=... LLM_MODEL=qwen3.5-flash \
        uv run pytest tests/live/test_trigger_script_live.py -v
"""

from __future__ import annotations

import os

import pytest

from ecs_agent.components import (
    ConversationComponent,
    LLMComponent,
    UserPromptConfigComponent,
)
from ecs_agent.components.definitions import KVStoreComponent
from ecs_agent.core import Runner, World
from ecs_agent.prompts.contracts import TriggerSpec
from ecs_agent.providers import OpenAIModel
from ecs_agent.providers.config import ApiFormat, ProviderConfig
from ecs_agent.systems.error_handling import ErrorHandlingSystem
from ecs_agent.systems.reasoning import ReasoningSystem
from ecs_agent.systems.user_prompt_normalization_system import (
    UserPromptNormalizationSystem,
)
from ecs_agent.types import EntityId, Message


def _make_provider(api_key: str, base_url: str, model: str) -> OpenAIModel:
    return OpenAIModel(
        config=ProviderConfig(
            provider_id="aliyun",
            base_url=base_url,
            api_key=api_key,
            api_format=ApiFormat.OPENAI_CHAT_COMPLETIONS,
        ),
        model=model,
    )


@pytest.mark.asyncio
async def test_live_script_action_rewrites_prompt_before_llm(live_api_key: str) -> None:
    base_url = os.getenv(
        "LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    model = os.getenv("LLM_MODEL") or "qwen3.5-flash"
    model = _make_provider(api_key=live_api_key, base_url=base_url, model=model)

    async def tag_handler(
        world: World,
        entity_id: EntityId,
        user_text: str,
    ) -> str | None:
        return f"{user_text}\n\n(Reply with exactly: TAGGED_OK)"

    trigger = TriggerSpec(
        pattern="@tag",
        match_mode="keyword",
        action="script",
        content="tag_handler",
    )

    world = World()
    entity = world.create_entity()
    world.add_component(
        entity,
        LLMComponent(
            model=model, system_prompt="You are a helpful assistant."
        ),
    )
    world.add_component(
        entity,
        ConversationComponent(
            messages=[Message(role="user", content="@tag test message")]
        ),
    )
    world.add_component(
        entity,
        UserPromptConfigComponent(
            triggers=[trigger],
            script_handlers={"tag_handler": tag_handler},
        ),
    )

    world.register_system(UserPromptNormalizationSystem(), priority=-10)
    world.register_system(ReasoningSystem(priority=0), priority=0)
    world.register_system(ErrorHandlingSystem(priority=99), priority=99)

    runner = Runner()
    await runner.run(world, max_ticks=3)

    conv = world.get_component(entity, ConversationComponent)
    assert conv is not None
    last_assistant = next(
        (m for m in reversed(conv.messages) if m.role == "assistant"), None
    )
    assert last_assistant is not None
    assert "TAGGED_OK" in last_assistant.content


@pytest.mark.asyncio
async def test_live_script_action_world_mutation_persists(live_api_key: str) -> None:
    base_url = os.getenv(
        "LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    model = os.getenv("LLM_MODEL") or "qwen3.5-flash"
    model = _make_provider(api_key=live_api_key, base_url=base_url, model=model)

    async def store_handler(
        world: World,
        entity_id: EntityId,
        user_text: str,
    ) -> str | None:
        world.add_component(
            entity_id,
            KVStoreComponent(store={"script_ran": True, "original": user_text}),
        )
        return "Say: MUTATION_CONFIRMED"

    trigger = TriggerSpec(
        pattern="@store",
        match_mode="keyword",
        action="script",
        content="store_handler",
    )

    world = World()
    entity = world.create_entity()
    world.add_component(
        entity,
        LLMComponent(
            model=model, system_prompt="You are a helpful assistant."
        ),
    )
    world.add_component(
        entity,
        ConversationComponent(
            messages=[Message(role="user", content="@store my data")]
        ),
    )
    world.add_component(
        entity,
        UserPromptConfigComponent(
            triggers=[trigger],
            script_handlers={"store_handler": store_handler},
        ),
    )

    world.register_system(UserPromptNormalizationSystem(), priority=-10)
    world.register_system(ReasoningSystem(priority=0), priority=0)
    world.register_system(ErrorHandlingSystem(priority=99), priority=99)

    runner = Runner()
    await runner.run(world, max_ticks=3)

    kv = world.get_component(entity, KVStoreComponent)
    assert kv is not None
    assert kv.store["script_ran"] is True
    assert kv.store["original"] == "@store my data"

    conv = world.get_component(entity, ConversationComponent)
    assert conv is not None
    last_assistant = next(
        (m for m in reversed(conv.messages) if m.role == "assistant"), None
    )
    assert last_assistant is not None
    assert "MUTATION_CONFIRMED" in last_assistant.content
