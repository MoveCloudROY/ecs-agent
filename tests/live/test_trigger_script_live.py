"""Live LLM tests for TriggerSpec script action.

Select a profile with --live; see docs/testing.md.
"""

from __future__ import annotations


import pytest

from ecs_agent.components import (
    ConversationComponent,
    LLMComponent,
    UserPromptConfigComponent,
)
from ecs_agent.components.definitions import KVStoreComponent
from ecs_agent.core import Runner, World
from ecs_agent.prompts.contracts import TriggerSpec
from ecs_agent.systems.error_handling import ErrorHandlingSystem
from ecs_agent.systems.reasoning import ReasoningSystem
from ecs_agent.systems.user_prompt_normalization_system import (
    UserPromptNormalizationSystem,
)
from ecs_agent.types import EntityId, Message

from tests.support.models import LiveModelFactory


@pytest.mark.live(
    formats=["openai_chat_completions", "openai_responses"],
    capabilities=["text", "tools"],
)
@pytest.mark.prompts
@pytest.mark.asyncio
async def test_live_script_action_rewrites_prompt_before_llm(
    *, live_model_factory: LiveModelFactory
) -> None:
    model = live_model_factory.openai()

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
        LLMComponent(model=model, system_prompt="You are a helpful assistant."),
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


@pytest.mark.live(
    formats=["openai_chat_completions", "openai_responses"],
    capabilities=["text", "tools"],
)
@pytest.mark.prompts
@pytest.mark.asyncio
async def test_live_script_action_world_mutation_persists(
    *, live_model_factory: LiveModelFactory
) -> None:
    model = live_model_factory.openai()

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
        LLMComponent(model=model, system_prompt="You are a helpful assistant."),
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
