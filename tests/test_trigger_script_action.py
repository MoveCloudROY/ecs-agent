from __future__ import annotations

from typing import get_args

from ecs_agent.components import ConversationComponent, UserPromptConfigComponent
from ecs_agent.components.definitions import (
    KVStoreComponent,
    RenderedUserPromptComponent,
)
from ecs_agent.core import World
from ecs_agent.prompts.contracts import TriggerSpec
from ecs_agent.systems.user_prompt_normalization_system import (
    UserPromptNormalizationSystem,
)
from ecs_agent.types import EntityId, Message


def test_trigger_spec_accepts_script_action() -> None:
    assert "script" in get_args(TriggerSpec.__annotations__["action"])

    spec = TriggerSpec(
        pattern="@run",
        match_mode="keyword",
        action="script",
        content="my_handler",
    )
    assert spec.action == "script"
    assert spec.content == "my_handler"


def test_user_prompt_config_has_script_handlers_field() -> None:
    comp = UserPromptConfigComponent()
    assert hasattr(comp, "script_handlers")
    assert isinstance(comp.script_handlers, dict)
    assert len(comp.script_handlers) == 0


def test_user_prompt_config_accepts_script_handler() -> None:
    async def my_handler(
        world: object, entity_id: object, user_text: str
    ) -> str | None:
        return "rewritten"

    trigger = TriggerSpec(
        pattern="@run",
        match_mode="keyword",
        action="script",
        content="my_handler",
    )
    comp = UserPromptConfigComponent(
        triggers=[trigger],
        script_handlers={"my_handler": my_handler},
    )
    assert "my_handler" in comp.script_handlers


def test_user_prompt_config_positional_backward_compat() -> None:
    triggers = []
    comp = UserPromptConfigComponent(triggers, True, 4096)
    assert comp.enable_context_pool is True
    assert comp.context_pool_max_chars == 4096
    assert comp.script_handlers == {}


async def test_script_action_returns_rewritten_text() -> None:
    async def rewrite_handler(
        world: World,
        entity_id: EntityId,
        user_text: str,
    ) -> str | None:
        return f"[rewritten] {user_text}"

    trigger = TriggerSpec(
        pattern="@run",
        match_mode="keyword",
        action="script",
        content="rewrite_handler",
    )
    world = World()
    entity_id = world.create_entity()
    world.add_component(
        entity_id,
        ConversationComponent(
            messages=[Message(role="user", content="@run do something")]
        ),
    )
    world.add_component(
        entity_id,
        UserPromptConfigComponent(
            triggers=[trigger],
            script_handlers={"rewrite_handler": rewrite_handler},
        ),
    )

    await UserPromptNormalizationSystem().process(world)

    rendered = world.get_component(entity_id, RenderedUserPromptComponent)
    assert rendered is not None
    assert rendered.text == "[rewritten] @run do something"


async def test_script_action_none_return_keeps_original() -> None:
    async def passthrough_handler(
        world: World,
        entity_id: EntityId,
        user_text: str,
    ) -> str | None:
        return None

    trigger = TriggerSpec(
        pattern="@noop",
        match_mode="keyword",
        action="script",
        content="passthrough_handler",
    )
    world = World()
    entity_id = world.create_entity()
    world.add_component(
        entity_id,
        ConversationComponent(messages=[Message(role="user", content="@noop hello")]),
    )
    world.add_component(
        entity_id,
        UserPromptConfigComponent(
            triggers=[trigger],
            script_handlers={"passthrough_handler": passthrough_handler},
        ),
    )

    await UserPromptNormalizationSystem().process(world)

    rendered = world.get_component(entity_id, RenderedUserPromptComponent)
    assert rendered is not None
    assert rendered.text == "@noop hello"


async def test_script_action_world_mutation() -> None:
    async def mutating_handler(
        world: World,
        entity_id: EntityId,
        user_text: str,
    ) -> str | None:
        world.add_component(entity_id, KVStoreComponent(store={"triggered": True}))
        return "mutated"

    trigger = TriggerSpec(
        pattern="@mutate",
        match_mode="keyword",
        action="script",
        content="mutating_handler",
    )
    world = World()
    entity_id = world.create_entity()
    world.add_component(
        entity_id,
        ConversationComponent(messages=[Message(role="user", content="@mutate now")]),
    )
    world.add_component(
        entity_id,
        UserPromptConfigComponent(
            triggers=[trigger],
            script_handlers={"mutating_handler": mutating_handler},
        ),
    )

    await UserPromptNormalizationSystem().process(world)

    rendered = world.get_component(entity_id, RenderedUserPromptComponent)
    assert rendered is not None
    assert rendered.text == "mutated"

    kv = world.get_component(entity_id, KVStoreComponent)
    assert kv is not None
    assert kv.store["triggered"] is True


async def test_script_action_missing_handler_keeps_original() -> None:
    trigger = TriggerSpec(
        pattern="@ghost",
        match_mode="keyword",
        action="script",
        content="nonexistent_handler",
    )
    world = World()
    entity_id = world.create_entity()
    world.add_component(
        entity_id,
        ConversationComponent(messages=[Message(role="user", content="@ghost call")]),
    )
    world.add_component(
        entity_id,
        UserPromptConfigComponent(
            triggers=[trigger],
            script_handlers={},
        ),
    )

    await UserPromptNormalizationSystem().process(world)

    rendered = world.get_component(entity_id, RenderedUserPromptComponent)
    assert rendered is not None
    assert rendered.text == "@ghost call"


async def test_script_action_priority_over_inject() -> None:
    async def high_handler(
        world: World,
        entity_id: EntityId,
        user_text: str,
    ) -> str | None:
        return "script_won"

    script_trigger = TriggerSpec(
        pattern="@x",
        match_mode="keyword",
        action="script",
        content="high_handler",
        priority=10,
    )
    inject_trigger = TriggerSpec(
        pattern="@x",
        match_mode="keyword",
        action="inject",
        content="inject content",
        priority=5,
    )
    world = World()
    entity_id = world.create_entity()
    world.add_component(
        entity_id,
        ConversationComponent(messages=[Message(role="user", content="@x test")]),
    )
    world.add_component(
        entity_id,
        UserPromptConfigComponent(
            triggers=[inject_trigger, script_trigger],
            script_handlers={"high_handler": high_handler},
        ),
    )

    await UserPromptNormalizationSystem().process(world)

    rendered = world.get_component(entity_id, RenderedUserPromptComponent)
    assert rendered is not None
    assert rendered.text == "script_won"
