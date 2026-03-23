from __future__ import annotations

from ecs_agent.components import ConversationComponent, UserPromptConfigComponent
from ecs_agent.components.definitions import (
    ContextEntry,
    PromptContextQueueComponent,
    RenderedUserPromptComponent,
)
from ecs_agent.core import World
from ecs_agent.prompts.contracts import TriggerSpec
from ecs_agent.systems.user_prompt_normalization_system import (
    UserPromptNormalizationSystem,
)
from ecs_agent.types import Message


async def test_no_trigger_passthrough() -> None:
    world = World()
    entity_id = world.create_entity()
    world.add_component(
        entity_id,
        ConversationComponent(messages=[Message(role="user", content="hello")]),
    )

    await UserPromptNormalizationSystem().process(world)

    rendered = world.get_component(entity_id, RenderedUserPromptComponent)
    assert rendered is not None
    assert rendered.text == "hello"


async def test_keyword_trigger_prepends_content() -> None:
    world = World()
    entity_id = world.create_entity()
    world.add_component(
        entity_id,
        ConversationComponent(messages=[Message(role="user", content="@greet please")]),
    )
    world.add_component(
        entity_id,
        UserPromptConfigComponent(triggers={"@greet": "Be greeting"}),
    )

    await UserPromptNormalizationSystem().process(world)

    rendered = world.get_component(entity_id, RenderedUserPromptComponent)
    assert rendered is not None
    assert rendered.text.startswith("[PROMPT_INJECT:@greet]\nBe greeting")
    assert rendered.text.endswith("@greet please")


async def test_trigger_replace_action() -> None:
    trigger = TriggerSpec(
        pattern="@rewrite",
        match_mode="keyword",
        action="replace",
        content="Replacement prompt",
    )

    rendered_text = UserPromptNormalizationSystem.apply_trigger_specs(
        user_text="@rewrite this",
        trigger_specs=[trigger],
    )

    assert rendered_text == "Replacement prompt"


async def test_no_user_message_skips() -> None:
    world = World()
    entity_id = world.create_entity()
    world.add_component(
        entity_id,
        ConversationComponent(messages=[Message(role="assistant", content="hello")]),
    )

    await UserPromptNormalizationSystem().process(world)

    rendered = world.get_component(entity_id, RenderedUserPromptComponent)
    assert rendered is None


async def test_rendered_text_is_transient_not_stored() -> None:
    world = World()
    entity_id = world.create_entity()
    original_messages = [Message(role="user", content="@greet please")]
    world.add_component(
        entity_id, ConversationComponent(messages=list(original_messages))
    )
    world.add_component(
        entity_id,
        UserPromptConfigComponent(triggers={"@greet": "Be greeting"}),
    )

    await UserPromptNormalizationSystem().process(world)

    conversation = world.get_component(entity_id, ConversationComponent)
    rendered = world.get_component(entity_id, RenderedUserPromptComponent)
    assert conversation is not None
    assert rendered is not None
    assert conversation.messages == original_messages
    assert rendered.text != original_messages[-1].content


async def test_duplicate_injection_marker_not_doubled() -> None:
    world = World()
    entity_id = world.create_entity()
    already_injected = "[PROMPT_INJECT:@greet]\nBe greeting\n\n@greet please"
    world.add_component(
        entity_id,
        ConversationComponent(
            messages=[Message(role="user", content=already_injected)]
        ),
    )
    world.add_component(
        entity_id,
        UserPromptConfigComponent(triggers={"@greet": "Be greeting"}),
    )

    await UserPromptNormalizationSystem().process(world)

    rendered = world.get_component(entity_id, RenderedUserPromptComponent)
    assert rendered is not None
    assert rendered.text.count("[PROMPT_INJECT:") == 1


async def test_empty_triggers_passthrough() -> None:
    world = World()
    entity_id = world.create_entity()
    world.add_component(
        entity_id,
        ConversationComponent(messages=[Message(role="user", content="hello")]),
    )
    world.add_component(entity_id, UserPromptConfigComponent(triggers={}))

    await UserPromptNormalizationSystem().process(world)

    rendered = world.get_component(entity_id, RenderedUserPromptComponent)
    assert rendered is not None
    assert rendered.text == "hello"


async def test_context_pool_entries_are_not_injected_by_normalization_system() -> None:
    world = World()
    entity_id = world.create_entity()
    world.add_component(
        entity_id,
        ConversationComponent(messages=[Message(role="user", content="Need summary")]),
    )
    world.add_component(entity_id, UserPromptConfigComponent(enable_context_pool=True))
    world.add_component(
        entity_id,
        PromptContextQueueComponent(
            entries=[
                ContextEntry(
                    entry_id="tool-one-0",
                    priority=30,
                    registration_order=0,
                    source_label="tool:one",
                    content="source: tool:one\nresult: A",
                ),
                ContextEntry(
                    entry_id="subagent-two-1",
                    priority=20,
                    registration_order=1,
                    source_label="subagent:two",
                    content="source: subagent:two\nresult: B",
                ),
            ]
        ),
    )

    await UserPromptNormalizationSystem().process(world)

    rendered = world.get_component(entity_id, RenderedUserPromptComponent)
    assert rendered is not None
    assert rendered.text == "Need summary"


async def test_stale_rendered_prompt_removed_when_no_latest_user_message() -> None:
    world = World()
    entity_id = world.create_entity()
    conversation = ConversationComponent(
        messages=[Message(role="user", content="hello")]
    )
    world.add_component(entity_id, conversation)

    system = UserPromptNormalizationSystem()
    await system.process(world)

    initial_rendered = world.get_component(entity_id, RenderedUserPromptComponent)
    assert initial_rendered is not None
    assert initial_rendered.text == "hello"

    conversation.messages = [Message(role="assistant", content="done")]
    await system.process(world)

    stale_rendered = world.get_component(entity_id, RenderedUserPromptComponent)
    assert stale_rendered is None



async def test_rendered_user_prompt_component_accepts_only_text() -> None:
    """Verify RenderedUserPromptComponent can be constructed with only text field."""
    component = RenderedUserPromptComponent(text="hello world")
    assert component.text == "hello world"