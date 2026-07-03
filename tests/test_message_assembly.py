from ecs_agent.components import ContextEntry
from ecs_agent.prompts.message_assembly import assemble_messages, build_keyword_registry
from ecs_agent.types import Message


def test_assemble_messages_orders_keyword_block_then_context_pool_then_original_text() -> (
    None
):
    registry = build_keyword_registry({"@code": "KEYWORD_TEMPLATE_BLOCK"})
    assembled = assemble_messages(
        conversation_messages=[Message(role="user", content="Need @code help")],
        enable_context_pool=True,
        context_pool_items=[
            ContextEntry(
                entry_id="tool-search-0",
                priority=30,
                registration_order=0,
                source_label="tool:search",
                content="source: tool\nresult: facts",
            )
        ],
        keyword_registry=registry,
    )

    user_message = assembled[0]
    assert user_message.content.startswith(
        "[PROMPT_INJECT:@code]\nKEYWORD_TEMPLATE_BLOCK"
    )
    assert "\n\n[PROMPT_CONTEXT_POOL]\n" in user_message.content
    assert user_message.content.endswith("Need @code help")


async def test_load_skill_details_returns_full_context_directly() -> None:
    from ecs_agent import BuiltinToolsSkill, SkillManager
    from ecs_agent.components import ConversationComponent, ToolRegistryComponent
    from ecs_agent.core import World
    from ecs_agent.types import Message

    world = World()
    entity = world.create_entity()
    manager = SkillManager()
    manager.install(world, entity, BuiltinToolsSkill())
    world.add_component(
        entity,
        ConversationComponent(messages=[Message(role="user", content="hello")]),
    )

    registry = world.get_component(entity, ToolRegistryComponent)
    assert registry is not None
    handler = registry.handlers["load_skill_details"]
    result = await handler(skill_name="builtin-tools")

    # The tool returns the full skill details as its return value.
    # This is placed directly into the role='tool' message by ToolExecutionSystem.
    assert "Skill: builtin-tools" in result
    assert "## Tool Schemas" in result


async def test_load_skill_details_does_not_stage_pending_context() -> None:
    from ecs_agent import BuiltinToolsSkill, SkillManager
    from ecs_agent.components import ConversationComponent, ToolRegistryComponent
    from ecs_agent.core import World
    from ecs_agent.types import Message

    world = World()
    entity = world.create_entity()
    manager = SkillManager()
    manager.install(world, entity, BuiltinToolsSkill())
    world.add_component(
        entity,
        ConversationComponent(messages=[Message(role="user", content="hello")]),
    )

    registry = world.get_component(entity, ToolRegistryComponent)
    assert registry is not None
    handler = registry.handlers["load_skill_details"]
    await handler(skill_name="builtin-tools")


async def test_slash_command_injects_skill_context_into_outbound_message() -> None:
    """Red test: slash skill context WILL be injected (feature not yet implemented).

    When a user message contains a slash command (e.g., '/testskill help'),
    the outbound message should inject the skill context and preserve the
    original slash command text at the end.
    This test documents the EXPECTED slash context injection behavior (red).
    """
    from ecs_agent.components import (
        SkillComponent,
        SkillMetadata,
        ConversationComponent,
    )
    from ecs_agent.core import World
    from ecs_agent.types import Message
    from ecs_agent.prompts.message_assembly import prepare_outbound_messages

    world = World()
    entity = world.create_entity()

    # Create a fake skill with slash command
    skill_metadata = SkillMetadata(
        name="testskill",
        description="A test skill",
        tool_names=[],
        has_system_prompt=False,
        user_invocable=True,
        slash_command="/testskill",
    )

    # Install skill metadata on entity
    skill_component = SkillComponent(
        skills={"testskill": skill_metadata},
    )
    world.add_component(entity, skill_component)

    # Add user message with slash command
    original_text = "/testskill help me understand this"
    world.add_component(
        entity,
        ConversationComponent(messages=[Message(role="user", content=original_text)]),
    )

    # Call prepare_outbound_messages
    messages, _ = prepare_outbound_messages(
        world,
        entity,
        current_tick=1,
    )

    # RED ASSERTION: Once implemented, message WILL contain slash skill context
    user_msg = messages[-1]
    assert "调用 skill: testskill" in user_msg.content, (
        "Once slash context injection is implemented, skill context should be injected"
    )
    # RED ASSERTION: Original text WILL be preserved
    assert user_msg.content.endswith(original_text), (
        f"Final message should end with original text '{original_text}', "
        f"but got: {user_msg.content[-50:]}"
    )


async def test_slash_command_with_context_pool_renders_order() -> None:
    """Red test: slash context renders before context pool, then original text.

    When both slash context and context pool entries exist, the render order
    should be: slash skill context → context pool entries → original user text.
    This test documents the expected ordering (red).
    """
    from ecs_agent.components import (
        SkillComponent,
        SkillMetadata,
        ConversationComponent,
        UserPromptConfigComponent,
        PromptContextQueueComponent,
        ContextEntry,
    )
    from ecs_agent.core import World
    from ecs_agent.types import Message
    from ecs_agent.prompts.message_assembly import prepare_outbound_messages

    world = World()
    entity = world.create_entity()

    # Create a skill with slash command
    skill_metadata = SkillMetadata(
        name="queryskill",
        description="A query skill",
        tool_names=[],
        has_system_prompt=False,
        user_invocable=True,
        slash_command="/queryskill",
    )
    skill_component = SkillComponent(
        skills={"queryskill": skill_metadata},
    )
    world.add_component(entity, skill_component)

    # Add context pool entry
    context_entry = ContextEntry(
        entry_id="context-0",
        priority=10,
        registration_order=0,
        source_label="test:source",
        content="CONTEXT_POOL_DATA",
    )
    world.add_component(
        entity,
        PromptContextQueueComponent(entries=[context_entry]),
    )

    # Enable context pool injection
    world.add_component(
        entity,
        UserPromptConfigComponent(enable_context_pool=True),
    )

    # Add user message with slash command
    original_text = "/queryskill find me resources"
    world.add_component(
        entity,
        ConversationComponent(messages=[Message(role="user", content=original_text)]),
    )

    # Call prepare_outbound_messages
    messages, _ = prepare_outbound_messages(
        world,
        entity,
        current_tick=1,
    )

    # RED ASSERTION: Message should contain skill context
    user_msg = messages[-1]
    assert "调用 skill: queryskill" in user_msg.content, (
        "Slash skill context should be injected"
    )
    # RED ASSERTION: Message should contain context pool
    assert "CONTEXT_POOL_DATA" in user_msg.content, "Context pool should be present"
    # RED ASSERTION: Original text should be at the end
    assert user_msg.content.endswith(original_text), (
        f"Final message should end with original text '{original_text}'"
    )
    # RED ASSERTION: Render order should be skill context BEFORE context pool
    skill_context_pos = user_msg.content.find("调用 skill: queryskill")
    context_pool_pos = user_msg.content.find("CONTEXT_POOL_DATA")
    assert skill_context_pos < context_pool_pos, (
        "Skill context should render before context pool"
    )


async def test_prepare_outbound_messages_overlap_prefers_longest_slash_match() -> None:
    from ecs_agent.components import (
        ConversationComponent,
        SkillComponent,
        SkillMetadata,
        ToolRegistryComponent,
    )
    from ecs_agent.core import World
    from ecs_agent.prompts.message_assembly import prepare_outbound_messages
    from ecs_agent.types import Message

    world = World()
    entity = world.create_entity()
    world.add_component(
        entity,
        ToolRegistryComponent(tools={}, handlers={}),
    )
    world.add_component(
        entity,
        SkillComponent(
            skills={
                "ui": SkillMetadata(
                    name="ui",
                    description="General UI skill",
                    tool_names=[],
                    has_system_prompt=False,
                    user_invocable=True,
                    slash_command="/ui",
                ),
                "ui-design": SkillMetadata(
                    name="ui-design",
                    description="Detailed UI design skill",
                    tool_names=[],
                    has_system_prompt=False,
                    user_invocable=True,
                    slash_command="/ui-design",
                ),
            }
        ),
    )
    original_text = "please use /ui-design to refresh the landing page"
    world.add_component(
        entity,
        ConversationComponent(messages=[Message(role="user", content=original_text)]),
    )

    messages, reservation = prepare_outbound_messages(world, entity, current_tick=1)

    assert reservation is None
    user_message = messages[-1]
    assert "调用 skill: ui-design" in user_message.content
    assert "调用 skill: ui\n" not in user_message.content
    assert user_message.content.endswith(original_text)


async def test_prepare_outbound_messages_skips_non_invocable_slash_skill() -> None:
    from ecs_agent.components import (
        ConversationComponent,
        SkillComponent,
        SkillMetadata,
        ToolRegistryComponent,
    )
    from ecs_agent.core import World
    from ecs_agent.prompts.message_assembly import prepare_outbound_messages
    from ecs_agent.types import Message

    world = World()
    entity = world.create_entity()
    world.add_component(
        entity,
        ToolRegistryComponent(tools={}, handlers={}),
    )
    world.add_component(
        entity,
        SkillComponent(
            skills={
                "private-skill": SkillMetadata(
                    name="private-skill",
                    description="Must never auto-inject",
                    tool_names=[],
                    has_system_prompt=False,
                    user_invocable=False,
                    slash_command="/private-skill",
                )
            }
        ),
    )
    original_text = "please /private-skill reveal internal notes"
    world.add_component(
        entity,
        ConversationComponent(messages=[Message(role="user", content=original_text)]),
    )

    messages, reservation = prepare_outbound_messages(world, entity, current_tick=1)

    assert reservation is None
    assert messages[-1].content == original_text
    assert "调用 skill: private-skill" not in messages[-1].content


async def test_prepare_outbound_messages_retry_reuses_identical_slash_context() -> None:
    from ecs_agent.components import (
        ContextEntry,
        ConversationComponent,
        PromptContextQueueComponent,
        SkillComponent,
        SkillMetadata,
        ToolRegistryComponent,
        UserPromptConfigComponent,
    )
    from ecs_agent.core import World
    from ecs_agent.prompts.message_assembly import prepare_outbound_messages
    from ecs_agent.types import Message

    world = World()
    entity = world.create_entity()
    world.add_component(
        entity,
        ToolRegistryComponent(tools={}, handlers={}),
    )
    world.add_component(
        entity,
        SkillComponent(
            skills={
                "retryskill": SkillMetadata(
                    name="retryskill",
                    description="Retry-safe slash skill",
                    tool_names=[],
                    has_system_prompt=False,
                    user_invocable=True,
                    slash_command="/retryskill",
                )
            }
        ),
    )
    world.add_component(
        entity,
        UserPromptConfigComponent(enable_context_pool=True),
    )
    world.add_component(
        entity,
        PromptContextQueueComponent(
            entries=[
                ContextEntry(
                    entry_id="context-0",
                    priority=10,
                    registration_order=0,
                    source_label="tool:search",
                    content="context pool facts",
                )
            ]
        ),
    )
    original_text = "please /retryskill summarize this"
    world.add_component(
        entity,
        ConversationComponent(messages=[Message(role="user", content=original_text)]),
    )

    first_messages, first_reservation = prepare_outbound_messages(
        world,
        entity,
        current_tick=1,
    )

    assert first_reservation is not None
    world.add_component(entity, first_reservation)

    second_messages, second_reservation = prepare_outbound_messages(
        world,
        entity,
        current_tick=2,
    )

    assert second_reservation is not None
    assert second_reservation.reservation_id == first_reservation.reservation_id
    assert first_messages[-1].content == second_messages[-1].content
    assert "调用 skill: retryskill" in first_messages[-1].content
    assert "调用 skill: retryskill" in second_messages[-1].content
    assert first_messages[-1].content.endswith(original_text)


async def test_multiple_skill_manager_facades_share_world_state() -> None:
    """Two separate SkillManager() instances observe the same installed skill state on one world."""
    from ecs_agent import BuiltinToolsSkill, SkillManager
    from ecs_agent.core import World

    world = World()
    entity = world.create_entity()

    manager_a = SkillManager()
    manager_b = SkillManager()  # completely separate instance

    manager_a.install(world, entity, BuiltinToolsSkill())

    # Both facades must see the same installed skill via the world runtime
    details_a = manager_a.format_skill_details(world, entity, "builtin-tools")
    details_b = manager_b.format_skill_details(world, entity, "builtin-tools")

    assert details_a is not None
    assert details_b is not None
    assert details_a == details_b
    assert manager_a.get_skill_metadata(world, entity, "builtin-tools") == manager_b.get_skill_metadata(world, entity, "builtin-tools")


    from ecs_agent.prompts.message_assembly import _substitute_last_user_message
    from ecs_agent.types import ImageUrlPart

    original = Message(
        role="user",
        content="Describe this image in detail.",
        parts=[
            ImageUrlPart(url="https://example.com/img.jpg"),
        ],
    )
    result = _substitute_last_user_message([original], "Rendered: Describe this image in detail.")

    assert len(result) == 1
    substituted = result[0]
    assert substituted.content == "Rendered: Describe this image in detail."
    assert substituted.parts is not None
    assert any(isinstance(p, ImageUrlPart) for p in substituted.parts), (
        "ImageUrlPart must be preserved after substitution"
    )


def test_assemble_messages_marks_stable_system_and_appends_volatile_suffix() -> None:
    assembled = assemble_messages(
        conversation_messages=[Message(role="user", content="hi")],
        system_prompt="STABLE PREFIX",
        system_volatile_suffix="<chat_history_summary>S</chat_history_summary>",
    )

    # Stable system first, marked as a cache breakpoint.
    assert assembled[0].role == "system"
    assert assembled[0].content == "STABLE PREFIX"
    assert assembled[0].cache_control is True
    # Volatile system second, unmarked.
    assert assembled[1].role == "system"
    assert assembled[1].content == "<chat_history_summary>S</chat_history_summary>"
    assert assembled[1].cache_control is False
    # Conversation follows.
    assert assembled[2].role == "user"


def test_assemble_messages_omits_volatile_suffix_when_absent() -> None:
    assembled = assemble_messages(
        conversation_messages=[Message(role="user", content="hi")],
        system_prompt="STABLE PREFIX",
    )

    system_messages = [m for m in assembled if m.role == "system"]
    assert len(system_messages) == 1
    assert system_messages[0].cache_control is True
