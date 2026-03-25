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

