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
