"""Canonical message assembly for LLM provider calls."""

from __future__ import annotations

from typing import Protocol

from ecs_agent.prompts.contracts import PromptTemplate
from ecs_agent.prompts.keyword_injection import inject_keywords
from ecs_agent.prompts.registry import PromptRegistry
from ecs_agent.types import Message

CONTEXT_POOL_DELIMITER = "\n\n---\n\n"
CONTEXT_POOL_MARKER = "[PROMPT_CONTEXT_POOL]"


class ContextPoolReservationProtocol(Protocol):
    items: list[tuple[int, int, str, str]]
    reserved_items: list[tuple[int, int, str, str]]
    state: str
    reserved_turn_id: str
    reserved_counter_snapshot: int
    _counter: int


def build_keyword_registry(keyword_templates: dict[str, str]) -> PromptRegistry:
    """Build a keyword registry from keyword-to-template-content mapping."""
    registry = PromptRegistry()
    for index, (keyword, template_content) in enumerate(keyword_templates.items()):
        template_id = f"keyword-template-{index}"
        registry.register(
            PromptTemplate(template_id=template_id, content=template_content)
        )
        registry.register_keyword(keyword, template_id)
    return registry


def assemble_messages(
    *,
    conversation_messages: list[Message],
    system_prompt: str | None = None,
    prefix_messages: list[Message] | None = None,
    enable_context_pool: bool = False,
    context_pool_items: list[tuple[int, int, str, str]] | None = None,
    keyword_registry: PromptRegistry | None = None,
) -> list[Message]:
    """Assemble provider-call messages with stable ordering.

    Ordering is always:
    1) optional system prompt
    2) optional prefix/system context messages
    3) conversation messages (with transient final-user injection when enabled)
    """
    assembled: list[Message] = []
    if system_prompt is not None:
        assembled.append(Message(role="system", content=system_prompt))

    if prefix_messages is not None:
        assembled.extend(prefix_messages)

    transient_conversation = _with_transient_user_injection(
        conversation_messages,
        keyword_registry=keyword_registry,
        enable_context_pool=enable_context_pool,
        context_pool_items=context_pool_items,
    )
    assembled.extend(transient_conversation)
    return assembled


def reserve_context_pool_items(
    *,
    pool: ContextPoolReservationProtocol,
    turn_id: str,
) -> list[tuple[int, int, str, str]]:
    if pool.state == "reserved" and pool.reserved_turn_id == turn_id:
        return list(pool.reserved_items)

    reserved_items = list(pool.items)
    pool.reserved_items = reserved_items
    pool.state = "reserved"
    pool.reserved_turn_id = turn_id
    pool.reserved_counter_snapshot = pool._counter
    return list(reserved_items)


def commit_context_pool_reservation(
    *,
    pool: ContextPoolReservationProtocol,
    turn_id: str,
) -> None:
    if pool.state == "committed" and pool.reserved_turn_id == turn_id:
        return

    if pool.state != "reserved" or pool.reserved_turn_id != turn_id:
        return

    pool.items.clear()
    pool.reserved_items.clear()
    pool.state = "committed"


def _with_transient_user_injection(
    conversation_messages: list[Message],
    *,
    keyword_registry: PromptRegistry | None,
    enable_context_pool: bool,
    context_pool_items: list[tuple[int, int, str, str]] | None,
) -> list[Message]:
    if not conversation_messages:
        return []

    last_user_index = -1
    for index in range(len(conversation_messages) - 1, -1, -1):
        if conversation_messages[index].role == "user":
            last_user_index = index
            break

    if last_user_index < 0:
        return list(conversation_messages)

    last_user_message = conversation_messages[last_user_index]
    original_text = last_user_message.content
    transformed_text = original_text

    if keyword_registry is not None:
        transformed_text = inject_keywords(transformed_text, keyword_registry)

    if enable_context_pool:
        context_block = _render_context_pool_block(context_pool_items)
        if context_block:
            transformed_text = _inject_context_block(
                text_with_keyword=transformed_text,
                original_user_text=original_text,
                context_block=context_block,
            )

    if transformed_text == original_text:
        return list(conversation_messages)

    mutated = list(conversation_messages)
    mutated[last_user_index] = Message(
        role=last_user_message.role,
        content=transformed_text,
        tool_calls=last_user_message.tool_calls,
        tool_call_id=last_user_message.tool_call_id,
    )
    return mutated


def _render_context_pool_block(
    context_pool_items: list[tuple[int, int, str, str]] | None,
) -> str:
    if not context_pool_items:
        return ""

    sorted_items = sorted(context_pool_items, key=lambda item: (-item[0], item[1]))
    rendered_entries = [item[3] for item in sorted_items if item[3]]
    if not rendered_entries:
        return ""

    joined_entries = CONTEXT_POOL_DELIMITER.join(rendered_entries)
    return f"{CONTEXT_POOL_MARKER}\n{joined_entries}"


def _inject_context_block(
    *, text_with_keyword: str, original_user_text: str, context_block: str
) -> str:
    if text_with_keyword != original_user_text and text_with_keyword.endswith(
        original_user_text
    ):
        prefix = text_with_keyword[: -len(original_user_text)].rstrip("\n")
        return f"{prefix}\n\n{context_block}\n\n{original_user_text}"

    return f"{context_block}\n\n{text_with_keyword}"


__all__ = ["assemble_messages", "build_keyword_registry"]
