"""Canonical message assembly for LLM provider calls."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol

from ecs_agent.prompts.contracts import (
    PromptTemplate,
    TriggerSpec,
)
from ecs_agent.prompts.keyword_injection import inject_triggers
from ecs_agent.prompts.registry import PromptRegistry
from ecs_agent.prompts.user_prompt_rendering import render_user_prompt_text
from ecs_agent.types import Message
from ecs_agent.logging import get_logger

logger = get_logger(__name__)

CONTEXT_POOL_DELIMITER = "\n\n---\n\n"
CONTEXT_POOL_MARKER = "[PROMPT_CONTEXT_POOL]"

if TYPE_CHECKING:
    from ecs_agent.core import World
    from ecs_agent.types import EntityId
    from ecs_agent.components.definitions import (
        PromptContextQueueComponent,
        PromptContextReservationComponent,
        SkillMetadata,
    )


class ContextEntryProtocol(Protocol):
    entry_id: str
    priority: int
    source_label: str
    content: str
    registration_order: int


def build_keyword_registry(triggers: dict[str, str]) -> PromptRegistry:
    """Build a keyword registry from trigger-to-template-content mapping."""
    registry = PromptRegistry()
    for index, (trigger_key, template_content) in enumerate(triggers.items()):
        template_id = f"keyword-template-{index}"
        registry.register(
            PromptTemplate(template_id=template_id, content=template_content)
        )
        registry.register_keyword(trigger_key, template_id)
    return registry


def build_trigger_specs(triggers: dict[str, str]) -> list[TriggerSpec]:
    trigger_specs: list[TriggerSpec] = []
    for index, trigger_key in enumerate(triggers):
        trigger_specs.append(
            TriggerSpec(
                pattern=trigger_key,
                match_mode="keyword",
                action="inject",
                content=f"keyword-template-{index}",
                priority=0,
            )
        )
    return trigger_specs


def assemble_messages(
    *,
    conversation_messages: list[Message],
    system_prompt: str | None = None,
    prefix_messages: list[Message] | None = None,
    enable_context_pool: bool = False,
    context_pool_items: Sequence[ContextEntryProtocol] | None = None,
    keyword_registry: PromptRegistry | None = None,
    trigger_specs: list[TriggerSpec] | None = None,
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
        trigger_specs=trigger_specs,
        enable_context_pool=enable_context_pool,
        context_pool_items=context_pool_items,
    )
    assembled.extend(transient_conversation)
    return assembled


def reserve_prompt_context_reservation(
    *,
    queue: PromptContextQueueComponent,
    reservation: PromptContextReservationComponent | None,
    current_tick: int,
) -> PromptContextReservationComponent:
    if reservation is not None:
        return reservation

    from ecs_agent.components.definitions import PromptContextReservationComponent

    return PromptContextReservationComponent(
        reservation_id=uuid.uuid4().hex,
        created_at_tick=current_tick,
        reserved_entries=list(queue.entries),
    )


def commit_prompt_context_reservation(
    *,
    queue: PromptContextQueueComponent,
    reservation: PromptContextReservationComponent,
) -> None:
    reserved_ids = {entry.entry_id for entry in reservation.reserved_entries}
    if not reserved_ids:
        return

    queue.entries = [
        entry for entry in queue.entries if entry.entry_id not in reserved_ids
    ]


def prepare_outbound_messages(
    world: World,
    entity_id: EntityId,
    *,
    system_prompt: str | None = None,
    prefix_messages: list[Message] | None = None,
    current_tick: int,
    conversation_override: list[Message] | None = None,
) -> tuple[list[Message], PromptContextReservationComponent | None]:
    """Build the final message list for an LLM provider call.

    Args:
        world: The ECS world.
        entity_id: Target entity.
        system_prompt: Optional system prompt override.
        prefix_messages: Extra messages inserted after the system prompt.
        current_tick: Current runner tick (used for context reservation).
        conversation_override: When supplied, used **instead** of the entity's
            conversation history.  ``RenderedUserPromptComponent`` substitution
            and trigger injection via the pre-rendered path are skipped;
            keyword triggers are applied inline at call-time so the override
            text still receives trigger processing.

    Returns:
        ``(messages, context_reservation | None)``
    """
    from ecs_agent.components.definitions import (
        ConversationComponent,
        ConversationTreeComponent,
        PromptContextQueueComponent,
        PromptContextReservationComponent,
        RenderedUserPromptComponent,
        UserPromptConfigComponent,
    )
    from ecs_agent.conversation_tree import get_active_leaf, linearize

    prompt_config = world.get_component(entity_id, UserPromptConfigComponent)
    context_queue = world.get_component(entity_id, PromptContextQueueComponent)
    context_reservation = world.get_component(
        entity_id, PromptContextReservationComponent
    )

    context_pool_enabled = (
        prompt_config.enable_context_pool if prompt_config is not None else False
    )
    slash_skill_context: str | None = None

    # -------------------------------------------------------------------
    # Resolve conversation messages
    # -------------------------------------------------------------------
    if conversation_override is not None:
        # Override path: skip World read, skip RenderedUserPromptComponent.
        # Convert config triggers to trigger_specs for inline injection.
        messages_for_assembly = list(conversation_override)
        trigger_specs: list[TriggerSpec] | None = None
        if prompt_config is not None and prompt_config.triggers:
            trigger_specs = list(prompt_config.triggers)
    else:
        # Normal path: read from World, apply RenderedUserPromptComponent.
        tree = world.get_component(entity_id, ConversationTreeComponent)
        conversation = world.get_component(entity_id, ConversationComponent)

        conversation_messages: list[Message] = []
        if tree is not None:
            active_leaf_id = get_active_leaf(tree)
            if active_leaf_id is not None:
                conversation_messages.extend(linearize(tree, active_leaf_id))
        elif conversation is not None:
            conversation_messages.extend(conversation.messages)

        raw_last_user_text = _last_user_text(conversation_messages)
        if raw_last_user_text is not None:
            slash_skill_context = _resolve_slash_skill_context(
                world,
                entity_id,
                raw_last_user_text,
            )

        rendered_user_prompt = world.get_component(
            entity_id, RenderedUserPromptComponent
        )

        messages_for_assembly = list(conversation_messages)
        if rendered_user_prompt is not None:
            messages_for_assembly = _substitute_last_user_message(
                messages_for_assembly,
                rendered_user_prompt.text,
            )

        # Inline fallback: if no pre-rendered prompt, apply triggers at call-time
        trigger_specs = None
        if (
            rendered_user_prompt is None
            and prompt_config is not None
            and prompt_config.triggers
        ):
            trigger_specs = list(prompt_config.triggers)
    # -------------------------------------------------------------------
    # Context pool reservation (always at call-time)
    # -------------------------------------------------------------------
    reserved_context_pool_items = None
    if context_pool_enabled and context_queue is not None:
        context_reservation = reserve_prompt_context_reservation(
            queue=context_queue,
            reservation=context_reservation,
            current_tick=current_tick,
        )
        reserved_context_pool_items = context_reservation.reserved_entries

    context_pool_items_for_assembly = reserved_context_pool_items
    if slash_skill_context is not None:
        from ecs_agent.components.definitions import ContextEntry

        slash_context_entry = ContextEntry(
            entry_id="slash-skill-context",
            priority=1_000_000,
            source_label="slash:skill",
            content=slash_skill_context,
            registration_order=-1,
        )
        existing_items = (
            list(reserved_context_pool_items)
            if reserved_context_pool_items is not None
            else []
        )
        context_pool_items_for_assembly = [slash_context_entry, *existing_items]

    enable_context_pool_for_assembly = context_pool_enabled or (
        slash_skill_context is not None
    )

    messages = assemble_messages(
        system_prompt=system_prompt,
        prefix_messages=prefix_messages,
        conversation_messages=messages_for_assembly,
        enable_context_pool=enable_context_pool_for_assembly,
        context_pool_items=context_pool_items_for_assembly,
        trigger_specs=trigger_specs,
    )

    logger.debug(
        "outbound_messages_assembled",
        world_name=world.name,
        entity_id=entity_id,
        message_count=len(messages),
        system_prompt_preview=f"{messages[0].content[:200]} ... (only the first 200)"
        if messages and messages[0].role == "system"
        else None,
        last_user_prompt=messages[-1].content
        if messages and messages[-1].role == "user"
        else None,
    )
    return messages, context_reservation


def _with_transient_user_injection(
    conversation_messages: list[Message],
    *,
    keyword_registry: PromptRegistry | None,
    trigger_specs: list[TriggerSpec] | None,
    enable_context_pool: bool,
    context_pool_items: Sequence[ContextEntryProtocol] | None,
) -> list[Message]:
    """Apply transient trigger + context-pool injection to the last user message.

    Trigger injection is performed via one of two paths:
    * **trigger_specs** (preferred) — delegates to ``render_user_prompt_text``.
    * **keyword_registry** (legacy) — delegates to ``inject_triggers`` for
      backward compatibility with direct ``assemble_messages`` callers.
    """
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

    # Trigger injection: prefer shared helper, fall back to legacy registry
    if keyword_registry is not None:
        transformed_text = inject_triggers(
            transformed_text,
            keyword_registry,
            trigger_specs=trigger_specs,
        )
    elif trigger_specs is not None:
        transformed_text = render_user_prompt_text(
            transformed_text,
            trigger_specs=trigger_specs,
        )

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
        parts=last_user_message.parts,
        tool_calls=last_user_message.tool_calls,
        tool_call_id=last_user_message.tool_call_id,
    )
    return mutated


def _substitute_last_user_message(
    conversation_messages: list[Message],
    replacement_text: str,
) -> list[Message]:
    messages = list(conversation_messages)
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if message.role == "user":
            messages[index] = Message(
                role="user",
                content=replacement_text,
                parts=message.parts,
                tool_calls=message.tool_calls,
                tool_call_id=message.tool_call_id,
            )
            break
    return messages


def _append_to_last_user_message(
    conversation_messages: list[Message],
    *,
    suffix: str,
) -> list[Message]:
    if not suffix:
        return list(conversation_messages)

    messages = list(conversation_messages)
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if message.role != "user":
            continue
        messages[index] = Message(
            role="user",
            content=f"{message.content}\n\n{suffix}",
            tool_calls=message.tool_calls,
            tool_call_id=message.tool_call_id,
        )
        break
    return messages


def _last_user_text(conversation_messages: list[Message]) -> str | None:
    for index in range(len(conversation_messages) - 1, -1, -1):
        message = conversation_messages[index]
        if message.role == "user":
            return message.content
    return None


def _resolve_slash_skill_context(
    world: World,
    entity_id: EntityId,
    raw_user_text: str,
) -> str | None:
    from ecs_agent.components.definitions import SkillComponent
    from ecs_agent.skills.manager import SkillManager

    skill_component = world.get_component(entity_id, SkillComponent)
    if skill_component is None:
        return None

    matches: list[SkillMetadata] = []
    skill_manager = SkillManager()
    for metadata in skill_component.skills.values():
        slash_command = metadata.slash_command.strip()
        if not slash_command or not skill_manager.can_invoke_via_slash(
            world, entity_id, slash_command
        ):
            continue
        if slash_command in raw_user_text:
            matches.append(metadata)

    if not matches:
        return None

    winning_skill = max(matches, key=lambda metadata: len(metadata.slash_command))
    return _format_slash_skill_context(world, entity_id, winning_skill.name)


def _format_slash_skill_context(
    world: World,
    entity_id: EntityId,
    skill_name: str,
) -> str:
    from ecs_agent.skills.manager import SkillManager

    header = f"调用 skill: {skill_name}"
    details = SkillManager().format_skill_details(world, entity_id, skill_name)
    if details is None:
        return header
    return f"{header}\n\n{details}"


def _render_context_pool_block(
    context_pool_items: Sequence[ContextEntryProtocol] | None,
) -> str:
    if not context_pool_items:
        return ""

    sorted_items = sorted(
        context_pool_items,
        key=lambda entry: (-entry.priority, entry.registration_order),
    )
    rendered_entries = [entry.content for entry in sorted_items if entry.content]
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


__all__ = [
    "assemble_messages",
    "build_keyword_registry",
    "build_trigger_specs",
    "commit_prompt_context_reservation",
    "prepare_outbound_messages",
    "reserve_prompt_context_reservation",
]
