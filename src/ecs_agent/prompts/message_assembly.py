"""Canonical message assembly for LLM model calls."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from ecs_agent.token_counting import count_tokens
from typing import TYPE_CHECKING, Protocol, TypeVar

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
    from ecs_agent.components.definitions import (
        ContextTrimConfig,
        ContextCacheComponent,
        ContextEntry,
        ConversationComponent,
    )
    from ecs_agent.core import World
    from ecs_agent.types import DroppableContextKind
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
    droppable_kind: DroppableContextKind | None


ContextEntryT = TypeVar("ContextEntryT", bound=ContextEntryProtocol)


def trim_context_to_fit(
    messages: list[Message],
    system_prompt: str,
    context_entries: list[ContextEntry],
    config: ContextTrimConfig,
    *,
    cache_component: ContextCacheComponent | None = None,
) -> list[Message]:
    reduced_messages = list(messages)
    reduced_context_entries = list(context_entries)

    # No explicit budget on this (transient outbound) path -> nothing to trim
    # here. Model-window-derived budgeting is handled by the CompactionSystem
    # trim step, which has the model id.
    budget = config.max_tokens
    if budget is None:
        return reduced_messages

    if (
        _estimate_total_tokens(
            reduced_messages,
            system_prompt=system_prompt,
            context_entries=reduced_context_entries,
            chars_per_token=config.token_estimation_chars_per_token,
        )
        <= budget
    ):
        return reduced_messages

    if config.trim_tool_results:
        while True:
            if (
                _estimate_total_tokens(
                    reduced_messages,
                    system_prompt=system_prompt,
                    context_entries=reduced_context_entries,
                    chars_per_token=config.token_estimation_chars_per_token,
                )
                <= budget
            ):
                return reduced_messages
            next_messages = _drop_oldest_tool_span(reduced_messages)
            if len(next_messages) == len(reduced_messages):
                break
            reduced_messages = next_messages

        while True:
            if (
                _estimate_total_tokens(
                    reduced_messages,
                    system_prompt=system_prompt,
                    context_entries=reduced_context_entries,
                    chars_per_token=config.token_estimation_chars_per_token,
                )
                <= budget
            ):
                return reduced_messages

            next_context_entries = _drop_oldest_context_entries_by_kind(
                reduced_context_entries,
                kind="tool_result",
                cache_component=cache_component,
            )
            if len(next_context_entries) == len(reduced_context_entries):
                break
            dropped_ids = {e.entry_id for e in next_context_entries}
            for entry in reduced_context_entries:
                if entry.entry_id not in dropped_ids:
                    logger.info(
                        "context_entry_pruned",
                        reason="tool_result",
                        entry_id=entry.entry_id,
                        source_label=entry.source_label,
                    )
            reduced_context_entries = next_context_entries
            reduced_messages = _replace_context_pool_block(
                reduced_messages,
                original_entries=context_entries,
                remaining_entries=reduced_context_entries,
                cache_component=cache_component,
            )

    if config.trim_reasoning:
        reasoning_entries = [
            entry
            for entry in reduced_context_entries
            if _context_entry_kind(entry) == "reasoning"
        ]
        if reasoning_entries:
            reduced_context_entries = [
                entry
                for entry in reduced_context_entries
                if _context_entry_kind(entry) != "reasoning"
            ]
            reduced_messages = _replace_context_pool_block(
                reduced_messages,
                original_entries=context_entries,
                remaining_entries=reduced_context_entries,
                cache_component=cache_component,
            )
            if (
                _estimate_total_tokens(
                    reduced_messages,
                    system_prompt=system_prompt,
                    context_entries=reduced_context_entries,
                    chars_per_token=config.token_estimation_chars_per_token,
                )
                <= budget
            ):
                return reduced_messages

    estimated_tokens = _estimate_total_tokens(
        reduced_messages,
        system_prompt=system_prompt,
        context_entries=reduced_context_entries,
        chars_per_token=config.token_estimation_chars_per_token,
    )
    if config.overflow_behavior == "error":
        raise ValueError("Protected content exceeds configured budget")

    logger.warning(
        "outbound_budget_exceeded",
        estimated_tokens=estimated_tokens,
        max_tokens=budget,
    )
    return reduced_messages


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
    system_volatile_suffix: str | None = None,
    prefix_messages: list[Message] | None = None,
    enable_context_pool: bool = False,
    context_pool_items: Sequence[ContextEntryProtocol] | None = None,
    keyword_registry: PromptRegistry | None = None,
    trigger_specs: list[TriggerSpec] | None = None,
) -> list[Message]:
    """Assemble model-call messages with stable ordering.

    Ordering is always:
    1) optional system prompt (cache-stable prefix, marked as a cache breakpoint)
    2) optional volatile system suffix (compaction summary / workflow state)
    3) optional prefix/system context messages
    4) conversation messages (with transient trigger injection on the last
       user message)
    5) optional context-pool block as a trailing user message

    The stable system prompt carries ``cache_control=True`` so caching-capable
    adapters place a prompt-cache breakpoint after it; the volatile suffix sits
    after the breakpoint and is never marked.

    The context-pool block is appended as its own final user message — never
    merged into an earlier message. Rewriting a message that previous calls
    already sent (the last *user* message sits mid-history during a tool loop)
    would invalidate the provider prompt cache for everything after it on
    every call. Tail placement keeps all previously-sent bytes intact;
    ``commit_prompt_context_reservation`` persists the block into the
    conversation so later calls replay the same bytes.
    """
    assembled: list[Message] = []
    if system_prompt is not None:
        assembled.append(
            Message(role="system", content=system_prompt, cache_control=True)
        )
    if system_volatile_suffix:
        assembled.append(Message(role="system", content=system_volatile_suffix))

    if prefix_messages is not None:
        assembled.extend(prefix_messages)

    transient_conversation = _with_transient_user_injection(
        conversation_messages,
        keyword_registry=keyword_registry,
        trigger_specs=trigger_specs,
    )
    assembled.extend(transient_conversation)

    if enable_context_pool:
        context_block = _render_context_pool_block(context_pool_items)
        if context_block:
            assembled.append(Message(role="user", content=context_block))
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
    queue: PromptContextQueueComponent | None,
    reservation: PromptContextReservationComponent,
    conversation: ConversationComponent | None = None,
) -> None:
    """Consume reserved queue entries and persist the sent context block.

    When ``conversation`` is given and the reservation carries a rendered
    block, the block is appended to the history as the user message the model
    just saw. Leaving it transient would make the next call's prompt diverge
    at the block's position, invalidating the provider prompt cache for every
    later token; persisting keeps outbound requests append-only.
    """
    if queue is not None:
        reserved_ids = {entry.entry_id for entry in reservation.reserved_entries}
        if reserved_ids:
            queue.entries = [
                entry for entry in queue.entries if entry.entry_id not in reserved_ids
            ]

    if conversation is not None and reservation.rendered_block:
        conversation.messages.append(
            Message(role="user", content=reservation.rendered_block)
        )


def resolve_system_prompt_parts(
    world: World, entity_id: EntityId
) -> tuple[str | None, str | None]:
    """Resolve ``(stable_prefix, volatile_suffix)`` for an entity's system prompt.

    Prefers the split :class:`RenderedSystemPromptComponent` (ISSUE-6); falls back
    to the legacy ``SystemPromptComponent`` / ``LLMComponent.system_prompt`` with an
    empty volatile suffix. Shared by the reasoning/planning/replanning systems so
    the cache-stable prefix is derived identically everywhere.
    """
    from ecs_agent.components.definitions import (
        LLMComponent,
        RenderedSystemPromptComponent,
        SystemPromptComponent,
    )

    rendered = world.get_component(entity_id, RenderedSystemPromptComponent)
    if rendered is not None:
        # A split render sets stable_text and/or volatile_text. When the whole
        # prompt body is volatile (e.g. a workflow-driven prompt), stable_text is
        # empty and there is simply no cacheable prefix — send the volatile tail
        # as the sole system message rather than falling back to the full text
        # (which would duplicate the volatile content).
        if rendered.stable_text or rendered.volatile_text:
            return (rendered.stable_text or None, rendered.volatile_text or None)
        return (rendered.text or None, None)

    system_prompt = world.get_component(entity_id, SystemPromptComponent)
    if system_prompt is not None:
        return (system_prompt.content, None)

    llm_component = world.get_component(entity_id, LLMComponent)
    if llm_component is not None:
        return (llm_component.system_prompt or None, None)

    return (None, None)


def prepare_outbound_messages(
    world: World,
    entity_id: EntityId,
    *,
    system_prompt: str | None = None,
    system_volatile_suffix: str | None = None,
    prefix_messages: list[Message] | None = None,
    current_tick: int,
    conversation_override: list[Message] | None = None,
) -> tuple[list[Message], PromptContextReservationComponent | None]:
    """Build the final message list for an LLM model call.

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
        ContextTrimConfig,
        PromptContextQueueComponent,
        PromptContextReservationComponent,
        RenderedUserPromptComponent,
        UserPromptConfigComponent,
    )
    from ecs_agent.conversation_tree import get_active_leaf, linearize

    prompt_config = world.get_component(entity_id, UserPromptConfigComponent)
    budget_config = world.get_component(entity_id, ContextTrimConfig)
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
    tree = world.get_component(entity_id, ConversationTreeComponent)

    if conversation_override is not None:
        # Override path: skip World read, skip RenderedUserPromptComponent.
        # Convert config triggers to trigger_specs for inline injection.
        messages_for_assembly = list(conversation_override)
        trigger_specs: list[TriggerSpec] | None = None
        if prompt_config is not None and prompt_config.triggers:
            trigger_specs = list(prompt_config.triggers)
    else:
        # Normal path: read from World, apply RenderedUserPromptComponent.
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
        system_volatile_suffix=system_volatile_suffix,
        prefix_messages=prefix_messages,
        conversation_messages=messages_for_assembly,
        enable_context_pool=enable_context_pool_for_assembly,
        context_pool_items=context_pool_items_for_assembly,
        trigger_specs=trigger_specs,
    )

    if budget_config is not None and tree is None:
        from ecs_agent.components.definitions import ContextCacheComponent

        messages = trim_context_to_fit(
            messages,
            system_prompt=(system_prompt or "") + (system_volatile_suffix or ""),
            context_entries=list(context_pool_items_for_assembly or []),
            config=budget_config,
            cache_component=world.get_component(entity_id, ContextCacheComponent),
        )

    # Record the context block exactly as sent (post-trim) so the commit step
    # can persist it into the conversation. A reservation is created even
    # without a queue (e.g. slash-skill context only) — otherwise the block
    # would vanish from the next call's prompt and break the cache prefix.
    sent_block = ""
    if messages and messages[-1].role == "user":
        tail_content = messages[-1].content or ""
        if tail_content.startswith(CONTEXT_POOL_MARKER):
            sent_block = tail_content
    if sent_block:
        if context_reservation is None:
            context_reservation = PromptContextReservationComponent(
                reservation_id=uuid.uuid4().hex,
                created_at_tick=current_tick,
                reserved_entries=[],
            )
        context_reservation.rendered_block = sent_block

    logger.debug(
        "outbound_messages_assembled",
        world_name=world.name,
        entity_id=entity_id,
        message_count=len(messages),
        system_prompt_preview=f"{messages[0].content[:200]} ... (only the first 200)"
        if messages and messages[0].role == "system"
        else None,
        last_user_prompt=messages[-1].content[:200]
        if messages and messages[-1].role == "user"
        else None,
    )
    return messages, context_reservation


def _with_transient_user_injection(
    conversation_messages: list[Message],
    *,
    keyword_registry: PromptRegistry | None,
    trigger_specs: list[TriggerSpec] | None,
) -> list[Message]:
    """Apply transient trigger injection to the last user message.

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
    *,
    cache_component: ContextCacheComponent | None = None,
) -> str:
    if not context_pool_items:
        return ""

    sorted_items = sorted(
        context_pool_items,
        key=lambda entry: (-entry.priority, entry.registration_order),
    )
    rendered_entries = [
        _render_context_entry(entry, cache_component=cache_component)
        for entry in sorted_items
        if _render_context_entry(entry, cache_component=cache_component)
    ]
    if not rendered_entries:
        return ""

    joined_entries = CONTEXT_POOL_DELIMITER.join(rendered_entries)
    return f"{CONTEXT_POOL_MARKER}\n{joined_entries}"


def _estimate_total_tokens(
    messages: list[Message],
    *,
    system_prompt: str,
    context_entries: Sequence[ContextEntryProtocol],
    chars_per_token: float,
) -> int:
    # Real BPE count when tiktoken is available; the CJK-aware fallback reduces
    # to ceil(total_chars / chars_per_token) for ASCII text (ISSUE-8).
    parts = [message.content or "" for message in messages]
    parts.append(system_prompt)
    parts.extend(entry.content for entry in context_entries)
    return count_tokens("".join(parts), fallback_chars_per_token=chars_per_token)


def _drop_oldest_tool_span(
    messages: list[Message], *, protect_from: int | None = None
) -> list[Message]:
    """Drop the oldest complete tool span (assistant tool-call + its results).

    When ``protect_from`` is given, only spans that end at or before that index
    are eligible, so the most recent messages are never dropped.
    """
    for index, message in enumerate(messages):
        if message.role != "assistant" or not message.tool_calls:
            continue

        tool_call_ids = {tool_call.id for tool_call in message.tool_calls}
        if not tool_call_ids:
            continue

        end_index = index + 1
        matched_tool_result = False
        while end_index < len(messages):
            candidate = messages[end_index]
            if candidate.role != "tool":
                break
            if candidate.tool_call_id in tool_call_ids:
                matched_tool_result = True
                end_index += 1
                continue
            break

        if not matched_tool_result:
            continue

        if protect_from is not None and end_index > protect_from:
            # Span reaches into the protected recent window; keep it whole.
            continue

        return [*messages[:index], *messages[end_index:]]

    return list(messages)


def _context_entry_kind(entry: ContextEntryProtocol) -> DroppableContextKind | None:
    if entry.droppable_kind is not None:
        return entry.droppable_kind
    if entry.source_label == "reasoning":
        return "reasoning"
    if entry.source_label.startswith("tool:") or entry.source_label.startswith(
        "structured_output:"
    ):
        return "tool_result"
    return None


def _drop_oldest_context_entries_by_kind(
    entries: Sequence[ContextEntryT],
    *,
    kind: DroppableContextKind,
    cache_component: ContextCacheComponent | None = None,
) -> list[ContextEntryT]:
    matching_entries = [
        entry for entry in entries if _context_entry_kind(entry) == kind
    ]
    if not matching_entries:
        return list(entries)

    oldest_entry = min(
        matching_entries,
        key=lambda entry: (
            0
            if _cached_artifact_path_for_entry(entry, cache_component=cache_component)
            else 1,
            entry.registration_order,
        ),
    )
    return [entry for entry in entries if entry.entry_id != oldest_entry.entry_id]


def _replace_context_pool_block(
    messages: list[Message],
    *,
    original_entries: Sequence[ContextEntryProtocol],
    remaining_entries: Sequence[ContextEntryProtocol],
    cache_component: ContextCacheComponent | None = None,
) -> list[Message]:
    original_block = _render_context_pool_block(
        original_entries,
        cache_component=cache_component,
    )
    if not original_block:
        return list(messages)

    remaining_block = _render_context_pool_block(
        remaining_entries,
        cache_component=cache_component,
    )
    replaced_messages = list(messages)
    for index in range(len(replaced_messages) - 1, -1, -1):
        message = replaced_messages[index]
        if message.role != "user" or original_block not in message.content:
            continue

        updated_content = message.content.replace(original_block, remaining_block, 1)
        if not remaining_block:
            updated_content = updated_content.replace("\n\n\n\n", "\n\n")
            updated_content = updated_content.replace(f"{CONTEXT_POOL_MARKER}\n", "", 1)
        if not updated_content.strip():
            # The block was the whole message (tail-injected pool message):
            # drop it rather than sending an empty user turn.
            del replaced_messages[index]
            break
        replaced_messages[index] = Message(
            role=message.role,
            content=updated_content,
            parts=message.parts,
            tool_calls=message.tool_calls,
            tool_call_id=message.tool_call_id,
            compaction_metadata=message.compaction_metadata,
        )
        break
    return replaced_messages


def _render_context_entry(
    entry: ContextEntryProtocol,
    *,
    cache_component: ContextCacheComponent | None,
) -> str:
    artifact_path = _cached_artifact_path_for_entry(
        entry,
        cache_component=cache_component,
    )
    if artifact_path is not None:
        return (
            f"source: {entry.source_label}\n"
            "status: cached\n"
            f"result: [Tool result cached - retrieve full content from {artifact_path}]"
        )
    return entry.content


def _cached_artifact_path_for_entry(
    entry: ContextEntryProtocol,
    *,
    cache_component: ContextCacheComponent | None,
) -> str | None:
    if cache_component is None or not entry.source_label.startswith("tool:"):
        return None

    tool_call_id = entry.source_label.split(":", 1)[1]
    for cached_result in cache_component.cached_tool_results:
        if cached_result.tool_call_id == tool_call_id:
            return cached_result.artifact_path
    return None


__all__ = [
    "trim_context_to_fit",
    "assemble_messages",
    "build_keyword_registry",
    "build_trigger_specs",
    "commit_prompt_context_reservation",
    "prepare_outbound_messages",
    "reserve_prompt_context_reservation",
]
