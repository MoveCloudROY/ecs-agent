"""User prompt normalization system."""

from __future__ import annotations

import uuid

from ecs_agent.components import (
    ConversationComponent,
    ConversationTreeComponent,
    OneShotContextPoolComponent,
    UserPromptConfigComponent,
    RenderedUserPromptComponent,
    TurnStateComponent,
)
from ecs_agent.conversation_tree import get_active_leaf, linearize
from ecs_agent.core.world import World
from ecs_agent.logging import get_logger
from ecs_agent.prompts.contracts import TriggerSpec
from ecs_agent.prompts.message_assembly import (
    assemble_messages,
    build_keyword_registry,
    build_trigger_specs,
    collect_active_events,
)
from ecs_agent.types import EntityId, Message

logger = get_logger(__name__)


class UserPromptNormalizationSystem:
    """Normalizes user prompts with triggers and context pool injection."""

    def __init__(
        self,
        priority: int = 0,
        trigger_specs: list[TriggerSpec] | None = None,
    ) -> None:
        self.priority = priority
        self._trigger_specs = trigger_specs or []

    async def process(self, world: World) -> None:
        entity_ids: set[EntityId] = set()
        entity_ids.update(
            entity_id for entity_id, _ in world.query(ConversationComponent)
        )
        entity_ids.update(
            entity_id for entity_id, _ in world.query(ConversationTreeComponent)
        )

        for entity_id in entity_ids:
            conversation_messages = self._resolve_conversation_messages(
                world, entity_id
            )
            if not conversation_messages:
                continue

            raw_user_text = self._find_last_user_text(conversation_messages)
            if raw_user_text is None:
                continue

            prompt_config = world.get_component(entity_id, UserPromptConfigComponent)
            context_pool = world.get_component(entity_id, OneShotContextPoolComponent)
            turn_state = world.get_component(entity_id, TurnStateComponent)
            turn_id = (
                turn_state.current_turn_id
                if turn_state is not None and turn_state.current_turn_id
                else uuid.uuid4().hex
            )

            keyword_registry = None
            trigger_specs = None
            context_pool_enabled = False
            context_pool_items: list[tuple[int, int, str, str]] | None = None

            if prompt_config is not None:
                context_pool_enabled = prompt_config.enable_context_pool
                if prompt_config.triggers:
                    keyword_registry = build_keyword_registry(
                        prompt_config.triggers
                    )
                    trigger_specs = build_trigger_specs(prompt_config.triggers)
                if context_pool_enabled and context_pool is not None:
                    context_pool_items = list(context_pool.items)

            active_events = collect_active_events(context_pool_items)
            assembled = assemble_messages(
                conversation_messages=[Message(role="user", content=raw_user_text)],
                keyword_registry=keyword_registry,
                trigger_specs=trigger_specs,
                enable_context_pool=context_pool_enabled,
                context_pool_items=context_pool_items,
                active_events=active_events,
            )
            normalized_text = assembled[-1].content if assembled else raw_user_text
            normalized_text = self.apply_trigger_specs(
                user_text=normalized_text,
                trigger_specs=self._trigger_specs,
            )

            world.add_component(
                entity_id,
                RenderedUserPromptComponent(text=normalized_text, turn_id=turn_id),
            )

    @staticmethod
    def apply_trigger_specs(user_text: str, trigger_specs: list[TriggerSpec]) -> str:
        """Apply TriggerSpec rules over normalized user text."""
        if not trigger_specs or "[PROMPT_INJECT:" in user_text:
            return user_text

        ordered_specs = sorted(trigger_specs, key=lambda spec: -spec.priority)
        for spec in ordered_specs:
            if not UserPromptNormalizationSystem._matches(spec=spec, text=user_text):
                continue
            if spec.action == "replace":
                return spec.content
            return f"{spec.content}\n\n{user_text}"

        return user_text

    @staticmethod
    def _matches(*, spec: TriggerSpec, text: str) -> bool:
        if spec.match_mode == "keyword":
            return spec.pattern in text
        if spec.match_mode == "prefix":
            return text.startswith(spec.pattern)
        return spec.pattern in text

    @staticmethod
    def _find_last_user_text(messages: list[Message]) -> str | None:
        for message in reversed(messages):
            if message.role == "user":
                return message.content
        return None

    @staticmethod
    def _resolve_conversation_messages(
        world: World, entity_id: EntityId
    ) -> list[Message]:
        tree = world.get_component(entity_id, ConversationTreeComponent)
        if tree is not None:
            active_leaf_id = get_active_leaf(tree)
            if active_leaf_id is not None:
                return linearize(tree, active_leaf_id)

        conversation = world.get_component(entity_id, ConversationComponent)
        if conversation is not None:
            return list(conversation.messages)

        return []


__all__ = ["UserPromptNormalizationSystem"]
