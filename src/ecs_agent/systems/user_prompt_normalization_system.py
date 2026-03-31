"""User prompt normalization system."""

from __future__ import annotations


from ecs_agent.components import (
    ConversationComponent,
    ConversationTreeComponent,
    RenderedUserPromptComponent,
    UserPromptConfigComponent,
)
from ecs_agent.conversation_tree import get_active_leaf, linearize
from ecs_agent.core.world import World
from ecs_agent.logging import get_logger
from ecs_agent.prompts.contracts import TriggerSpec
from ecs_agent.prompts.user_prompt_rendering import (
    render_user_prompt_text,
    _apply_trigger_specs,
    _matches,
)
from ecs_agent.types import EntityId, Message

logger = get_logger(__name__)


class UserPromptNormalizationSystem:
    """Normalizes user prompts with triggers and context pool injection."""

    def __init__(
        self,
        priority: int = 0,
    ) -> None:
        self.priority = priority

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
                self._clear_rendered_user_prompt(world=world, entity_id=entity_id)
                continue

            raw_user_text = self._find_last_user_text(conversation_messages)
            if raw_user_text is None:
                self._clear_rendered_user_prompt(world=world, entity_id=entity_id)
                continue

            prompt_config = world.get_component(entity_id, UserPromptConfigComponent)

            trigger_specs = (
                prompt_config.triggers
                if prompt_config is not None and prompt_config.triggers
                else None
            )
            normalized_text = await self._apply_triggers(
                world=world,
                entity_id=entity_id,
                raw_user_text=raw_user_text,
                trigger_specs=trigger_specs,
                prompt_config=prompt_config,
            )

            logger.debug(
                "user_prompt_normalized",
                entity_id=entity_id,
                raw_length=len(raw_user_text),
                normalized_length=len(normalized_text),
                trigger_count=len(trigger_specs) if trigger_specs else 0,
                prompt_text=normalized_text,
            )

            world.add_component(
                entity_id,
                RenderedUserPromptComponent(text=normalized_text),
            )

    @staticmethod
    def apply_trigger_specs(user_text: str, trigger_specs: list[TriggerSpec]) -> str:
        """Apply TriggerSpec rules over normalized user text.

        Thin wrapper kept for backward compatibility.  Delegates to the
        shared ``_apply_trigger_specs`` helper in ``user_prompt_rendering``.
        """
        return _apply_trigger_specs(user_text, trigger_specs)

    async def _apply_triggers(
        self,
        *,
        world: World,
        entity_id: EntityId,
        raw_user_text: str,
        trigger_specs: list[TriggerSpec] | None,
        prompt_config: UserPromptConfigComponent | None,
    ) -> str:
        if not trigger_specs:
            return raw_user_text

        ordered = sorted(trigger_specs, key=lambda spec: -spec.priority)
        for spec in ordered:
            if not _matches(spec=spec, text=raw_user_text):
                continue
            if spec.action == "script":
                return await self._run_script_handler(
                    world=world,
                    entity_id=entity_id,
                    spec=spec,
                    raw_user_text=raw_user_text,
                    prompt_config=prompt_config,
                )

            return render_user_prompt_text(
                raw_user_text,
                trigger_specs=trigger_specs,
            )

        return raw_user_text

    async def _run_script_handler(
        self,
        *,
        world: World,
        entity_id: EntityId,
        spec: TriggerSpec,
        raw_user_text: str,
        prompt_config: UserPromptConfigComponent | None,
    ) -> str:
        handlers = prompt_config.script_handlers if prompt_config is not None else {}
        handler = handlers.get(spec.content)
        if handler is None:
            logger.error(
                "trigger_script_handler_missing",
                entity_id=entity_id,
                handler_key=spec.content,
                pattern=spec.pattern,
            )
            return raw_user_text

        result = await handler(world, entity_id, raw_user_text)
        if result is None:
            return raw_user_text
        return result

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

    @staticmethod
    def _clear_rendered_user_prompt(world: World, entity_id: EntityId) -> None:
        if world.has_component(entity_id, RenderedUserPromptComponent):
            world.remove_component(entity_id, RenderedUserPromptComponent)


__all__ = ["UserPromptNormalizationSystem"]
