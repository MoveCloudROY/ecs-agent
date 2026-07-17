"""User prompt normalization system."""

from __future__ import annotations

import hashlib
import re

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
from ecs_agent.scratchbook import ArtifactRegistry, ScratchbookService
from ecs_agent.types import EntityId, Message, PromptReplacementEvent

logger = get_logger(__name__)

_PLAN_TRIGGER_PATTERN = re.compile(r"(^|[^a-z0-9])(plan|replan|planning)([^a-z0-9]|$)")


class UserPromptNormalizationSystem:
    """Normalizes user prompts with triggers and context pool injection."""

    def __init__(
        self,
        priority: int = 0,
        service: ScratchbookService | None = None,
        registry: ArtifactRegistry | None = None,
    ) -> None:
        self.priority = priority
        self._registry: ArtifactRegistry | None
        if registry is not None:
            self._registry = registry
        elif service is not None:
            self._registry = ArtifactRegistry(root=service.root)
        else:
            self._registry = None

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

            last_user_message = self._find_last_user_message(conversation_messages)
            if last_user_message is None:
                self._clear_rendered_user_prompt(world=world, entity_id=entity_id)
                continue
            last_user_index, raw_user_text = last_user_message

            self._freeze_superseded_render(
                world=world,
                entity_id=entity_id,
                last_user_index=last_user_index,
            )

            prompt_config = world.get_component(entity_id, UserPromptConfigComponent)

            trigger_specs = (
                prompt_config.triggers
                if prompt_config is not None and prompt_config.triggers
                else None
            )
            trigger_key = self._build_trigger_key(
                trigger_specs,
                prompt_config=prompt_config,
            )
            source_fingerprint = self._fingerprint_text(raw_user_text)
            rendered_user_prompt = world.get_component(
                entity_id, RenderedUserPromptComponent
            )
            if (
                rendered_user_prompt is not None
                and rendered_user_prompt.source_fingerprint == source_fingerprint
                and rendered_user_prompt.trigger_key == trigger_key
                and rendered_user_prompt.source_message_index == last_user_index
            ):
                continue

            normalized_text = await self._apply_triggers(
                world=world,
                entity_id=entity_id,
                raw_user_text=raw_user_text,
                trigger_specs=trigger_specs,
                prompt_config=prompt_config,
            )

            updated_source_index = last_user_index
            updated_messages = self._resolve_conversation_messages(world, entity_id)
            updated_last_user_message = self._find_last_user_message(updated_messages)
            if updated_last_user_message is not None:
                updated_source_index, _ = updated_last_user_message

            logger.debug(
                "user_prompt_normalized",
                entity_id=entity_id,
                raw_length=len(raw_user_text),
                normalized_length=len(normalized_text),
                trigger_count=len(trigger_specs) if trigger_specs else 0,
                prompt_text=normalized_text[:200],
            )

            world.add_component(
                entity_id,
                RenderedUserPromptComponent(
                    text=normalized_text,
                    source_fingerprint=source_fingerprint,
                    trigger_key=trigger_key,
                    source_message_index=updated_source_index,
                ),
            )
            if normalized_text != raw_user_text:
                matched_spec = self._matching_trigger_spec(
                    raw_user_text,
                    trigger_specs,
                )
                replacements: dict[str, str] = {}
                metadata = {
                    "system_name": (
                        "ecs_agent.systems.user_prompt_normalization_system."
                        "UserPromptNormalizationSystem"
                    ),
                }
                if matched_spec is not None:
                    replacements["trigger_content"] = matched_spec.content
                    metadata["trigger_action"] = matched_spec.action
                    metadata["trigger_pattern"] = matched_spec.pattern
                await world.event_bus.publish(
                    PromptReplacementEvent(
                        entity_id=entity_id,
                        prompt_kind="user",
                        source_text=raw_user_text,
                        rendered_text=normalized_text,
                        replacements=replacements,
                        metadata=metadata,
                    )
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

        if self._should_create_boulder(spec=spec):
            self._create_initial_boulder(
                world=world,
                entity_id=entity_id,
                spec=spec,
                raw_user_text=raw_user_text,
            )

        result = await handler(world, entity_id, raw_user_text)
        if result is None:
            return raw_user_text
        return result

    def _create_initial_boulder(
        self,
        *,
        world: World,
        entity_id: EntityId,
        spec: TriggerSpec,
        raw_user_text: str,
    ) -> None:
        if self._registry is None:
            return

        _ = entity_id

        world_names = [world.name] if world.name is not None else []
        worktree_path: str | None = None

        active_plan = self._resolve_active_plan_name(
            spec=spec,
            raw_user_text=raw_user_text,
        )
        self._registry.create_boulder(
            plan_name=active_plan,
            initial_data={
                "trigger_pattern": spec.pattern,
                "status": "created",
                "world_names": world_names,
                "worktree_path": worktree_path,
            },
        )

    @staticmethod
    def _should_create_boulder(*, spec: TriggerSpec) -> bool:
        trigger_signature = f"{spec.pattern} {spec.content}".lower()
        return _PLAN_TRIGGER_PATTERN.search(trigger_signature) is not None

    @staticmethod
    def _resolve_active_plan_name(*, spec: TriggerSpec, raw_user_text: str) -> str:
        trimmed = raw_user_text.strip()
        if trimmed:
            return trimmed
        pattern = spec.pattern.strip()
        if pattern:
            return pattern
        return spec.content

    @staticmethod
    def _find_last_user_message(messages: list[Message]) -> tuple[int, str] | None:
        for index in range(len(messages) - 1, -1, -1):
            message = messages[index]
            if message.role == "user":
                return index, message.content
        return None

    @staticmethod
    def _build_trigger_key(
        trigger_specs: list[TriggerSpec] | None,
        *,
        prompt_config: UserPromptConfigComponent | None,
    ) -> str | None:
        if not trigger_specs:
            return None
        return "\n".join(
            (
                f"{spec.pattern!r}\0{spec.match_mode}\0{spec.action}\0"
                f"{spec.content!r}\0{spec.priority}\0"
                f"{UserPromptNormalizationSystem._handler_signature(spec, prompt_config)}"
            )
            for spec in sorted(
                trigger_specs,
                key=lambda spec: -spec.priority,
            )
        )

    @staticmethod
    def _matching_trigger_spec(
        raw_user_text: str,
        trigger_specs: list[TriggerSpec] | None,
    ) -> TriggerSpec | None:
        if not trigger_specs:
            return None
        for spec in sorted(trigger_specs, key=lambda item: -item.priority):
            if _matches(spec=spec, text=raw_user_text):
                return spec
        return None

    @staticmethod
    def _handler_signature(
        spec: TriggerSpec,
        prompt_config: UserPromptConfigComponent | None,
    ) -> str:
        if spec.action != "script" or prompt_config is None:
            return ""
        handler = prompt_config.script_handlers.get(spec.content)
        if handler is None:
            return "missing"
        return f"{handler.__module__}.{handler.__qualname__}:{id(handler)}"

    @staticmethod
    def _fingerprint_text(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _freeze_superseded_render(
        self,
        *,
        world: World,
        entity_id: EntityId,
        last_user_index: int,
    ) -> None:
        """Persist a superseded transient render into the flat conversation.

        The rendered user prompt (trigger injection / script replacement) is
        substituted at call time, so the stored history keeps the raw text
        while the model saw the rendered text. Once a *newer* user message
        exists, the old slot would silently revert to its raw bytes in the
        next outbound prompt — flipping a mid-history message and forfeiting
        the provider prompt cache for everything after it, every turn.
        Freezing the rendered text into the message the moment it stops being
        the last user message keeps outbound prompts append-only.

        Tree conversations are not writable here and keep the transient
        behaviour.
        """
        rendered = world.get_component(entity_id, RenderedUserPromptComponent)
        if (
            rendered is None
            or rendered.source_message_index is None
            or rendered.source_message_index == last_user_index
            or rendered.source_fingerprint is None
        ):
            return
        conversation = world.get_component(entity_id, ConversationComponent)
        if conversation is None:
            return
        index = rendered.source_message_index
        if not 0 <= index < len(conversation.messages):
            return
        message = conversation.messages[index]
        if message.role != "user":
            return
        # Only freeze the exact message the render was produced from —
        # compaction or trimming may have rewritten the slot since.
        if self._fingerprint_text(message.content) != rendered.source_fingerprint:
            return
        if message.content == rendered.text:
            return
        conversation.messages[index] = Message(
            role="user",
            content=rendered.text,
            parts=message.parts,
            tool_calls=message.tool_calls,
            tool_call_id=message.tool_call_id,
            compaction_metadata=message.compaction_metadata,
        )
        logger.debug(
            "user_prompt_render_frozen",
            entity_id=entity_id,
            message_index=index,
            rendered_length=len(rendered.text),
        )

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
