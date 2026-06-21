"""Conversation compaction system."""

from __future__ import annotations

import hashlib
import math
from typing import cast

from ecs_agent.accounting.instrumentation import complete_with_llm_invocation_event
from ecs_agent.components import (
    CompactionConfigComponent,
    ContextBudgetConfig,
    CurrentCompactionSummaryComponent,
    ConversationArchiveComponent,
    ConversationComponent,
    EntityRegistryComponent,
    LLMComponent,
    RenderedUserPromptComponent,
    RenderedSystemPromptComponent,
    SubagentNotificationQueueComponent,
    SubagentSessionTableComponent,
)
from ecs_agent.core import World
from ecs_agent.logging import get_logger
from ecs_agent.prompts.message_assembly import apply_outbound_budget
from ecs_agent.providers.protocol import LLMModel
from ecs_agent.providers.registry import ProviderRegistry, get_model
from ecs_agent.systems.subagent_wait import build_subagent_compaction_state
from ecs_agent.systems.system_prompt_render_system import render_compaction_prompt
from ecs_agent.types import (
    CompactionCompleteEvent,
    CompletionResult,
    EntityId,
    Message,
)

logger = get_logger(__name__)

DEFAULT_COMPACTION_PROMPT = (
    "Please summarize the following conversation history. Focus on key "
    "decisions, important context, tool calls made and their results, and any "
    "pending tasks or subagents. Be concise but preserve all critical "
    "information needed to continue the conversation."
)


class CompactionSystem:
    async def process(self, world: World) -> None:
        for entity_id, (config, conversation) in world.query(
            CompactionConfigComponent, ConversationComponent
        ):
            llm_component = world.get_component(entity_id, LLMComponent)
            if llm_component is None:
                continue

            original_tokens = self._estimate_tokens(conversation.messages)
            if original_tokens <= config.threshold_tokens:
                continue

            system_message: Message | None = None
            working_messages = conversation.messages
            if working_messages and working_messages[0].role == "system":
                system_message = working_messages[0]
                working_messages = working_messages[1:]

            working_messages = [
                message for message in working_messages if message.role != "compaction"
            ]

            if len(working_messages) < 2:
                continue

            messages_to_summarize, retained_messages = self._select_compaction_strategy(
                config=config,
                messages=working_messages,
            )

            summary_model = self._resolve_summary_target(
                world=world,
                entity_id=entity_id,
                llm_component=llm_component,
                config=config,
            )
            summary_prompt = self._resolve_compaction_prompt(
                world=world,
                entity_id=entity_id,
                config=config,
            )
            current_summary = world.get_component(
                entity_id, CurrentCompactionSummaryComponent
            )
            summary = await self._summarize(
                world=world,
                entity_id=entity_id,
                model=summary_model,
                messages=self._build_summary_input_messages(
                    previous_summary=(
                        current_summary.summary
                        if current_summary is not None and current_summary.summary
                        else None
                    ),
                    messages_to_summarize=messages_to_summarize,
                ),
                system_prompt=summary_prompt,
                subagent_state=self._render_subagent_summary_state(world, entity_id),
            )

            archive = world.get_component(entity_id, ConversationArchiveComponent)
            if archive is None:
                archive = ConversationArchiveComponent()
                world.add_component(entity_id, archive)
            archive.archived_summaries.append(summary)

            world.add_component(
                entity_id,
                CurrentCompactionSummaryComponent(summary=summary),
            )
            if (
                world.get_component(entity_id, RenderedSystemPromptComponent)
                is not None
            ):
                world.remove_component(entity_id, RenderedSystemPromptComponent)

            new_messages: list[Message] = []
            if system_message is not None:
                new_messages.append(system_message)
            new_messages.extend(retained_messages)
            if not any(message.role != "system" for message in new_messages):
                anchor = self._build_continuation_anchor(
                    world=world,
                    entity_id=entity_id,
                    messages=conversation.messages,
                )
                if anchor is not None:
                    new_messages.append(anchor)
            conversation.messages = new_messages

            compacted_tokens = self._estimate_tokens(new_messages)
            await world.event_bus.publish(
                CompactionCompleteEvent(
                    entity_id=entity_id,
                    original_tokens=original_tokens,
                    compacted_tokens=compacted_tokens,
                )
            )
            logger.info(
                "conversation_compacted",
                entity_id=entity_id,
                original_tokens=original_tokens,
                compacted_tokens=compacted_tokens,
                summary_model=config.summary_model,
                summary_model_id=config.summary_model_id,
                resolved_summary_model=summary_model.model_id,
            )

    def _estimate_tokens(self, messages: list[Message]) -> int:
        word_count = sum(len(message.content.split()) for message in messages)
        return int(math.ceil(word_count * 1.3))

    def _build_continuation_anchor(
        self,
        *,
        world: World,
        entity_id: EntityId,
        messages: list[Message],
    ) -> Message | None:
        last_user = self._find_last_user_message(messages)
        if last_user is None:
            return None

        last_user_index, last_user_message = last_user
        rendered_user_prompt = world.get_component(entity_id, RenderedUserPromptComponent)
        if (
            rendered_user_prompt is not None
            and self._rendered_prompt_matches(
                rendered_user_prompt,
                source_message=last_user_message,
                source_message_index=last_user_index,
            )
        ):
            return Message(role="user", content=rendered_user_prompt.text)

        return Message(
            role="user",
            content=last_user_message.content,
            parts=list(last_user_message.parts) if last_user_message.parts else None,
        )

    def _find_last_user_message(
        self, messages: list[Message]
    ) -> tuple[int, Message] | None:
        for index in range(len(messages) - 1, -1, -1):
            message = messages[index]
            if message.role == "user":
                return index, message
        return None

    def _rendered_prompt_matches(
        self,
        rendered_user_prompt: RenderedUserPromptComponent,
        *,
        source_message: Message,
        source_message_index: int,
    ) -> bool:
        if rendered_user_prompt.source_message_index != source_message_index:
            return False
        return rendered_user_prompt.source_fingerprint == self._fingerprint_text(
            source_message.content
        )

    def _fingerprint_text(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _select_compaction_strategy(
        self,
        *,
        config: CompactionConfigComponent,
        messages: list[Message],
    ) -> tuple[list[Message], list[Message]]:
        if config.compaction_method == "full_history":
            return list(messages), []

        if config.compaction_method == "predrop_then_compact":
            pruned_messages = apply_outbound_budget(
                list(messages),
                system_prompt="",
                context_entries=[],
                config=ContextBudgetConfig(
                    max_tokens=config.threshold_tokens,
                    prune_tool_results=True,
                    prune_reasoning=False,
                    overflow_behavior="truncate",
                ),
            )
            return list(pruned_messages), []

        raise ValueError(f"Unsupported compaction method: {config.compaction_method}")

    def _build_summary_input_messages(
        self,
        *,
        previous_summary: str | None,
        messages_to_summarize: list[Message],
    ) -> list[Message]:
        if previous_summary is None:
            return list(messages_to_summarize)

        return [
            Message(
                role="user",
                content=(
                    "Previous summary:\n\n"
                    f"{previous_summary}\n\n"
                    "Conversation to summarize:"
                ),
            ),
            *messages_to_summarize,
        ]

    def _render_subagent_summary_state(
        self,
        world: World,
        entity_id: EntityId,
    ) -> str | None:
        table = world.get_component(entity_id, SubagentSessionTableComponent)
        queue = world.get_component(entity_id, SubagentNotificationQueueComponent)
        state = build_subagent_compaction_state(table, queue)
        if not state.pending and not state.completed and not state.notifications:
            return None

        lines = ["Subagent session state:"]
        for session_id in state.pending:
            lines.append(f"Pending: {session_id}")
        for session_id, status in state.completed:
            lines.append(f"Completed ({status}): {session_id}")
        lines.extend(state.notifications)
        return "\n".join(lines)

    def _resolve_summary_target(
        self,
        *,
        world: World,
        entity_id: EntityId,
        llm_component: LLMComponent,
        config: CompactionConfigComponent,
    ) -> LLMModel:
        if config.summary_model_id is not None:
            registry = self._resolve_provider_registry(world, entity_id, llm_component)
            return get_model(
                config.summary_model_id,
                registry=registry,
                api_key=self._resolve_api_key(llm_component.model),
            )

        if config.summary_model is not None:
            logger.warning(
                "compaction_summary_model_legacy_deprecated",
                entity_id=entity_id,
                summary_model=config.summary_model,
            )

        return cast(LLMModel, llm_component.model)

    def _resolve_provider_registry(
        self,
        world: World,
        entity_id: EntityId,
        llm_component: LLMComponent,
    ) -> ProviderRegistry:
        provider_registry = getattr(llm_component.model, "registry", None)
        if isinstance(provider_registry, ProviderRegistry):
            return provider_registry

        entity_registry = world.get_component(entity_id, EntityRegistryComponent)
        if entity_registry is not None:
            registry = entity_registry.metadata.get("provider_registry")
            if isinstance(registry, ProviderRegistry):
                return registry

        raise ValueError(
            "summary_model_id requires a ProviderRegistry on the current model "
            "or entity metadata"
        )

    def _resolve_api_key(self, model: LLMModel) -> str | None:
        provider_config = getattr(model, "_provider_config", None)
        api_key = getattr(provider_config, "api_key", None)
        return cast(str | None, api_key)

    def _resolve_compaction_prompt(
        self,
        *,
        world: World,
        entity_id: EntityId,
        config: CompactionConfigComponent,
    ) -> str:
        if config.compaction_prompt_template is None:
            return DEFAULT_COMPACTION_PROMPT

        return render_compaction_prompt(
            config.compaction_prompt_template,
            world,
            entity_id,
        )

    async def _summarize(
        self,
        *,
        world: World,
        entity_id: EntityId,
        model: LLMModel,
        messages: list[Message],
        system_prompt: str,
        subagent_state: str | None,
    ) -> str:
        formatted_messages = "\n".join(
            f"{message.role}: {message.content}" for message in messages
        )
        if subagent_state is not None:
            formatted_messages = f"{formatted_messages}\n\n{subagent_state}"
        result = await complete_with_llm_invocation_event(
            event_bus=world.event_bus,
            entity_id=entity_id,
            model=model,
            messages=[
                Message(role="system", content=system_prompt),
                Message(role="user", content=formatted_messages),
            ],
            tools=None,
            stream=False,
            operation="compaction",
        )
        if not isinstance(result, CompletionResult):
            raise RuntimeError("Provider returned stream iterator for compaction")
        return result.message.content


__all__ = ["DEFAULT_COMPACTION_PROMPT", "CompactionSystem"]
