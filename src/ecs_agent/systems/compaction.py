"""Conversation compaction system."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import replace
from typing import cast

from ecs_agent.accounting.instrumentation import complete_with_llm_invocation_event
from ecs_agent.context_windows import resolve_context_budget
from ecs_agent.components import (
    CompactionConfigComponent,
    ContextTrimConfig,
    CurrentCompactionSummaryComponent,
    ConversationArchiveComponent,
    ConversationComponent,
    EntityRegistryComponent,
    LLMComponent,
    RenderedUserPromptComponent,
    TokenUsageComponent,
)
from ecs_agent.core import World
from ecs_agent.logging import get_logger
from ecs_agent.token_counting import count_tokens
from ecs_agent.prompts.message_assembly import (
    _drop_oldest_tool_span,
    trim_context_to_fit,
)
from ecs_agent.providers.protocol import LLMModel
from ecs_agent.providers.registry import ProviderRegistry, get_model
from ecs_agent.prompts.compaction_context import (
    CompactionContextProvider,
    DEFAULT_COMPACTION_CONTEXT_PROVIDERS,
    render_compaction_context_blocks,
)
from ecs_agent.prompts.template_render import render_compaction_prompt
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
    def __init__(
        self,
        context_providers: Sequence[CompactionContextProvider] | None = None,
    ) -> None:
        self._context_providers: tuple[CompactionContextProvider, ...] = (
            tuple(context_providers)
            if context_providers is not None
            else DEFAULT_COMPACTION_CONTEXT_PROVIDERS
        )

    async def process(self, world: World) -> None:
        for entity_id, (config, conversation) in world.query(
            CompactionConfigComponent, ConversationComponent
        ):
            llm_component = world.get_component(entity_id, LLMComponent)
            if llm_component is None:
                continue

            # ISSUE-5 pipeline: estimate -> trim -> (still over) summarize.
            trim_config = world.get_component(entity_id, ContextTrimConfig)
            budget = self._resolve_trim_budget(trim_config, llm_component)
            chars_per_token = (
                trim_config.token_estimation_chars_per_token
                if trim_config is not None
                else 4.0
            )
            # When a trim budget is known it also acts as the summary trigger;
            # otherwise fall back to the compaction threshold (legacy behaviour).
            trigger = budget if budget is not None else config.threshold_tokens

            original_tokens = self._current_context_tokens(
                world, entity_id, conversation, chars_per_token
            )
            if original_tokens <= trigger:
                continue

            # Trim step: permanently drop droppable content to try to fit under
            # budget before paying for an LLM summary.
            if trim_config is not None and budget is not None:
                trimmed, changed = self._trim_history(
                    conversation.messages, budget, trim_config
                )
                if changed:
                    dropped = len(conversation.messages) - len(trimmed)
                    conversation.messages = trimmed
                    self._invalidate_usage_anchor(world, entity_id)
                    logger.info(
                        "context_trimmed",
                        entity_id=int(entity_id),
                        dropped_messages=dropped,
                        budget=budget,
                    )
                # Use the same estimator the trim loop targeted (counts replayed
                # reasoning) to decide whether trimming freed enough space.
                if (
                    self._estimate_local_tokens(conversation.messages, chars_per_token)
                    <= budget
                ):
                    # Trimming freed enough space; no summary needed this turn.
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
                context_state=render_compaction_context_blocks(
                    world, entity_id, self._context_providers
                ),
            )

            archive = world.get_component(entity_id, ConversationArchiveComponent)
            if archive is None:
                archive = ConversationArchiveComponent()
                world.add_component(entity_id, archive)
            archive.archived_summaries.append(summary)

            # RenderedSystemPromptComponent is deliberately NOT invalidated here:
            # the render system re-renders when the summary fingerprint in its
            # cache key changes. Only the render system may delete that component.
            world.add_component(
                entity_id,
                CurrentCompactionSummaryComponent(summary=summary),
            )

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

    def _current_context_tokens(
        self,
        world: World,
        entity_id: EntityId,
        conversation: ConversationComponent,
        chars_per_token: float = 4.0,
    ) -> int:
        """Best estimate of the next call's input token count.

        Calibrated against ground truth when available: the provider-reported
        ``last_prompt_tokens`` (the real size of the last input — system, tools
        and history included) plus a local estimate of only the messages appended
        since that call. Falls back to a pure local estimate before the first
        call, or after compaction shrinks the conversation below the anchor
        (``last_prompt_message_count``).
        """
        messages = conversation.messages
        usage = world.get_component(entity_id, TokenUsageComponent)
        if (
            usage is not None
            and usage.call_count > 0
            and 0 <= usage.last_prompt_message_count <= len(messages)
        ):
            appended = messages[usage.last_prompt_message_count :]
            return usage.last_prompt_tokens + self._estimate_tokens(
                appended, chars_per_token
            )
        return self._estimate_tokens(messages, chars_per_token)

    def _estimate_tokens(
        self, messages: list[Message], chars_per_token: float = 4.0
    ) -> int:
        # Real BPE count when tiktoken is available; CJK-aware fallback otherwise.
        # (A word count catastrophically under-counts CJK/code — ISSUE-8.)
        text = "".join(message.content or "" for message in messages)
        return count_tokens(text, fallback_chars_per_token=chars_per_token)

    @staticmethod
    def _resolve_trim_budget(
        trim_config: ContextTrimConfig | None, llm_component: LLMComponent
    ) -> int | None:
        """Budget for the trim step: explicit ``max_tokens`` or model-derived."""
        if trim_config is None:
            return None
        if trim_config.max_tokens is not None:
            return trim_config.max_tokens
        model = getattr(llm_component, "model", None)
        model_id = getattr(model, "model_id", "") if model is not None else ""
        return resolve_context_budget(model_id)

    def _estimate_local_tokens(
        self, messages: list[Message], chars_per_token: float = 4.0
    ) -> int:
        """Local estimate that also counts replayed ``reasoning_content`` (which
        is sent to the model), so trimming reasoning actually reduces the count."""
        parts: list[str] = []
        for message in messages:
            if message.content:
                parts.append(message.content)
            if message.reasoning_content:
                parts.append(message.reasoning_content)
        return count_tokens("".join(parts), fallback_chars_per_token=chars_per_token)

    def _trim_history(
        self,
        messages: list[Message],
        budget: int,
        trim_config: ContextTrimConfig,
    ) -> tuple[list[Message], bool]:
        """Permanently drop droppable content toward ``budget``.

        Oldest tool spans first (atomic assistant-tool-call + results), then
        (optionally) strip replayed reasoning from the oldest assistant messages.
        The most recent ``protect_recent_turns`` messages are never touched.
        Returns ``(trimmed_messages, changed)``.
        """
        result = list(messages)
        changed = False
        protect = max(0, trim_config.protect_recent_turns)
        cpt = trim_config.token_estimation_chars_per_token

        if trim_config.trim_tool_results:
            while self._estimate_local_tokens(result, cpt) > budget:
                # Recompute the protected boundary each round (result shrinks).
                protect_from = max(0, len(result) - protect)
                nxt = _drop_oldest_tool_span(result, protect_from=protect_from)
                if len(nxt) == len(result):
                    break
                result = nxt
                changed = True

        if trim_config.trim_reasoning:
            # Keep the newest reasoning-bearing assistant message (latest
            # thinking), and never strip reasoning from a tool-calling message —
            # its thinking + signature is load-bearing for extended-thinking
            # tool-use replay (C).
            last_reasoning_idx = max(
                (
                    index
                    for index, message in enumerate(result)
                    if message.role == "assistant"
                    and (message.reasoning_content or message.reasoning_signature)
                ),
                default=-1,
            )
            protect_from = max(0, len(result) - protect)
            for index, message in enumerate(result):
                if index >= protect_from:
                    break
                if self._estimate_local_tokens(result, cpt) <= budget:
                    break
                if index == last_reasoning_idx:
                    continue
                if message.role != "assistant" or message.tool_calls:
                    continue
                if not (message.reasoning_content or message.reasoning_signature):
                    continue
                result[index] = replace(
                    message, reasoning_content=None, reasoning_signature=None
                )
                changed = True

        return result, changed

    @staticmethod
    def _invalidate_usage_anchor(world: World, entity_id: EntityId) -> None:
        """Drop the compaction calibration anchor after trimming rewrites history
        (the recorded ``last_prompt_tokens`` no longer matches the message list)."""
        usage = world.get_component(entity_id, TokenUsageComponent)
        if usage is not None:
            usage.last_prompt_message_count = -1

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
            pruned_messages = trim_context_to_fit(
                list(messages),
                system_prompt="",
                context_entries=[],
                config=ContextTrimConfig(
                    max_tokens=config.threshold_tokens,
                    trim_tool_results=True,
                    trim_reasoning=False,
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
        context_state: str | None,
    ) -> str:
        formatted_messages = "\n".join(
            f"{message.role}: {message.content}" for message in messages
        )
        if context_state is not None:
            formatted_messages = f"{formatted_messages}\n\n{context_state}"
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
