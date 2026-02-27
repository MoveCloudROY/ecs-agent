"""Conversation compaction system using LLM summarization."""

from __future__ import annotations

import math

from ecs_agent.components import (
    CompactionConfigComponent,
    ConversationArchiveComponent,
    ConversationComponent,
    LLMComponent,
)
from ecs_agent.core import World
from ecs_agent.logging import get_logger
from ecs_agent.types import CompactionCompleteEvent, CompletionResult, Message

logger = get_logger(__name__)


class CompactionSystem:
    """LLM-based conversation summarization using bisect algorithm."""
    def __init__(self, bisect_ratio: float = 0.5) -> None:
        if bisect_ratio <= 0 or bisect_ratio >= 1:
            raise ValueError("bisect_ratio must be between 0 and 1")
        self.bisect_ratio = bisect_ratio

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

            if len(working_messages) < 2:
                continue

            split_index = self._split_index(len(working_messages))
            older_half = working_messages[:split_index]
            recent_half = working_messages[split_index:]

            summary = await self._summarize(llm_component, older_half)

            archive = world.get_component(entity_id, ConversationArchiveComponent)
            if archive is None:
                archive = ConversationArchiveComponent()
                world.add_component(entity_id, archive)
            archive.archived_summaries.append(summary)

            new_messages: list[Message] = []
            if system_message is not None:
                new_messages.append(system_message)
            new_messages.append(
                Message(
                    role="user",
                    content=f"Previous conversation summary: {summary}",
                )
            )
            new_messages.extend(recent_half)
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
            )

    def _split_index(self, message_count: int) -> int:
        split_index = int(math.floor(message_count * self.bisect_ratio))
        return min(max(split_index, 1), message_count - 1)

    def _estimate_tokens(self, messages: list[Message]) -> int:
        word_count = sum(len(message.content.split()) for message in messages)
        return int(math.ceil(word_count * 1.3))

    async def _summarize(
        self, llm_component: LLMComponent, messages: list[Message]
    ) -> str:
        formatted_messages = "\n".join(
            f"{message.role}: {message.content}" for message in messages
        )
        prompt = (
            "Summarize the following conversation, preserving key decisions, "
            f"facts, and context: {formatted_messages}"
        )
        result = await llm_component.provider.complete(
            messages=[Message(role="user", content=prompt)],
            tools=None,
            stream=False,
        )
        if not isinstance(result, CompletionResult):
            raise RuntimeError("Provider returned stream iterator for compaction")
        return result.message.content


__all__ = ["CompactionSystem"]
