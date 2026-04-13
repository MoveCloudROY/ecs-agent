from __future__ import annotations

from ecs_agent.components import ConversationComponent
from ecs_agent.core import World
from ecs_agent.types import ConversationTruncatedEvent


class MemorySystem:
    async def process(self, world: World) -> None:
        for entity_id, components in world.query(ConversationComponent):
            conversation = components[0]
            messages = conversation.messages
            max_messages = conversation.max_messages

            if len(messages) <= max_messages:
                continue

            has_system = len(messages) > 0 and messages[0].role == "system"

            last_compaction_index = -1
            for index in range(len(messages) - 1, -1, -1):
                if messages[index].role == "compaction":
                    last_compaction_index = index
                    break

            leading_messages = messages[:1] if has_system else []
            if last_compaction_index >= 0:
                compaction_boundary = messages[last_compaction_index:]
                if has_system and last_compaction_index == 0:
                    compaction_boundary = messages[1:]
                new_messages = leading_messages + compaction_boundary
            else:
                keep_count = max_messages - len(leading_messages)
                keep_count = max(keep_count, 0)
                trailing = messages[-keep_count:] if keep_count > 0 else []
                new_messages = leading_messages + trailing
            removed_count = len(messages) - len(new_messages)

            if removed_count <= 0:
                continue

            conversation.messages = new_messages
            await world.event_bus.publish(
                ConversationTruncatedEvent(
                    entity_id=entity_id, removed_count=removed_count
                )
            )


__all__ = ["MemorySystem", "ConversationTruncatedEvent"]
