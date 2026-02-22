from __future__ import annotations

from ecs_agent.components import CollaborationComponent, ConversationComponent
from ecs_agent.core import World
from ecs_agent.types import Message, MessageDeliveredEvent


class CollaborationSystem:
    def __init__(self, priority: int = 0) -> None:
        self.priority = priority

    async def process(self, world: World) -> None:
        for entity_id, (collaboration,) in world.query(CollaborationComponent):
            if not collaboration.inbox:
                continue

            conversation = world.get_component(entity_id, ConversationComponent)
            if conversation is None:
                continue

            for sender_id, message in collaboration.inbox:
                conversation.messages.append(
                    Message(
                        role="user", content=f"From: {sender_id}: {message.content}"
                    )
                )
                await world.event_bus.publish(
                    MessageDeliveredEvent(
                        from_entity=sender_id,
                        to_entity=entity_id,
                        message=message,
                    )
                )

            collaboration.inbox.clear()


__all__ = ["CollaborationSystem"]
