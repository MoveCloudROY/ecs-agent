"""Coordinated history replacement and remote response invalidation."""

from uuid import uuid4

from ecs_agent.components import ConversationComponent, ResponsesAPIStateComponent
from ecs_agent.core.world import World
from ecs_agent.logging import get_logger
from ecs_agent.types import EntityId, Message

logger = get_logger(__name__)


def invalidate_response_chain(world: World, entity: EntityId, *, reason: str) -> None:
    """Forget a remote anchor whose provenance or covered history changed."""
    if world.has_component(entity, ResponsesAPIStateComponent):
        world.remove_component(entity, ResponsesAPIStateComponent)
        logger.info("responses_chain_invalidated", entity_id=int(entity), reason=reason)


def replace_conversation_history(
    world: World, entity: EntityId, messages: list[Message], *, reason: str, new_conversation: bool = False,
) -> None:
    """Replace history atomically with a new revision and no stale remote chain."""
    conversation = world.get_component(entity, ConversationComponent)
    if conversation is None:
        conversation = ConversationComponent(messages=list(messages))
        world.add_component(entity, conversation)
    else:
        conversation.messages = list(messages)
        conversation.revision += 1
        if new_conversation:
            conversation.history_id = uuid4().hex
    invalidate_response_chain(world, entity, reason=reason)
