"""Durable message writes, coordinated history changes, and response invalidation."""

from dataclasses import asdict
from uuid import NAMESPACE_URL, uuid4, uuid5
from typing import Any

from ecs_agent.components import ChildStubComponent, ConversationComponent, ResponsesAPIStateComponent, SessionRefComponent
from ecs_agent.core.world import World
from ecs_agent.logging import get_logger
from ecs_agent.observability.context import current_run_id
from ecs_agent.sessions.types import SessionRecord
from ecs_agent.types import EntityId, Message

logger = get_logger(__name__)


def record_session_event(world: World, entity: EntityId, kind: str, payload: dict[str, Any], *, record_id: str | None = None) -> SessionRecord | None:
    """Commit an execution event before publishing its observation counterpart."""
    if world.session_service is None or world.has_component(entity, ChildStubComponent):
        return None
    ref = world.get_component(entity, SessionRefComponent)
    if ref is None or entity not in world._session_writer_tokens:
        raise RuntimeError("Recording requires an installed active session writer")
    conversation = world.get_component(entity, ConversationComponent)
    return world.session_service.append(
        ref, kind, payload, writer_id=world._session_writer_tokens[entity],
        history_id=conversation.history_id if conversation else "",
        revision=conversation.revision if conversation else 0, run_id=current_run_id(), record_id=record_id,
    )


def tool_record_id(world: World, entity: EntityId, tool_call_id: str, stage: str) -> str | None:
    """Derive an execution identity that survives checkpoint restore and process restart."""
    ref = world.get_component(entity, SessionRefComponent)
    if world.session_service is None or ref is None:
        return None
    return "rec_" + uuid5(NAMESPACE_URL, f"{ref.session_id}:{tool_call_id}:{stage}").hex


def record_conversation_message(world: World, entity: EntityId, message: Message, *, record_id: str | None = None) -> SessionRecord | None:
    """Persist an original message independently of its compact model-facing view."""
    if world.session_service is None or world.has_component(entity, ChildStubComponent):
        return None
    ref = world.get_component(entity, SessionRefComponent)
    if ref is None or entity not in world._session_writer_tokens:
        raise RuntimeError("Recording requires an installed active session writer")
    conversation = world.get_component(entity, ConversationComponent)
    return world.session_service.append_message(
        ref, message, writer_id=world._session_writer_tokens[entity],
        history_id=conversation.history_id if conversation else "",
        revision=conversation.revision if conversation else 0, run_id=current_run_id(), record_id=record_id,
    )


def append_conversation_message(
    world: World, entity: EntityId, message: Message, *, recorded_record_id: str | None = None,
) -> None:
    """Commit a message before changing model context, with no cancellation gap."""
    conversation = world.get_component(entity, ConversationComponent)
    if conversation is None:
        conversation = ConversationComponent(messages=[])
        world.add_component(entity, conversation)
    ref = world.get_component(entity, SessionRefComponent)
    if world.session_service is not None and not world.has_component(entity, ChildStubComponent):
        if ref is None or entity not in world._session_writer_tokens:
            raise RuntimeError("Recording requires an installed active session writer")
        if recorded_record_id is None:
            record_conversation_message(world, entity, message)
        else:
            record = world.session_service.store.get_record(recorded_record_id)
            if record.session_id != ref.session_id or record.payload.get("role") != message.role or record.payload.get("tool_call_id") != message.tool_call_id:
                raise ValueError("Recorded message does not belong to this session/tool call")
    conversation.messages.append(message)


def invalidate_response_chain(world: World, entity: EntityId, *, reason: str) -> None:
    """Forget a remote anchor whose provenance or covered history changed."""
    if world.has_component(entity, ResponsesAPIStateComponent):
        world.remove_component(entity, ResponsesAPIStateComponent)
        logger.info("responses_chain_invalidated", entity_id=int(entity), reason=reason)


def replace_conversation_history(
    world: World, entity: EntityId, messages: list[Message], *, reason: str, new_conversation: bool = False,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Replace history atomically with a new revision and no stale remote chain."""
    conversation = world.get_component(entity, ConversationComponent)
    next_history_id = uuid4().hex if new_conversation else (conversation.history_id if conversation else uuid4().hex)
    ref = world.get_component(entity, SessionRefComponent)
    service = world.session_service
    if service is not None and not world.has_component(entity, ChildStubComponent):
        if ref is None or entity not in world._session_writer_tokens:
            raise RuntimeError("Recording requires an installed active session writer")
        writer = world._session_writer_tokens[entity]
        service.append(
            ref, "reset" if new_conversation else "history_replaced",
            {
                **(metadata or {}),
                "reason": reason, "previous_head": ref.head_record_id,
                "previous_message_count": len(conversation.messages) if conversation else 0,
                "replacement_message_count": len(messages),
                "messages": [] if new_conversation else [asdict(message) for message in messages if conversation is None or message not in conversation.messages],
            },
            writer_id=writer, history_id=next_history_id,
            revision=conversation.revision + 1 if conversation else 0, run_id=current_run_id(),
        )
        if new_conversation:
            previous = service.store.get_session(ref.session_id)
            info = service.create_session(agent_name=previous.agent_name, workspace_id=previous.workspace_id, parent_session_id=previous.parent_session_id, source_session_id=previous.session_id)
            next_ref = SessionRefComponent(info.session_id)
            next_writer = service.claim_writer(info.session_id)
            try:
                for message in messages:
                    service.append_message(next_ref, message, writer_id=next_writer, history_id=next_history_id, run_id=current_run_id())
            except Exception as exc:
                service.store.release_writer(next_ref.session_id, next_writer)
                logger.error("session_reset_failed", session_id=next_ref.session_id, exception=str(exc))
                raise
            service.store.release_writer(ref.session_id, writer)
            world.add_component(entity, next_ref)
            world._session_writer_tokens[entity] = next_writer
    if conversation is None:
        conversation = ConversationComponent(messages=list(messages), history_id=next_history_id)
        world.add_component(entity, conversation)
    else:
        conversation.messages = list(messages)
        conversation.revision += 1
        if new_conversation:
            conversation.history_id = next_history_id
    invalidate_response_chain(world, entity, reason=reason)
