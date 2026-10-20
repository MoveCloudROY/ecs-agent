"""Durable message writes, coordinated history changes, and response invalidation."""

from dataclasses import asdict, dataclass, replace
from copy import deepcopy
from uuid import NAMESPACE_URL, uuid4, uuid5
from typing import Any

from ecs_agent.components import AgentTurnComponent, ChildStubComponent, ConversationComponent, ResponsesAPIStateComponent, SessionRefComponent
from ecs_agent.core.world import World
from ecs_agent.logging import get_logger
from ecs_agent.observability.context import current_run_id
from ecs_agent.sessions.types import SessionRecord
from ecs_agent.types import EntityId, Message

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class HistorySnapshot:
    """An isolated copy of one model-facing history revision."""

    history_id: str
    revision: int
    messages: tuple[Message, ...]


def record_session_event(world: World, entity: EntityId, kind: str, payload: dict[str, Any], *, record_id: str | None = None) -> SessionRecord | None:
    """Commit an execution event before publishing its observation counterpart."""
    if world.session_service is None or world.has_component(entity, ChildStubComponent):
        return None
    ref = world.get_component(entity, SessionRefComponent)
    if ref is None or entity not in world._session_writer_tokens:
        raise RuntimeError("Recording requires an installed active session writer")
    conversation = world.get_component(entity, ConversationComponent)
    turn = world.get_component(entity, AgentTurnComponent)
    if turn is not None:
        payload = {**payload, "turn_id": turn.turn_id}
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


def input_normalization_record_id(world: World, entity: EntityId, turn_id: str, stage: str) -> str | None:
    """Identify an input script attempt across checkpoint branches and runs."""
    ref = world.get_component(entity, SessionRefComponent)
    if world.session_service is None or ref is None:
        return None
    return "rec_" + uuid5(NAMESPACE_URL, f"{ref.session_id}:input_normalization:{turn_id}:{stage}").hex


def record_conversation_message(world: World, entity: EntityId, message: Message, *, record_id: str | None = None) -> SessionRecord | None:
    """Persist an original message independently of its compact model-facing view."""
    message = _with_turn(world, entity, message)
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
    message = _with_turn(world, entity, message)
    conversation = world.get_component(entity, ConversationComponent)
    new_conversation = conversation is None
    if conversation is None:
        conversation = ConversationComponent(messages=[])
    ref = world.get_component(entity, SessionRefComponent)
    if world.session_service is not None and not world.has_component(entity, ChildStubComponent):
        if ref is None or entity not in world._session_writer_tokens:
            raise RuntimeError("Recording requires an installed active session writer")
        if recorded_record_id is None:
            world.session_service.append_message(
                ref, message, writer_id=world._session_writer_tokens[entity],
                history_id=conversation.history_id, revision=conversation.revision, run_id=current_run_id(),
            )
        else:
            record = world.session_service.store.get_record(recorded_record_id)
            if record.session_id != ref.session_id or record.payload.get("role") != message.role or record.payload.get("tool_call_id") != message.tool_call_id:
                raise ValueError("Recorded message does not belong to this session/tool call")
    if new_conversation:
        world.add_component(entity, conversation)
    conversation.messages.append(deepcopy(message))
    turn = world.get_component(entity, AgentTurnComponent)
    if turn is not None and message.role == "assistant" and message.turn_id == turn.turn_id:
        turn.last_message = deepcopy(message)


def _with_turn(world: World, entity: EntityId, message: Message) -> Message:
    turn = world.get_component(entity, AgentTurnComponent)
    if turn is not None and turn.status == "running":
        # Providers and callers may reuse a Message instance. Assigning a new
        # turn must never rewrite a message already retained by an earlier one.
        message = deepcopy(message)
        message.turn_id = turn.turn_id
    return message


def invalidate_response_chain(world: World, entity: EntityId, *, reason: str) -> None:
    """Forget a remote anchor whose provenance or covered history changed."""
    if world.has_component(entity, ResponsesAPIStateComponent):
        world.remove_component(entity, ResponsesAPIStateComponent)
        logger.info("responses_chain_invalidated", entity_id=int(entity), reason=reason)


def replace_conversation_history(
    world: World, entity: EntityId, messages: list[Message], *, reason: str, new_conversation: bool = False,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Publish a complete history revision after its durable commit succeeds."""
    from ecs_agent.serialization import WorldSerializer

    if not reason.strip():
        raise ValueError("History replacement requires an explicit reason")
    messages = deepcopy(messages)
    conversation = world.get_component(entity, ConversationComponent)
    next_history_id = uuid4().hex if new_conversation else (conversation.history_id if conversation else uuid4().hex)
    next_revision = conversation.revision + 1 if conversation else 0
    ref = world.get_component(entity, SessionRefComponent)
    service = world.session_service
    if service is not None and not world.has_component(entity, ChildStubComponent):
        if ref is None or entity not in world._session_writer_tokens:
            raise RuntimeError("Recording requires an installed active session writer")
        writer = world._session_writer_tokens[entity]
        payload = {
            **(metadata or {}), "reason": reason, "previous_head": ref.head_record_id,
            "previous_message_count": len(conversation.messages) if conversation else 0,
            "replacement_message_count": len(messages), "snapshot_format": "complete",
            "messages": [WorldSerializer._message_to_dict(message) for message in messages],
        }
        if new_conversation:
            previous = service.store.get_session(ref.session_id)
            next_ref: SessionRefComponent | None = None
            next_writer: str | None = None
            reset_id = "rec_" + uuid4().hex
            transaction = getattr(service.store, "transaction", None)
            if transaction is None:
                raise RuntimeError("Session reset requires a store with atomic transaction support")
            try:
                with transaction():
                    info = service.create_session(agent_name=previous.agent_name, workspace_id=previous.workspace_id, parent_session_id=previous.parent_session_id, source_session_id=previous.session_id)
                    next_ref = SessionRefComponent(info.session_id)
                    next_writer = service.claim_writer(info.session_id)
                    for message in messages:
                        service.append_message(next_ref, message, writer_id=next_writer, history_id=next_history_id, revision=next_revision, run_id=current_run_id())
                    service.append(next_ref, "history_replaced", payload, writer_id=next_writer, history_id=next_history_id, revision=next_revision)
                    service.append(
                        replace(ref), "reset", {"reason": reason, "next_session_id": info.session_id, "next_head": next_ref.head_record_id},
                        writer_id=writer, history_id=next_history_id, revision=next_revision, record_id=reset_id,
                    )
                    service.store.release_writer(ref.session_id, writer)
            except Exception as exc:
                logger.error("session_reset_failed", session_id=ref.session_id, exception=str(exc))
                # A lost commit acknowledgement does not imply rollback.
                try:
                    committed = service.store.get_record(reset_id)
                except Exception as probe_error:
                    logger.error("session_reset_unconfirmed", session_id=ref.session_id, exception=str(probe_error))
                    raise exc from probe_error
                if next_ref is None or next_writer is None or committed.session_id != ref.session_id or committed.payload.get("next_session_id") != next_ref.session_id:
                    raise
                service.store.claim_writer(next_ref.session_id, next_writer)
                service.store.release_writer(ref.session_id, writer)
            assert next_ref is not None and next_writer is not None
            world.add_component(entity, next_ref)
            world._session_writer_tokens[entity] = next_writer
        else:
            service.append(ref, "history_replaced", payload, writer_id=writer, history_id=next_history_id, revision=next_revision, run_id=current_run_id())
    if conversation is None:
        conversation = ConversationComponent(messages=messages, history_id=next_history_id, revision=next_revision)
        world.add_component(entity, conversation)
    else:
        conversation.messages = messages
        conversation.revision = next_revision
        if new_conversation:
            conversation.history_id = next_history_id
    invalidate_response_chain(world, entity, reason=reason)
