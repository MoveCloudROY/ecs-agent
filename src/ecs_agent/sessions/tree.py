"""Runtime binding between existing conversation-tree mutations and record paths."""

from dataclasses import replace
from typing import cast
from uuid import NAMESPACE_URL, uuid5
from weakref import WeakValueDictionary

from ecs_agent.components import (
    ChildStubComponent, ConversationComponent, ConversationTreeComponent, SessionRefComponent,
)
from ecs_agent.core.world import World
from ecs_agent.observability.context import current_run_id
from ecs_agent.types import AcceptedMessageRole, ConversationMessage, EntityId, Message

# These bindings contain no persistent state. Worlds own components; weak values
# remove entries when a World is collected, and identity checks prevent id reuse.
_TREE_WORLDS: WeakValueDictionary[int, World] = WeakValueDictionary()


def _binding(tree: ConversationTreeComponent) -> tuple[World, EntityId, SessionRefComponent] | None:
    world = _TREE_WORLDS.get(id(tree))
    if world is None or world.session_service is None:
        return None
    for entity, (candidate, ref) in world.query(ConversationTreeComponent, SessionRefComponent):
        if candidate is tree and not world.has_component(entity, ChildStubComponent):
            if entity not in world._session_writer_tokens:
                raise RuntimeError("Tree mutation requires an active session writer")
            return world, entity, ref
    return None


def tree_record_id(session_id: str, message_id: str) -> str:
    """Retain node identity across checkpoint restore and import retries."""
    return "rec_" + uuid5(NAMESPACE_URL, f"{session_id}:tree:{message_id}").hex


def record_tree_message(
    tree: ConversationTreeComponent, message: ConversationMessage, *, incomplete: bool = False,
) -> None:
    """Persist a node without selecting its branch or changing the active head."""
    binding = _binding(tree)
    if binding is None:
        return
    world, entity, ref = binding
    service = world.session_service
    assert service is not None
    record_id = tree_record_id(ref.session_id, message.id)
    try:
        existing = service.store.get_record(record_id)
    except FileNotFoundError:
        existing = None
    if existing is not None:
        if existing.session_id != ref.session_id or existing.payload.get("source", {}).get("tree_message_id") != message.id:
            raise ValueError("Tree record identity conflict")
        return
    parent_id = tree_record_id(ref.session_id, message.parent_message_id) if message.parent_message_id else None
    working = world.get_component(entity, ConversationComponent)
    service.append_message(
        SessionRefComponent(ref.session_id, parent_id),
        Message(role=cast(AcceptedMessageRole, message.role), content=message.content or "", tool_calls=message.tool_calls, tool_call_id=message.tool_call_id),
        writer_id=world._session_writer_tokens[entity], record_id=record_id,
        history_id=working.history_id if working else "", revision=working.revision if working else 0,
        run_id=current_run_id(), branch=True, select_head=False,
        metadata={"tree_message_id": message.id, "parent_message_id": message.parent_message_id, "tree_metadata": message.metadata, "history_incomplete": incomplete},
    )


def record_tree_branch(
    tree: ConversationTreeComponent, branch_id: str, leaf_message_id: str,
    *, operation: str, select: bool, update_revision: bool = True,
) -> None:
    """Commit branch changes before mutating the tree's selected model path."""
    binding = _binding(tree)
    if binding is None:
        return
    world, entity, ref = binding
    service = world.session_service
    assert service is not None
    leaf = tree_record_id(ref.session_id, leaf_message_id)
    service.store.get_record(leaf)
    candidate = replace(ref, head_record_id=leaf if select else ref.head_record_id)
    conversation = world.get_component(entity, ConversationComponent)
    record = service.append(
        candidate, "branch", {"operation": operation, "branch_id": branch_id, "tree_leaf_id": leaf_message_id, "previous_head": ref.head_record_id},
        writer_id=world._session_writer_tokens[entity], branch=True, select_head=select,
        history_id=conversation.history_id if conversation else "",
        revision=(conversation.revision + int(update_revision and select)) if conversation else 0,
        run_id=current_run_id(),
    )
    if select:
        ref.head_record_id = record.record_id
        if update_revision:
            from ecs_agent.history import invalidate_response_chain

            if conversation is not None:
                conversation.revision += 1
            invalidate_response_chain(world, entity, reason=operation)


def bind_conversation_tree(
    world: World, entity: EntityId, tree: ConversationTreeComponent, *, select: bool = True,
    incomplete: bool = False,
) -> None:
    """Import existing nodes once, then bind the tree's synchronous write APIs."""
    if world.session_service is None or world.has_component(entity, ChildStubComponent):
        return
    _TREE_WORLDS[id(tree)] = world
    imported: set[str] = set()
    remaining = dict(tree.messages)
    while remaining:
        ready = [node for node in remaining.values() if node.parent_message_id is None or node.parent_message_id in imported]
        if not ready:
            raise ValueError("Conversation tree has a missing parent or cycle")
        for node in ready:
            record_tree_message(tree, node, incomplete=incomplete)
            imported.add(node.id)
            del remaining[node.id]
    if select and tree.current_branch_id is not None:
        branch = tree.branches[tree.current_branch_id]
        record_tree_branch(tree, branch.branch_id, branch.leaf_message_id, operation="tree_import", select=True, update_revision=False)
