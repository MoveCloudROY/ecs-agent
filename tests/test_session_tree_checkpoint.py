"""Conversation-tree recording and in-place checkpoint recovery contracts."""

import pytest

from ecs_agent.components import (
    CheckpointComponent, ConversationComponent, ConversationTreeComponent,
    ResponsesAPIStateComponent, SessionPolicyComponent, SessionRefComponent,
)
from ecs_agent.conversation_tree import add_message, create_branch, switch_branch, revert_to_message
from ecs_agent.core import World
from ecs_agent.history import append_conversation_message
from ecs_agent.sessions import InMemorySessionStore, SQLiteSessionStore, SessionService, install_session_recording
from ecs_agent.sessions.resources import read_session_resource
from ecs_agent.systems.checkpoint import CheckpointSystem
from ecs_agent.types import Message


@pytest.fixture(params=["memory", "sqlite"])
def service(request, tmp_path):
    store = InMemorySessionStore() if request.param == "memory" else SQLiteSessionStore(tmp_path / "sessions.sqlite3")
    result = SessionService(store)
    yield result
    result.close()


def setup(service, tree=None):
    world = World()
    entity = world.create_entity()
    world.add_component(entity, ConversationComponent(messages=[]))
    if tree is not None:
        world.add_component(entity, tree)
    install_session_recording(world, entity, service, workspace_id="project", agent_name="main", register_tools=False)
    return world, entity


def transcript(service, ref):
    uri = service.transcript_uri(service.store.get_session(ref.session_id))
    return uri, read_session_resource(service, ref.session_id, SessionPolicyComponent(), uri, limit=500)


def test_tree_import_and_branch_switch_preserve_nodes_and_selected_path(service):
    tree = ConversationTreeComponent()
    root = add_message(tree, "user", "root")
    left = add_message(tree, "assistant", "left evidence", parent_id=root.id)
    right = add_message(tree, "assistant", "right evidence", parent_id=root.id)
    create_branch(tree, "left", left.id)
    create_branch(tree, "right", right.id)
    switch_branch(tree, "left")
    world, entity = setup(service, tree)
    ref = world.get_component(entity, SessionRefComponent)
    old_uri, left_text = transcript(service, ref)
    assert "left evidence" in left_text and "right evidence" not in left_text
    world.add_component(entity, ResponsesAPIStateComponent(previous_response_id="old"))
    switch_branch(tree, "right")
    _, right_text = transcript(service, ref)
    assert "right evidence" in right_text and "left evidence" not in right_text
    assert world.get_component(entity, ResponsesAPIStateComponent) is None
    assert world.get_component(entity, ConversationComponent).revision == 1
    assert read_session_resource(service, ref.session_id, SessionPolicyComponent(), old_uri, limit=500) == left_text
    hits = service.search(ref.session_id, SessionPolicyComponent(), "evidence").items
    assert len(hits) == 1
    records = list(service.store.records(ref.session_id))
    assert {r.payload.get("source", {}).get("tree_message_id") for r in records if r.kind == "message"} == {root.id, left.id, right.id}


def test_detached_tree_node_does_not_change_selected_transcript(service):
    tree = ConversationTreeComponent()
    world, entity = setup(service, tree)
    root = add_message(tree, "user", "initial root")
    create_branch(tree, "main", root.id)
    switch_branch(tree, "main")
    ref = world.get_component(entity, SessionRefComponent)
    _, before = transcript(service, ref)
    detached = add_message(tree, "assistant", "detached evidence", parent_id=root.id)
    assert transcript(service, ref)[1] == before
    create_branch(tree, "other", detached.id)
    switch_branch(tree, "other")
    assert "detached evidence" in transcript(service, ref)[1]
    revert_to_message(tree, root.id)
    assert "detached evidence" not in transcript(service, ref)[1]


def test_tree_write_failure_leaves_nodes_and_branch_unmodified(service, monkeypatch):
    tree = ConversationTreeComponent()
    world, entity = setup(service, tree)
    root = add_message(tree, "user", "root")
    create_branch(tree, "main", root.id)
    switch_branch(tree, "main")
    before = dict(tree.messages)

    def fail(*args, **kwargs):
        raise OSError("cannot persist tree")

    monkeypatch.setattr(service.store, "append", fail)
    with pytest.raises(OSError):
        add_message(tree, "assistant", "not committed", parent_id=root.id)
    assert tree.messages == before
    with pytest.raises(OSError):
        create_branch(tree, "uncommitted", root.id)
    assert "uncommitted" not in tree.branches


async def test_checkpoint_undo_rebinds_writer_to_saved_head_and_keeps_future(service):
    world, entity = setup(service)
    world.add_component(entity, CheckpointComponent())
    append_conversation_message(world, entity, Message(role="user", content="saved message"))
    checkpoint = CheckpointSystem()
    await checkpoint.process(world)
    saved_ref = world.get_component(entity, SessionRefComponent)
    saved_head = saved_ref.head_record_id
    append_conversation_message(world, entity, Message(role="assistant", content="future message"))
    future_head = saved_ref.head_record_id
    await checkpoint.process(world)
    await checkpoint.undo(world, providers={}, tool_handlers={})
    append_conversation_message(world, entity, Message(role="assistant", content="new branch"))
    ref = world.get_component(entity, SessionRefComponent)
    path = list(service.store.path(ref.session_id, ref.head_record_id))
    assert saved_head in [r.record_id for r in path]
    assert future_head not in [r.record_id for r in path]
    assert service.store.get_record(future_head).payload["content"] == "future message"
    assert world.get_component(entity, ConversationComponent).messages[-1].content == "new branch"


def test_legacy_tree_checkpoint_import_marks_incomplete_history(service):
    tree = ConversationTreeComponent()
    node = add_message(tree, "user", "remaining text only")
    create_branch(tree, "saved", node.id)
    switch_branch(tree, "saved")
    world = World()
    entity = world.create_entity()
    world.add_component(entity, tree)
    install_session_recording(world, entity, service, workspace_id="project", agent_name="legacy", register_tools=False, restore=True)
    ref = world.get_component(entity, SessionRefComponent)
    text = transcript(service, ref)[1]
    assert "remaining text only" in text
    assert "history_incomplete" in text
