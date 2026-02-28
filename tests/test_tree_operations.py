"""Tests for tree conversation utility functions."""

import uuid
from datetime import datetime, timezone

import pytest

from ecs_agent.components.definitions import ConversationTreeComponent
from ecs_agent.conversation_tree import (
    add_message,
    create_branch,
    get_active_leaf,
    get_branch_path,
    get_siblings,
    linearize,
    switch_branch,
)
from ecs_agent.types import ConversationBranch, ConversationMessage, Message, ToolCall


def test_add_message_creates_message_with_uuid() -> None:
    """add_message generates uuid for message id and stores in tree."""
    tree = ConversationTreeComponent()

    msg = add_message(tree, role="user", content="hello", parent_id=None)

    assert msg.id in tree.messages
    assert tree.messages[msg.id] == msg
    assert msg.role == "user"
    assert msg.content == "hello"
    assert msg.parent_message_id is None
    # UUID format check (36 chars with hyphens)
    assert len(msg.id) == 36
    assert msg.id.count("-") == 4


def test_add_message_sets_created_at_timestamp() -> None:
    """add_message sets created_at to ISO timestamp."""
    tree = ConversationTreeComponent()

    before = datetime.now(timezone.utc)
    msg = add_message(tree, role="assistant", content="hi", parent_id=None)
    after = datetime.now(timezone.utc)

    assert msg.created_at != ""
    # Parse ISO timestamp
    created = datetime.fromisoformat(msg.created_at.replace("Z", "+00:00"))
    assert before <= created <= after


def test_add_message_with_parent_creates_child() -> None:
    """add_message with parent_id creates child message."""
    tree = ConversationTreeComponent()

    root = add_message(tree, role="user", content="hello", parent_id=None)
    child = add_message(tree, role="assistant", content="hi there", parent_id=root.id)

    assert child.parent_message_id == root.id
    assert root.id in tree.messages
    assert child.id in tree.messages


def test_add_message_with_tool_calls() -> None:
    """add_message supports tool_calls and tool_call_id."""
    tree = ConversationTreeComponent()

    tc = ToolCall(id="call_123", name="search", arguments={"query": "test"})
    msg = add_message(
        tree,
        role="assistant",
        content=None,
        parent_id=None,
        tool_calls=[tc],
        tool_call_id=None,
    )

    assert msg.tool_calls == [tc]
    assert msg.tool_call_id is None

    result_msg = add_message(
        tree,
        role="tool",
        content="result data",
        parent_id=msg.id,
        tool_calls=None,
        tool_call_id="call_123",
    )

    assert result_msg.tool_call_id == "call_123"
    assert result_msg.tool_calls is None


def test_linearize_returns_root_to_leaf_path() -> None:
    """linearize builds root-to-leaf Message list from conversation tree."""
    tree = ConversationTreeComponent()

    root = add_message(tree, role="user", content="hello", parent_id=None)
    msg2 = add_message(tree, role="assistant", content="hi", parent_id=root.id)
    msg3 = add_message(tree, role="user", content="how are you?", parent_id=msg2.id)

    result = linearize(tree, msg3.id)

    assert len(result) == 3
    assert result[0].role == "user"
    assert result[0].content == "hello"
    assert result[1].role == "assistant"
    assert result[1].content == "hi"
    assert result[2].role == "user"
    assert result[2].content == "how are you?"
    # Verify return type is list[Message]
    assert all(isinstance(msg, Message) for msg in result)


def test_linearize_follows_correct_branch() -> None:
    """linearize follows parent pointer, skipping sibling branches."""
    tree = ConversationTreeComponent()

    root = add_message(tree, role="user", content="root", parent_id=None)
    # Two branches from root
    branch_a = add_message(
        tree, role="assistant", content="branch A", parent_id=root.id
    )
    branch_b = add_message(
        tree, role="assistant", content="branch B", parent_id=root.id
    )
    # Continue branch B
    leaf_b = add_message(tree, role="user", content="leaf B", parent_id=branch_b.id)

    result = linearize(tree, leaf_b.id)

    assert len(result) == 3
    assert result[0].content == "root"
    assert result[1].content == "branch B"
    assert result[2].content == "leaf B"
    # Verify branch_a is NOT in the linearized path
    assert not any(msg.content == "branch A" for msg in result)


def test_linearize_converts_to_message_list() -> None:
    """linearize output is list[Message] compatible with existing systems."""
    tree = ConversationTreeComponent()

    tc = ToolCall(id="call_1", name="foo", arguments={})
    root = add_message(tree, role="user", content="test", parent_id=None)
    child = add_message(
        tree,
        role="assistant",
        content=None,
        parent_id=root.id,
        tool_calls=[tc],
        tool_call_id=None,
    )

    result = linearize(tree, child.id)

    assert isinstance(result, list)
    assert isinstance(result[0], Message)
    assert isinstance(result[1], Message)
    assert result[1].tool_calls == [tc]
    # Message type has: role, content, tool_calls, tool_call_id
    assert hasattr(result[0], "role")
    assert hasattr(result[0], "content")
    assert hasattr(result[0], "tool_calls")
    assert hasattr(result[0], "tool_call_id")


def test_create_branch_adds_branch_to_tree() -> None:
    """create_branch stores ConversationBranch in tree.branches."""
    tree = ConversationTreeComponent()

    msg = add_message(tree, role="user", content="test", parent_id=None)
    branch = create_branch(tree, branch_id="main", leaf_message_id=msg.id)

    assert "main" in tree.branches
    assert tree.branches["main"] == branch
    assert branch.branch_id == "main"
    assert branch.leaf_message_id == msg.id


def test_create_branch_verifies_leaf_exists() -> None:
    """create_branch raises KeyError if leaf message doesn't exist."""
    tree = ConversationTreeComponent()

    with pytest.raises(KeyError, match="nonexistent"):
        create_branch(tree, branch_id="main", leaf_message_id="nonexistent")


def test_switch_branch_updates_current_branch() -> None:
    """switch_branch updates tree.current_branch_id."""
    tree = ConversationTreeComponent()

    msg = add_message(tree, role="user", content="test", parent_id=None)
    create_branch(tree, branch_id="dev", leaf_message_id=msg.id)

    switch_branch(tree, "dev")

    assert tree.current_branch_id == "dev"


def test_switch_branch_verifies_branch_exists() -> None:
    """switch_branch raises KeyError if branch doesn't exist."""
    tree = ConversationTreeComponent()

    with pytest.raises(KeyError, match="missing"):
        switch_branch(tree, "missing")


def test_get_siblings_returns_all_children_of_parent() -> None:
    """get_siblings returns all messages with same parent_message_id."""
    tree = ConversationTreeComponent()

    root = add_message(tree, role="user", content="root", parent_id=None)
    child1 = add_message(tree, role="assistant", content="child1", parent_id=root.id)
    child2 = add_message(tree, role="assistant", content="child2", parent_id=root.id)
    child3 = add_message(tree, role="assistant", content="child3", parent_id=root.id)

    siblings = get_siblings(tree, child1.id)

    assert len(siblings) == 3
    sibling_ids = {s.id for s in siblings}
    assert child1.id in sibling_ids
    assert child2.id in sibling_ids
    assert child3.id in sibling_ids


def test_get_siblings_for_root_returns_all_roots() -> None:
    """get_siblings for message with parent_id=None returns all root messages."""
    tree = ConversationTreeComponent()

    root1 = add_message(tree, role="user", content="root1", parent_id=None)
    root2 = add_message(tree, role="user", content="root2", parent_id=None)
    child = add_message(tree, role="assistant", content="child", parent_id=root1.id)

    siblings = get_siblings(tree, root1.id)

    assert len(siblings) == 2
    sibling_ids = {s.id for s in siblings}
    assert root1.id in sibling_ids
    assert root2.id in sibling_ids
    assert child.id not in sibling_ids


def test_get_branch_path_returns_message_ids() -> None:
    """get_branch_path returns list of message IDs from root to leaf."""
    tree = ConversationTreeComponent()

    root = add_message(tree, role="user", content="root", parent_id=None)
    msg2 = add_message(tree, role="assistant", content="msg2", parent_id=root.id)
    msg3 = add_message(tree, role="user", content="msg3", parent_id=msg2.id)
    create_branch(tree, branch_id="main", leaf_message_id=msg3.id)

    path = get_branch_path(tree, "main")

    assert path == [root.id, msg2.id, msg3.id]


def test_get_branch_path_verifies_branch_exists() -> None:
    """get_branch_path raises KeyError if branch doesn't exist."""
    tree = ConversationTreeComponent()

    with pytest.raises(KeyError, match="invalid"):
        get_branch_path(tree, "invalid")


def test_get_active_leaf_returns_current_branch_leaf() -> None:
    """get_active_leaf returns leaf_message_id of current_branch_id."""
    tree = ConversationTreeComponent()

    msg = add_message(tree, role="user", content="test", parent_id=None)
    create_branch(tree, branch_id="active", leaf_message_id=msg.id)
    switch_branch(tree, "active")

    leaf = get_active_leaf(tree)

    assert leaf == msg.id


def test_get_active_leaf_returns_none_when_no_branch() -> None:
    """get_active_leaf returns None if current_branch_id is None."""
    tree = ConversationTreeComponent()

    leaf = get_active_leaf(tree)

    assert leaf is None
