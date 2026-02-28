"""Tests for tree-structured conversation support."""

import pytest

from ecs_agent.components.definitions import ConversationTreeComponent
from ecs_agent.types import (
    ConversationMessage,
    ConversationBranch,
    BranchCreatedEvent,
    EntityId,
)


def test_conversation_message_has_required_fields() -> None:
    """Test ConversationMessage has all required fields."""
    msg = ConversationMessage(
        id="msg_1",
        parent_message_id=None,
        role="user",
        content="Hello",
    )
    assert msg.id == "msg_1"
    assert msg.parent_message_id is None
    assert msg.role == "user"
    assert msg.content == "Hello"
    assert msg.tool_calls is None
    assert msg.tool_call_id is None
    assert msg.created_at == ""
    assert msg.metadata == {}


def test_conversation_message_root_has_none_parent() -> None:
    """Test root message has parent_message_id=None."""
    root = ConversationMessage(
        id="root",
        parent_message_id=None,
        role="system",
        content="You are a helpful assistant",
    )
    assert root.parent_message_id is None


def test_conversation_branch_dataclass() -> None:
    """Test ConversationBranch dataclass fields."""
    branch = ConversationBranch(
        branch_id="main",
        leaf_message_id="msg_5",
    )
    assert branch.branch_id == "main"
    assert branch.leaf_message_id == "msg_5"


def test_conversation_tree_component_defaults() -> None:
    """Test ConversationTreeComponent has correct defaults."""
    tree = ConversationTreeComponent()
    assert tree.messages == {}
    assert tree.current_branch_id is None
    assert tree.branches == {}


def test_conversation_tree_component_is_dataclass_with_slots() -> None:
    """Test ConversationTreeComponent is a slotted dataclass."""
    tree = ConversationTreeComponent()
    assert hasattr(tree, "__slots__")
    assert not hasattr(tree, "__dict__")


def test_conversation_tree_add_message() -> None:
    """Test adding message to tree stores it in messages dict."""
    tree = ConversationTreeComponent()
    msg = ConversationMessage(
        id="msg_1",
        parent_message_id=None,
        role="user",
        content="Hello",
    )
    tree.messages[msg.id] = msg

    assert "msg_1" in tree.messages
    assert tree.messages["msg_1"] == msg


def test_conversation_tree_add_child_message() -> None:
    """Test adding child message with parent relationship."""
    tree = ConversationTreeComponent()

    parent = ConversationMessage(
        id="msg_1",
        parent_message_id=None,
        role="user",
        content="Hello",
    )
    tree.messages[parent.id] = parent

    child = ConversationMessage(
        id="msg_2",
        parent_message_id="msg_1",
        role="assistant",
        content="Hi there!",
    )
    tree.messages[child.id] = child

    assert tree.messages["msg_2"].parent_message_id == "msg_1"
    assert tree.messages["msg_1"].id == tree.messages["msg_2"].parent_message_id


def test_branch_created_event_creation() -> None:
    """Test BranchCreatedEvent can be created."""
    event = BranchCreatedEvent(
        entity_id=EntityId(1),
        branch_id="feature-branch",
        parent_message_id="msg_3",
    )
    assert event.entity_id == EntityId(1)
    assert event.branch_id == "feature-branch"
    assert event.parent_message_id == "msg_3"


def test_branch_created_event_is_dataclass_with_slots() -> None:
    """Test BranchCreatedEvent is a slotted dataclass."""
    event = BranchCreatedEvent(
        entity_id=EntityId(1),
        branch_id="test",
        parent_message_id="msg_1",
    )
    assert hasattr(event, "__slots__")
    assert not hasattr(event, "__dict__")
