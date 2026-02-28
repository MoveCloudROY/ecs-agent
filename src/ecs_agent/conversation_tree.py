"""Utility functions for tree-structured conversation operations."""

import uuid
from datetime import datetime, timezone

from ecs_agent.components.definitions import ConversationTreeComponent
from ecs_agent.types import ConversationBranch, ConversationMessage, Message, ToolCall


def add_message(
    tree: ConversationTreeComponent,
    role: str,
    content: str | None,
    parent_id: str | None = None,
    tool_calls: list[ToolCall] | None = None,
    tool_call_id: str | None = None,
    metadata: dict[str, object] | None = None,
) -> ConversationMessage:
    """Add a new message to the conversation tree.

    Args:
        tree: ConversationTreeComponent to add message to
        role: Message role ('user', 'assistant', 'tool', 'system')
        content: Message content (can be None for tool calls)
        parent_id: Parent message ID (None for root messages)
        tool_calls: Optional list of tool calls
        tool_call_id: Optional tool call ID for tool role messages
        metadata: Optional metadata dict

    Returns:
        Created ConversationMessage with generated ID and timestamp
    """
    message_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    msg = ConversationMessage(
        id=message_id,
        parent_message_id=parent_id,
        role=role,
        content=content,
        tool_calls=tool_calls,
        tool_call_id=tool_call_id,
        created_at=created_at,
        metadata=metadata or {},
    )

    tree.messages[message_id] = msg
    return msg


def linearize(tree: ConversationTreeComponent, leaf_message_id: str) -> list[Message]:
    """Convert tree branch to flat message list for LLM consumption.

    Walks from leaf to root via parent_message_id chain, then reverses
    to produce chronological order. Converts ConversationMessage to Message
    for compatibility with existing systems.

    Args:
        tree: ConversationTreeComponent to traverse
        leaf_message_id: ID of leaf message to start from

    Returns:
        List of Messages from root to leaf in chronological order
    """
    # Walk from leaf to root (reversed chronological order)
    path: list[ConversationMessage] = []
    current_id: str | None = leaf_message_id

    while current_id is not None:
        conv_msg = tree.messages[current_id]
        path.append(conv_msg)
        current_id = conv_msg.parent_message_id

    # Reverse to get root-to-leaf chronological order
    path.reverse()

    # Convert ConversationMessage → Message
    messages: list[Message] = []
    for conv_msg in path:
        msg = Message(
            role=conv_msg.role,
            content=conv_msg.content or "",
            tool_calls=conv_msg.tool_calls,
            tool_call_id=conv_msg.tool_call_id,
        )
        messages.append(msg)

    return messages


def create_branch(
    tree: ConversationTreeComponent,
    branch_id: str,
    leaf_message_id: str,
) -> ConversationBranch:
    """Create a new branch pointing to a leaf message.

    Args:
        tree: ConversationTreeComponent to add branch to
        branch_id: Unique identifier for the branch
        leaf_message_id: ID of the leaf message this branch points to

    Returns:
        Created ConversationBranch

    Raises:
        KeyError: If leaf_message_id doesn't exist in tree.messages
    """
    # Verify leaf exists
    if leaf_message_id not in tree.messages:
        raise KeyError(f"Leaf message not found: {leaf_message_id}")

    branch = ConversationBranch(
        branch_id=branch_id,
        leaf_message_id=leaf_message_id,
    )

    tree.branches[branch_id] = branch
    return branch


def switch_branch(tree: ConversationTreeComponent, branch_id: str) -> None:
    """Switch the current active branch.

    Args:
        tree: ConversationTreeComponent to modify
        branch_id: ID of branch to switch to

    Raises:
        KeyError: If branch_id doesn't exist in tree.branches
    """
    if branch_id not in tree.branches:
        raise KeyError(f"Branch not found: {branch_id}")

    tree.current_branch_id = branch_id


def get_siblings(
    tree: ConversationTreeComponent,
    message_id: str,
) -> list[ConversationMessage]:
    """Get all messages with the same parent as the given message.

    Includes the message itself in the result.

    Args:
        tree: ConversationTreeComponent to query
        message_id: ID of message to find siblings for

    Returns:
        List of all messages with the same parent_message_id
    """
    msg = tree.messages[message_id]
    parent_id = msg.parent_message_id

    siblings: list[ConversationMessage] = []
    for conv_msg in tree.messages.values():
        if conv_msg.parent_message_id == parent_id:
            siblings.append(conv_msg)

    return siblings


def get_branch_path(tree: ConversationTreeComponent, branch_id: str) -> list[str]:
    """Get list of message IDs from root to branch leaf.

    Args:
        tree: ConversationTreeComponent to query
        branch_id: Branch to get path for

    Returns:
        List of message IDs in chronological order (root to leaf)

    Raises:
        KeyError: If branch_id doesn't exist in tree.branches
    """
    if branch_id not in tree.branches:
        raise KeyError(f"Branch not found: {branch_id}")

    branch = tree.branches[branch_id]
    leaf_id = branch.leaf_message_id

    # Walk from leaf to root
    path: list[str] = []
    current_id: str | None = leaf_id

    while current_id is not None:
        path.append(current_id)
        conv_msg = tree.messages[current_id]
        current_id = conv_msg.parent_message_id

    # Reverse to get root-to-leaf order
    path.reverse()
    return path


def get_active_leaf(tree: ConversationTreeComponent) -> str | None:
    """Get the leaf message ID of the current active branch.

    Args:
        tree: ConversationTreeComponent to query

    Returns:
        Leaf message ID of current branch, or None if no branch is active
    """
    if tree.current_branch_id is None:
        return None

    branch = tree.branches[tree.current_branch_id]
    return branch.leaf_message_id
