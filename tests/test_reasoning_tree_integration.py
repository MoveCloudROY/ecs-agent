"""Tests for ReasoningSystem integration with ConversationTreeComponent."""

import pytest

from ecs_agent.components import (
    ConversationComponent,
    ConversationTreeComponent,
    LLMComponent,
    SystemPromptComponent,
)
from ecs_agent.conversation_tree import add_message, create_branch, revert_to_message
from ecs_agent.core import World
from ecs_agent.providers import FakeProvider
from ecs_agent.systems.reasoning import ReasoningSystem
from ecs_agent.types import CompletionResult, Message


class RecordingFakeProvider(FakeProvider):
    """FakeProvider that records all messages sent to it."""

    def __init__(self, responses: list[CompletionResult]) -> None:
        super().__init__(responses=responses)
        self.calls: list[list[Message]] = []

    async def complete(
        self,
        messages: list[Message],
        tools: list | None = None,
        stream: bool = False,
    ) -> CompletionResult:
        self.calls.append(list(messages))
        return await super().complete(messages, tools, stream)


@pytest.mark.asyncio
async def test_reasoning_uses_tree_when_tree_component_exists() -> None:
    """ReasoningSystem uses linearize(tree) when ConversationTreeComponent exists."""
    world = World()
    provider = RecordingFakeProvider(
        responses=[CompletionResult(message=Message(role="assistant", content="reply"))]
    )

    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(provider=provider, model="fake"))

    # Create tree with messages
    tree = ConversationTreeComponent()
    root = add_message(tree, role="user", content="hello", parent_id=None)
    child = add_message(tree, role="assistant", content="hi", parent_id=root.id)
    leaf = add_message(tree, role="user", content="how are you?", parent_id=child.id)

    # Create branch pointing to leaf
    create_branch(tree, branch_id="main", leaf_message_id=leaf.id)
    tree.current_branch_id = "main"

    world.add_component(entity_id, tree)

    await ReasoningSystem().process(world)

    # Verify provider received linearized tree messages (3 messages from tree)
    assert len(provider.calls) == 1
    sent_messages = provider.calls[0]
    assert len(sent_messages) == 3
    assert sent_messages[0].content == "hello"
    assert sent_messages[1].content == "hi"
    assert sent_messages[2].content == "how are you?"


@pytest.mark.asyncio
async def test_reasoning_falls_back_to_flat_conversation_when_no_tree() -> None:
    """ReasoningSystem uses ConversationComponent when tree is absent (backward compat)."""
    world = World()
    provider = RecordingFakeProvider(
        responses=[CompletionResult(message=Message(role="assistant", content="reply"))]
    )

    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(provider=provider, model="fake"))
    world.add_component(
        entity_id,
        ConversationComponent(
            messages=[
                Message(role="user", content="test message 1"),
                Message(role="assistant", content="test message 2"),
            ]
        ),
    )

    await ReasoningSystem().process(world)

    # Verify provider received flat conversation messages
    assert len(provider.calls) == 1
    sent_messages = provider.calls[0]
    assert len(sent_messages) == 2
    assert sent_messages[0].content == "test message 1"
    assert sent_messages[1].content == "test message 2"


@pytest.mark.asyncio
async def test_reasoning_tree_with_system_prompt() -> None:
    """ReasoningSystem prepends system prompt before tree messages."""
    world = World()
    provider = RecordingFakeProvider(
        responses=[CompletionResult(message=Message(role="assistant", content="ok"))]
    )

    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(provider=provider, model="fake"))
    world.add_component(entity_id, SystemPromptComponent(content="You are helpful"))

    tree = ConversationTreeComponent()
    root = add_message(tree, role="user", content="test", parent_id=None)
    create_branch(tree, branch_id="main", leaf_message_id=root.id)
    tree.current_branch_id = "main"

    world.add_component(entity_id, tree)

    await ReasoningSystem().process(world)

    # Verify system prompt comes first
    assert len(provider.calls) == 1
    sent_messages = provider.calls[0]
    assert len(sent_messages) == 2
    assert sent_messages[0].role == "system"
    assert sent_messages[0].content == "You are helpful"
    assert sent_messages[1].role == "user"
    assert sent_messages[1].content == "test"


@pytest.mark.asyncio
async def test_reasoning_after_revert_uses_reverted_branch_context() -> None:
    """After revert_to_message(), next reasoning uses reverted branch context."""
    world = World()
    provider = RecordingFakeProvider(
        responses=[
            CompletionResult(message=Message(role="assistant", content="first")),
            CompletionResult(message=Message(role="assistant", content="second")),
        ]
    )

    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(provider=provider, model="fake"))

    # Build tree: root -> child1 -> child2
    tree = ConversationTreeComponent()
    root = add_message(tree, role="user", content="root message", parent_id=None)
    child1 = add_message(tree, role="assistant", content="child1", parent_id=root.id)
    child2 = add_message(tree, role="user", content="child2", parent_id=child1.id)

    create_branch(tree, branch_id="main", leaf_message_id=child2.id)
    tree.current_branch_id = "main"

    world.add_component(entity_id, tree)

    # First reasoning call (should use all 3 messages: root, child1, child2)
    await ReasoningSystem().process(world)
    assert len(provider.calls) == 1
    assert len(provider.calls[0]) == 3

    # Revert to child1 (drops child2 from active branch)
    revert_to_message(tree, child1.id)

    # Second reasoning call (should use only 2 messages: root, child1)
    await ReasoningSystem().process(world)
    assert len(provider.calls) == 2
    sent_messages = provider.calls[1]
    assert len(sent_messages) == 2
    assert sent_messages[0].content == "root message"
    assert sent_messages[1].content == "child1"
    # child2 should NOT appear
    assert not any(msg.content == "child2" for msg in sent_messages)


@pytest.mark.asyncio
async def test_reasoning_tree_follows_active_branch() -> None:
    """ReasoningSystem follows active branch when multiple branches exist."""
    world = World()
    provider = RecordingFakeProvider(
        responses=[CompletionResult(message=Message(role="assistant", content="ok"))]
    )

    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(provider=provider, model="fake"))

    # Create tree with two branches
    tree = ConversationTreeComponent()
    root = add_message(tree, role="user", content="root", parent_id=None)
    branch_a = add_message(
        tree, role="assistant", content="branch A", parent_id=root.id
    )
    branch_b = add_message(
        tree, role="assistant", content="branch B", parent_id=root.id
    )

    # Create two branches
    create_branch(tree, branch_id="branch-a", leaf_message_id=branch_a.id)
    create_branch(tree, branch_id="branch-b", leaf_message_id=branch_b.id)

    # Set branch-b as active
    tree.current_branch_id = "branch-b"

    world.add_component(entity_id, tree)

    await ReasoningSystem().process(world)

    # Verify provider received branch-b messages (not branch-a)
    assert len(provider.calls) == 1
    sent_messages = provider.calls[0]
    assert len(sent_messages) == 2
    assert sent_messages[0].content == "root"
    assert sent_messages[1].content == "branch B"
    # branch A should NOT appear
    assert not any(msg.content == "branch A" for msg in sent_messages)
