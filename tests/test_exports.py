"""Test that all new public symbols are importable from ecs_agent."""

import pytest


def test_all_new_symbols_importable_from_ecs_agent() -> None:
    """Verify all new symbols from five-features are exported in __init__.py."""
    # Types from types.py
    from ecs_agent import (
        ConversationMessage,
        ConversationBranch,
        BranchCreatedEvent,
        ResponsesAPICallEvent,
        SubagentConfig,
        DelegationStartedEvent,
        DelegationCompletedEvent,
    )

    # Components from definitions.py
    from ecs_agent import (
        ResponsesAPIStateComponent,
        ConversationTreeComponent,
        SubagentRegistryComponent,
    )

    # Skills from markdown_skill.py
    from ecs_agent import MarkdownSkill

    # Functions from conversation_tree.py
    from ecs_agent import (
        add_message,
        linearize,
        create_branch,
        switch_branch,
    )

    # Systems from subagent.py
    from ecs_agent import SubagentSystem

    # Verify all imports succeeded (no exceptions raised)
    assert ConversationMessage is not None
    assert ConversationBranch is not None
    assert BranchCreatedEvent is not None
    assert ResponsesAPICallEvent is not None
    assert SubagentConfig is not None
    assert DelegationStartedEvent is not None
    assert DelegationCompletedEvent is not None
    assert ResponsesAPIStateComponent is not None
    assert ConversationTreeComponent is not None
    assert SubagentRegistryComponent is not None
    assert MarkdownSkill is not None
    assert add_message is not None
    assert linearize is not None
    assert create_branch is not None
    assert switch_branch is not None
    assert SubagentSystem is not None


def test_collaboration_symbols_absent_from_systems_exports() -> None:
    with pytest.raises(ImportError):
        exec("from ecs_agent.systems import CollaborationSystem", {})


def test_collaboration_symbols_absent_from_components_exports() -> None:
    with pytest.raises(ImportError):
        exec("from ecs_agent.components import CollaborationComponent", {})


def test_message_bus_system_exported_and_constructible() -> None:
    from ecs_agent.systems import MessageBusSystem

    system = MessageBusSystem(priority=5)
    assert system.priority == 5
