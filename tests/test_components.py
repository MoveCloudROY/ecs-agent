"""Tests for Component dataclasses."""

import dataclasses
from typing import TYPE_CHECKING

import pytest

from ecs_agent.types import EntityId, Message, ToolCall, ToolSchema
from ecs_agent.components import (
    LLMComponent,
    ConversationComponent,
    KVStoreComponent,
    ToolRegistryComponent,
    PendingToolCallsComponent,
    ToolResultsComponent,
    PlanComponent,
    CollaborationComponent,
    OwnerComponent,
    ErrorComponent,
    TerminalComponent,
    SystemPromptComponent,
)

if TYPE_CHECKING:
    from ecs_agent.providers.protocol import LLMProvider


class MockLLMProvider:
    """Mock LLMProvider for testing."""

    async def complete(self, messages, tools=None):
        pass


@pytest.fixture
def mock_llm():
    return MockLLMProvider()


class TestLLMComponent:
    """Test LLMComponent."""

    def test_instantiation(self, mock_llm):
        """Test LLMComponent can be instantiated."""
        comp = LLMComponent(provider=mock_llm, model="gpt-4")
        assert comp.provider is mock_llm
        assert comp.model == "gpt-4"
        assert comp.system_prompt == ""

    def test_system_prompt_default(self, mock_llm):
        """Test system_prompt defaults to empty string."""
        comp = LLMComponent(provider=mock_llm, model="gpt-3.5-turbo")
        assert comp.system_prompt == ""

    def test_system_prompt_custom(self, mock_llm):
        """Test system_prompt can be set."""
        prompt = "You are a helpful assistant."
        comp = LLMComponent(provider=mock_llm, model="gpt-4", system_prompt=prompt)
        assert comp.system_prompt == prompt

    def test_dataclass_slots(self):
        """Test LLMComponent uses slots."""
        assert hasattr(LLMComponent, "__slots__")


class TestConversationComponent:
    """Test ConversationComponent."""

    def test_instantiation(self):
        """Test ConversationComponent can be instantiated."""
        messages = [Message(role="user", content="hello")]
        comp = ConversationComponent(messages=messages)
        assert comp.messages == messages
        assert comp.max_messages == 100

    def test_max_messages_default(self):
        """Test max_messages defaults to 100."""
        comp = ConversationComponent(messages=[])
        assert comp.max_messages == 100

    def test_max_messages_custom(self):
        """Test max_messages can be set."""
        comp = ConversationComponent(messages=[], max_messages=50)
        assert comp.max_messages == 50

    def test_dataclass_slots(self):
        """Test ConversationComponent uses slots."""
        assert hasattr(ConversationComponent, "__slots__")


class TestKVStoreComponent:
    """Test KVStoreComponent."""

    def test_instantiation(self):
        """Test KVStoreComponent can be instantiated."""
        store = {"key": "value"}
        comp = KVStoreComponent(store=store)
        assert comp.store == store

    def test_empty_store(self):
        """Test KVStoreComponent with empty store."""
        comp = KVStoreComponent(store={})
        assert comp.store == {}

    def test_dataclass_slots(self):
        """Test KVStoreComponent uses slots."""
        assert hasattr(KVStoreComponent, "__slots__")


class TestToolRegistryComponent:
    """Test ToolRegistryComponent."""

    def test_instantiation(self):
        """Test ToolRegistryComponent can be instantiated."""
        tools = {"tool1": ToolSchema(name="tool1", description="desc", parameters={})}
        handlers = {"tool1": lambda: "result"}
        comp = ToolRegistryComponent(tools=tools, handlers=handlers)
        assert comp.tools == tools
        assert comp.handlers == handlers

    def test_empty_registry(self):
        """Test ToolRegistryComponent with empty tools and handlers."""
        comp = ToolRegistryComponent(tools={}, handlers={})
        assert comp.tools == {}
        assert comp.handlers == {}

    def test_dataclass_slots(self):
        """Test ToolRegistryComponent uses slots."""
        assert hasattr(ToolRegistryComponent, "__slots__")


class TestPendingToolCallsComponent:
    """Test PendingToolCallsComponent."""

    def test_instantiation(self):
        """Test PendingToolCallsComponent can be instantiated."""
        calls = [ToolCall(id="1", name="tool", arguments={})]
        comp = PendingToolCallsComponent(tool_calls=calls)
        assert comp.tool_calls == calls

    def test_empty_calls(self):
        """Test PendingToolCallsComponent with empty calls."""
        comp = PendingToolCallsComponent(tool_calls=[])
        assert comp.tool_calls == []

    def test_dataclass_slots(self):
        """Test PendingToolCallsComponent uses slots."""
        assert hasattr(PendingToolCallsComponent, "__slots__")


class TestToolResultsComponent:
    """Test ToolResultsComponent."""

    def test_instantiation(self):
        """Test ToolResultsComponent can be instantiated."""
        results = {"call_id_1": "result_1"}
        comp = ToolResultsComponent(results=results)
        assert comp.results == results

    def test_empty_results(self):
        """Test ToolResultsComponent with empty results."""
        comp = ToolResultsComponent(results={})
        assert comp.results == {}

    def test_dataclass_slots(self):
        """Test ToolResultsComponent uses slots."""
        assert hasattr(ToolResultsComponent, "__slots__")


class TestPlanComponent:
    """Test PlanComponent."""

    def test_instantiation(self):
        """Test PlanComponent can be instantiated."""
        steps = ["step1", "step2", "step3"]
        comp = PlanComponent(steps=steps)
        assert comp.steps == steps
        assert comp.current_step == 0
        assert comp.completed is False

    def test_custom_current_step(self):
        """Test current_step can be set."""
        comp = PlanComponent(steps=["a", "b"], current_step=1)
        assert comp.current_step == 1

    def test_completed_flag(self):
        """Test completed flag can be set."""
        comp = PlanComponent(steps=["a"], completed=True)
        assert comp.completed is True

    def test_dataclass_slots(self):
        """Test PlanComponent uses slots."""
        assert hasattr(PlanComponent, "__slots__")


class TestCollaborationComponent:
    """Test CollaborationComponent."""

    def test_instantiation(self):
        """Test CollaborationComponent can be instantiated."""
        peers = [EntityId(1), EntityId(2)]
        inbox = [(EntityId(1), Message(role="user", content="msg"))]
        comp = CollaborationComponent(peers=peers, inbox=inbox)
        assert comp.peers == peers
        assert comp.inbox == inbox

    def test_empty_peers_and_inbox(self):
        """Test CollaborationComponent with empty peers and inbox."""
        comp = CollaborationComponent(peers=[], inbox=[])
        assert comp.peers == []
        assert comp.inbox == []

    def test_dataclass_slots(self):
        """Test CollaborationComponent uses slots."""
        assert hasattr(CollaborationComponent, "__slots__")


class TestOwnerComponent:
    """Test OwnerComponent."""

    def test_instantiation(self):
        """Test OwnerComponent can be instantiated."""
        owner_id = EntityId(1)
        comp = OwnerComponent(owner_id=owner_id)
        assert comp.owner_id == owner_id

    def test_dataclass_slots(self):
        """Test OwnerComponent uses slots."""
        assert hasattr(OwnerComponent, "__slots__")


class TestErrorComponent:
    """Test ErrorComponent."""

    def test_instantiation(self):
        """Test ErrorComponent can be instantiated."""
        comp = ErrorComponent(error="oops", system_name="test", timestamp=1.0)
        assert comp.error == "oops"
        assert comp.system_name == "test"
        assert comp.timestamp == 1.0

    def test_dataclass_slots(self):
        """Test ErrorComponent uses slots."""
        assert hasattr(ErrorComponent, "__slots__")


class TestTerminalComponent:
    """Test TerminalComponent."""

    def test_instantiation(self):
        """Test TerminalComponent can be instantiated."""
        comp = TerminalComponent(reason="done")
        assert comp.reason == "done"

    def test_dataclass_slots(self):
        """Test TerminalComponent uses slots."""
        assert hasattr(TerminalComponent, "__slots__")


class TestSystemPromptComponent:
    """Test SystemPromptComponent."""

    def test_instantiation(self):
        """Test SystemPromptComponent can be instantiated."""
        content = "Be helpful"
        comp = SystemPromptComponent(content=content)
        assert comp.content == content

    def test_dataclass_slots(self):
        """Test SystemPromptComponent uses slots."""
        assert hasattr(SystemPromptComponent, "__slots__")


class TestComponentCount:
    """Test component count limit."""

    def test_component_count_limit(self):
        """Test that component count does not exceed 12."""
        import ecs_agent.components.definitions as d
        count = sum(
            1
            for name in dir(d)
            if not name.startswith("_")
            and dataclasses.is_dataclass(getattr(d, name, None))
            and getattr(d, name).__module__ == "ecs_agent.components.definitions"
        )
        assert count <= 12, f"Component count {count} exceeds limit of 12"


class TestComponentsExportedInInit:
    """Test that all components are exported from __init__.py."""

    def test_all_components_exported(self):
        """Test all 12 components can be imported from ecs_agent.components."""
        from ecs_agent import components

        component_names = [
            "LLMComponent",
            "ConversationComponent",
            "KVStoreComponent",
            "ToolRegistryComponent",
            "PendingToolCallsComponent",
            "ToolResultsComponent",
            "PlanComponent",
            "CollaborationComponent",
            "OwnerComponent",
            "ErrorComponent",
            "TerminalComponent",
            "SystemPromptComponent",
        ]

        for name in component_names:
            assert hasattr(components, name), f"{name} not exported"
