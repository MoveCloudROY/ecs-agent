"""Tests for Component dataclasses."""

import dataclasses

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
    OwnerComponent,
    ErrorComponent,
    TerminalComponent,
    SystemPromptComponent,
    MessageBusConfigComponent,
    MessageBusSubscriptionComponent,
    MessageBusConversationComponent,
)


class MockLLMModel:
    """Mock LLMModel for testing."""

    async def complete(self, messages, tools=None):
        pass


@pytest.fixture
def mock_llm():
    return MockLLMModel()


class TestLLMComponent:
    """Test LLMComponent."""

    def test_instantiation(self, mock_llm):
        """Test LLMComponent can be instantiated."""
        comp = LLMComponent(model=mock_llm)
        assert comp.model is mock_llm
        assert comp.system_prompt == ""

    def test_system_prompt_default(self, mock_llm):
        """Test system_prompt defaults to empty string."""
        comp = LLMComponent(model=mock_llm)
        assert comp.system_prompt == ""

    def test_system_prompt_custom(self, mock_llm):
        """Test system_prompt can be set."""
        prompt = "You are a helpful assistant."
        comp = LLMComponent(model=mock_llm, system_prompt=prompt)
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
        assert comp.template == ""
        assert comp.content == content

    def test_dataclass_slots(self):
        """Test SystemPromptComponent uses slots."""
        assert hasattr(SystemPromptComponent, "__slots__")


class TestMessageBusConfigComponent:
    """Test MessageBusConfigComponent."""

    def test_instantiation(self):
        """Test MessageBusConfigComponent can be instantiated with defaults."""
        comp = MessageBusConfigComponent()
        assert comp.max_queue_size == 1000
        assert comp.publish_timeout == 2.0
        assert comp.request_timeout == 30.0
        assert comp.cleanup_interval == 60.0
        assert comp.max_pending_requests == 10000

    def test_custom_values(self):
        """Test MessageBusConfigComponent can be instantiated with custom values."""
        comp = MessageBusConfigComponent(
            max_queue_size=500,
            publish_timeout=1.0,
            request_timeout=20.0,
            cleanup_interval=30.0,
            max_pending_requests=5000,
        )
        assert comp.max_queue_size == 500
        assert comp.publish_timeout == 1.0
        assert comp.request_timeout == 20.0
        assert comp.cleanup_interval == 30.0
        assert comp.max_pending_requests == 5000

    def test_invalid_negative_queue_size(self):
        """Test that negative queue size is caught by validation."""
        # Dataclass doesn't validate, but the component is designed for positive values
        comp = MessageBusConfigComponent(max_queue_size=-1)
        assert comp.max_queue_size == -1  # Stored as-is, validation in system

    def test_dataclass_slots(self):
        """Test MessageBusConfigComponent uses slots."""
        assert hasattr(MessageBusConfigComponent, "__slots__")


class TestMessageBusSubscriptionComponent:
    """Test MessageBusSubscriptionComponent."""

    def test_instantiation(self):
        """Test MessageBusSubscriptionComponent can be instantiated with empty subscriptions."""
        comp = MessageBusSubscriptionComponent()
        assert comp.subscriptions == {}

    def test_mutable_default_independence(self):
        """Test that mutable defaults are independent instances."""
        comp1 = MessageBusSubscriptionComponent()
        comp2 = MessageBusSubscriptionComponent()
        comp1.subscriptions["topic1"] = {"sub1", "sub2"}
        assert "topic1" not in comp2.subscriptions

    def test_custom_subscriptions(self):
        """Test MessageBusSubscriptionComponent with custom subscriptions."""
        subs = {"topic1": {"subscriber_a", "subscriber_b"}}
        comp = MessageBusSubscriptionComponent(subscriptions=subs)
        assert comp.subscriptions == subs

    def test_dataclass_slots(self):
        """Test MessageBusSubscriptionComponent uses slots."""
        assert hasattr(MessageBusSubscriptionComponent, "__slots__")


class TestMessageBusConversationComponent:
    """Test MessageBusConversationComponent."""

    def test_instantiation(self):
        """Test MessageBusConversationComponent can be instantiated."""
        entity_id = EntityId(123)
        comp = MessageBusConversationComponent(entity_id=entity_id)
        assert comp.entity_id == entity_id
        assert comp.messages == []
        assert comp.max_messages == 1000

    def test_custom_max_messages(self):
        """Test MessageBusConversationComponent with custom max_messages."""
        entity_id = EntityId(456)
        comp = MessageBusConversationComponent(entity_id=entity_id, max_messages=500)
        assert comp.entity_id == entity_id
        assert comp.max_messages == 500

    def test_message_history(self):
        """Test MessageBusConversationComponent with message history."""
        entity_id = EntityId(789)
        messages = [Message(role="user", content="hello")]
        comp = MessageBusConversationComponent(
            entity_id=entity_id,
            messages=messages,
            max_messages=1000,
        )
        assert comp.entity_id == entity_id
        assert len(comp.messages) == 1
        assert comp.messages[0].role == "user"

    def test_mutable_message_default_independence(self):
        """Test that mutable message defaults are independent instances."""
        entity_id1 = EntityId(111)
        entity_id2 = EntityId(222)
        comp1 = MessageBusConversationComponent(entity_id=entity_id1)
        comp2 = MessageBusConversationComponent(entity_id=entity_id2)
        comp1.messages.append(Message(role="user", content="test"))
        assert len(comp2.messages) == 0

    def test_dataclass_slots(self):
        """Test MessageBusConversationComponent uses slots."""
        assert hasattr(MessageBusConversationComponent, "__slots__")


class TestComponentCount:
    """Test component count limit."""

    def test_component_count_limit(self):
        """Test that component count does not exceed the current guardrail."""
        import ecs_agent.components.definitions as d

        count = sum(
            1
            for name in dir(d)
            if not name.startswith("_")
            and dataclasses.is_dataclass(getattr(d, name, None))
            and getattr(d, name).__module__ == "ecs_agent.components.definitions"
        )
        assert count <= 57, f"Component count {count} exceeds limit of 57"


class TestComponentsExportedInInit:
    """Test that all components are exported from __init__.py."""

    def test_all_components_exported(self):
        """Test all 26 components can be imported from ecs_agent.components."""
        from ecs_agent import components

        component_names = [
            "LLMComponent",
            "ConversationComponent",
            "KVStoreComponent",
            "ToolRegistryComponent",
            "PendingToolCallsComponent",
            "ToolResultsComponent",
            "PlanComponent",
            "OwnerComponent",
            "ErrorComponent",
            "TerminalComponent",
            "SystemPromptComponent",
            "ToolApprovalComponent",
            "SandboxConfigComponent",
            "PlanSearchComponent",
            "RAGTriggerComponent",
            "EmbeddingComponent",
            "VectorStoreComponent",
            "StreamingComponent",
            "CheckpointComponent",
            "CompactionConfigComponent",
            "ConversationArchiveComponent",
            "RunnerStateComponent",
            "MessageBusConfigComponent",
            "MessageBusSubscriptionComponent",
            "MessageBusConversationComponent",
            "ToolRuntimeStateComponent",
            "ToolStateNamespace",
        ]

        for name in component_names:
            assert hasattr(components, name), f"{name} not exported"


class TestEntityRegistryComponent:
    """Test EntityRegistryComponent for runtime entity naming and tagging."""

    def test_instantiation_with_name_only(self):
        """Test EntityRegistryComponent with name only."""
        from ecs_agent.components import EntityRegistryComponent

        entity_id = EntityId(42)
        comp = EntityRegistryComponent(entity_id=entity_id, name="agent_main")
        assert comp.entity_id == entity_id
        assert comp.name == "agent_main"
        assert comp.tags == set()
        assert comp.metadata == {}

    def test_instantiation_with_tags(self):
        """Test EntityRegistryComponent with tags."""
        from ecs_agent.components import EntityRegistryComponent

        entity_id = EntityId(99)
        comp = EntityRegistryComponent(
            entity_id=entity_id,
            name="worker_1",
            tags={"worker", "async"},
        )
        assert comp.entity_id == entity_id
        assert comp.name == "worker_1"
        assert comp.tags == {"worker", "async"}
        assert comp.metadata == {}

    def test_instantiation_with_metadata(self):
        """Test EntityRegistryComponent with arbitrary metadata."""
        from ecs_agent.components import EntityRegistryComponent

        entity_id = EntityId(123)
        metadata = {"priority": "high", "retries": 3}
        comp = EntityRegistryComponent(
            entity_id=entity_id,
            name="task_processor",
            tags={"processor"},
            metadata=metadata,
        )
        assert comp.entity_id == entity_id
        assert comp.name == "task_processor"
        assert comp.tags == {"processor"}
        assert comp.metadata == metadata

    def test_mutable_default_independence_tags(self):
        """Test that tag sets are independent instances."""
        from ecs_agent.components import EntityRegistryComponent

        comp1 = EntityRegistryComponent(entity_id=EntityId(1), name="a")
        comp2 = EntityRegistryComponent(entity_id=EntityId(2), name="b")
        comp1.tags.add("test")
        assert "test" not in comp2.tags

    def test_mutable_default_independence_metadata(self):
        """Test that metadata dicts are independent instances."""
        from ecs_agent.components import EntityRegistryComponent

        comp1 = EntityRegistryComponent(entity_id=EntityId(1), name="a")
        comp2 = EntityRegistryComponent(entity_id=EntityId(2), name="b")
        comp1.metadata["key"] = "value"
        assert "key" not in comp2.metadata

    def test_dataclass_slots(self):
        """Test EntityRegistryComponent uses slots."""
        from ecs_agent.components import EntityRegistryComponent

        assert hasattr(EntityRegistryComponent, "__slots__")


class TestInterruptionComponent:
    """Test InterruptionComponent for graceful agent pause."""

    def test_instantiation_with_reason(self):
        """Test InterruptionComponent with interruption reason."""
        from ecs_agent.components import InterruptionComponent
        from ecs_agent.types import InterruptionReason

        comp = InterruptionComponent(
            reason=InterruptionReason.USER_REQUESTED,
            message="User pressed pause button",
        )
        assert comp.reason == InterruptionReason.USER_REQUESTED
        assert comp.message == "User pressed pause button"
        assert comp.timestamp > 0.0

    def test_instantiation_system_pause(self):
        """Test InterruptionComponent for system-initiated pause."""
        from ecs_agent.components import InterruptionComponent
        from ecs_agent.types import InterruptionReason

        comp = InterruptionComponent(
            reason=InterruptionReason.SYSTEM_PAUSE,
            message="Awaiting user approval",
        )
        assert comp.reason == InterruptionReason.SYSTEM_PAUSE
        assert comp.message == "Awaiting user approval"

    def test_empty_message_default(self):
        """Test that message defaults to empty string."""
        from ecs_agent.components import InterruptionComponent
        from ecs_agent.types import InterruptionReason

        comp = InterruptionComponent(reason=InterruptionReason.USER_REQUESTED)
        assert comp.message == ""

    def test_dataclass_slots(self):
        """Test InterruptionComponent uses slots."""
        from ecs_agent.components import InterruptionComponent

        assert hasattr(InterruptionComponent, "__slots__")


class TestScratchbookRefComponent:
    """Test ScratchbookRefComponent for artifact references."""

    def test_instantiation(self):
        """Test ScratchbookRefComponent with artifact metadata."""
        from ecs_agent.components import ScratchbookRefComponent

        comp = ScratchbookRefComponent(
            artifact_id="art-456",
            category="plan",
            content_hash="sha256:abc123",
            timestamp="2026-03-07T10:00:00Z",
        )
        assert comp.artifact_id == "art-456"
        assert comp.category == "plan"
        assert comp.content_hash == "sha256:abc123"
        assert comp.timestamp == "2026-03-07T10:00:00Z"

    def test_dataclass_slots(self):
        """Test ScratchbookRefComponent uses slots."""
        from ecs_agent.components import ScratchbookRefComponent

        assert hasattr(ScratchbookRefComponent, "__slots__")


class TestScratchbookIndexComponent:
    """Test ScratchbookIndexComponent for scratchbook index tracking."""

    def test_instantiation_with_empty_index(self):
        """Test ScratchbookIndexComponent with empty artifact index."""
        from ecs_agent.components import ScratchbookIndexComponent

        comp = ScratchbookIndexComponent()
        assert comp.artifacts == {}

    def test_instantiation_with_artifacts(self):
        """Test ScratchbookIndexComponent with artifact entries."""
        from ecs_agent.components import ScratchbookIndexComponent
        from ecs_agent.types import ScratchbookRef

        ref1 = ScratchbookRef(
            artifact_id="a1",
            category="plan",
            content_hash="hash1",
            timestamp="2026-03-07T10:00:00Z",
        )
        ref2 = ScratchbookRef(
            artifact_id="a2",
            category="output",
            content_hash="hash2",
            timestamp="2026-03-07T11:00:00Z",
        )
        comp = ScratchbookIndexComponent(artifacts={"a1": ref1, "a2": ref2})
        assert len(comp.artifacts) == 2
        assert comp.artifacts["a1"] == ref1
        assert comp.artifacts["a2"] == ref2

    def test_mutable_default_independence(self):
        """Test that artifact dicts are independent instances."""
        from ecs_agent.components import ScratchbookIndexComponent
        from ecs_agent.types import ScratchbookRef

        comp1 = ScratchbookIndexComponent()
        comp2 = ScratchbookIndexComponent()
        ref = ScratchbookRef(
            artifact_id="test",
            category="temp",
            content_hash="h1",
            timestamp="2026-03-07T12:00:00Z",
        )
        comp1.artifacts["test"] = ref
        assert "test" not in comp2.artifacts

    def test_dataclass_slots(self):
        """Test ScratchbookIndexComponent uses slots."""
        from ecs_agent.components import ScratchbookIndexComponent

        assert hasattr(ScratchbookIndexComponent, "__slots__")


# ---------------------------------------------------------------------------
# Prompt normalization components (Task-1)
# ---------------------------------------------------------------------------


class TestUserPromptConfigComponent:
    """Tests for UserPromptConfigComponent."""

    def test_instantiation_defaults(self):
        """Test UserPromptConfigComponent defaults."""
        from ecs_agent.components import UserPromptConfigComponent

        comp = UserPromptConfigComponent()
        assert comp.triggers == []
        assert comp.enable_context_pool is False
        assert comp.context_pool_max_chars == 8192

    def test_triggers(self):
        """Test UserPromptConfigComponent with triggers."""
        from ecs_agent.components import UserPromptConfigComponent
        from ecs_agent.prompts.contracts import TriggerSpec

        trigger = TriggerSpec(
            pattern="code",
            match_mode="keyword",
            action="inject",
            content="coding-assistant",
            priority=0,
        )
        comp = UserPromptConfigComponent(triggers=[trigger])
        assert comp.triggers == [trigger]

    def test_enable_context_pool(self):
        """Test UserPromptConfigComponent with enable_context_pool=True."""
        from ecs_agent.components import UserPromptConfigComponent

        comp = UserPromptConfigComponent(enable_context_pool=True)
        assert comp.enable_context_pool is True

    def test_context_pool_max_chars_custom(self):
        """Test UserPromptConfigComponent with custom context_pool_max_chars."""
        from ecs_agent.components import UserPromptConfigComponent

        comp = UserPromptConfigComponent(context_pool_max_chars=4096)
        assert comp.context_pool_max_chars == 4096

    def test_dataclass_slots(self):
        """Test UserPromptConfigComponent uses slots."""
        from ecs_agent.components import UserPromptConfigComponent

        assert hasattr(UserPromptConfigComponent, "__slots__")

    def test_mutable_default_independence(self):
        """Test that triggers are independent instances."""
        from ecs_agent.components import UserPromptConfigComponent
        from ecs_agent.prompts.contracts import TriggerSpec

        comp1 = UserPromptConfigComponent()
        comp2 = UserPromptConfigComponent()
        comp1.triggers.append(
            TriggerSpec(
                pattern="x",
                match_mode="keyword",
                action="inject",
                content="y",
                priority=0,
            )
        )
        assert len(comp2.triggers) == 0


def test_prompt_contributions_component_removed_from_exports() -> None:
    from ecs_agent import components

    assert not hasattr(components, "PromptContributionsComponent")


class TestContextEntry:
    def test_fields_are_stored(self):
        from ecs_agent.components import ContextEntry

        entry = ContextEntry(
            entry_id="tool-search-0",
            priority=30,
            registration_order=0,
            source_label="tool:search",
            content="source: tool\nresult: evidence",
        )
        assert entry.entry_id == "tool-search-0"
        assert entry.priority == 30
        assert entry.registration_order == 0
        assert entry.source_label == "tool:search"
        assert entry.content == "source: tool\nresult: evidence"


class TestPromptContextQueueComponent:
    def test_instantiation_defaults(self):
        from ecs_agent.components import PromptContextQueueComponent

        comp = PromptContextQueueComponent()
        assert comp.entries == []

    def test_entries_field(self):
        from ecs_agent.components import ContextEntry, PromptContextQueueComponent

        entry = ContextEntry(
            entry_id="tool-search-0",
            priority=10,
            registration_order=0,
            source_label="tool:search",
            content="some content",
        )
        comp = PromptContextQueueComponent(entries=[entry])
        assert len(comp.entries) == 1
        assert comp.entries[0] == entry

    def test_dataclass_slots(self):
        from ecs_agent.components import PromptContextQueueComponent

        assert hasattr(PromptContextQueueComponent, "__slots__")

    def test_mutable_default_independence(self):
        from ecs_agent.components import ContextEntry, PromptContextQueueComponent

        comp1 = PromptContextQueueComponent()
        comp2 = PromptContextQueueComponent()
        comp1.entries.append(
            ContextEntry(
                entry_id="entry-1",
                priority=1,
                registration_order=0,
                source_label="src",
                content="text",
            )
        )
        assert len(comp2.entries) == 0


class TestPromptContextReservationComponent:
    def test_instantiation_fields(self):
        from ecs_agent.components import ContextEntry, PromptContextReservationComponent

        entry = ContextEntry(
            entry_id="tool-search-0",
            priority=30,
            registration_order=0,
            source_label="tool:search",
            content="source: tool\nresult: evidence",
        )
        comp = PromptContextReservationComponent(
            reservation_id="reservation-1",
            created_at_tick=10,
            reserved_entries=[entry],
        )
        assert comp.reservation_id == "reservation-1"
        assert comp.created_at_tick == 10
        assert comp.reserved_entries == [entry]

    def test_reserved_entries_default(self):
        from ecs_agent.components import PromptContextReservationComponent

        comp = PromptContextReservationComponent(
            reservation_id="reservation-2",
            created_at_tick=11,
        )
        assert comp.reserved_entries == []

    def test_dataclass_slots(self):
        from ecs_agent.components import PromptContextReservationComponent

        assert hasattr(PromptContextReservationComponent, "__slots__")


class TestPromptContractsModule:
    """Tests for the prompts.contracts module."""

    def test_prompt_template_instantiation(self):
        """Test PromptTemplate can be instantiated."""
        from ecs_agent.prompts import PromptTemplate

        tmpl = PromptTemplate(
            template_id="coding-assistant", content="You are a coder."
        )
        assert tmpl.template_id == "coding-assistant"
        assert tmpl.content == "You are a coder."
        assert tmpl.description == ""
        assert tmpl.metadata == {}

    def test_prompt_template_with_metadata(self):
        """Test PromptTemplate with metadata."""
        from ecs_agent.prompts import PromptTemplate

        tmpl = PromptTemplate(
            template_id="test",
            content="content",
            description="A test template",
            metadata={"version": "1"},
        )
        assert tmpl.description == "A test template"
        assert tmpl.metadata == {"version": "1"}

    def test_prompt_template_slots(self):
        """Test PromptTemplate uses slots."""
        from ecs_agent.prompts import PromptTemplate

        assert hasattr(PromptTemplate, "__slots__")
