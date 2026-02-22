"""Tests for core type definitions."""

import pytest
from ecs_agent.types import (
    EntityId,
    Message,
    ToolCall,
    ToolSchema,
    CompletionResult,
    Usage,
)


class TestMessage:
    """Test Message dataclass."""

    def test_message_basic_fields(self) -> None:
        """Test Message with required fields."""
        msg = Message(role="user", content="hello")
        assert msg.role == "user"
        assert msg.content == "hello"

    def test_message_optional_fields_default_none(self) -> None:
        """Test that optional fields default to None."""
        msg = Message(role="assistant", content="response")
        assert msg.tool_calls is None
        assert msg.tool_call_id is None

    def test_message_with_tool_calls(self) -> None:
        """Test Message with tool_calls."""
        tc = ToolCall(id="1", name="search", arguments="{}")
        msg = Message(role="assistant", content="searching", tool_calls=[tc])
        assert msg.tool_calls == [tc]
        assert len(msg.tool_calls) == 1

    def test_message_with_tool_call_id(self) -> None:
        """Test Message with tool_call_id."""
        msg = Message(role="tool", content="result", tool_call_id="call_123")
        assert msg.tool_call_id == "call_123"

    def test_message_slots_prevent_extra_attributes(self) -> None:
        """Test that slots=True prevents adding arbitrary attributes."""
        msg = Message(role="user", content="test")
        with pytest.raises(AttributeError):
            msg.extra_field = "should fail"  # type: ignore


class TestToolCall:
    """Test ToolCall dataclass."""

    def test_toolcall_basic_fields(self) -> None:
        """Test ToolCall with required fields."""
        tc = ToolCall(id="call_1", name="search", arguments='{"q": "test"}')
        assert tc.id == "call_1"
        assert tc.name == "search"
        assert tc.arguments == '{"q": "test"}'

    def test_toolcall_slots_prevent_extra_attributes(self) -> None:
        """Test that slots=True prevents adding arbitrary attributes."""
        tc = ToolCall(id="1", name="test", arguments="{}")
        with pytest.raises(AttributeError):
            tc.extra = "bad"  # type: ignore


class TestToolSchema:
    """Test ToolSchema dataclass."""

    def test_toolschema_basic_fields(self) -> None:
        """Test ToolSchema with required fields."""
        schema = ToolSchema(
            name="search",
            description="Search the web",
            parameters={"type": "object"},
        )
        assert schema.name == "search"
        assert schema.description == "Search the web"
        assert schema.parameters == {"type": "object"}

    def test_toolschema_empty_parameters(self) -> None:
        """Test ToolSchema with empty parameters dict."""
        schema = ToolSchema(name="ping", description="Ping", parameters={})
        assert schema.parameters == {}

    def test_toolschema_slots_prevent_extra_attributes(self) -> None:
        """Test that slots=True prevents adding arbitrary attributes."""
        schema = ToolSchema(name="test", description="test", parameters={})
        with pytest.raises(AttributeError):
            schema.extra = "bad"  # type: ignore


class TestUsage:
    """Test Usage dataclass."""

    def test_usage_basic_fields(self) -> None:
        """Test Usage with token counts."""
        usage = Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        assert usage.prompt_tokens == 10
        assert usage.completion_tokens == 5
        assert usage.total_tokens == 15

    def test_usage_slots_prevent_extra_attributes(self) -> None:
        """Test that slots=True prevents adding arbitrary attributes."""
        usage = Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2)
        with pytest.raises(AttributeError):
            usage.extra = "bad"  # type: ignore


class TestCompletionResult:
    """Test CompletionResult dataclass."""

    def test_completionresult_with_message_and_usage(self) -> None:
        """Test CompletionResult with both message and usage."""
        msg = Message(role="assistant", content="result")
        usage = Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        result = CompletionResult(message=msg, usage=usage)
        assert result.message == msg
        assert result.usage == usage

    def test_completionresult_message_only(self) -> None:
        """Test CompletionResult with message only (usage optional)."""
        msg = Message(role="assistant", content="result")
        result = CompletionResult(message=msg)
        assert result.message == msg
        assert result.usage is None

    def test_completionresult_slots_prevent_extra_attributes(self) -> None:
        """Test that slots=True prevents adding arbitrary attributes."""
        msg = Message(role="assistant", content="test")
        result = CompletionResult(message=msg)
        with pytest.raises(AttributeError):
            result.extra = "bad"  # type: ignore


class TestEntityId:
    """Test EntityId type alias."""

    def test_entityid_is_newtype(self) -> None:
        """Test that EntityId is a valid NewType."""
        eid = EntityId(42)
        assert eid == 42

    def test_entityid_numeric_values(self) -> None:
        """Test EntityId with various numeric values."""
        eid1 = EntityId(0)
        eid2 = EntityId(1)
        eid3 = EntityId(999999)
        assert eid1 == 0
        assert eid2 == 1
        assert eid3 == 999999

    def test_entityid_type_hint_compatibility(self) -> None:
        """Test that EntityId works as a type hint."""

        def process_entity(entity_id: EntityId) -> int:
            return entity_id

        result = process_entity(EntityId(123))
        assert result == 123


class TestDataclassFeatures:
    """Test dataclass-specific features."""

    def test_all_types_use_slots(self) -> None:
        """Verify all dataclasses use slots=True."""
        # This is verified by the AttributeError tests above
        # Additional check: verify __slots__ is defined
        assert hasattr(Message, "__slots__")
        assert hasattr(ToolCall, "__slots__")
        assert hasattr(ToolSchema, "__slots__")
        assert hasattr(Usage, "__slots__")
        assert hasattr(CompletionResult, "__slots__")

    def test_message_field_ordering(self) -> None:
        """Test Message field initialization order."""
        msg = Message(role="user", content="test")
        # Verify fields can be accessed in expected order
        assert msg.role == "user"
        assert msg.content == "test"
        assert msg.tool_calls is None
        assert msg.tool_call_id is None
