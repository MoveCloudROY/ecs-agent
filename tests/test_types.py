"""Tests for core type definitions."""

import pytest
from ecs_agent.types import (
    EntityId,
    FileRefPart,
    ImageUrlPart,
    Message,
    ToolCall,
    ToolSchema,
    CompletionResult,
    DelegationCompletedEvent,
    Usage,
    StreamDelta,
    RetryConfig,
    SubagentSessionRecord,
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
        assert msg.parts is None
        assert msg.tool_calls is None
        assert msg.tool_call_id is None

    def test_message_with_tool_calls(self) -> None:
        """Test Message with tool_calls."""
        tc = ToolCall(id="1", name="search", arguments={})
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
        tc = ToolCall(id="call_1", name="search", arguments={"q": "test"})
        assert tc.id == "call_1"
        assert tc.name == "search"
        assert tc.arguments == {"q": "test"}

    def test_toolcall_slots_prevent_extra_attributes(self) -> None:
        """Test that slots=True prevents adding arbitrary attributes."""
        tc = ToolCall(id="1", name="test", arguments={})
        with pytest.raises(AttributeError):
            tc.extra = "bad"  # type: ignore


class TestMessagePart:
    """Test multimodal message part dataclasses."""

    def test_message_parts_can_be_constructed(self) -> None:
        image = ImageUrlPart(url="https://example.com/a.png", detail="high")
        file_ref = FileRefPart(file_id="file_123", filename="a.txt")

        assert image.url == "https://example.com/a.png"
        assert image.detail == "high"
        assert file_ref.file_id == "file_123"
        assert file_ref.filename == "a.txt"

    def test_message_with_multimodal_parts(self) -> None:
        msg = Message(
            role="user",
            content="describe this image",
            parts=[
                ImageUrlPart(url="https://example.com/a.png", detail="auto"),
                FileRefPart(file_id="file_abc", filename="notes.pdf"),
            ],
        )

        assert msg.role == "user"
        assert msg.content == "describe this image"
        assert msg.parts is not None
        assert len(msg.parts) == 2
        assert isinstance(msg.parts[0], ImageUrlPart)
        assert isinstance(msg.parts[1], FileRefPart)
    def test_message_part_slots_prevent_extra_attributes(self) -> None:
        image = ImageUrlPart(url="https://example.com/img.png")
        file_ref = FileRefPart(file_id="file_1")

        with pytest.raises(AttributeError):
            image.extra = "bad"  # type: ignore
        with pytest.raises(AttributeError):
            file_ref.extra = "bad"  # type: ignore


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


class TestStreamDelta:
    """Test StreamDelta dataclass."""

    def test_stream_delta_default(self) -> None:
        """Test StreamDelta with default None values."""
        delta = StreamDelta()
        assert delta.content is None
        assert delta.reasoning_content is None
        assert delta.tool_calls is None
        assert delta.finish_reason is None
        assert delta.usage is None

    def test_stream_delta_with_content(self) -> None:
        """Test StreamDelta with content."""
        delta = StreamDelta(content="Hello ")
        assert delta.content == "Hello "
        assert delta.reasoning_content is None
        assert delta.tool_calls is None
        assert delta.finish_reason is None
        assert delta.usage is None

    def test_stream_delta_with_tool_calls(self) -> None:
        """Test StreamDelta with tool_calls."""
        tc = ToolCall(id="call_1", name="search", arguments={"q": "test"})
        delta = StreamDelta(tool_calls=[tc], finish_reason="stop")
        assert delta.content is None
        assert delta.reasoning_content is None
        assert delta.tool_calls == [tc]
        assert delta.finish_reason == "stop"
        assert delta.usage is None


class TestRetryConfig:
    """Test RetryConfig dataclass."""

    def test_retry_config_defaults(self) -> None:
        """Test RetryConfig with default values."""
        config = RetryConfig()
        assert config.max_attempts == 3
        assert config.multiplier == 1.0
        assert config.min_wait == 4.0
        assert config.max_wait == 60.0
        assert config.retry_status_codes == (429, 500, 502, 503, 504)


def test_subagent_session_record_positional_fields_remain_compatible() -> None:
    """New launch trace fields do not shift existing positional arguments."""
    record = SubagentSessionRecord(
        "session-1",
        "worker",
        "do work",
        EntityId(1),
        "created",
        "updated",
        ["skill-a"],
        True,
        True,
        "running",
        "corr-1",
        "traceparent-1",
        30.0,
        "deadline",
        "excerpt",
        "summary",
        "artifact",
        "record-path",
        "inline",
        None,
        "started",
        "finished",
    )

    assert record.timeout_seconds == 30.0
    assert record.deadline_at == "deadline"
    assert record.finished_at == "finished"
    assert record.launch_trace_id is None


def test_delegation_completed_event_positional_fields_remain_compatible() -> None:
    """New fallback trace fields do not shift existing positional arguments."""
    event = DelegationCompletedEvent(
        EntityId(1),
        "worker",
        "result",
        True,
        None,
        "corr-1",
        "traceparent-1",
        "child-world",
        "obs-1",
        None,
        1.5,
        "completed",
        "success",
    )

    assert event.correlation_id == "corr-1"
    assert event.traceparent == "traceparent-1"
    assert event.observation_id == "obs-1"
    assert event.duration_seconds == 1.5
    assert event.status == "success"
    assert event.task == ""
    assert event.trace_id is None

    def test_retry_config_custom(self) -> None:
        """Test RetryConfig with custom values."""
        config = RetryConfig(
            max_attempts=5,
            multiplier=2.0,
            min_wait=1.0,
            max_wait=30.0,
            retry_status_codes=(408, 429, 500),
        )
        assert config.max_attempts == 5
        assert config.multiplier == 2.0
        assert config.min_wait == 1.0
        assert config.max_wait == 30.0
        assert config.retry_status_codes == (408, 429, 500)


class TestSystemHandle:
    """Test SystemHandle type for stable system identity."""

    def test_systemhandle_is_newtype(self) -> None:
        """Test that SystemHandle is a valid NewType."""
        from ecs_agent.types import SystemHandle

        handle = SystemHandle("reasoning_system_1")
        assert handle == "reasoning_system_1"

    def test_systemhandle_with_various_values(self) -> None:
        """Test SystemHandle with different string values."""
        from ecs_agent.types import SystemHandle

        h1 = SystemHandle("sys_001")
        h2 = SystemHandle("planning_v2")
        h3 = SystemHandle("tool_exec_primary")
        assert h1 == "sys_001"
        assert h2 == "planning_v2"
        assert h3 == "tool_exec_primary"

    def test_systemhandle_type_hint_compatibility(self) -> None:
        """Test that SystemHandle works as a type hint."""
        from ecs_agent.types import SystemHandle

        def register_system(handle: SystemHandle) -> str:
            return handle

        result = register_system(SystemHandle("test_system"))
        assert result == "test_system"


class TestInterruptionReason:
    """Test InterruptionReason enum for interruption categorization."""

    def test_interruption_reason_user_requested(self) -> None:
        """Test USER_REQUESTED enum value."""
        from ecs_agent.types import InterruptionReason

        assert InterruptionReason.USER_REQUESTED.value == "user_requested"

    def test_interruption_reason_system_pause(self) -> None:
        """Test SYSTEM_PAUSE enum value."""
        from ecs_agent.types import InterruptionReason

        assert InterruptionReason.SYSTEM_PAUSE.value == "system_pause"

    def test_interruption_reason_error(self) -> None:
        """Test ERROR enum value."""
        from ecs_agent.types import InterruptionReason

        assert InterruptionReason.ERROR.value == "error"

    def test_interruption_reason_completion(self) -> None:
        """Test COMPLETION enum value."""
        from ecs_agent.types import InterruptionReason

        assert InterruptionReason.COMPLETION.value == "completion"


class TestRevertRequest:
    """Test RevertRequest for conversation tree revert operations."""

    def test_revert_request_basic(self) -> None:
        """Test RevertRequest with target branch."""
        from ecs_agent.types import RevertRequest, EntityId

        req = RevertRequest(
            entity_id=EntityId(42),
            target_branch_id="branch_v1",
        )
        assert req.entity_id == EntityId(42)
        assert req.target_branch_id == "branch_v1"

    def test_revert_request_slots(self) -> None:
        """Test RevertRequest uses slots."""
        from ecs_agent.types import RevertRequest, EntityId

        req = RevertRequest(entity_id=EntityId(1), target_branch_id="main")
        assert hasattr(type(req), "__slots__")


class TestRevertResult:
    """Test RevertResult for revert operation outcomes."""

    def test_revert_result_success(self) -> None:
        """Test RevertResult for successful revert."""
        from ecs_agent.types import RevertResult, EntityId

        result = RevertResult(
            entity_id=EntityId(99),
            success=True,
            new_branch_id="branch_v2",
            message="Reverted to branch_v2",
        )
        assert result.entity_id == EntityId(99)
        assert result.success is True
        assert result.new_branch_id == "branch_v2"
        assert result.message == "Reverted to branch_v2"

    def test_revert_result_failure(self) -> None:
        """Test RevertResult for failed revert."""
        from ecs_agent.types import RevertResult, EntityId

        result = RevertResult(
            entity_id=EntityId(10),
            success=False,
            message="Branch not found",
        )
        assert result.entity_id == EntityId(10)
        assert result.success is False
        assert result.new_branch_id is None
        assert result.message == "Branch not found"

    def test_revert_result_default_message(self) -> None:
        """Test RevertResult with default empty message."""
        from ecs_agent.types import RevertResult, EntityId

        result = RevertResult(entity_id=EntityId(1), success=True)
        assert result.message == ""

    def test_revert_result_slots(self) -> None:
        """Test RevertResult uses slots."""
        from ecs_agent.types import RevertResult, EntityId

        result = RevertResult(entity_id=EntityId(1), success=True)
        assert hasattr(type(result), "__slots__")
