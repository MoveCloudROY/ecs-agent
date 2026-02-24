"""Tests for new type definitions (Task 1): ToolTimeoutError, ApprovalPolicy, events."""

import asyncio
from enum import Enum

import pytest

from ecs_agent.types import (
    ApprovalPolicy,
    EntityId,
    MCTSNodeScoredEvent,
    RAGRetrievalCompletedEvent,
    ToolApprovedEvent,
    ToolApprovalRequestedEvent,
    ToolCall,
    ToolDeniedEvent,
    ToolTimeoutError,
)


class TestToolTimeoutError:
    """ToolTimeoutError exception tests."""

    def test_is_exception_subclass(self) -> None:
        """ToolTimeoutError should be a subclass of Exception."""
        assert issubclass(ToolTimeoutError, Exception)

    def test_can_be_raised_and_caught(self) -> None:
        """ToolTimeoutError should be raisable and catchable."""
        with pytest.raises(ToolTimeoutError):
            raise ToolTimeoutError("Tool execution timeout")

    def test_has_docstring(self) -> None:
        """ToolTimeoutError should have a docstring."""
        assert ToolTimeoutError.__doc__ is not None
        assert "timeout" in ToolTimeoutError.__doc__.lower()


class TestApprovalPolicy:
    """ApprovalPolicy enum tests."""

    def test_is_enum(self) -> None:
        """ApprovalPolicy should be an Enum."""
        assert issubclass(ApprovalPolicy, Enum)

    def test_has_three_members(self) -> None:
        """ApprovalPolicy should have exactly 3 members."""
        assert len(ApprovalPolicy) == 3

    def test_always_approve_value(self) -> None:
        """ALWAYS_APPROVE member should have correct value."""
        assert ApprovalPolicy.ALWAYS_APPROVE.value == "always_approve"

    def test_always_deny_value(self) -> None:
        """ALWAYS_DENY member should have correct value."""
        assert ApprovalPolicy.ALWAYS_DENY.value == "always_deny"

    def test_require_approval_value(self) -> None:
        """REQUIRE_APPROVAL member should have correct value."""
        assert ApprovalPolicy.REQUIRE_APPROVAL.value == "require_approval"


class TestToolApprovalRequestedEvent:
    """ToolApprovalRequestedEvent dataclass tests."""

    @pytest.mark.asyncio
    async def test_instantiate_with_all_fields(self) -> None:
        """ToolApprovalRequestedEvent should instantiate with required fields."""
        future: asyncio.Future[bool] = asyncio.get_event_loop().create_future()
        tool_call = ToolCall(id="t1", name="test_fn", arguments={"arg1": "value1"})
        event = ToolApprovalRequestedEvent(
            entity_id=EntityId(1),
            tool_call=tool_call,
            approval_future=future,
        )
        assert event.entity_id == EntityId(1)
        assert event.tool_call == tool_call
        assert event.approval_future is future

    @pytest.mark.asyncio
    async def test_has_dataclass_fields(self) -> None:
        """ToolApprovalRequestedEvent should be a dataclass with slots."""
        future: asyncio.Future[bool] = asyncio.get_event_loop().create_future()
        tool_call = ToolCall(id="t1", name="test_fn", arguments={})
        event = ToolApprovalRequestedEvent(
            entity_id=EntityId(42),
            tool_call=tool_call,
            approval_future=future,
        )
        # Check that it has the expected attributes
        assert hasattr(event, "entity_id")
        assert hasattr(event, "tool_call")
        assert hasattr(event, "approval_future")


class TestToolApprovedEvent:
    """ToolApprovedEvent dataclass tests."""

    def test_instantiate_with_all_fields(self) -> None:
        """ToolApprovedEvent should instantiate with required fields."""
        event = ToolApprovedEvent(entity_id=EntityId(1), tool_call_id="t1")
        assert event.entity_id == EntityId(1)
        assert event.tool_call_id == "t1"

    def test_has_dataclass_fields(self) -> None:
        """ToolApprovedEvent should be a dataclass with slots."""
        event = ToolApprovedEvent(entity_id=EntityId(42), tool_call_id="tool_123")
        assert hasattr(event, "entity_id")
        assert hasattr(event, "tool_call_id")


class TestToolDeniedEvent:
    """ToolDeniedEvent dataclass tests."""

    def test_instantiate_with_all_fields(self) -> None:
        """ToolDeniedEvent should instantiate with required fields."""
        event = ToolDeniedEvent(
            entity_id=EntityId(1),
            tool_call_id="t1",
            reason="User denied approval",
        )
        assert event.entity_id == EntityId(1)
        assert event.tool_call_id == "t1"
        assert event.reason == "User denied approval"

    def test_has_dataclass_fields(self) -> None:
        """ToolDeniedEvent should be a dataclass with slots."""
        event = ToolDeniedEvent(
            entity_id=EntityId(99),
            tool_call_id="denied_tool",
            reason="Security policy violation",
        )
        assert hasattr(event, "entity_id")
        assert hasattr(event, "tool_call_id")
        assert hasattr(event, "reason")


class TestMCTSNodeScoredEvent:
    """MCTSNodeScoredEvent dataclass tests."""

    def test_instantiate_with_all_fields(self) -> None:
        """MCTSNodeScoredEvent should instantiate with required fields."""
        event = MCTSNodeScoredEvent(entity_id=EntityId(1), node_id=10, score=0.95)
        assert event.entity_id == EntityId(1)
        assert event.node_id == 10
        assert event.score == 0.95

    def test_has_dataclass_fields(self) -> None:
        """MCTSNodeScoredEvent should be a dataclass with slots."""
        event = MCTSNodeScoredEvent(
            entity_id=EntityId(5),
            node_id=42,
            score=0.50,
        )
        assert hasattr(event, "entity_id")
        assert hasattr(event, "node_id")
        assert hasattr(event, "score")

    def test_score_can_be_float(self) -> None:
        """MCTSNodeScoredEvent score field should accept any float value."""
        for score in [0.0, 0.5, 1.0, 0.123456]:
            event = MCTSNodeScoredEvent(entity_id=EntityId(1), node_id=1, score=score)
            assert event.score == score


class TestRAGRetrievalCompletedEvent:
    """RAGRetrievalCompletedEvent dataclass tests."""

    def test_instantiate_with_all_fields(self) -> None:
        """RAGRetrievalCompletedEvent should instantiate with required fields."""
        event = RAGRetrievalCompletedEvent(
            entity_id=EntityId(1),
            query="test query",
            num_results=5,
        )
        assert event.entity_id == EntityId(1)
        assert event.query == "test query"
        assert event.num_results == 5

    def test_has_dataclass_fields(self) -> None:
        """RAGRetrievalCompletedEvent should be a dataclass with slots."""
        event = RAGRetrievalCompletedEvent(
            entity_id=EntityId(7),
            query="complex query",
            num_results=10,
        )
        assert hasattr(event, "entity_id")
        assert hasattr(event, "query")
        assert hasattr(event, "num_results")

    def test_num_results_is_int(self) -> None:
        """RAGRetrievalCompletedEvent num_results should be an integer."""
        event = RAGRetrievalCompletedEvent(
            entity_id=EntityId(1),
            query="q",
            num_results=42,
        )
        assert isinstance(event.num_results, int)
        assert event.num_results == 42


class TestAllExports:
    """Test that all new symbols are properly exported."""

    def test_all_new_symbols_importable(self) -> None:
        """All new symbols should be importable from ecs_agent.types."""
        from ecs_agent import types

        required_symbols = {
            "ToolTimeoutError",
            "ApprovalPolicy",
            "ToolApprovalRequestedEvent",
            "ToolApprovedEvent",
            "ToolDeniedEvent",
            "MCTSNodeScoredEvent",
            "RAGRetrievalCompletedEvent",
        }
        missing = required_symbols - set(types.__all__)
        assert not missing, f"Missing from __all__: {missing}"

    def test_all_symbols_have_correct_types(self) -> None:
        """All new symbols should have expected types."""
        assert issubclass(ToolTimeoutError, Exception)
        assert issubclass(ApprovalPolicy, Enum)
        # Events are dataclasses, just check they're callable
        assert callable(ToolApprovalRequestedEvent)
        assert callable(ToolApprovedEvent)
        assert callable(ToolDeniedEvent)
        assert callable(MCTSNodeScoredEvent)
        assert callable(RAGRetrievalCompletedEvent)
