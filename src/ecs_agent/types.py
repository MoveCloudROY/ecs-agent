"""Core type definitions for ECS-based LLM Agent."""

import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import Any, NewType

EntityId = NewType("EntityId", int)


@dataclass(slots=True)
class ToolCall:
    """Represents a call to a tool/function."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(slots=True)
class Message:
    """Represents a message in the conversation."""

    role: str
    content: str
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None


@dataclass(slots=True)
class ToolSchema:
    """Describes the schema of a tool."""

    name: str
    description: str
    parameters: dict[str, Any]


@dataclass(slots=True)
class Usage:
    """Token usage statistics."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass(slots=True)
class CompletionResult:
    """Result from LLM completion."""

    message: Message
    usage: Usage | None = None


@dataclass(slots=True)
class ConversationTruncatedEvent:
    entity_id: EntityId
    removed_count: int


@dataclass(slots=True)
class ErrorOccurredEvent:
    entity_id: EntityId
    error: str
    system_name: str


@dataclass(slots=True)
class MessageDeliveredEvent:
    from_entity: EntityId
    to_entity: EntityId
    message: Message


@dataclass(slots=True)
class PlanStepCompletedEvent:
    entity_id: EntityId
    step_index: int
    step_description: str


@dataclass(slots=True)
class PlanRevisedEvent:
    """Event emitted when the plan is dynamically revised during execution."""

    entity_id: EntityId
    old_steps: list[str]
    new_steps: list[str]


@dataclass(slots=True)
class StreamDelta:
    """Represents a chunk of streamed response data."""

    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    finish_reason: str | None = None
    usage: Usage | None = None


@dataclass(slots=True)
class RetryConfig:
    """Configuration for retry behavior."""

    max_attempts: int = 3
    multiplier: float = 1.0
    min_wait: float = 4.0
    max_wait: float = 60.0
    retry_status_codes: tuple[int, ...] = (429, 500, 502, 503, 504)


class ToolTimeoutError(Exception):
    """Raised when a sandboxed tool exceeds its timeout."""

    pass


class ApprovalPolicy(Enum):
    """Policy for tool approval decisions."""

    ALWAYS_APPROVE = "always_approve"
    ALWAYS_DENY = "always_deny"
    REQUIRE_APPROVAL = "require_approval"


@dataclass(slots=True)
class ToolApprovalRequestedEvent:
    """Event emitted when a tool call requires approval."""

    entity_id: EntityId
    tool_call: ToolCall
    approval_future: asyncio.Future[bool]


@dataclass(slots=True)
class ToolApprovedEvent:
    """Event emitted when a tool call is approved."""

    entity_id: EntityId
    tool_call_id: str


@dataclass(slots=True)
class ToolDeniedEvent:
    """Event emitted when a tool call is denied."""

    entity_id: EntityId
    tool_call_id: str
    reason: str


@dataclass(slots=True)
class MCTSNodeScoredEvent:
    """Event emitted when an MCTS node is scored."""

    entity_id: EntityId
    node_id: int
    score: float


@dataclass(slots=True)
class RAGRetrievalCompletedEvent:
    """Event emitted when RAG retrieval completes."""

    entity_id: EntityId
    query: str
    num_results: int

__all__ = [
    "ApprovalPolicy",
    "CompletionResult",
    "ConversationTruncatedEvent",
    "EntityId",
    "ErrorOccurredEvent",
    "MCTSNodeScoredEvent",
    "Message",
    "MessageDeliveredEvent",
    "PlanRevisedEvent",
    "PlanStepCompletedEvent",
    "RAGRetrievalCompletedEvent",
    "RetryConfig",
    "StreamDelta",
    "ToolApprovedEvent",
    "ToolApprovalRequestedEvent",
    "ToolCall",
    "ToolDeniedEvent",
    "ToolSchema",
    "ToolTimeoutError",
    "Usage",
]
