"""Core type definitions for ECS-based LLM Agent."""

from dataclasses import dataclass
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

__all__ = [
    "EntityId",
    "ToolCall",
    "Message",
    "ToolSchema",
    "Usage",
    "CompletionResult",
    "StreamDelta",
    "RetryConfig",
    "ErrorOccurredEvent",
    "ConversationTruncatedEvent",
    "PlanStepCompletedEvent",
    "MessageDeliveredEvent",
    "PlanRevisedEvent",
]
