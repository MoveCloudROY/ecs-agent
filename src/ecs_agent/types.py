"""Core type definitions for ECS-based LLM Agent."""

from dataclasses import dataclass
from typing import Any, NewType

EntityId = NewType("EntityId", int)


@dataclass(slots=True)
class ToolCall:
    """Represents a call to a tool/function."""

    id: str
    name: str
    arguments: str


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


__all__ = [
    "EntityId",
    "ToolCall",
    "Message",
    "ToolSchema",
    "Usage",
    "CompletionResult",
    "ErrorOccurredEvent",
    "ConversationTruncatedEvent",
    "PlanStepCompletedEvent",
    "MessageDeliveredEvent",
]
