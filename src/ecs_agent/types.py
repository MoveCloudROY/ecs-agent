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
