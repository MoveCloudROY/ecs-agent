"""Raw observability events for adapter-neutral telemetry mapping."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from ecs_agent.types import EntityId, Message, ToolSchema, Usage


LLMObservationStatus = Literal["success", "error", "cancelled", "unknown"]


@dataclass(slots=True)
class LLMObservationStartedEvent:
    """Raw LLM generation observation start event."""

    entity_id: EntityId | int
    provider_id: str
    model: str
    operation: str
    messages: list[Message]
    tools: list[ToolSchema] | None = None
    streaming: bool = False
    model_parameters: dict[str, Any] | None = None


@dataclass(slots=True)
class LLMObservationCompletedEvent:
    """Raw LLM generation observation completion event."""

    entity_id: EntityId | int
    provider_id: str
    model: str
    operation: str
    messages: list[Message]
    tools: list[ToolSchema] | None = None
    streaming: bool = False
    model_parameters: dict[str, Any] | None = None
    response_message: Message | None = None
    reasoning_content: str | None = None
    usage: Usage | None = None
    response_id: str | None = None
    status: LLMObservationStatus = "success"
    error: str | None = None
    duration_seconds: float | None = None
    cost_details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class UserInputReceivedEvent:
    """Raw user input text received by UserInputSystem."""

    entity_id: EntityId | int
    prompt: str
    text: str


__all__ = [
    "LLMObservationCompletedEvent",
    "LLMObservationStartedEvent",
    "LLMObservationStatus",
    "UserInputReceivedEvent",
]
