"""Raw observability events for adapter-neutral telemetry mapping."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

from ecs_agent.types import EntityId, Message, ToolSchema, Usage, UserInputReceivedEvent


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
    start_time: datetime | None = None
    end_time: datetime | None = None
    cost_details: dict[str, Any] = field(default_factory=dict)


__all__ = [
    "LLMObservationCompletedEvent",
    "LLMObservationStartedEvent",
    "LLMObservationStatus",
    "UserInputReceivedEvent",
]
