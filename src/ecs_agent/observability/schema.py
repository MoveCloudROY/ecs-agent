"""Internal telemetry schema and JSON-safe serialization helpers."""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from datetime import date, datetime
from enum import Enum
from typing import Any, Literal, TypeAlias


TelemetryRecordKind: TypeAlias = Literal["trace", "span", "generation", "tool", "event"]
TelemetryStatus: TypeAlias = Literal[
    "success",
    "error",
    "cancelled",
    "running",
    "unknown",
]
JsonSafe: TypeAlias = None | bool | int | float | str | list["JsonSafe"] | dict[str, "JsonSafe"]


def json_safe(value: Any) -> JsonSafe:
    """Convert a Python value into deterministic JSON-safe data."""
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, Enum):
        return json_safe(value.value)
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: json_safe(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, dict):
        return {str(json_safe(key)): json_safe(item) for key, item in value.items()}
    if isinstance(value, tuple | list | set | frozenset):
        return [json_safe(item) for item in value]
    return repr(value)


@dataclass(slots=True)
class TelemetryRecord:
    """One internal trace/span/generation/tool/event telemetry observation."""

    trace_id: str
    run_id: str
    observation_id: str
    name: str
    kind: TelemetryRecordKind
    schema_version: str = "1.0"
    parent_observation_id: str | None = None
    entity_id: int | None = None
    tick: int | None = None
    system_name: str | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None
    latency_ms: float | None = None
    status: TelemetryStatus = "unknown"
    input: Any = None
    output: Any = None
    metadata: dict[str, Any] | None = None
    error: str | dict[str, Any] | None = None
    model: str | None = None
    model_parameters: dict[str, Any] | None = None
    usage_details: dict[str, Any] | None = None
    cost_details: dict[str, Any] | None = None
    redaction: dict[str, Any] | None = None

    def to_payload(self) -> dict[str, JsonSafe]:
        """Serialize the record into a JSON-safe payload."""
        return {
            "schema_version": json_safe(self.schema_version),
            "trace_id": json_safe(self.trace_id),
            "run_id": json_safe(self.run_id),
            "observation_id": json_safe(self.observation_id),
            "parent_observation_id": json_safe(self.parent_observation_id),
            "entity_id": json_safe(self.entity_id),
            "tick": json_safe(self.tick),
            "system_name": json_safe(self.system_name),
            "name": json_safe(self.name),
            "kind": json_safe(self.kind),
            "start_time": json_safe(self.start_time),
            "end_time": json_safe(self.end_time),
            "latency_ms": json_safe(self.latency_ms),
            "status": json_safe(self.status),
            "input": json_safe(self.input),
            "output": json_safe(self.output),
            "metadata": json_safe(self.metadata),
            "error": json_safe(self.error),
            "model": json_safe(self.model),
            "model_parameters": json_safe(self.model_parameters),
            "usage_details": json_safe(self.usage_details),
            "cost_details": json_safe(self.cost_details),
            "redaction": json_safe(self.redaction),
        }


@dataclass(slots=True)
class TelemetryScore:
    """One score attached to an observation or trace."""

    trace_id: str
    run_id: str
    observation_id: str
    name: str
    value: bool | int | float | str
    schema_version: str = "1.0"
    comment: str | None = None
    metadata: dict[str, Any] | None = None

    def to_payload(self) -> dict[str, JsonSafe]:
        """Serialize the score into a JSON-safe payload."""
        return {
            "schema_version": json_safe(self.schema_version),
            "trace_id": json_safe(self.trace_id),
            "run_id": json_safe(self.run_id),
            "observation_id": json_safe(self.observation_id),
            "name": json_safe(self.name),
            "value": json_safe(self.value),
            "comment": json_safe(self.comment),
            "metadata": json_safe(self.metadata),
        }


__all__ = [
    "JsonSafe",
    "TelemetryRecord",
    "TelemetryRecordKind",
    "TelemetryScore",
    "TelemetryStatus",
    "json_safe",
]
