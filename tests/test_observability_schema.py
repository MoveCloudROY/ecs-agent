import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone

import pytest

from ecs_agent.observability.context import (
    current_observation_id,
    current_observation_stack,
    current_run_id,
    current_trace_id,
    push_observation,
    reset_observation,
    reset_run_context,
    set_run_context,
)
from ecs_agent.observability.schema import TelemetryRecord, TelemetryScore, json_safe
from ecs_agent.observability.sinks import NoOpTelemetrySink, RecordingTelemetrySink
from ecs_agent.types import EntityId, Message, ToolCall


@dataclass(slots=True)
class NestedPayload:
    label: str
    tool_call: ToolCall


class ReprOnly:
    def __repr__(self) -> str:
        return "<ReprOnly stable>"


def test_telemetry_record_defaults_are_explicit() -> None:
    record = TelemetryRecord(
        trace_id="trace-default",
        run_id="run-default",
        observation_id="obs-default",
        name="default span",
        kind="span",
    )

    assert record.schema_version == "1.0"
    assert record.parent_observation_id is None
    assert record.entity_id is None
    assert record.tick is None
    assert record.system_name is None
    assert record.start_time is None
    assert record.end_time is None
    assert record.latency_ms is None
    assert record.status == "unknown"
    assert record.input is None
    assert record.output is None
    assert record.metadata is None
    assert record.error is None
    assert record.model is None
    assert record.model_parameters is None
    assert record.usage_details is None
    assert record.cost_details is None
    assert record.redaction is None


def test_telemetry_score_defaults_are_explicit() -> None:
    score = TelemetryScore(
        trace_id="trace-default",
        run_id="run-default",
        observation_id="obs-default",
        name="quality",
        value=1.0,
    )

    assert score.schema_version == "1.0"
    assert score.comment is None
    assert score.metadata is None


def test_telemetry_record_serializes_raw_input_output() -> None:
    raw_input = {
        "messages": [Message(role="user", content="raw prompt before redaction")],
        "tool_calls": [
            ToolCall(id="call-raw", name="lookup", arguments={"query": "raw value"})
        ],
    }
    raw_output = {
        "message": Message(role="assistant", content="raw response before redaction")
    }
    record = TelemetryRecord(
        trace_id="trace-raw",
        run_id="run-raw",
        observation_id="obs-raw",
        name="raw generation",
        kind="generation",
        input=raw_input,
        output=raw_output,
        redaction={"applied": False, "fields": []},
    )

    payload = record.to_payload()

    assert payload["input"] == {
        "messages": [
            {
                "role": "user",
                "content": "raw prompt before redaction",
                "parts": None,
                "tool_calls": None,
                "tool_call_id": None,
                "reasoning_content": None,
                "reasoning_signature": None,
                "compaction_metadata": None,
                "cache_control": False,
            }
        ],
        "tool_calls": [
            {
                "id": "call-raw",
                "name": "lookup",
                "arguments": {"query": "raw value"},
            }
        ],
    }
    assert payload["output"] == {
        "message": {
            "role": "assistant",
            "content": "raw response before redaction",
            "parts": None,
            "tool_calls": None,
            "tool_call_id": None,
            "reasoning_content": None,
            "reasoning_signature": None,
            "compaction_metadata": None,
            "cache_control": False,
        }
    }
    assert payload["redaction"] == {"applied": False, "fields": []}


def test_telemetry_record_serializes_non_json_values_safely() -> None:
    record = TelemetryRecord(
        trace_id="trace-safe",
        run_id="run-safe",
        observation_id="obs-safe",
        name="safe serialization",
        kind="event",
        input={"object": ReprOnly()},
        output=ReprOnly(),
    )

    payload = record.to_payload()

    assert payload["input"] == {"object": "<ReprOnly stable>"}
    assert payload["output"] == "<ReprOnly stable>"


def test_telemetry_record_payload_contains_schema_fields_and_json_safe_values() -> None:
    tool_call = ToolCall(
        id="call-1",
        name="lookup_weather",
        arguments={"city": "Paris", "fallback": ReprOnly()},
    )
    message = Message(
        role="assistant",
        content="Checking weather",
        tool_calls=[tool_call],
    )
    record = TelemetryRecord(
        trace_id="trace-1",
        run_id="run-1",
        observation_id="obs-1",
        parent_observation_id="parent-1",
        entity_id=EntityId(7),
        tick=3,
        system_name="ReasoningSystem",
        name="llm.call",
        kind="generation",
        start_time=datetime(2026, 5, 5, 12, 0, tzinfo=timezone.utc),
        end_time=datetime(2026, 5, 5, 12, 0, 1, 250000, tzinfo=timezone.utc),
        latency_ms=1250.0,
        status="success",
        input=[message, {"nested": NestedPayload("payload", tool_call)}],
        output={"message": message, "object": ReprOnly()},
        metadata={"attempt": 1, "tags": ["reasoning"]},
        error=None,
        model="gpt-test",
        model_parameters={"temperature": 0.2},
        usage_details={"prompt_tokens": 10, "completion_tokens": 5},
        cost_details={"total_cost": 0.01, "currency": "USD"},
        redaction={"applied": False, "fields": []},
    )

    payload = record.to_payload()

    assert payload == {
        "schema_version": "1.0",
        "trace_id": "trace-1",
        "run_id": "run-1",
        "observation_id": "obs-1",
        "parent_observation_id": "parent-1",
        "entity_id": 7,
        "tick": 3,
        "system_name": "ReasoningSystem",
        "name": "llm.call",
        "kind": "generation",
        "start_time": "2026-05-05T12:00:00+00:00",
        "end_time": "2026-05-05T12:00:01.250000+00:00",
        "latency_ms": 1250.0,
        "status": "success",
        "input": [
            {
                "role": "assistant",
                "content": "Checking weather",
                "parts": None,
                "tool_calls": [
                    {
                        "id": "call-1",
                        "name": "lookup_weather",
                        "arguments": {
                            "city": "Paris",
                            "fallback": "<ReprOnly stable>",
                        },
                    }
                ],
                "tool_call_id": None,
                "reasoning_content": None,
                "reasoning_signature": None,
                "compaction_metadata": None,
                "cache_control": False,
            },
            {
                "nested": {
                    "label": "payload",
                    "tool_call": {
                        "id": "call-1",
                        "name": "lookup_weather",
                        "arguments": {
                            "city": "Paris",
                            "fallback": "<ReprOnly stable>",
                        },
                    },
                }
            },
        ],
        "output": {
            "message": {
                "role": "assistant",
                "content": "Checking weather",
                "parts": None,
                "tool_calls": [
                    {
                        "id": "call-1",
                        "name": "lookup_weather",
                        "arguments": {
                            "city": "Paris",
                            "fallback": "<ReprOnly stable>",
                        },
                    }
                ],
                "tool_call_id": None,
                "reasoning_content": None,
                "reasoning_signature": None,
                "compaction_metadata": None,
                "cache_control": False,
            },
            "object": "<ReprOnly stable>",
        },
        "metadata": {"attempt": 1, "tags": ["reasoning"]},
        "error": None,
        "model": "gpt-test",
        "model_parameters": {"temperature": 0.2},
        "usage_details": {"prompt_tokens": 10, "completion_tokens": 5},
        "cost_details": {"total_cost": 0.01, "currency": "USD"},
        "redaction": {"applied": False, "fields": []},
    }


def test_json_safe_handles_primitives_nested_dataclasses_and_unknown_objects() -> None:
    value = {
        "primitive": ["text", 1, 2.5, True, None],
        "dataclass": NestedPayload(
            label="wrapped",
            tool_call=ToolCall(id="call-2", name="search", arguments={"limit": 3}),
        ),
        9: ReprOnly(),
    }

    assert json_safe(value) == {
        "primitive": ["text", 1, 2.5, True, None],
        "dataclass": {
            "label": "wrapped",
            "tool_call": {
                "id": "call-2",
                "name": "search",
                "arguments": {"limit": 3},
            },
        },
        "9": "<ReprOnly stable>",
    }


def test_telemetry_score_payload_is_separate_from_record_payload() -> None:
    score = TelemetryScore(
        trace_id="trace-1",
        run_id="run-1",
        observation_id="obs-1",
        name="quality",
        value=0.9,
        comment="useful",
        metadata={"source": "test"},
    )

    assert score.to_payload() == {
        "schema_version": "1.0",
        "trace_id": "trace-1",
        "run_id": "run-1",
        "observation_id": "obs-1",
        "name": "quality",
        "value": 0.9,
        "comment": "useful",
        "metadata": {"source": "test"},
    }


@pytest.mark.asyncio
async def test_context_helpers_are_isolated_between_async_contexts() -> None:
    async def scoped_run(run_id: str) -> tuple[str | None, str | None, tuple[str, ...]]:
        run_token = set_run_context(trace_id=f"trace-{run_id}", run_id=run_id)
        obs_token = push_observation(f"obs-{run_id}")
        try:
            await asyncio.sleep(0)
            return current_trace_id(), current_run_id(), current_observation_stack()
        finally:
            reset_observation(obs_token)
            reset_run_context(run_token)

    first, second = await asyncio.gather(scoped_run("one"), scoped_run("two"))

    assert first == ("trace-one", "one", ("obs-one",))
    assert second == ("trace-two", "two", ("obs-two",))
    assert current_trace_id() is None
    assert current_run_id() is None
    assert current_observation_id() is None
    assert current_observation_stack() == ()


@pytest.mark.asyncio
async def test_recording_sink_preserves_operation_order_and_noop_is_silent() -> None:
    sink = RecordingTelemetrySink()
    record = TelemetryRecord(
        trace_id="trace-1",
        run_id="run-1",
        observation_id="obs-1",
        name="span",
        kind="span",
    )
    score = TelemetryScore(
        trace_id="trace-1",
        run_id="run-1",
        observation_id="obs-1",
        name="quality",
        value=1.0,
    )

    await sink.emit(record)
    await sink.score(score)
    await sink.emit(record)
    await sink.flush()
    await sink.shutdown()

    assert sink.records == [record, record]
    assert sink.scores == [score]
    assert sink.operations == [("emit", record), ("score", score), ("emit", record)]

    noop = NoOpTelemetrySink()
    await noop.emit(record)
    await noop.score(score)
    await noop.flush()
    await noop.shutdown()
