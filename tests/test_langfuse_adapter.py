"""Optional Langfuse adapter tests."""

from __future__ import annotations

import builtins
from datetime import datetime, timezone
from typing import Any

import pytest

from ecs_agent.core import World
from ecs_agent.observability.redaction import SecretRedactor
from ecs_agent.observability.schema import TelemetryRecord, TelemetryScore
from ecs_agent.observability.sinks import RecordingTelemetrySink


class FakeLangfuseClient:
    """Langfuse-like client that records adapter calls without network access."""

    def __init__(self, *, fail: bool = False, **kwargs: Any) -> None:
        self.init_kwargs = kwargs
        self.fail = fail
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def trace(self, **kwargs: Any) -> None:
        """Record a trace call."""
        self._record("trace", kwargs)

    def observation(self, **kwargs: Any) -> None:
        """Record an observation call."""
        self._record("observation", kwargs)

    def create_score(self, **kwargs: Any) -> None:
        """Record a score call."""
        self._record("score", kwargs)

    def flush(self) -> None:
        """Record a flush call."""
        self._record("flush", {})

    def shutdown(self) -> None:
        """Record a shutdown call."""
        self._record("shutdown", {})

    def _record(self, name: str, kwargs: dict[str, Any]) -> None:
        self.calls.append((name, kwargs))
        if self.fail:
            raise RuntimeError(f"{name} failed")


def test_langfuse_module_import_does_not_import_optional_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Importing the integration module does not require the Langfuse SDK."""
    original_import = builtins.__import__

    def guarded_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "langfuse" or name.startswith("langfuse."):
            raise AssertionError("langfuse import attempted too early")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    import ecs_agent.integrations.langfuse as langfuse_adapter

    assert langfuse_adapter.LangfuseConfig().enabled is True


def test_install_without_sdk_raises_actionable_import_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Client creation raises a clear optional-extra ImportError only on use."""
    from ecs_agent.integrations.langfuse import install_langfuse_observability

    original_import = builtins.__import__

    def missing_langfuse(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "langfuse" or name.startswith("langfuse."):
            raise ImportError("not installed")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_langfuse)

    with pytest.raises(
        ImportError,
        match="Install ecs-agent\\[langfuse\\] to use Langfuse observability",
    ):
        install_langfuse_observability(World())


def test_install_reads_env_only_for_absent_config_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Install resolves Langfuse env aliases without overriding explicit config."""
    from ecs_agent.integrations.langfuse import (
        LangfuseConfig,
        install_langfuse_observability,
    )

    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "env-public-value")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "env-private-value")
    monkeypatch.setenv("LANGFUSE_BASE_URL", "env-base-url-value")
    monkeypatch.setenv("LANGFUSE_HOST", "env-host-value")
    sink = RecordingTelemetrySink()

    handle = install_langfuse_observability(
        World(),
        config=LangfuseConfig(public_key="explicit-public", host=None),
        sink=sink,
    )

    resolved = handle.config
    assert resolved.public_key == "explicit-public"
    assert resolved.secret_key == "env-private-value"
    assert resolved.host == "env-host-value"
    assert resolved.enabled is True
    assert handle.sink is sink


def test_base_url_fills_host_when_host_env_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LANGFUSE_BASE_URL is a sensible host alias when LANGFUSE_HOST is absent."""
    from ecs_agent.integrations.langfuse import LangfuseConfig

    monkeypatch.delenv("LANGFUSE_HOST", raising=False)
    monkeypatch.setenv("LANGFUSE_BASE_URL", "env-base-url-value")

    config = LangfuseConfig().with_env()

    assert config.host == "env-base-url-value"


@pytest.mark.asyncio
async def test_langfuse_adapter_maps_generation_and_scores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Generations and scores are mapped to Langfuse calls after redaction."""
    from ecs_agent.integrations.langfuse import LangfuseConfig, LangfuseTelemetrySink

    private_value = "private-redaction-value"
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", private_value)
    client = FakeLangfuseClient()
    sink = LangfuseTelemetrySink(
        client=client,
        config=LangfuseConfig(
            environment="test",
            release="release-one",
            user_id="user-one",
            session_id="session-one",
            tags=["unit"],
            metadata={"suite": "langfuse"},
        ),
    )

    await sink.emit(
        TelemetryRecord(
            trace_id="trace-one",
            run_id="run-one",
            observation_id="generation-one",
            parent_observation_id="trace-root",
            name="llm.complete",
            kind="generation",
            status="success",
            start_time=datetime(2026, 1, 2, tzinfo=timezone.utc),
            end_time=datetime(2026, 1, 2, 0, 0, 1, tzinfo=timezone.utc),
            latency_ms=1000.0,
            input={"messages": [{"content": private_value}]},
            output={"message": "safe response"},
            metadata={"api_key": private_value, "safe": "value"},
            model="model-one",
            model_parameters={"temperature": 0},
            usage_details={"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
            cost_details={"total": 0.0},
        )
    )
    await sink.score(
        TelemetryScore(
            trace_id="trace-one",
            run_id="run-one",
            observation_id="generation-one",
            name="quality",
            value=0.75,
            comment="safe comment",
            metadata={"note": private_value},
        )
    )

    observation_call = client.calls[0]
    assert observation_call[0] == "observation"
    observation_payload = observation_call[1]
    assert observation_payload["trace_id"] == "trace-one"
    assert observation_payload["id"] == "generation-one"
    assert observation_payload["parent_observation_id"] == "trace-root"
    assert observation_payload["as_type"] == "generation"
    assert observation_payload["model"] == "model-one"
    assert observation_payload["usage_details"] == {
        "prompt_tokens": 1,
        "completion_tokens": 2,
        "total_tokens": 3,
    }
    assert observation_payload["metadata"]["environment"] == "test"
    assert observation_payload["metadata"]["redaction"]["total_redactions"] == 2
    assert client.calls[1][0] == "score"
    assert client.calls[1][1]["observation_id"] == "generation-one"

    emitted_text = str(client.calls)
    assert private_value not in emitted_text
    assert "[REDACTED:value:LANGFUSE_SECRET_KEY]" in emitted_text
    assert "[REDACTED:key:api_key]" in emitted_text


@pytest.mark.asyncio
async def test_langfuse_config_metadata_uses_sink_redactor_extra_values() -> None:
    """Config metadata is sanitized with the sink's configured redactor."""
    from ecs_agent.integrations.langfuse import LangfuseConfig, LangfuseTelemetrySink

    synthetic_marker = "synthetic-config-metadata-secret-value"
    client = FakeLangfuseClient()
    sink = LangfuseTelemetrySink(
        client=client,
        config=LangfuseConfig(metadata={"note": synthetic_marker}),
        redactor=SecretRedactor(extra_secret_values=[synthetic_marker]),
    )

    await sink.emit(
        TelemetryRecord(
            trace_id="trace-one",
            run_id="run-one",
            observation_id="event-one",
            name="safe.event",
            kind="event",
        )
    )

    observation_metadata = client.calls[0][1]["metadata"]
    assert observation_metadata["note"] == "[REDACTED:value:extra_secret]"
    assert observation_metadata["redaction"] == {
        "total_redactions": 1,
        "counts_by_rule": {"value:extra_secret": 1},
    }
    assert synthetic_marker not in str(client.calls)


@pytest.mark.asyncio
async def test_langfuse_config_fields_are_redacted_in_trace_exports() -> None:
    """Config-derived trace kwargs and metadata are sanitized before export."""
    from ecs_agent.integrations.langfuse import LangfuseConfig, LangfuseTelemetrySink

    synthetic_marker = "synthetic-config-export-secret-value"
    client = FakeLangfuseClient()
    sink = LangfuseTelemetrySink(
        client=client,
        config=LangfuseConfig(
            environment=synthetic_marker,
            release=synthetic_marker,
            user_id=synthetic_marker,
            session_id=synthetic_marker,
            tags=[synthetic_marker],
        ),
        redactor=SecretRedactor(extra_secret_values=[synthetic_marker]),
    )

    await sink.emit(
        TelemetryRecord(
            trace_id="trace-one",
            run_id="run-one",
            observation_id="trace-one",
            name="runner.run",
            kind="trace",
            status="success",
        )
    )

    trace_payload = client.calls[0][1]
    assert trace_payload["environment"] == "[REDACTED:value:extra_secret]"
    assert trace_payload["release"] == "[REDACTED:value:extra_secret]"
    assert trace_payload["user_id"] == "[REDACTED:value:extra_secret]"
    assert trace_payload["session_id"] == "[REDACTED:value:extra_secret]"
    assert trace_payload["tags"] == ["[REDACTED:value:extra_secret]"]
    assert trace_payload["metadata"]["environment"] == "[REDACTED:value:extra_secret]"
    assert trace_payload["metadata"]["release"] == "[REDACTED:value:extra_secret]"
    assert trace_payload["metadata"]["user_id"] == "[REDACTED:value:extra_secret]"
    assert trace_payload["metadata"]["session_id"] == "[REDACTED:value:extra_secret]"
    assert trace_payload["metadata"]["tags"] == ["[REDACTED:value:extra_secret]"]
    assert trace_payload["metadata"]["redaction"] == {
        "total_redactions": 5,
        "counts_by_rule": {"value:extra_secret": 5},
    }
    assert synthetic_marker not in str(client.calls)


@pytest.mark.asyncio
async def test_langfuse_adapter_maps_trace_span_tool_and_event_records() -> None:
    """Every internal record kind maps to a Langfuse trace or observation call."""
    from ecs_agent.integrations.langfuse import LangfuseTelemetrySink

    client = FakeLangfuseClient()
    sink = LangfuseTelemetrySink(client=client)

    records = [
        TelemetryRecord(
            trace_id="trace-one",
            run_id="run-one",
            observation_id="trace-root",
            name="runner.run",
            kind="trace",
            status="success",
            input={"start": True},
            output={"done": True},
        ),
        TelemetryRecord(
            trace_id="trace-one",
            run_id="run-one",
            observation_id="span-one",
            parent_observation_id="trace-root",
            name="runner.tick",
            kind="span",
            status="success",
        ),
        TelemetryRecord(
            trace_id="trace-one",
            run_id="run-one",
            observation_id="tool-one",
            parent_observation_id="trace-root",
            name="tool.lookup",
            kind="tool",
            status="error",
            error="tool failed",
        ),
        TelemetryRecord(
            trace_id="trace-one",
            run_id="run-one",
            observation_id="event-one",
            parent_observation_id="trace-root",
            name="stream.end",
            kind="event",
            status="success",
        ),
    ]

    for record in records:
        await sink.emit(record)

    assert [name for name, _ in client.calls] == [
        "trace",
        "observation",
        "observation",
        "observation",
    ]
    assert client.calls[0][1]["id"] == "trace-one"
    assert client.calls[1][1]["as_type"] == "span"
    assert client.calls[2][1]["as_type"] == "tool"
    assert client.calls[2][1]["level"] == "ERROR"
    assert client.calls[3][1]["as_type"] == "event"


@pytest.mark.asyncio
async def test_langfuse_adapter_flush_shutdown_and_client_errors_are_safe() -> None:
    """Langfuse client failures never escape telemetry sink operations."""
    from ecs_agent.integrations.langfuse import LangfuseTelemetrySink

    client = FakeLangfuseClient(fail=True)
    sink = LangfuseTelemetrySink(client=client)

    await sink.emit(
        TelemetryRecord(
            trace_id="trace-one",
            run_id="run-one",
            observation_id="event-one",
            name="safe.event",
            kind="event",
        )
    )
    await sink.score(
        TelemetryScore(
            trace_id="trace-one",
            run_id="run-one",
            observation_id="event-one",
            name="safe_score",
            value=True,
        )
    )
    await sink.flush()
    await sink.shutdown()

    assert [name for name, _ in client.calls] == [
        "observation",
        "score",
        "flush",
        "shutdown",
    ]


def test_install_delegates_to_generic_observability_with_langfuse_sink() -> None:
    """Langfuse install returns the generic ObservabilityHandle shape."""
    from ecs_agent.integrations.langfuse import (
        LangfuseConfig,
        LangfuseTelemetrySink,
        install_langfuse_observability,
    )
    from ecs_agent.observability import ObservabilityHandle

    world = World()
    sink = LangfuseTelemetrySink(client=FakeLangfuseClient())

    handle = install_langfuse_observability(
        world,
        config=LangfuseConfig(enabled=True),
        sink=sink,
    )

    assert isinstance(handle, ObservabilityHandle)
    assert handle.sink is sink
    assert handle.config.enabled is True
    assert getattr(world, "_ecs_agent_observability_sink") is sink
