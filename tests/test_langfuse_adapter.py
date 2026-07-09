"""Optional Langfuse adapter tests."""

from __future__ import annotations

import builtins
import sys
import types
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


class FakeLangfuseV4Observation:
    """Langfuse v4-like observation object requiring explicit end."""

    def __init__(self, client: FakeLangfuseV4Client, kwargs: dict[str, Any]) -> None:
        self._client = client
        self._kwargs = kwargs

    def __enter__(self) -> FakeLangfuseV4Observation:
        """Record that the observation became the active v4 span."""
        self._client.active_observations.append(self._kwargs)
        self._client.calls.append(("observation_enter", self._kwargs))
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object | None,
    ) -> None:
        """Record that the active v4 span context exited."""
        self._client.calls.append(("observation_exit", self._kwargs))
        self._client.active_observations.pop()

    def end(self, *, end_time: int | None = None) -> None:
        """Record that the v4 observation was ended."""
        self._client.calls.append(("end", {"kwargs": self._kwargs, "end_time": end_time}))

    def update(self, **kwargs: Any) -> None:
        """Record updates applied to an existing v4 observation."""
        self._kwargs.update(kwargs)
        self._client.calls.append(("update", kwargs))


class FakeLangfuseV4Client:
    """Langfuse v4-like client using start_observation instead of observation."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.active_observations: list[dict[str, Any]] = []

    def propagate_attributes(self, **kwargs: Any) -> FakeLangfuseV4Propagation:
        """Record v4 trace-level propagated attributes."""
        self.calls.append(("propagate_attributes", kwargs))
        return FakeLangfuseV4Propagation(self)

    def start_observation(self, **kwargs: Any) -> FakeLangfuseV4Observation:
        """Record a v4 start_observation call."""
        if "trace_id" in kwargs:
            raise TypeError("start_observation got an unexpected keyword argument 'trace_id'")
        if "parent_observation_id" in kwargs:
            raise TypeError(
                "start_observation got an unexpected keyword argument "
                "'parent_observation_id'"
            )
        self.calls.append(("start_observation", kwargs))
        return FakeLangfuseV4Observation(self, kwargs)

    def start_as_current_observation(self, **kwargs: Any) -> FakeLangfuseV4Observation:
        """Record a v4 current observation context-manager call."""
        if "trace_id" in kwargs:
            raise TypeError(
                "start_as_current_observation got an unexpected keyword argument 'trace_id'"
            )
        if "parent_observation_id" in kwargs:
            raise TypeError(
                "start_as_current_observation got an unexpected keyword argument "
                "'parent_observation_id'"
            )
        self.calls.append(("start_as_current_observation", kwargs))
        return FakeLangfuseV4Observation(self, kwargs)

    def create_event(self, **kwargs: Any) -> None:
        """Record a v4 create_event call."""
        if "trace_id" in kwargs:
            raise TypeError("create_event got an unexpected keyword argument 'trace_id'")
        self.calls.append(("create_event", kwargs))


class FakeLangfuseV4AmbientParentClient(FakeLangfuseV4Client):
    """v4-like client that models ambient current-observation parent inference."""

    def start_as_current_observation(self, **kwargs: Any) -> FakeLangfuseV4Observation:
        """Record when a global current start inherits the active observation."""
        trace_context = kwargs.get("trace_context")
        if (
            self.active_observations
            and isinstance(trace_context, dict)
            and "parent_span_id" not in trace_context
        ):
            kwargs["ambient_parent_name"] = self.active_observations[-1].get("name")
        return super().start_as_current_observation(**kwargs)


class FakeLangfuseV4OtelParentClient(FakeLangfuseV4Client):
    """v4-like client that records exported OTel parent relationships."""

    def __init__(self) -> None:
        super().__init__()
        self._next_span_id = 0
        self.exported_spans: list[dict[str, Any]] = []

    def start_observation(self, **kwargs: Any) -> FakeLangfuseV4Observation:
        """Record the parent span that v4 would assign at OTel span creation."""
        observation = super().start_observation(**kwargs)
        span_id = self._span_id()
        trace_context = kwargs.get("trace_context")
        parent_span_id = self._parent_span_id(trace_context)
        trace_id = self._trace_id(trace_context, span_id)
        kwargs["otel_span_id"] = span_id
        kwargs["otel_trace_id"] = trace_id
        observation.id = span_id
        observation.trace_id = trace_id
        self.exported_spans.append(
            {
                "name": kwargs.get("name"),
                "span_id": span_id,
                "trace_id": trace_id,
                "parent_span_id": parent_span_id,
            }
        )
        return observation

    def _span_id(self) -> str:
        self._next_span_id += 1
        return f"{self._next_span_id:016x}"

    def _parent_span_id(self, trace_context: Any) -> str | None:
        if isinstance(trace_context, dict):
            parent_span_id = trace_context.get("parent_span_id")
            if isinstance(parent_span_id, str):
                return parent_span_id
            if isinstance(trace_context.get("trace_id"), str):
                return "synthetic-parent"
        if self.active_observations:
            active_span_id = self.active_observations[-1].get("otel_span_id")
            if isinstance(active_span_id, str):
                return active_span_id
        return None

    def _trace_id(self, trace_context: Any, span_id: str) -> str:
        if isinstance(trace_context, dict):
            trace_id = trace_context.get("trace_id")
            if isinstance(trace_id, str):
                return trace_id
        if self.active_observations:
            trace_id = self.active_observations[-1].get("otel_trace_id")
            if isinstance(trace_id, str):
                return trace_id
        return f"trace-{span_id}"


class FakeLangfuseV4ClientRejectingObservationTimes(FakeLangfuseV4Client):
    """Langfuse v4 client shape that rejects observation start/end kwargs."""

    def start_as_current_observation(self, **kwargs: Any) -> FakeLangfuseV4Observation:
        """Reject deprecated timing kwargs on current observation start."""
        if "start_time" in kwargs:
            raise TypeError(
                "start_as_current_observation got an unexpected keyword argument "
                "'start_time'"
            )
        if "end_time" in kwargs:
            raise TypeError(
                "start_as_current_observation got an unexpected keyword argument "
                "'end_time'"
            )
        return super().start_as_current_observation(**kwargs)


class FakeLangfuseV4MappedIdClient(FakeLangfuseV4Client):
    """Langfuse v4 client that exposes SDK-generated observation IDs."""

    def __init__(self, observation_ids: list[str]) -> None:
        super().__init__()
        self._observation_ids = observation_ids

    def start_observation(self, **kwargs: Any) -> FakeLangfuseV4Observation:
        """Return an observation with the next SDK-generated ID."""
        observation = super().start_observation(**kwargs)
        observation.id = self._observation_ids.pop(0)
        return observation


class FakeLangfuseV4Propagation:
    """Langfuse v4-like propagation context manager."""

    def __init__(self, client: FakeLangfuseV4Client) -> None:
        self._client = client

    def __enter__(self) -> None:
        """Record entering v4 trace attribute propagation."""
        if self._client.active_observations:
            self._client.active_observations[-1]["propagated_attributes"] = dict(
                self._client.calls[-1][1]
            )
        self._client.calls.append(("propagation_enter", {}))

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object | None,
    ) -> None:
        """Record exiting v4 trace attribute propagation."""
        self._client.calls.append(("propagation_exit", {}))


class FakeLangfuseV4ModulePropagation:
    """Module-level Langfuse v4 propagation context manager."""

    def __init__(self, client: FakeLangfuseV4Client, kwargs: dict[str, Any]) -> None:
        self._client = client
        self._kwargs = kwargs

    def __enter__(self) -> None:
        """Record module-level propagation before span creation."""
        self._client.active_propagations.append(self._kwargs)
        self._client.calls.append(("module_propagation_enter", {}))

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object | None,
    ) -> None:
        """Record exiting module-level propagation."""
        self._client.calls.append(("module_propagation_exit", {}))
        self._client.active_propagations.pop()


class FakeLangfuseV4ModuleClient(FakeLangfuseV4Client):
    """Real SDK v4 shape: propagation exists on langfuse module, not client."""

    def __init__(self) -> None:
        super().__init__()
        self.active_propagations: list[dict[str, Any]] = []

    def __getattribute__(self, name: str) -> Any:
        """Mimic real SDK v4 by omitting a client-level propagation API."""
        if name == "propagate_attributes":
            raise AttributeError("Langfuse v4 client has no propagate_attributes method")
        return super().__getattribute__(name)

    def start_as_current_observation(self, **kwargs: Any) -> FakeLangfuseV4Observation:
        """Record whether module-level propagation was active at span creation."""
        if self.active_propagations:
            kwargs["propagated_at_start"] = dict(self.active_propagations[-1])
        return super().start_as_current_observation(**kwargs)


class FakeLangfuseV4TimedClient:
    """Langfuse v4-like client exposing explicit observation start/end timing."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def start_observation(self, **kwargs: Any) -> FakeLangfuseV4Observation:
        """Record non-current observation creation for explicit timing tests."""
        if "start_time" in kwargs:
            raise TypeError("start_observation got an unexpected keyword argument 'start_time'")
        if "end_time" in kwargs:
            raise TypeError("start_observation got an unexpected keyword argument 'end_time'")
        self.calls.append(("start_observation", kwargs))
        return FakeLangfuseV4Observation(self, kwargs)


class FakeLangfuseV4HistoricalSpan:
    """OTel-like span that records attributes set by the adapter."""

    def __init__(self, client: FakeLangfuseV4HistoricalClient, name: str) -> None:
        self._client = client
        self.name = name
        self.attributes: dict[str, Any] = {}

    def set_attribute(self, key: str, value: Any) -> None:
        """Record OTel attributes set on the historical span."""
        self.attributes[key] = value
        self._client.calls.append(("otel_set_attribute", {key: value}))


class FakeLangfuseV4HistoricalTracer:
    """OTel-like tracer that records manual historical start times."""

    def __init__(self, client: FakeLangfuseV4HistoricalClient) -> None:
        self._client = client

    def start_span(
        self,
        *,
        name: str,
        start_time: int | None = None,
    ) -> FakeLangfuseV4HistoricalSpan:
        """Record the span creation request and return a fake OTel span."""
        self._client.calls.append(
            ("otel_start_span", {"name": name, "start_time": start_time})
        )
        return FakeLangfuseV4HistoricalSpan(self._client, name)


class FakeLangfuseV4HistoricalClient:
    """Langfuse v4-like client that can backdate OTel span starts."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.active_observations: list[dict[str, Any]] = []
        self._otel_tracer = FakeLangfuseV4HistoricalTracer(self)

    def start_observation(self, **kwargs: Any) -> FakeLangfuseV4Observation:
        """Expose the normal v4 method; historical path should bypass it."""
        self.calls.append(("start_observation", kwargs))
        return FakeLangfuseV4Observation(self, kwargs)

    def propagate_attributes(self, **kwargs: Any) -> FakeLangfuseV4Propagation:
        """Record trace-level attributes around historical observation creation."""
        self.calls.append(("propagate_attributes", kwargs))
        return FakeLangfuseV4Propagation(self)

    def _create_remote_parent_span(
        self,
        *,
        trace_id: str,
        parent_span_id: str | None = None,
    ) -> str:
        """Record parent trace context creation."""
        self.calls.append(
            (
                "create_remote_parent_span",
                {"trace_id": trace_id, "parent_span_id": parent_span_id},
            )
        )
        return "remote-parent"

    def _create_observation_from_otel_span(self, **kwargs: Any) -> FakeLangfuseV4Observation:
        """Create a fake Langfuse observation from the historical OTel span."""
        self.calls.append(("create_observation_from_otel_span", kwargs))
        return FakeLangfuseV4Observation(self, kwargs)


class FakeLangfuseV4PartialHistoricalClient:
    """Client with partial historical hooks that must use public fallback."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.active_observations: list[dict[str, Any]] = []
        self._otel_tracer = FakeLangfuseV4HistoricalTracer(self)

    def start_observation(self, **kwargs: Any) -> FakeLangfuseV4Observation:
        """Record the safe public fallback path."""
        self.calls.append(("start_observation", kwargs))
        return FakeLangfuseV4Observation(self, kwargs)

    def _create_observation_from_otel_span(self, **kwargs: Any) -> FakeLangfuseV4Observation:
        """Record an unsafe path that should not be used without parent support."""
        self.calls.append(("create_observation_from_otel_span", kwargs))
        return FakeLangfuseV4Observation(self, kwargs)


class FakeLangfuseV4HistoricalModuleClient(FakeLangfuseV4HistoricalClient):
    """Historical client with module-level propagation like real SDK v4.5."""

    def __init__(self) -> None:
        super().__init__()
        self.active_propagations: list[dict[str, Any]] = []

    def __getattribute__(self, name: str) -> Any:
        """Mimic real SDK v4.5 by omitting client-level propagation."""
        if name == "propagate_attributes":
            raise AttributeError("Langfuse v4 client has no propagate_attributes method")
        return super().__getattribute__(name)

    def _create_observation_from_otel_span(self, **kwargs: Any) -> FakeLangfuseV4Observation:
        """Record whether module propagation was active at observation creation."""
        if self.active_propagations:
            kwargs["propagated_at_start"] = dict(self.active_propagations[-1])
        return super()._create_observation_from_otel_span(**kwargs)


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

    import ecs_agent.plugins.langfuse as langfuse_adapter

    assert langfuse_adapter.LangfuseConfig().enabled is True


def test_base_url_fills_host_when_host_env_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LANGFUSE_BASE_URL is a sensible host alias when LANGFUSE_HOST is absent."""
    from ecs_agent.plugins.langfuse import LangfuseConfig

    monkeypatch.delenv("LANGFUSE_HOST", raising=False)
    monkeypatch.setenv("LANGFUSE_BASE_URL", "env-base-url-value")

    config = LangfuseConfig().with_env()

    assert config.host == "env-base-url-value"


def test_langfuse_client_creation_passes_timeout_to_sdk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Self-hosted Langfuse installs can raise the SDK export timeout."""
    from ecs_agent.plugins.langfuse import LangfuseConfig, _create_langfuse_client

    fake_module = types.ModuleType("langfuse")
    fake_module.Langfuse = FakeLangfuseClient
    monkeypatch.setitem(sys.modules, "langfuse", fake_module)

    client = _create_langfuse_client(
        LangfuseConfig(
            public_key="public-one",
            secret_key="secret-one",
            host="https://trace.example.test",
            timeout=30,
        )
    )

    assert client.init_kwargs["timeout"] == 30


@pytest.mark.asyncio
async def test_langfuse_adapter_maps_generation_and_scores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Generations and scores are mapped to Langfuse calls after redaction."""
    from ecs_agent.plugins.langfuse import LangfuseConfig, LangfuseTelemetrySink

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
    assert observation_payload["start_time"] == "2026-01-02T00:00:00+00:00"
    assert observation_payload["end_time"] == "2026-01-02T00:00:01+00:00"
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
    from ecs_agent.plugins.langfuse import LangfuseConfig, LangfuseTelemetrySink

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
    from ecs_agent.plugins.langfuse import LangfuseConfig, LangfuseTelemetrySink

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
    from ecs_agent.plugins.langfuse import LangfuseTelemetrySink

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
async def test_langfuse_trace_record_with_parent_exports_as_child_span() -> None:
    """Parented trace records are root observations, not top-level trace containers."""
    from ecs_agent.plugins.langfuse import LangfuseTelemetrySink

    client = FakeLangfuseClient()
    sink = LangfuseTelemetrySink(client=client)

    await sink.emit(
        TelemetryRecord(
            trace_id="trace-one",
            run_id="run-one",
            observation_id="child-root",
            parent_observation_id="parent-span",
            name="runner.run",
            kind="trace",
            status="success",
        )
    )

    assert [name for name, _ in client.calls] == ["observation"]
    observation_payload = client.calls[0][1]
    assert observation_payload["trace_id"] == "trace-one"
    assert observation_payload["id"] == "child-root"
    assert observation_payload["parent_observation_id"] == "parent-span"
    assert observation_payload["as_type"] == "span"


@pytest.mark.asyncio
async def test_langfuse_user_turn_trace_record_exports_as_root_span_observation() -> None:
    """User turns are root observations inside a trace, not trace containers."""
    from ecs_agent.plugins.langfuse import LangfuseTelemetrySink

    client = FakeLangfuseClient()
    sink = LangfuseTelemetrySink(client=client)

    await sink.emit(
        TelemetryRecord(
            trace_id="trace-one",
            run_id="run-one",
            observation_id="turn-root",
            name="user.turn",
            kind="trace",
            status="running",
        )
    )

    assert [name for name, _ in client.calls] == ["observation"]
    observation_payload = client.calls[0][1]
    assert observation_payload["id"] == "turn-root"
    assert observation_payload["trace_id"] == "trace-one"
    assert "parent_observation_id" not in observation_payload
    assert observation_payload["as_type"] == "span"


@pytest.mark.asyncio
async def test_langfuse_v4_user_turn_final_emit_updates_existing_root_observation() -> None:
    """Finalizing a user turn must not create a nested duplicate root observation."""
    from ecs_agent.plugins.langfuse import LangfuseTelemetrySink

    client = FakeLangfuseV4Client()
    sink = LangfuseTelemetrySink(client=client)
    started_at = datetime(2026, 1, 2, 0, 0, 0, tzinfo=timezone.utc)
    ended_at = datetime(2026, 1, 2, 0, 0, 1, tzinfo=timezone.utc)

    await sink.emit(
        TelemetryRecord(
            trace_id="trace-one",
            run_id="run-one",
            observation_id="turn-root",
            name="user.turn",
            kind="trace",
            status="running",
            start_time=started_at,
            input={"text": "hello"},
        )
    )
    await sink.emit(
        TelemetryRecord(
            trace_id="trace-one",
            run_id="run-one",
            observation_id="turn-root",
            name="user.turn",
            kind="trace",
            status="cancelled",
            start_time=started_at,
            end_time=ended_at,
            latency_ms=1000.0,
            input={"text": "hello"},
            metadata={"reason": "external_cancellation"},
        )
    )

    start_calls = [name for name, _ in client.calls if name.startswith("start_")]
    assert start_calls == ["start_observation"]
    assert any(name == "update" for name, _ in client.calls)
    assert client.calls[-1][0] == "end"


@pytest.mark.asyncio
async def test_langfuse_v4_root_observation_uses_non_current_start_to_avoid_ambient_parent() -> None:
    """Root observations must not inherit a lingering SDK current observation."""
    from ecs_agent.plugins.langfuse import LangfuseTelemetrySink

    client = FakeLangfuseV4AmbientParentClient()
    client.active_observations.append({"name": "workflow.state"})
    sink = LangfuseTelemetrySink(client=client)

    await sink.emit(
        TelemetryRecord(
            trace_id="trace-one",
            run_id="run-one",
            observation_id="turn-root",
            name="user.turn",
            kind="trace",
            status="running",
            input={"text": "continue"},
        )
    )

    assert client.calls[0] == ("propagate_attributes", {"trace_name": "user.turn"})
    start_call = next(payload for name, payload in client.calls if name == "start_observation")
    assert "ambient_parent_name" not in start_call


@pytest.mark.asyncio
async def test_langfuse_v4_user_turn_root_has_no_synthetic_parent() -> None:
    """Root user turns should export as real OTel roots, not synthetic children."""
    from ecs_agent.plugins.langfuse import LangfuseTelemetrySink

    client = FakeLangfuseV4OtelParentClient()
    sink = LangfuseTelemetrySink(client=client)

    await sink.emit(
        TelemetryRecord(
            trace_id="internal-trace-one",
            run_id="run-one",
            observation_id="turn-root",
            name="user.turn",
            kind="trace",
            status="running",
            input={"text": "continue"},
        )
    )
    await sink.emit(
        TelemetryRecord(
            trace_id="internal-trace-one",
            run_id="run-one",
            observation_id="workflow-state",
            parent_observation_id="turn-root",
            name="workflow.state",
            kind="event",
            status="success",
        )
    )

    user_turn_span = next(
        span for span in client.exported_spans if span["name"] == "user.turn"
    )
    assert user_turn_span["parent_span_id"] is None
    workflow_call = next(payload for name, payload in client.calls if name == "create_event")
    assert workflow_call["trace_context"] == {
        "trace_id": user_turn_span["trace_id"],
        "parent_span_id": user_turn_span["span_id"],
    }


@pytest.mark.asyncio
async def test_langfuse_v4_real_sdk_user_turn_root_ignores_ambient_workflow_span() -> None:
    """Real Langfuse v4 SDK exports user.turn as root even with ambient workflow span."""
    langfuse_module = pytest.importorskip("langfuse")
    trace_module = pytest.importorskip("opentelemetry.sdk.trace")
    export_module = pytest.importorskip("opentelemetry.sdk.trace.export")
    try:
        memory_export_module = pytest.importorskip(
            "opentelemetry.sdk.trace.export.in_memory_span_exporter"
        )
        exporter_class = memory_export_module.InMemorySpanExporter
    except pytest.skip.Exception:
        exporter_class = export_module.InMemorySpanExporter
    from ecs_agent.plugins.langfuse import LangfuseTelemetrySink

    exporter = exporter_class()
    provider = trace_module.TracerProvider()
    provider.add_span_processor(export_module.SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("ecs-agent-langfuse-test")
    client = langfuse_module.Langfuse(
        public_key="pk-lf-00000000000000000000000000000000",
        secret_key="sk-lf-00000000000000000000000000000000",
        host="http://127.0.0.1:9",
        flush_at=1000,
    )
    client._otel_tracer = tracer
    sink = LangfuseTelemetrySink(client=client)
    started_at = datetime(2026, 1, 2, 0, 0, 0, tzinfo=timezone.utc)
    ended_at = datetime(2026, 1, 2, 0, 0, 1, tzinfo=timezone.utc)

    with client.start_as_current_observation(
        name="workflow.state",
        as_type="span",
    ):
        await sink.emit(
            TelemetryRecord(
                trace_id="internal-trace-one",
                run_id="run-one",
                observation_id="turn-root",
                name="user.turn",
                kind="trace",
                status="running",
                start_time=started_at,
            )
        )
        await sink.emit(
            TelemetryRecord(
                trace_id="internal-trace-one",
                run_id="run-one",
                observation_id="workflow-child",
                parent_observation_id="turn-root",
                name="workflow.state",
                kind="event",
                status="success",
            )
        )
        await sink.emit(
            TelemetryRecord(
                trace_id="internal-trace-one",
                run_id="run-one",
                observation_id="turn-root",
                name="user.turn",
                kind="trace",
                status="success",
                start_time=started_at,
                end_time=ended_at,
            )
        )

    spans_by_name = {span.name: span for span in exporter.get_finished_spans()}
    user_turn = spans_by_name["user.turn"]
    workflow_child = next(
        span
        for span in exporter.get_finished_spans()
        if span.name == "workflow.state" and span.parent is not None
    )
    assert user_turn.parent is None
    assert user_turn.attributes["langfuse.trace.name"] == "user.turn"
    assert workflow_child.parent is not None
    assert workflow_child.parent.span_id == user_turn.get_span_context().span_id


@pytest.mark.asyncio
async def test_langfuse_v4_trace_context_shortens_internal_uuid_parent_ids() -> None:
    """v4 trace_context parent_span_id must use Langfuse's 16-hex span ID shape."""
    from ecs_agent.plugins.langfuse import LangfuseTelemetrySink

    client = FakeLangfuseV4Client()
    sink = LangfuseTelemetrySink(client=client)
    parent_observation_id = "215222198bf24da4a4effcf01c89521a"

    await sink.emit(
        TelemetryRecord(
            trace_id="trace-one",
            run_id="run-one",
            observation_id="span-one",
            parent_observation_id=parent_observation_id,
            name="runner.tick",
            kind="span",
            status="success",
            start_time=datetime(2026, 1, 2, tzinfo=timezone.utc),
            end_time=datetime(2026, 1, 2, 0, 0, 1, tzinfo=timezone.utc),
        )
    )

    observation_payload = client.calls[0][1]
    assert observation_payload["trace_context"] == {
        "trace_id": "trace-one",
        "parent_span_id": parent_observation_id[:16],
    }


@pytest.mark.asyncio
async def test_langfuse_v4_historical_remote_parent_shortens_internal_uuid_parent_ids() -> None:
    """Historical remote parent creation must not pass 32-hex observation IDs as span IDs."""
    from ecs_agent.plugins.langfuse import LangfuseConfig, LangfuseTelemetrySink

    client = FakeLangfuseV4HistoricalClient()
    sink = LangfuseTelemetrySink(
        client=client,
        config=LangfuseConfig(enable_private_v4_historical_otel=True),
    )
    parent_observation_id = "215222198bf24da4a4effcf01c89521a"

    await sink.emit(
        TelemetryRecord(
            trace_id="trace-one",
            run_id="run-one",
            observation_id="generation-one",
            parent_observation_id=parent_observation_id,
            name="llm.reasoning",
            kind="generation",
            status="success",
            start_time=datetime(2026, 1, 2, tzinfo=timezone.utc),
            end_time=datetime(2026, 1, 2, 0, 0, 1, tzinfo=timezone.utc),
        )
    )

    assert client.calls[0] == (
        "create_remote_parent_span",
        {"trace_id": "trace-one", "parent_span_id": parent_observation_id[:16]},
    )


@pytest.mark.asyncio
async def test_langfuse_v4_tool_parent_uses_generation_sdk_observation_id() -> None:
    """Tool spans should parent to the actual v4 generation observation ID."""
    from ecs_agent.plugins.langfuse import LangfuseTelemetrySink

    generation_observation_id = "215222198bf24da4a4effcf01c89521a"
    root_observation_id = "3bd19b7975c845a2920fcd8b68b95a62"
    generation_sdk_id = "aaaaaaaaaaaaaaaa"
    client = FakeLangfuseV4MappedIdClient(
        observation_ids=[generation_sdk_id, "bbbbbbbbbbbbbbbb"]
    )
    sink = LangfuseTelemetrySink(client=client)

    await sink.emit(
        TelemetryRecord(
            trace_id="trace-one",
            run_id="run-one",
            observation_id=generation_observation_id,
            parent_observation_id=root_observation_id,
            name="llm.reasoning",
            kind="generation",
            status="success",
            start_time=datetime(2026, 1, 2, tzinfo=timezone.utc),
            end_time=datetime(2026, 1, 2, 0, 0, 1, tzinfo=timezone.utc),
        )
    )
    await sink.emit(
        TelemetryRecord(
            trace_id="trace-one",
            run_id="run-one",
            observation_id="tool-observation",
            parent_observation_id=generation_observation_id,
            name="tool.lookup",
            kind="tool",
            status="success",
            start_time=datetime(2026, 1, 2, 0, 0, 2, tzinfo=timezone.utc),
            end_time=datetime(2026, 1, 2, 0, 0, 3, tzinfo=timezone.utc),
        )
    )

    tool_payload = client.calls[2][1]
    assert tool_payload["trace_context"] == {
        "trace_id": "trace-one",
        "parent_span_id": generation_sdk_id,
    }


@pytest.mark.asyncio
async def test_langfuse_adapter_uses_v4_trace_context_methods() -> None:
    """Langfuse v4 manual observations receive trace_context, not trace_id kwargs."""
    from ecs_agent.plugins.langfuse import LangfuseTelemetrySink

    client = FakeLangfuseV4Client()
    sink = LangfuseTelemetrySink(client=client)

    await sink.emit(
        TelemetryRecord(
            trace_id="trace-one",
            run_id="run-one",
            observation_id="span-one",
            parent_observation_id="trace-root",
            name="runner.tick",
            kind="span",
            status="success",
            start_time=datetime(2026, 1, 2, tzinfo=timezone.utc),
            end_time=datetime(2026, 1, 2, 0, 0, 1, tzinfo=timezone.utc),
        )
    )
    await sink.emit(
        TelemetryRecord(
            trace_id="trace-one",
            run_id="run-one",
            observation_id="event-one",
            parent_observation_id="span-one",
            name="stream.end",
            kind="event",
            status="success",
        )
    )

    assert client.calls[0] == (
        "start_observation",
        {
            "trace_context": {
                "trace_id": "trace-one",
                "parent_span_id": "trace-root",
            },
            "name": "runner.tick",
            "as_type": "span",
            "metadata": {
                "redaction": {"total_redactions": 0, "counts_by_rule": {}},
                "observation_id": "span-one",
            },
            "level": "DEFAULT",
        },
    )
    assert client.calls[1] == (
        "end",
        {
            "kwargs": client.calls[0][1],
            "end_time": int(datetime(2026, 1, 2, 0, 0, 1, tzinfo=timezone.utc).timestamp() * 1_000_000_000),
        },
    )
    assert client.calls[2][0] == "create_event"
    event_payload = client.calls[2][1]
    assert event_payload["trace_context"] == {
        "trace_id": "trace-one",
        "parent_span_id": "span-one",
    }
    assert event_payload["name"] == "stream.end"
    assert event_payload["metadata"]["observation_id"] == "event-one"


@pytest.mark.asyncio
async def test_langfuse_v4_current_observation_does_not_receive_timing_kwargs() -> None:
    """v4 observation creation in newer SDKs rejects start_time/end_time kwargs."""
    from ecs_agent.plugins.langfuse import LangfuseTelemetrySink

    client = FakeLangfuseV4ClientRejectingObservationTimes()
    sink = LangfuseTelemetrySink(client=client)

    await sink.emit(
        TelemetryRecord(
            trace_id="trace-one",
            run_id="run-one",
            observation_id="span-one",
            parent_observation_id="trace-root",
            name="runner.tick",
            kind="span",
            status="success",
            start_time=datetime(2026, 1, 2, tzinfo=timezone.utc),
            end_time=datetime(2026, 1, 2, 0, 0, 1, tzinfo=timezone.utc),
        )
    )

    assert client.calls[0][0] == "start_observation"
    observation_payload = client.calls[0][1]
    assert "start_time" not in observation_payload
    assert "end_time" not in observation_payload
    assert client.calls[1][0] == "end"


@pytest.mark.asyncio
async def test_langfuse_v4_observation_uses_explicit_record_timing() -> None:
    """v4 exports should preserve record timing instead of export-context timing."""
    from ecs_agent.plugins.langfuse import LangfuseTelemetrySink

    client = FakeLangfuseV4TimedClient()
    sink = LangfuseTelemetrySink(client=client)
    started_at = datetime(2026, 1, 2, 0, 0, 0, 123456, tzinfo=timezone.utc)
    ended_at = datetime(2026, 1, 2, 0, 0, 1, 654321, tzinfo=timezone.utc)

    await sink.emit(
        TelemetryRecord(
            trace_id="trace-one",
            run_id="run-one",
            observation_id="generation-one",
            parent_observation_id="trace-root",
            name="llm.reasoning",
            kind="generation",
            status="success",
            start_time=started_at,
            end_time=ended_at,
            input={"messages": []},
            output={"message": "ok"},
            model="model-one",
            usage_details={"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
        )
    )

    assert client.calls[0][0] == "start_observation"
    observation_payload = client.calls[0][1]
    assert observation_payload["trace_context"] == {
        "trace_id": "trace-one",
        "parent_span_id": "trace-root",
    }
    assert observation_payload["as_type"] == "generation"
    assert observation_payload["model"] == "model-one"
    assert observation_payload["usage_details"] == {
        "prompt_tokens": 1,
        "completion_tokens": 2,
        "total_tokens": 3,
    }
    assert client.calls[1] == (
        "end",
        {
            "kwargs": observation_payload,
            "end_time": int(ended_at.timestamp() * 1_000_000_000),
        },
    )


@pytest.mark.asyncio
async def test_langfuse_v4_timed_generation_uses_manual_lifecycle_when_current_api_exists() -> None:
    """Completed records use manual v4 lifecycle so Langfuse receives real timing."""
    from ecs_agent.plugins.langfuse import LangfuseTelemetrySink

    client = FakeLangfuseV4Client()
    sink = LangfuseTelemetrySink(client=client)
    started_at = datetime(2026, 1, 2, 0, 0, 0, tzinfo=timezone.utc)
    ended_at = datetime(2026, 1, 2, 0, 0, 2, 500000, tzinfo=timezone.utc)

    await sink.emit(
        TelemetryRecord(
            trace_id="trace-one",
            run_id="run-one",
            observation_id="generation-one",
            parent_observation_id="trace-root",
            name="llm.reasoning",
            kind="generation",
            status="success",
            start_time=started_at,
            end_time=ended_at,
            input={"messages": []},
            output={"message": "ok"},
            model="deepseek-v4-flash",
            model_parameters={"temperature": 0},
        )
    )

    assert [name for name, _ in client.calls] == ["start_observation", "end"]
    observation_payload = client.calls[0][1]
    assert observation_payload["as_type"] == "generation"
    assert observation_payload["model"] == "deepseek-v4-flash"
    assert observation_payload["model_parameters"] == {"temperature": 0}
    assert client.calls[1][1]["end_time"] == int(ended_at.timestamp() * 1_000_000_000)


@pytest.mark.asyncio
async def test_langfuse_v4_private_historical_otel_is_opt_in() -> None:
    """Private v4 OTel hooks are not used unless explicitly enabled."""
    from ecs_agent.plugins.langfuse import LangfuseTelemetrySink

    client = FakeLangfuseV4HistoricalClient()
    sink = LangfuseTelemetrySink(client=client)
    started_at = datetime(2026, 1, 2, 0, 0, 0, tzinfo=timezone.utc)
    ended_at = datetime(2026, 1, 2, 0, 0, 1, tzinfo=timezone.utc)

    await sink.emit(
        TelemetryRecord(
            trace_id="trace-one",
            run_id="run-one",
            observation_id="generation-one",
            parent_observation_id="trace-root",
            name="llm.reasoning",
            kind="generation",
            status="success",
            start_time=started_at,
            end_time=ended_at,
        )
    )

    assert [name for name, _ in client.calls] == ["start_observation", "end"]


@pytest.mark.asyncio
async def test_langfuse_v4_historical_observation_backdates_start_and_end() -> None:
    """v4 OTel-backed observations preserve elapsed record latency in Langfuse."""
    from ecs_agent.plugins.langfuse import LangfuseConfig, LangfuseTelemetrySink

    client = FakeLangfuseV4HistoricalClient()
    sink = LangfuseTelemetrySink(
        client=client,
        config=LangfuseConfig(enable_private_v4_historical_otel=True),
    )
    started_at = datetime(2026, 1, 2, 0, 0, 0, 123456, tzinfo=timezone.utc)
    ended_at = datetime(2026, 1, 2, 0, 0, 2, 654321, tzinfo=timezone.utc)

    await sink.emit(
        TelemetryRecord(
            trace_id="trace-one",
            run_id="run-one",
            observation_id="generation-one",
            parent_observation_id="trace-root",
            name="llm.reasoning",
            kind="generation",
            status="success",
            start_time=started_at,
            end_time=ended_at,
            input={"messages": []},
            output={"message": "ok"},
            model="deepseek-v4-flash",
            model_parameters={"temperature": 0},
            usage_details={"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
        )
    )

    assert client.calls[0] == (
        "create_remote_parent_span",
        {"trace_id": "trace-one", "parent_span_id": "trace-root"},
    )
    assert client.calls[1] == (
        "otel_start_span",
        {
            "name": "llm.reasoning",
            "start_time": int(started_at.timestamp() * 1_000_000_000),
        },
    )
    assert client.calls[2][0] == "create_observation_from_otel_span"
    observation_payload = client.calls[2][1]
    assert observation_payload["as_type"] == "generation"
    assert observation_payload["model"] == "deepseek-v4-flash"
    assert observation_payload["usage_details"] == {
        "prompt_tokens": 1,
        "completion_tokens": 2,
        "total_tokens": 3,
    }
    assert client.calls[3] == (
        "end",
        {
            "kwargs": observation_payload,
            "end_time": int(ended_at.timestamp() * 1_000_000_000),
        },
    )


@pytest.mark.asyncio
async def test_langfuse_v4_partial_historical_client_uses_public_fallback() -> None:
    """Partial private SDK hooks must not create orphaned historical spans."""
    from ecs_agent.plugins.langfuse import LangfuseTelemetrySink

    client = FakeLangfuseV4PartialHistoricalClient()
    sink = LangfuseTelemetrySink(client=client)
    started_at = datetime(2026, 1, 2, 0, 0, 0, tzinfo=timezone.utc)
    ended_at = datetime(2026, 1, 2, 0, 0, 1, tzinfo=timezone.utc)

    await sink.emit(
        TelemetryRecord(
            trace_id="trace-one",
            run_id="run-one",
            observation_id="generation-one",
            parent_observation_id="trace-root",
            name="llm.reasoning",
            kind="generation",
            status="success",
            start_time=started_at,
            end_time=ended_at,
        )
    )

    assert [name for name, _ in client.calls] == ["start_observation", "end"]
    observation_payload = client.calls[0][1]
    assert observation_payload["trace_context"] == {
        "trace_id": "trace-one",
        "parent_span_id": "trace-root",
    }


@pytest.mark.asyncio
async def test_langfuse_v4_historical_root_observation_propagates_session() -> None:
    """Historical root observations keep Langfuse Session attributes."""
    from ecs_agent.plugins.langfuse import LangfuseConfig, LangfuseTelemetrySink

    client = FakeLangfuseV4HistoricalClient()
    sink = LangfuseTelemetrySink(
        client=client,
        config=LangfuseConfig(
            session_id="session-one",
            enable_private_v4_historical_otel=True,
        ),
    )
    started_at = datetime(2026, 1, 2, 0, 0, 0, tzinfo=timezone.utc)
    ended_at = datetime(2026, 1, 2, 0, 0, 1, tzinfo=timezone.utc)

    await sink.emit(
        TelemetryRecord(
            trace_id="trace-one",
            run_id="run-one",
            observation_id="turn-root",
            name="user.turn",
            kind="trace",
            status="success",
            start_time=started_at,
            end_time=ended_at,
        )
    )

    assert client.calls[0] == (
        "propagate_attributes",
        {"session_id": "session-one", "trace_name": "user.turn"},
    )
    assert client.calls[1] == ("propagation_enter", {})
    observation_payload = next(
        payload for name, payload in client.calls if name == "create_observation_from_otel_span"
    )
    assert observation_payload["metadata"]["session_id"] == "session-one"
    assert ("propagation_exit", {}) in client.calls
    assert client.calls[-1][0] == "end"


@pytest.mark.asyncio
async def test_langfuse_v4_historical_module_propagation_is_active_at_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Historical observations use module-level propagation for SDK v4.5."""
    from ecs_agent.plugins.langfuse import LangfuseConfig, LangfuseTelemetrySink

    client = FakeLangfuseV4HistoricalModuleClient()

    def propagate_attributes(**kwargs: Any) -> FakeLangfuseV4ModulePropagation:
        client.calls.append(("module_propagate_attributes", kwargs))
        return FakeLangfuseV4ModulePropagation(client, kwargs)

    fake_module = types.ModuleType("langfuse")
    fake_module.propagate_attributes = propagate_attributes
    monkeypatch.setitem(sys.modules, "langfuse", fake_module)
    sink = LangfuseTelemetrySink(
        client=client,
        config=LangfuseConfig(
            session_id="session-one",
            enable_private_v4_historical_otel=True,
        ),
    )
    started_at = datetime(2026, 1, 2, 0, 0, 0, tzinfo=timezone.utc)
    ended_at = datetime(2026, 1, 2, 0, 0, 1, tzinfo=timezone.utc)

    await sink.emit(
        TelemetryRecord(
            trace_id="trace-one",
            run_id="run-one",
            observation_id="turn-root",
            name="user.turn",
            kind="trace",
            status="success",
            start_time=started_at,
            end_time=ended_at,
        )
    )

    assert client.calls[0] == (
        "module_propagate_attributes",
        {"session_id": "session-one", "trace_name": "user.turn"},
    )
    assert client.calls[1] == ("module_propagation_enter", {})
    observation_payload = next(
        payload for name, payload in client.calls if name == "create_observation_from_otel_span"
    )
    assert observation_payload["propagated_at_start"] == {
        "session_id": "session-one",
        "trace_name": "user.turn",
    }
    assert observation_payload["metadata"]["session_id"] == "session-one"


@pytest.mark.asyncio
async def test_langfuse_adapter_preserves_env_configured_model_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LLM_MODEL is configuration, not a secret, so generation.model stays readable."""
    from ecs_agent.plugins.langfuse import LangfuseTelemetrySink

    monkeypatch.setenv("LLM_MODEL", "deepseek-v4-flash")
    client = FakeLangfuseClient()
    sink = LangfuseTelemetrySink(client=client, redactor=SecretRedactor())

    await sink.emit(
        TelemetryRecord(
            trace_id="trace-one",
            run_id="run-one",
            observation_id="generation-one",
            parent_observation_id="trace-root",
            name="llm.reasoning",
            kind="generation",
            status="success",
            model="deepseek-v4-flash",
            metadata={"model_id": "deepseek-v4-flash"},
        )
    )

    observation_payload = client.calls[0][1]
    assert observation_payload["model"] == "deepseek-v4-flash"
    assert observation_payload["metadata"]["model_id"] == "deepseek-v4-flash"
    assert "[REDACTED:value:LLM_MODEL]" not in str(observation_payload)


@pytest.mark.asyncio
async def test_langfuse_adapter_propagates_v4_session_id_as_trace_attribute() -> None:
    """Langfuse v4 sessions use propagate_attributes, not metadata.session_id."""
    from ecs_agent.plugins.langfuse import LangfuseConfig, LangfuseTelemetrySink

    client = FakeLangfuseV4Client()
    sink = LangfuseTelemetrySink(
        client=client,
        config=LangfuseConfig(
            environment="test",
            release="release-one",
            user_id="user-one",
            session_id="session-one",
            tags=["unit"],
        ),
    )

    await sink.emit(
        TelemetryRecord(
            trace_id="trace-one",
            run_id="run-one",
            observation_id="trace-root",
            name="runner.run",
            kind="trace",
            status="success",
        )
    )

    assert client.calls[0][0] == "start_as_current_observation"
    assert client.calls[0][1]["trace_context"] == {"trace_id": "trace-one"}
    assert client.calls[1] == ("observation_enter", client.calls[0][1])
    assert client.calls[2] == (
        "propagate_attributes",
        {
            "user_id": "user-one",
            "session_id": "session-one",
            "tags": ["unit"],
        },
    )
    assert client.calls[3] == ("propagation_enter", {})
    assert client.calls[0][1]["propagated_attributes"] == {
        "user_id": "user-one",
        "session_id": "session-one",
        "tags": ["unit"],
    }
    assert client.calls[4] == ("propagation_exit", {})
    assert client.calls[5] == ("observation_exit", client.calls[0][1])


@pytest.mark.asyncio
async def test_langfuse_adapter_sets_session_on_active_v4_root_observation() -> None:
    """Sessions require propagation while the root observation is current."""
    from ecs_agent.plugins.langfuse import LangfuseConfig, LangfuseTelemetrySink

    client = FakeLangfuseV4Client()
    sink = LangfuseTelemetrySink(
        client=client,
        config=LangfuseConfig(session_id="session-one"),
    )

    await sink.emit(
        TelemetryRecord(
            trace_id="trace-one",
            run_id="run-one",
            observation_id="trace-root",
            name="runner.run",
            kind="trace",
            status="success",
        )
    )

    assert [name for name, _ in client.calls[:5]] == [
        "start_as_current_observation",
        "observation_enter",
        "propagate_attributes",
        "propagation_enter",
        "propagation_exit",
    ]
    root_payload = client.calls[0][1]
    assert root_payload["trace_context"] == {"trace_id": "trace-one"}
    assert root_payload["propagated_attributes"] == {"session_id": "session-one"}
    assert root_payload["metadata"]["session_id"] == "session-one"


@pytest.mark.asyncio
async def test_langfuse_adapter_uses_module_level_v4_session_propagation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Langfuse SDK v4.5 exposes propagate_attributes on the module."""
    from ecs_agent.plugins.langfuse import LangfuseConfig, LangfuseTelemetrySink

    client = FakeLangfuseV4ModuleClient()

    def propagate_attributes(**kwargs: Any) -> FakeLangfuseV4ModulePropagation:
        client.calls.append(("module_propagate_attributes", kwargs))
        return FakeLangfuseV4ModulePropagation(client, kwargs)

    fake_module = types.ModuleType("langfuse")
    fake_module.propagate_attributes = propagate_attributes
    monkeypatch.setitem(sys.modules, "langfuse", fake_module)
    sink = LangfuseTelemetrySink(
        client=client,
        config=LangfuseConfig(session_id="session-one"),
    )

    await sink.emit(
        TelemetryRecord(
            trace_id="trace-one",
            run_id="run-one",
            observation_id="trace-root",
            name="runner.run",
            kind="trace",
            status="success",
        )
    )

    assert client.calls[0] == (
        "module_propagate_attributes",
        {"session_id": "session-one"},
    )
    assert client.calls[1] == ("module_propagation_enter", {})
    root_payload = client.calls[2][1]
    assert root_payload["trace_context"] == {"trace_id": "trace-one"}
    assert root_payload["propagated_at_start"] == {"session_id": "session-one"}
    assert root_payload["metadata"]["session_id"] == "session-one"


@pytest.mark.asyncio
async def test_langfuse_adapter_flush_shutdown_and_client_errors_are_safe() -> None:
    """Langfuse client failures never escape telemetry sink operations."""
    from ecs_agent.plugins.langfuse import LangfuseTelemetrySink

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
    assert sink.failure_count == 4
    assert sink.failures_by_operation == {
        "emit": 1,
        "score": 1,
        "flush": 1,
        "shutdown": 1,
    }
    assert sink.last_error == "shutdown failed"


@pytest.mark.asyncio
async def test_langfuse_adapter_can_suppress_raw_inputs_and_outputs() -> None:
    """Export privacy controls omit raw payload fields without dropping metadata."""
    from ecs_agent.plugins.langfuse import LangfuseConfig, LangfuseTelemetrySink

    private_value = "private-redaction-value"
    client = FakeLangfuseClient()
    sink = LangfuseTelemetrySink(
        client=client,
        config=LangfuseConfig(capture_input=False, capture_output=False),
        redactor=SecretRedactor(extra_secret_values=[private_value]),
    )

    await sink.emit(
        TelemetryRecord(
            trace_id="trace-one",
            run_id="run-one",
            observation_id="generation-one",
            parent_observation_id="trace-root",
            name="llm.reasoning",
            kind="generation",
            input={"prompt": private_value},
            output={"response": private_value},
            metadata={"safe": "value"},
            model="model-one",
        )
    )

    observation_payload = client.calls[0][1]
    assert "input" not in observation_payload
    assert "output" not in observation_payload
    assert observation_payload["metadata"]["safe"] == "value"
    assert observation_payload["metadata"]["redaction"] == {
        "total_redactions": 2,
        "counts_by_rule": {"value:extra_secret": 2},
    }
    assert private_value not in str(client.calls)


@pytest.mark.asyncio
async def test_plugin_install_mounts_langfuse_sink_on_record_pipeline() -> None:
    """Plugin install mounts the Langfuse sink behind the world's composite."""
    from ecs_agent.plugins import CompositeTelemetrySink, install_plugins
    from ecs_agent.plugins.langfuse import (
        LangfuseConfig,
        LangfusePlugin,
        LangfuseTelemetrySink,
    )

    world = World()
    sink = LangfuseTelemetrySink(client=FakeLangfuseClient())

    handle = await install_plugins(
        world,
        [LangfusePlugin(LangfuseConfig(enabled=True), sink=sink)],
    )

    plugin = handle.plugin("langfuse")
    assert plugin is not None
    assert plugin.telemetry_sink() is sink
    composite = getattr(world, "_ecs_agent_observability_sink")
    assert isinstance(composite, CompositeTelemetrySink)
    assert dict(composite.sinks()) == {"langfuse": sink}
