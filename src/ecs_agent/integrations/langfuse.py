"""Optional Langfuse telemetry adapter."""

from __future__ import annotations

import inspect
import importlib
import os
from contextlib import nullcontext
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Any

from ecs_agent.logging import get_logger
from ecs_agent.observability.install import ObservabilityHandle, install_observability
from ecs_agent.observability.redaction import SecretRedactor, sanitize_payload
from ecs_agent.observability.schema import JsonSafe, TelemetryRecord, TelemetryScore
from ecs_agent.observability.sinks import NoOpTelemetrySink, TelemetrySink

logger = get_logger(__name__)

LANGFUSE_IMPORT_ERROR = "Install ecs-agent[langfuse] to use Langfuse observability"
_LANGFUSE_SPAN_ID_HEX_LENGTH = 16
_INTERNAL_OBSERVATION_ID_HEX_LENGTH = 32


@dataclass(slots=True)
class LangfuseConfig:
    """Configuration for optional Langfuse telemetry export."""

    public_key: str | None = None
    secret_key: str | None = None
    host: str | None = None
    environment: str | None = None
    release: str | None = None
    user_id: str | None = None
    session_id: str | None = None
    tags: list[str] | None = None
    metadata: dict[str, Any] | None = None
    flush_at: int | None = None
    flush_interval: float | None = None
    enabled: bool = True
    capture_input: bool = True
    capture_output: bool = True
    enable_private_v4_historical_otel: bool = False
    timeout: float | None = None

    def with_env(self, env: dict[str, str] | None = None) -> LangfuseConfig:
        """Return a copy with missing SDK credential fields loaded from env names."""
        source = os.environ if env is None else env
        host = self.host
        if host is None:
            host = source.get("LANGFUSE_HOST") or source.get("LANGFUSE_BASE_URL")
        return replace(
            self,
            public_key=self.public_key or source.get("LANGFUSE_PUBLIC_KEY"),
            secret_key=self.secret_key or source.get("LANGFUSE_SECRET_KEY"),
            host=host,
        )


class LangfuseTelemetrySink:
    """Telemetry sink that exports sanitized records to a Langfuse client."""

    def __init__(
        self,
        *,
        client: Any,
        config: LangfuseConfig | None = None,
        redactor: SecretRedactor | None = None,
    ) -> None:
        self.client = client
        self.config = LangfuseConfig() if config is None else config
        self.redactor = SecretRedactor() if redactor is None else redactor
        self._v4_observation_ids: dict[str, str] = {}
        self._v4_trace_ids: dict[str, str] = {}
        self._v4_trace_names: dict[str, str] = {}
        self._active_v4_root_observations: dict[str, Any] = {}
        self.failure_count = 0
        self.failures_by_operation: dict[str, int] = {}
        self.last_error: str | None = None

    async def emit(self, record: TelemetryRecord) -> None:
        """Emit one sanitized telemetry record to Langfuse."""
        if not self.config.enabled:
            return
        try:
            payload, redaction = sanitize_payload(record.to_payload(), self.redactor)
            record_payload = _apply_capture_policy(_as_dict(payload), self.config)
            redaction_payload = redaction.to_payload()
            if _is_trace_container_payload(record_payload):
                await self._emit_trace(record_payload, redaction_payload)
                return
            await self._emit_observation(record_payload, redaction_payload)
        except Exception as exc:
            self._record_failure("emit", exc)
            logger.error("langfuse_emit_failed", exception=str(exc))

    async def score(self, score: TelemetryScore) -> None:
        """Emit one sanitized score to Langfuse."""
        if not self.config.enabled:
            return
        try:
            payload, redaction = sanitize_payload(score.to_payload(), self.redactor)
            score_payload = _as_dict(payload)
            metadata = _metadata_with_context(
                self.config,
                score_payload.get("metadata"),
                redaction.to_payload(),
                _config_export_context(self.config, self.redactor),
            )
            await self._call_client(
                ("create_score", "score"),
                trace_id=score_payload.get("trace_id"),
                observation_id=score_payload.get("observation_id"),
                name=score_payload.get("name"),
                value=score_payload.get("value"),
                comment=score_payload.get("comment"),
                metadata=metadata,
            )
        except Exception as exc:
            self._record_failure("score", exc)
            logger.error("langfuse_score_failed", exception=str(exc))

    async def flush(self) -> None:
        """Flush the Langfuse client if supported."""
        try:
            await self._call_optional_client("flush")
        except Exception as exc:
            self._record_failure("flush", exc)
            logger.error("langfuse_flush_failed", exception=str(exc))

    async def shutdown(self) -> None:
        """Shutdown the Langfuse client if supported."""
        try:
            await self._call_optional_client("shutdown")
        except Exception as exc:
            self._record_failure("shutdown", exc)
            logger.error("langfuse_shutdown_failed", exception=str(exc))

    def _record_failure(self, operation: str, exc: Exception) -> None:
        """Record sink-local failure health without interrupting agent runs."""
        self.failure_count += 1
        self.failures_by_operation[operation] = (
            self.failures_by_operation.get(operation, 0) + 1
        )
        sanitized, _ = sanitize_payload(str(exc), self.redactor)
        self.last_error = sanitized if isinstance(sanitized, str) else repr(sanitized)

    async def _emit_trace(
        self,
        payload: dict[str, JsonSafe],
        redaction: dict[str, JsonSafe],
    ) -> None:
        config_context = _config_export_context(self.config, self.redactor)
        metadata = _metadata_with_context(
            self.config,
            payload.get("metadata"),
            redaction,
            config_context,
        )
        if _first_method(self.client, ("trace", "create_trace")) is None:
            await self._emit_v4_observation(payload, metadata, as_type="span")
            return
        await self._call_client(
            ("trace", "create_trace"),
            id=payload.get("trace_id"),
            name=payload.get("name"),
            input=payload.get("input"),
            output=payload.get("output"),
            metadata=metadata,
            user_id=config_context.metadata.get("user_id"),
            session_id=config_context.metadata.get("session_id"),
            tags=config_context.metadata.get("tags"),
            timestamp=payload.get("start_time"),
            release=config_context.metadata.get("release"),
            environment=config_context.metadata.get("environment"),
        )

    async def _emit_observation(
        self,
        payload: dict[str, JsonSafe],
        redaction: dict[str, JsonSafe],
    ) -> None:
        metadata = _metadata_with_context(
            self.config,
            payload.get("metadata"),
            redaction,
            _config_export_context(self.config, self.redactor),
        )
        if _first_method(self.client, ("observation", "create_observation")) is None:
            if payload.get("kind") == "event":
                await self._emit_v4_event(payload, metadata)
                return
            await self._emit_v4_observation(
                payload,
                metadata,
                as_type=_langfuse_observation_type(payload.get("kind")),
            )
            return
        await self._call_client(
            ("observation", "create_observation"),
            trace_id=payload.get("trace_id"),
            id=payload.get("observation_id"),
            parent_observation_id=payload.get("parent_observation_id"),
            name=payload.get("name"),
            as_type=_langfuse_observation_type(payload.get("kind")),
            input=payload.get("input"),
            output=payload.get("output"),
            metadata=metadata,
            level=_langfuse_level(payload.get("status"), payload.get("error")),
            status_message=payload.get("error"),
            model=payload.get("model"),
            model_parameters=payload.get("model_parameters"),
            usage_details=_langfuse_usage_details(payload.get("usage_details")),
            cost_details=_langfuse_cost_details(payload.get("cost_details")),
            start_time=payload.get("start_time"),
            end_time=payload.get("end_time"),
        )

    async def _call_client(self, method_names: tuple[str, ...], **kwargs: Any) -> None:
        method = _first_method(self.client, method_names)
        if method is None:
            raise AttributeError(f"Langfuse client has none of {method_names!r}")
        result = method(**_drop_none(kwargs))
        if inspect.isawaitable(result):
            await result

    async def _emit_v4_observation(
        self,
        payload: dict[str, JsonSafe],
        metadata: dict[str, JsonSafe],
        *,
        as_type: str,
    ) -> None:
        self._remember_v4_trace_name(payload)
        if await self._finalize_active_v4_root_observation(
            payload,
            metadata,
            as_type=as_type,
        ):
            return
        non_current_method = getattr(self.client, "start_observation", None)
        if _is_root_observation_payload(payload) and non_current_method is not None:
            await self._emit_v4_non_current_observation(
                non_current_method,
                payload,
                metadata,
                as_type=as_type,
            )
            return
        if payload.get("end_time") is not None and non_current_method is not None:
            await self._emit_v4_non_current_observation(
                non_current_method,
                payload,
                metadata,
                as_type=as_type,
            )
            return
        method = getattr(self.client, "start_as_current_observation", None)
        if method is None:
            if non_current_method is None:
                raise AttributeError("Langfuse client has no observation start method")
            await self._emit_v4_non_current_observation(
                non_current_method,
                payload,
                metadata,
                as_type=as_type,
            )
            return
        observation_metadata = dict(metadata)
        observation_id = payload.get("observation_id")
        if observation_id is not None:
            observation_metadata["observation_id"] = observation_id
        with self._v4_module_propagation_context(payload):
            observation_context = method(
                **_drop_none(
                    {
                        "trace_context": self._v4_trace_context(payload),
                        "name": payload.get("name"),
                        "as_type": as_type,
                        "end_on_exit": (
                            False if _is_root_observation_payload(payload) else None
                        ),
                        "input": payload.get("input"),
                        "output": payload.get("output"),
                        "metadata": observation_metadata,
                        "level": _langfuse_level(
                            payload.get("status"), payload.get("error")
                        ),
                        "status_message": payload.get("error"),
                        "model": payload.get("model"),
                        "model_parameters": payload.get("model_parameters"),
                        "usage_details": _langfuse_usage_details(
                            payload.get("usage_details")
                        ),
                        "cost_details": _langfuse_cost_details(
                            payload.get("cost_details")
                        ),
                    }
                )
            )
            with observation_context as observation:
                self._remember_v4_observation_id(payload, observation)
                self._remember_v4_trace_id(payload, observation)
                self._apply_v4_trace_name(payload, observation)
                self._remember_active_v4_root_observation(payload, observation)
                with self._v4_client_propagation_context(payload):
                    return

    async def _finalize_active_v4_root_observation(
        self,
        payload: dict[str, JsonSafe],
        metadata: dict[str, JsonSafe],
        *,
        as_type: str,
    ) -> bool:
        """Update and end an active root observation instead of creating a duplicate."""
        observation_id = payload.get("observation_id")
        if not isinstance(observation_id, str) or payload.get("end_time") is None:
            return False
        observation = self._active_v4_root_observations.pop(observation_id, None)
        if observation is None:
            return False

        update = getattr(observation, "update", None)
        if update is not None:
            result = update(
                **_drop_none(
                    {
                        "name": payload.get("name"),
                        "as_type": as_type,
                        "input": payload.get("input"),
                        "output": payload.get("output"),
                        "metadata": metadata,
                        "level": _langfuse_level(
                            payload.get("status"), payload.get("error")
                        ),
                        "status_message": payload.get("error"),
                        "model": payload.get("model"),
                        "model_parameters": payload.get("model_parameters"),
                        "usage_details": _langfuse_usage_details(
                            payload.get("usage_details")
                        ),
                        "cost_details": _langfuse_cost_details(
                            payload.get("cost_details")
                        ),
                    }
                )
            )
            if inspect.isawaitable(result):
                await result

        end = getattr(observation, "end", None)
        if end is not None:
            result = end(end_time=_datetime_to_epoch_ns(payload.get("end_time")))
            if inspect.isawaitable(result):
                await result
        return True

    def _remember_active_v4_root_observation(
        self,
        payload: dict[str, JsonSafe],
        observation: Any,
    ) -> None:
        """Remember open root observations so later completion updates them."""
        observation_id = payload.get("observation_id")
        if (
            isinstance(observation_id, str)
            and payload.get("end_time") is None
            and _is_root_observation_payload(payload)
        ):
            self._active_v4_root_observations[observation_id] = observation

    async def _emit_v4_non_current_observation(
        self,
        method: Any,
        payload: dict[str, JsonSafe],
        metadata: dict[str, JsonSafe],
        *,
        as_type: str,
    ) -> None:
        """Emit a v4 observation through start_observation and explicit end time."""
        observation_metadata = dict(metadata)
        observation_id = payload.get("observation_id")
        if observation_id is not None:
            observation_metadata["observation_id"] = observation_id
        if self._can_emit_v4_historical_observation(payload):
            await self._emit_v4_historical_observation(
                payload,
                observation_metadata,
                as_type=as_type,
            )
            return
        with self._v4_clean_root_context(payload):
            with self._v4_pre_start_propagation_context(payload):
                observation = method(
                    **_drop_none(
                        {
                            "trace_context": self._v4_trace_context(payload),
                            "name": payload.get("name"),
                            "as_type": as_type,
                            "input": payload.get("input"),
                            "output": payload.get("output"),
                            "metadata": observation_metadata,
                            "level": _langfuse_level(
                                payload.get("status"), payload.get("error")
                            ),
                            "status_message": payload.get("error"),
                            "model": payload.get("model"),
                            "model_parameters": payload.get("model_parameters"),
                            "usage_details": _langfuse_usage_details(
                                payload.get("usage_details")
                            ),
                            "cost_details": _langfuse_cost_details(
                                payload.get("cost_details")
                            ),
                        }
                    )
                )
                self._remember_v4_observation_id(payload, observation)
                self._remember_v4_trace_id(payload, observation)
                self._apply_v4_trace_name(payload, observation)
                self._remember_active_v4_root_observation(payload, observation)
                end = getattr(observation, "end", None)
                if end is not None and payload.get("end_time") is not None:
                    result = end(end_time=_datetime_to_epoch_ns(payload.get("end_time")))
                    if inspect.isawaitable(result):
                        await result

    def _can_emit_v4_historical_observation(
        self,
        payload: dict[str, JsonSafe],
    ) -> bool:
        """Return whether the client can create a v4 observation with start time."""
        return (
            self.config.enable_private_v4_historical_otel
            and payload.get("start_time") is not None
            and payload.get("end_time") is not None
            and getattr(self.client, "_otel_tracer", None) is not None
            and getattr(self.client, "_create_remote_parent_span", None) is not None
            and getattr(self.client, "_create_observation_from_otel_span", None)
            is not None
        )

    async def _emit_v4_historical_observation(
        self,
        payload: dict[str, JsonSafe],
        metadata: dict[str, JsonSafe],
        *,
        as_type: str,
    ) -> None:
        """Emit a v4 observation with historical start and end times."""
        tracer = getattr(self.client, "_otel_tracer")
        start_time = _datetime_to_epoch_ns(payload.get("start_time"))
        end_time = _datetime_to_epoch_ns(payload.get("end_time"))
        trace_context = self._v4_trace_context(payload) or {}
        with self._v4_historical_propagation_context(payload):
            remote_parent = self._v4_remote_parent_span(trace_context)
            with self._v4_use_span(remote_parent):
                otel_span = tracer.start_span(
                    name=str(payload.get("name", "ecs-agent.observation")),
                    start_time=start_time,
                )
                set_attribute = getattr(otel_span, "set_attribute", None)
                if set_attribute is not None and _is_root_observation_payload(payload):
                    set_attribute("langfuse.internal.as_root", True)
                observation_factory = getattr(
                    self.client,
                    "_create_observation_from_otel_span",
                )
                observation = observation_factory(
                    **_drop_none(
                        {
                            "otel_span": otel_span,
                            "as_type": as_type,
                            "input": payload.get("input"),
                            "output": payload.get("output"),
                            "metadata": metadata,
                            "level": _langfuse_level(
                                payload.get("status"), payload.get("error")
                            ),
                            "status_message": payload.get("error"),
                            "model": payload.get("model"),
                            "model_parameters": payload.get("model_parameters"),
                            "usage_details": _langfuse_usage_details(
                                payload.get("usage_details")
                            ),
                            "cost_details": _langfuse_cost_details(
                                payload.get("cost_details")
                            ),
                        }
                    )
                )
                self._remember_v4_observation_id(payload, observation)
                self._remember_v4_trace_id(payload, observation)
                self._apply_v4_trace_name(payload, observation)
        end = getattr(observation, "end", None)
        if end is not None:
            result = end(end_time=end_time)
            if inspect.isawaitable(result):
                await result

    def _v4_remote_parent_span(self, trace_context: dict[str, JsonSafe]) -> Any | None:
        """Create a v4 remote parent span when trace context is available."""
        trace_id = trace_context.get("trace_id")
        if not isinstance(trace_id, str):
            return None
        parent_span_id = trace_context.get("parent_span_id")
        if not isinstance(parent_span_id, str):
            parent_span_id = None
        create_remote_parent = getattr(self.client, "_create_remote_parent_span", None)
        if create_remote_parent is None:
            return None
        return create_remote_parent(
            trace_id=trace_id,
            parent_span_id=parent_span_id,
        )

    def _v4_trace_context(self, payload: dict[str, JsonSafe]) -> dict[str, JsonSafe] | None:
        """Return v4 trace context, omitting it for root observations."""
        if _is_root_observation_payload(payload):
            return None
        return _trace_context(payload, self._v4_observation_ids, self._v4_trace_ids)

    def _remember_v4_trace_id(
        self,
        payload: dict[str, JsonSafe],
        observation: Any,
    ) -> None:
        """Map internal trace IDs to SDK-generated Langfuse v4 trace IDs."""
        internal_trace_id = payload.get("trace_id")
        sdk_trace_id = getattr(observation, "trace_id", None)
        if isinstance(internal_trace_id, str) and isinstance(sdk_trace_id, str):
            self._v4_trace_ids[internal_trace_id] = sdk_trace_id

    def _apply_v4_trace_name(
        self,
        payload: dict[str, JsonSafe],
        observation: Any,
    ) -> None:
        """Set the v4 trace-name attribute on an observation's OTel span when exposed."""
        trace_name = self._v4_payload_trace_name(payload)
        if trace_name is None:
            return
        otel_span = getattr(observation, "_otel_span", None)
        set_attribute = getattr(otel_span, "set_attribute", None)
        if set_attribute is not None:
            set_attribute("langfuse.trace.name", trace_name)

    def _remember_v4_trace_name(self, payload: dict[str, JsonSafe]) -> None:
        """Remember the intended Langfuse trace name for later child observations."""
        trace_id = payload.get("trace_id")
        trace_name = self._v4_payload_trace_name(payload)
        if isinstance(trace_id, str) and trace_name is not None:
            self._v4_trace_names[trace_id] = trace_name

    def _v4_clean_root_context(self, payload: dict[str, JsonSafe]) -> Any:
        """Detach ambient OTel span context while creating v4 root observations."""
        if not _is_root_observation_payload(payload):
            return nullcontext()
        return _OtelDetachedContext()

    def _v4_use_span(self, span: Any | None) -> Any:
        """Return an OpenTelemetry use_span context or a no-op fallback."""
        if span is None:
            return nullcontext()
        try:
            trace_api = importlib.import_module("opentelemetry.trace")
        except ImportError:
            return nullcontext()
        use_span = getattr(trace_api, "use_span", None)
        if use_span is None:
            return nullcontext()
        return use_span(span)

    def _v4_historical_propagation_context(self, payload: dict[str, JsonSafe]) -> Any:
        """Return a trace-attribute propagation context for historical spans."""
        if getattr(self.client, "propagate_attributes", None) is not None:
            return self._v4_client_propagation_context(payload)
        return self._v4_module_propagation_context(payload)

    async def _emit_v4_event(
        self,
        payload: dict[str, JsonSafe],
        metadata: dict[str, JsonSafe],
    ) -> None:
        method = getattr(self.client, "create_event", None)
        if method is None:
            await self._emit_v4_observation(payload, metadata, as_type="span")
            return
        event_metadata = dict(metadata)
        observation_id = payload.get("observation_id")
        if observation_id is not None:
            event_metadata["observation_id"] = observation_id
        with self._v4_pre_start_propagation_context(payload):
            result = method(
                **_drop_none(
                    {
                            "trace_context": self._v4_trace_context(payload),
                        "name": payload.get("name"),
                        "input": payload.get("input"),
                        "output": payload.get("output"),
                        "metadata": event_metadata,
                        "level": _langfuse_level(
                            payload.get("status"), payload.get("error")
                        ),
                        "status_message": payload.get("error"),
                    }
                )
            )
            if inspect.isawaitable(result):
                await result

    def _remember_v4_observation_id(
        self,
        payload: dict[str, JsonSafe],
        observation: Any,
    ) -> None:
        """Map internal observation IDs to SDK-generated Langfuse v4 IDs."""
        internal_id = payload.get("observation_id")
        sdk_id = getattr(observation, "id", None)
        if not isinstance(internal_id, str) or not isinstance(sdk_id, str):
            return
        if _is_lower_hex(sdk_id, _LANGFUSE_SPAN_ID_HEX_LENGTH):
            self._v4_observation_ids[internal_id] = sdk_id

    def _v4_propagation_kwargs(
        self,
        payload: dict[str, JsonSafe] | None = None,
    ) -> dict[str, JsonSafe]:
        config_context = _config_export_context(self.config, self.redactor)
        return _drop_none(
            {
                "user_id": config_context.metadata.get("user_id"),
                "session_id": config_context.metadata.get("session_id"),
                "tags": config_context.metadata.get("tags"),
                "trace_name": self._v4_payload_trace_name(payload),
            }
        )

    def _v4_payload_trace_name(
        self,
        payload: dict[str, JsonSafe] | None,
    ) -> str | None:
        """Return the intended v4 trace-level name for this payload."""
        if payload is None:
            return None
        trace_id = payload.get("trace_id")
        if isinstance(trace_id, str) and trace_id in self._v4_trace_names:
            return self._v4_trace_names[trace_id]
        if payload.get("kind") == "trace" and payload.get("name") == "user.turn":
            return "user.turn"
        return None

    def _v4_pre_start_propagation_context(
        self,
        payload: dict[str, JsonSafe],
    ) -> Any:
        """Return propagation active before creating a v4 observation span."""
        if getattr(self.client, "propagate_attributes", None) is not None:
            return self._v4_client_propagation_context(payload)
        return self._v4_module_propagation_context(payload)

    def _v4_client_propagation_context(
        self,
        payload: dict[str, JsonSafe] | None = None,
    ) -> Any:
        method = getattr(self.client, "propagate_attributes", None)
        if method is None:
            return nullcontext()
        propagation = self._v4_propagation_kwargs(payload)
        if not propagation:
            return nullcontext()
        return method(**propagation)

    def _v4_module_propagation_context(
        self,
        payload: dict[str, JsonSafe] | None = None,
    ) -> Any:
        if getattr(self.client, "propagate_attributes", None) is not None:
            return nullcontext()
        propagation = self._v4_propagation_kwargs(payload)
        if not propagation:
            return nullcontext()
        try:
            langfuse_module = importlib.import_module("langfuse")
        except ImportError:
            return nullcontext()
        method = getattr(langfuse_module, "propagate_attributes", None)
        if method is None:
            return nullcontext()
        return method(**propagation)

    async def _call_optional_client(self, method_name: str) -> None:
        method = getattr(self.client, method_name, None)
        if method is None:
            return
        result = method()
        if inspect.isawaitable(result):
            await result


def install_langfuse_observability(
    world: Any,
    config: LangfuseConfig | None = None,
    sink: TelemetrySink | None = None,
) -> ObservabilityHandle:
    """Install generic observability with an optional Langfuse sink."""
    resolved_config = (LangfuseConfig() if config is None else config).with_env()
    telemetry_sink = sink
    if telemetry_sink is None:
        if resolved_config.enabled:
            telemetry_sink = LangfuseTelemetrySink(
                client=_create_langfuse_client(resolved_config),
                config=resolved_config,
            )
        else:
            telemetry_sink = NoOpTelemetrySink()
    return install_observability(world, telemetry_sink, config=resolved_config)


def _create_langfuse_client(config: LangfuseConfig) -> Any:
    try:
        langfuse_module = importlib.import_module("langfuse")
    except ImportError as exc:
        raise ImportError(LANGFUSE_IMPORT_ERROR) from exc
    langfuse_class = getattr(langfuse_module, "Langfuse")

    kwargs: dict[str, Any] = {
        "public_key": config.public_key,
        "secret_key": config.secret_key,
        "base_url": config.host,
        "environment": config.environment,
        "release": config.release,
        "flush_at": config.flush_at,
        "flush_interval": config.flush_interval,
        "timeout": config.timeout,
    }
    return langfuse_class(**_drop_none(kwargs))


def _first_method(client: Any, method_names: tuple[str, ...]) -> Any | None:
    for name in method_names:
        method = getattr(client, name, None)
        if method is not None:
            return method
    return None


def _drop_none(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if value is not None}


class _OtelDetachedContext:
    """Context manager that temporarily clears the active OpenTelemetry span."""

    def __init__(self) -> None:
        self._context_api: Any | None = None
        self._token: object | None = None

    def __enter__(self) -> None:
        try:
            context_api = importlib.import_module("opentelemetry.context")
        except ImportError:
            return
        context_factory = getattr(context_api, "Context", None)
        attach = getattr(context_api, "attach", None)
        if context_factory is None or attach is None:
            return
        self._context_api = context_api
        self._token = attach(context_factory())

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object | None,
    ) -> None:
        if self._context_api is None or self._token is None:
            return
        detach = getattr(self._context_api, "detach", None)
        if detach is not None:
            detach(self._token)


def _apply_capture_policy(
    payload: dict[str, JsonSafe],
    config: LangfuseConfig,
) -> dict[str, JsonSafe]:
    """Return a payload copy with raw input/output capture policy applied."""
    filtered = dict(payload)
    if not config.capture_input:
        filtered["input"] = None
    if not config.capture_output:
        filtered["output"] = None
    return filtered


def _langfuse_usage_details(value: JsonSafe) -> dict[str, int] | None:
    """Return Langfuse-compatible integer usage counters."""
    if not isinstance(value, dict):
        return None
    usage = {
        key: item
        for key, item in value.items()
        if isinstance(key, str) and isinstance(item, int) and not isinstance(item, bool)
    }
    return usage or None


def _langfuse_cost_details(value: JsonSafe) -> dict[str, float] | None:
    """Return Langfuse-compatible numeric cost counters."""
    if not isinstance(value, dict):
        return None
    cost = {
        key: float(item)
        for key, item in value.items()
        if isinstance(key, str)
        and isinstance(item, int | float)
        and not isinstance(item, bool)
    }
    return cost or None


def _datetime_to_epoch_ns(value: JsonSafe) -> int | None:
    """Convert an ISO datetime payload value to epoch nanoseconds."""
    if not isinstance(value, str):
        return None
    parsed = datetime.fromisoformat(value)
    return int(parsed.timestamp() * 1_000_000_000)


def _as_dict(payload: JsonSafe) -> dict[str, JsonSafe]:
    if isinstance(payload, dict):
        return payload
    raise TypeError("Langfuse telemetry payload must be a dictionary")


def _trace_context(
    payload: dict[str, JsonSafe],
    v4_observation_ids: dict[str, str] | None = None,
    v4_trace_ids: dict[str, str] | None = None,
) -> dict[str, JsonSafe]:
    context: dict[str, JsonSafe] = {}
    trace_id = payload.get("trace_id")
    if trace_id is not None:
        context["trace_id"] = (
            v4_trace_ids.get(trace_id, trace_id)
            if v4_trace_ids is not None and isinstance(trace_id, str)
            else trace_id
        )
    parent_observation_id = payload.get("parent_observation_id")
    parent_span_id = _langfuse_parent_span_id(parent_observation_id, v4_observation_ids)
    if parent_span_id is not None:
        context["parent_span_id"] = parent_span_id
    return context


def _langfuse_parent_span_id(
    value: JsonSafe,
    v4_observation_ids: dict[str, str] | None = None,
) -> str | None:
    """Return a Langfuse v4 parent span ID from an internal observation ID."""
    if not isinstance(value, str):
        return None
    if v4_observation_ids is not None and value in v4_observation_ids:
        return v4_observation_ids[value]
    if _is_lower_hex(value, _INTERNAL_OBSERVATION_ID_HEX_LENGTH):
        return value[:_LANGFUSE_SPAN_ID_HEX_LENGTH]
    return value


def _is_lower_hex(value: str, length: int) -> bool:
    return len(value) == length and all(char in "0123456789abcdef" for char in value)


def _is_root_trace_payload(payload: dict[str, JsonSafe]) -> bool:
    """Return whether a payload represents the top-level trace root."""
    return payload.get("kind") == "trace" and payload.get("parent_observation_id") is None


def _is_trace_container_payload(payload: dict[str, JsonSafe]) -> bool:
    """Return whether a payload should create/update a Langfuse trace container."""
    return _is_root_trace_payload(payload) and payload.get("name") != "user.turn"


def _is_root_observation_payload(payload: dict[str, JsonSafe]) -> bool:
    """Return whether a payload is an unparented Langfuse observation root."""
    return payload.get("parent_observation_id") is None and not _is_trace_container_payload(
        payload
    )


def _metadata_with_context(
    config: LangfuseConfig,
    metadata: JsonSafe,
    redaction: dict[str, JsonSafe],
    config_context: ConfigExportContext,
) -> dict[str, JsonSafe]:
    merged: dict[str, JsonSafe] = {}
    if isinstance(config.metadata, dict):
        config_metadata, metadata_redaction = _json_dict(
            config.metadata,
            config_context.redactor,
        )
        merged.update(config_metadata)
        redaction = _merge_redaction_reports(redaction, metadata_redaction)
    if isinstance(metadata, dict):
        merged.update(metadata)
    merged.update(config_context.metadata)
    redaction = _merge_redaction_reports(redaction, config_context.redaction)
    merged["redaction"] = redaction
    return merged


@dataclass(slots=True)
class ConfigExportContext:
    """Sanitized config-derived fields and their redaction report."""

    metadata: dict[str, JsonSafe]
    redaction: dict[str, JsonSafe]
    redactor: SecretRedactor


def _config_export_context(
    config: LangfuseConfig,
    redactor: SecretRedactor,
) -> ConfigExportContext:
    raw_metadata: dict[str, Any] = {}
    if config.environment is not None:
        raw_metadata["environment"] = config.environment
    if config.release is not None:
        raw_metadata["release"] = config.release
    if config.user_id is not None:
        raw_metadata["user_id"] = config.user_id
    if config.session_id is not None:
        raw_metadata["session_id"] = config.session_id
    if config.tags is not None:
        raw_metadata["tags"] = list(config.tags)

    sanitized, redaction = sanitize_payload(raw_metadata, redactor)
    return ConfigExportContext(
        metadata=_as_dict(sanitized),
        redaction=redaction.to_payload(),
        redactor=redactor,
    )


def _json_dict(
    value: dict[str, Any],
    redactor: SecretRedactor,
) -> tuple[dict[str, JsonSafe], dict[str, JsonSafe]]:
    sanitized, redaction = sanitize_payload(value, redactor)
    return _as_dict(sanitized), redaction.to_payload()


def _merge_redaction_reports(
    first: dict[str, JsonSafe],
    second: dict[str, JsonSafe],
) -> dict[str, JsonSafe]:
    total_redactions = _redaction_total(first) + _redaction_total(second)
    counts_by_rule: dict[str, JsonSafe] = {}
    for report in (first, second):
        counts = report.get("counts_by_rule")
        if not isinstance(counts, dict):
            continue
        for rule, count in counts.items():
            if not isinstance(rule, str) or not isinstance(count, int):
                continue
            existing = counts_by_rule.get(rule, 0)
            if not isinstance(existing, int):
                existing = 0
            counts_by_rule[rule] = existing + count
    return {
        "total_redactions": total_redactions,
        "counts_by_rule": counts_by_rule,
    }


def _redaction_total(report: dict[str, JsonSafe]) -> int:
    total_redactions = report.get("total_redactions")
    if isinstance(total_redactions, int):
        return total_redactions
    return 0


def _langfuse_observation_type(kind: JsonSafe) -> str:
    if kind == "generation":
        return "generation"
    if kind == "tool":
        return "tool"
    if kind == "event":
        return "event"
    if kind == "trace":
        return "span"
    return "span"


def _langfuse_level(status: JsonSafe, error: JsonSafe) -> str:
    if status == "error" or error is not None:
        return "ERROR"
    if status == "cancelled":
        return "WARNING"
    return "DEFAULT"


__all__ = [
    "LangfuseConfig",
    "LangfuseTelemetrySink",
    "install_langfuse_observability",
]
