"""Prometheus observability plugin (optional ``ecs-agent[prometheus]`` extra).

Provides the low-cardinality metric contract, the ``PrometheusMetrics``
recorder, metrics endpoint helpers, and ``PrometheusPlugin`` for mounting
metrics collection on a world through ``ecs_agent.plugins``.
"""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass, replace
from collections.abc import Iterator
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from typing import TYPE_CHECKING, Any, Literal

from ecs_agent.accounting.models import LLMInvocationEvent, LLMRetryEvent, UsageRecord
from ecs_agent.observability.install import EventSubscription
from ecs_agent.observability.sinks import TelemetrySink
from ecs_agent.types import (
    CheckpointCreatedEvent,
    CheckpointRestoredEvent,
    CompactionCompleteEvent,
    DelegationCompletedEvent,
    DelegationStartedEvent,
    ErrorOccurredEvent,
    MCTSNodeScoredEvent,
    MessageBusDeliveredEvent,
    MessageBusPublishedEvent,
    MessageBusResponseEvent,
    MessageBusTimeoutEvent,
    PlanStepCompletedEvent,
    PlanRevisedEvent,
    RunCompletedEvent,
    RunnerTickCompletedEvent,
    RunnerTickStartedEvent,
    RunStartedEvent,
    StreamContentDeltaEvent,
    StreamEndEvent,
    StreamReasoningDeltaEvent,
    StreamStartEvent,
    SystemExecutionCompletedEvent,
    SystemExecutionStartedEvent,
    ToolApprovedEvent,
    ToolDeniedEvent,
    ToolExecutionCompletedEvent,
    ToolExecutionStartedEvent,
    ToolResultCachedEvent,
)

if TYPE_CHECKING:
    from prometheus_client import CollectorRegistry

MetricType = Literal["counter", "histogram", "gauge"]

PROMETHEUS_IMPORT_ERROR = "Install ecs-agent[prometheus] to use Prometheus metrics"


def _prometheus_client() -> Any:
    """Import prometheus_client lazily with an actionable error message."""
    try:
        return importlib.import_module("prometheus_client")
    except ImportError as exc:
        raise ImportError(PROMETHEUS_IMPORT_ERROR) from exc


@dataclass(frozen=True, slots=True)
class MetricSpec:
    """Prometheus metric definition with fixed type and label names."""

    name: str
    metric_type: MetricType
    documentation: str
    labels: tuple[str, ...] = ()


@dataclass(slots=True)
class MetricsServerHandle:
    """Cleanup handle for a standalone Prometheus HTTP server."""

    server: Any
    thread: Any

    def __iter__(self) -> Iterator[Any]:
        """Yield the underlying ``(server, thread)`` for tuple-unpacking compatibility."""
        yield self.server
        yield self.thread

    def shutdown(self) -> None:
        """Stop the HTTP server loop."""
        self.server.shutdown()

    def server_close(self) -> None:
        """Close the HTTP server socket."""
        self.server.server_close()

    def join(self, timeout: float | None = None) -> None:
        """Join the background server thread."""
        self.thread.join(timeout)

    def close(self, timeout: float | None = None) -> None:
        """Shutdown, close, and join the standalone server deterministically."""
        self.shutdown()
        self.server_close()
        self.join(timeout)


ALLOWED_LABELS = frozenset(
    {
        "system",
        "status",
        "reason",
        "operation",
        "provider",
        "model",
        "streaming",
        "event",
        "phase",
        "policy",
        "tool",
        "error_type",
        "token_type",
    }
)

FORBIDDEN_LABELS = frozenset(
    {
        "entity_id",
        "tool_call_id",
        "request_id",
        "correlation_id",
        "trace_id",
        "session_id",
        "checkpoint_id",
        "node_id",
        "branch_id",
        "message_id",
        "world_name",
        "topic",
        "prompt_text",
        "response_text",
        "raw_prompt_text",
        "raw_response_text",
        "tool_args",
        "raw_tool_args",
        "tool_results",
        "raw_tool_results",
        "exception",
        "exception_string",
        "raw_exception",
        "artifact_path",
        "api_key",
        "api_token",
        "token",
    }
)

METRIC_CONTRACT = {
    "ecs_agent_runs_total": MetricSpec(
        "ecs_agent_runs_total", "counter", "Agent run outcomes.", ("status",)
    ),
    "ecs_agent_runner_ticks_total": MetricSpec(
        "ecs_agent_runner_ticks_total", "counter", "Runner tick outcomes.", ("status",)
    ),
    "ecs_agent_runner_tick_duration_seconds": MetricSpec(
        "ecs_agent_runner_tick_duration_seconds",
        "histogram",
        "Runner tick duration in seconds.",
        ("status",),
    ),
    "ecs_agent_system_executions_total": MetricSpec(
        "ecs_agent_system_executions_total",
        "counter",
        "System execution outcomes.",
        ("system", "status"),
    ),
    "ecs_agent_system_execution_duration_seconds": MetricSpec(
        "ecs_agent_system_execution_duration_seconds",
        "histogram",
        "System execution duration in seconds.",
        ("system", "status"),
    ),
    "ecs_agent_active_entities": MetricSpec(
        "ecs_agent_active_entities", "gauge", "Latest observed active entity count."
    ),
    "ecs_agent_llm_invocations_total": MetricSpec(
        "ecs_agent_llm_invocations_total",
        "counter",
        "Logical LLM invocation outcomes.",
        ("provider", "model", "operation", "status", "streaming"),
    ),
    "ecs_agent_llm_invocation_duration_seconds": MetricSpec(
        "ecs_agent_llm_invocation_duration_seconds",
        "histogram",
        "Logical LLM invocation duration in seconds.",
        ("provider", "model", "operation", "status", "streaming"),
    ),
    "ecs_agent_llm_tokens_total": MetricSpec(
        "ecs_agent_llm_tokens_total",
        "counter",
        "LLM token usage by normalized token type.",
        ("provider", "model", "token_type"),
    ),
    "ecs_agent_llm_retries_total": MetricSpec(
        "ecs_agent_llm_retries_total",
        "counter",
        "LLM retry attempts.",
        ("provider", "model", "reason"),
    ),
    "ecs_agent_tool_calls_total": MetricSpec(
        "ecs_agent_tool_calls_total", "counter", "Tool call outcomes.", ("tool", "status")
    ),
    "ecs_agent_tool_call_duration_seconds": MetricSpec(
        "ecs_agent_tool_call_duration_seconds",
        "histogram",
        "Tool call duration in seconds.",
        ("tool", "status"),
    ),
    "ecs_agent_tool_denied_total": MetricSpec(
        "ecs_agent_tool_denied_total", "counter", "Denied tool attempts.", ("tool", "reason")
    ),
    "ecs_agent_tool_approved_total": MetricSpec(
        "ecs_agent_tool_approved_total",
        "counter",
        "Approved tool attempts.",
        ("tool", "policy"),
    ),
    "ecs_agent_errors_total": MetricSpec(
        "ecs_agent_errors_total", "counter", "Captured error outcomes.", ("system", "error_type")
    ),
    "ecs_agent_terminals_total": MetricSpec(
        "ecs_agent_terminals_total", "counter", "Terminal run outcomes.", ("reason",)
    ),
    "ecs_agent_stream_events_total": MetricSpec(
        "ecs_agent_stream_events_total", "counter", "Stream lifecycle events.", ("event", "status")
    ),
    "ecs_agent_stream_first_delta_seconds": MetricSpec(
        "ecs_agent_stream_first_delta_seconds",
        "histogram",
        "Time to first stream delta in seconds.",
        ("provider", "model", "operation", "status"),
    ),
    "ecs_agent_stream_duration_seconds": MetricSpec(
        "ecs_agent_stream_duration_seconds",
        "histogram",
        "Total stream duration in seconds.",
        ("provider", "model", "operation", "status"),
    ),
    "ecs_agent_subagent_lifecycle_total": MetricSpec(
        "ecs_agent_subagent_lifecycle_total",
        "counter",
        "Subagent lifecycle events.",
        ("phase", "status"),
    ),
    "ecs_agent_message_bus_events_total": MetricSpec(
        "ecs_agent_message_bus_events_total",
        "counter",
        "Message bus operations.",
        ("event", "operation"),
    ),
    "ecs_agent_checkpoint_operations_total": MetricSpec(
        "ecs_agent_checkpoint_operations_total",
        "counter",
        "Checkpoint operations.",
        ("operation", "status"),
    ),
    "ecs_agent_compaction_operations_total": MetricSpec(
        "ecs_agent_compaction_operations_total",
        "counter",
        "Compaction operations.",
        ("operation", "status"),
    ),
    "ecs_agent_mcts_nodes_scored_total": MetricSpec(
        "ecs_agent_mcts_nodes_scored_total",
        "counter",
        "MCTS node scoring outcomes.",
        ("phase", "status"),
    ),
    "ecs_agent_plan_steps_total": MetricSpec(
        "ecs_agent_plan_steps_total", "counter", "Plan step operations.", ("operation", "status")
    ),
    "ecs_agent_tool_result_cached_total": MetricSpec(
        "ecs_agent_tool_result_cached_total",
        "counter",
        "Tool-result cache outcomes.",
        ("status",),
    ),
}


EVENT_SUBSCRIPTION_MATRIX = {
    "runner_lifecycle": (
        "RunStartedEvent",
        "RunnerTickStartedEvent",
        "RunnerTickCompletedEvent",
        "RunCompletedEvent",
    ),
    "system_lifecycle": (
        "SystemExecutionStartedEvent",
        "SystemExecutionCompletedEvent",
    ),
    "llm_lifecycle": (
        "LLMInvocationEvent",
        "LLMRetryEvent",
        "StreamStartEvent",
        "StreamContentDeltaEvent",
        "StreamReasoningDeltaEvent",
        "StreamEndEvent",
    ),
    "tool_lifecycle": (
        "ToolExecutionStartedEvent",
        "ToolExecutionCompletedEvent",
        "ToolDeniedEvent",
        "ToolApprovedEvent",
    ),
    "runtime_control_lifecycle": (
        "DelegationStartedEvent",
        "DelegationCompletedEvent",
        "MessageBusPublishedEvent",
        "MessageBusDeliveredEvent",
        "MessageBusTimeoutEvent",
        "MessageBusResponseEvent",
        "CheckpointCreatedEvent",
        "CheckpointRestoredEvent",
        "CompactionCompleteEvent",
        "MCTSNodeScoredEvent",
        "PlanStepCompletedEvent",
        "PlanRevisedEvent",
        "ToolResultCachedEvent",
    ),
}


def validate_metric_labels(labels: tuple[str, ...]) -> None:
    """Reject forbidden or non-contract Prometheus labels."""
    unknown_labels = set(labels) - ALLOWED_LABELS
    forbidden_labels = set(labels) & FORBIDDEN_LABELS
    invalid_labels = unknown_labels | forbidden_labels
    if invalid_labels:
        names = ", ".join(sorted(invalid_labels))
        raise ValueError(f"Prometheus metric labels are not allowed: {names}")


def _safe_label(value: object, *, default: str = "unknown", max_length: int = 64) -> str:
    text = str(value).strip() if value is not None else ""
    if not text:
        return default
    if len(text) > max_length:
        return default
    if any(char.isspace() for char in text):
        return default
    if any(char in text for char in {'"', "'", "\\", "/"}):
        return default
    return text


def _bool_label(value: bool) -> str:
    return "true" if value else "false"


def _inc(collector: Any, labels: dict[str, str] | None = None, amount: float = 1.0) -> None:
    if labels:
        collector.labels(**labels).inc(amount)
        return
    collector.inc(amount)


def _observe(collector: Any, labels: dict[str, str], amount: float | None) -> None:
    if amount is None:
        return
    collector.labels(**labels).observe(amount)


def _set(collector: Any, amount: float) -> None:
    collector.set(amount)


class PrometheusMetrics:
    """Owns an isolated Prometheus registry for ecs-agent metric families."""

    def __init__(
        self,
        *,
        registry: CollectorRegistry | None = None,
        metric_contract: dict[str, MetricSpec] | None = None,
    ) -> None:
        self.registry = (
            _prometheus_client().CollectorRegistry() if registry is None else registry
        )
        self.metric_contract = METRIC_CONTRACT if metric_contract is None else metric_contract
        self.collectors: dict[str, Any] = {}
        self._install_collectors()

    def _install_collectors(self) -> None:
        prometheus = _prometheus_client()
        for spec in self.metric_contract.values():
            validate_metric_labels(spec.labels)
            collector: Any
            if spec.metric_type == "counter":
                collector = prometheus.Counter(
                    spec.name,
                    spec.documentation,
                    spec.labels,
                    registry=self.registry,
                )
            elif spec.metric_type == "histogram":
                collector = prometheus.Histogram(
                    spec.name,
                    spec.documentation,
                    spec.labels,
                    registry=self.registry,
                )
            else:
                collector = prometheus.Gauge(
                    spec.name,
                    spec.documentation,
                    spec.labels,
                    registry=self.registry,
                )
            self.collectors[spec.name] = collector

    async def handle_run_started(self, event: RunStartedEvent) -> None:
        """Record runner start snapshot metrics."""
        _set(self.collectors["ecs_agent_active_entities"], float(event.active_entities))

    async def handle_runner_tick_started(self, event: RunnerTickStartedEvent) -> None:
        """Record runner tick start snapshot metrics."""
        _set(self.collectors["ecs_agent_active_entities"], float(event.active_entities))

    async def handle_runner_tick_completed(self, event: RunnerTickCompletedEvent) -> None:
        """Record runner tick outcome metrics."""
        labels = {"status": _safe_label(event.status)}
        _inc(self.collectors["ecs_agent_runner_ticks_total"], labels)
        _observe(
            self.collectors["ecs_agent_runner_tick_duration_seconds"],
            labels,
            event.duration_seconds,
        )
        _set(self.collectors["ecs_agent_active_entities"], float(event.active_entities))

    async def handle_run_completed(self, event: RunCompletedEvent) -> None:
        """Record runner completion and terminal outcome metrics."""
        _inc(
            self.collectors["ecs_agent_runs_total"],
            {"status": _safe_label(event.status)},
        )
        _inc(
            self.collectors["ecs_agent_terminals_total"],
            {"reason": _safe_label(event.reason)},
        )
        _set(self.collectors["ecs_agent_active_entities"], float(event.active_entities))

    async def handle_system_execution_started(
        self, event: SystemExecutionStartedEvent
    ) -> None:
        """Accept system start events for EventBus subscription completeness."""
        _ = event

    async def handle_system_execution_completed(
        self, event: SystemExecutionCompletedEvent
    ) -> None:
        """Record system execution outcome metrics."""
        labels = {
            "system": _safe_label(event.system),
            "status": _safe_label(event.status),
        }
        _inc(self.collectors["ecs_agent_system_executions_total"], labels)
        _observe(
            self.collectors["ecs_agent_system_execution_duration_seconds"],
            labels,
            event.duration_seconds,
        )

    async def handle_llm_invocation(self, event: LLMInvocationEvent) -> None:
        """Record logical LLM invocation, duration, and token usage metrics."""
        labels = {
            "provider": _safe_label(event.provider_id),
            "model": _safe_label(event.model),
            "operation": _safe_label(event.operation),
            "status": _safe_label(event.status),
            "streaming": _bool_label(event.streaming),
        }
        _inc(self.collectors["ecs_agent_llm_invocations_total"], labels)
        _observe(
            self.collectors["ecs_agent_llm_invocation_duration_seconds"],
            labels,
            event.duration_seconds,
        )
        self._record_usage(event.provider_id, event.model, event.usage)

    async def handle_llm_retry(self, event: LLMRetryEvent) -> None:
        """Record an LLM retry attempt without counting a logical invocation."""
        _inc(
            self.collectors["ecs_agent_llm_retries_total"],
            {
                "provider": _safe_label(event.provider_id),
                "model": _safe_label(event.model),
                "reason": _safe_label(event.reason),
            },
        )

    def _record_usage(self, provider: str, model: str, usage: UsageRecord) -> None:
        token_values = {
            "prompt": usage.prompt_tokens,
            "completion": usage.completion_tokens,
            "total": usage.total_tokens,
            "cached_input": usage.cached_input_tokens,
            "cache_creation": usage.cache_creation_tokens,
            "cache_read": usage.cache_read_tokens,
        }
        for token_type, value in token_values.items():
            if value is None:
                continue
            _inc(
                self.collectors["ecs_agent_llm_tokens_total"],
                {
                    "provider": _safe_label(provider),
                    "model": _safe_label(model),
                    "token_type": token_type,
                },
                float(value),
            )

    async def handle_tool_execution_started(
        self, event: ToolExecutionStartedEvent
    ) -> None:
        """Accept tool start events for EventBus subscription completeness."""
        _ = event

    async def handle_tool_execution_completed(
        self, event: ToolExecutionCompletedEvent
    ) -> None:
        """Record tool execution outcome metrics."""
        labels = {
            "tool": _safe_label(event.tool_name),
            "status": _safe_label(event.status),
        }
        _inc(self.collectors["ecs_agent_tool_calls_total"], labels)
        _observe(
            self.collectors["ecs_agent_tool_call_duration_seconds"],
            labels,
            event.duration_seconds,
        )

    async def handle_tool_approved(self, event: ToolApprovedEvent) -> None:
        """Record approved tool attempts."""
        _inc(
            self.collectors["ecs_agent_tool_approved_total"],
            {"tool": _safe_label(event.tool_name), "policy": _safe_label(event.policy)},
        )

    async def handle_tool_denied(self, event: ToolDeniedEvent) -> None:
        """Record denied tool attempts."""
        _inc(
            self.collectors["ecs_agent_tool_denied_total"],
            {"tool": _safe_label(event.tool_name), "reason": _safe_label(event.reason)},
        )

    async def handle_error_occurred(self, event: ErrorOccurredEvent) -> None:
        """Record captured error outcomes without raw exception text."""
        _ = event.error
        _inc(
            self.collectors["ecs_agent_errors_total"],
            {"system": _safe_label(event.system_name), "error_type": "error"},
        )

    async def handle_stream_start(self, event: StreamStartEvent) -> None:
        """Record stream start events with label-safe lifecycle counts."""
        _ = event
        self._record_stream_event("start", "started")

    async def handle_stream_content_delta(self, event: StreamContentDeltaEvent) -> None:
        """Record stream content delta events without exporting raw delta text."""
        _ = event
        self._record_stream_event("delta", "observed")

    async def handle_stream_reasoning_delta(
        self, event: StreamReasoningDeltaEvent
    ) -> None:
        """Record stream reasoning delta events without exporting raw reasoning text."""
        _ = event
        self._record_stream_event("delta", "observed")

    async def handle_stream_end(self, event: StreamEndEvent) -> None:
        """Record stream end events."""
        status = _safe_label(event.status)
        lifecycle_event = "end" if status == "success" else "interrupted"
        self._record_stream_event(lifecycle_event, status)
        _observe(
            self.collectors["ecs_agent_stream_duration_seconds"],
            {
                "provider": _safe_label(event.provider_id),
                "model": _safe_label(event.model),
                "operation": _safe_label(event.operation),
                "status": status,
            },
            event.duration_seconds,
        )
        _observe(
            self.collectors["ecs_agent_stream_first_delta_seconds"],
            {
                "provider": _safe_label(event.provider_id),
                "model": _safe_label(event.model),
                "operation": _safe_label(event.operation),
                "status": status,
            },
            event.first_delta_seconds,
        )

    def _record_stream_event(self, event: str, status: str) -> None:
        _inc(
            self.collectors["ecs_agent_stream_events_total"],
            {"event": event, "status": status},
        )

    async def handle_delegation_started(self, event: DelegationStartedEvent) -> None:
        """Record subagent delegation start using bounded lifecycle labels."""
        _inc(
            self.collectors["ecs_agent_subagent_lifecycle_total"],
            {"phase": _safe_label(event.phase), "status": _safe_label(event.status)},
        )

    async def handle_delegation_completed(self, event: DelegationCompletedEvent) -> None:
        """Record subagent delegation completion using success/failure only."""
        _inc(
            self.collectors["ecs_agent_subagent_lifecycle_total"],
            {
                "phase": _safe_label(event.phase),
                "status": _safe_label(
                    event.status or ("succeeded" if event.success else "failed")
                ),
            },
        )

    async def handle_message_bus_published(
        self, event: MessageBusPublishedEvent
    ) -> None:
        """Record message bus publish without exporting raw topics."""
        _ = event
        self._record_message_bus("message", "publish")

    async def handle_message_bus_delivered(
        self, event: MessageBusDeliveredEvent
    ) -> None:
        """Record message bus delivery without exporting subscriber IDs."""
        _ = event
        self._record_message_bus("message", "deliver")

    async def handle_message_bus_timeout(self, event: MessageBusTimeoutEvent) -> None:
        """Record message bus timeout without exporting correlation IDs."""
        _ = event
        self._record_message_bus("message", "timeout")

    async def handle_message_bus_response(self, event: MessageBusResponseEvent) -> None:
        """Record message bus response without exporting correlation IDs."""
        _ = event
        self._record_message_bus("message", "response")

    def _record_message_bus(self, event: str, operation: str) -> None:
        _inc(
            self.collectors["ecs_agent_message_bus_events_total"],
            {"event": event, "operation": operation},
        )

    async def handle_checkpoint_created(self, event: CheckpointCreatedEvent) -> None:
        """Record checkpoint creation without exporting checkpoint IDs."""
        self._record_checkpoint(event.operation, event.status)

    async def handle_checkpoint_restored(self, event: CheckpointRestoredEvent) -> None:
        """Record checkpoint restore without exporting checkpoint IDs."""
        self._record_checkpoint(event.operation, event.status)

    def _record_checkpoint(self, operation: str, status: str) -> None:
        _inc(
            self.collectors["ecs_agent_checkpoint_operations_total"],
            {"operation": operation, "status": status},
        )

    async def handle_compaction_complete(self, event: CompactionCompleteEvent) -> None:
        """Record compaction completion without token counts as labels."""
        _inc(
            self.collectors["ecs_agent_compaction_operations_total"],
            {
                "operation": _safe_label(event.operation),
                "status": _safe_label(event.status),
            },
        )

    async def handle_mcts_node_scored(self, event: MCTSNodeScoredEvent) -> None:
        """Record MCTS score operations without node IDs or scores as labels."""
        _inc(
            self.collectors["ecs_agent_mcts_nodes_scored_total"],
            {"phase": _safe_label(event.phase), "status": _safe_label(event.status)},
        )

    async def handle_plan_step_completed(self, event: PlanStepCompletedEvent) -> None:
        """Record plan-step completion without step text or indexes as labels."""
        _inc(
            self.collectors["ecs_agent_plan_steps_total"],
            {
                "operation": _safe_label(event.operation),
                "status": _safe_label(event.status),
            },
        )

    async def handle_plan_revised(self, event: PlanRevisedEvent) -> None:
        """Record replanning operations without old/new step text as labels."""
        _inc(
            self.collectors["ecs_agent_plan_steps_total"],
            {
                "operation": _safe_label(event.operation),
                "status": _safe_label(event.status),
            },
        )

    async def handle_tool_result_cached(self, event: ToolResultCachedEvent) -> None:
        """Record tool-result cache writes without raw artifact paths."""
        _inc(
            self.collectors["ecs_agent_tool_result_cached_total"],
            {"status": _safe_label(event.status)},
        )


def _event_subscriptions(
    metrics: PrometheusMetrics,
) -> tuple[tuple[type[Any], Any], ...]:
    return (
        (RunStartedEvent, metrics.handle_run_started),
        (RunnerTickStartedEvent, metrics.handle_runner_tick_started),
        (RunnerTickCompletedEvent, metrics.handle_runner_tick_completed),
        (RunCompletedEvent, metrics.handle_run_completed),
        (SystemExecutionStartedEvent, metrics.handle_system_execution_started),
        (SystemExecutionCompletedEvent, metrics.handle_system_execution_completed),
        (LLMInvocationEvent, metrics.handle_llm_invocation),
        (LLMRetryEvent, metrics.handle_llm_retry),
        (ToolExecutionStartedEvent, metrics.handle_tool_execution_started),
        (ToolExecutionCompletedEvent, metrics.handle_tool_execution_completed),
        (ToolApprovedEvent, metrics.handle_tool_approved),
        (ToolDeniedEvent, metrics.handle_tool_denied),
        (ErrorOccurredEvent, metrics.handle_error_occurred),
        (StreamStartEvent, metrics.handle_stream_start),
        (StreamContentDeltaEvent, metrics.handle_stream_content_delta),
        (StreamReasoningDeltaEvent, metrics.handle_stream_reasoning_delta),
        (StreamEndEvent, metrics.handle_stream_end),
        (DelegationStartedEvent, metrics.handle_delegation_started),
        (DelegationCompletedEvent, metrics.handle_delegation_completed),
        (MessageBusPublishedEvent, metrics.handle_message_bus_published),
        (MessageBusDeliveredEvent, metrics.handle_message_bus_delivered),
        (MessageBusTimeoutEvent, metrics.handle_message_bus_timeout),
        (MessageBusResponseEvent, metrics.handle_message_bus_response),
        (CheckpointCreatedEvent, metrics.handle_checkpoint_created),
        (CheckpointRestoredEvent, metrics.handle_checkpoint_restored),
        (CompactionCompleteEvent, metrics.handle_compaction_complete),
        (MCTSNodeScoredEvent, metrics.handle_mcts_node_scored),
        (PlanStepCompletedEvent, metrics.handle_plan_step_completed),
        (PlanRevisedEvent, metrics.handle_plan_revised),
        (ToolResultCachedEvent, metrics.handle_tool_result_cached),
    )


@dataclass(slots=True)
class PrometheusConfig:
    """Configuration for the Prometheus observability plugin.

    ``port``/``addr`` apply only to the embedded metrics server enabled by
    ``start_server``; when left unset they resolve from
    ``ECS_AGENT_PROMETHEUS_PORT`` / ``ECS_AGENT_PROMETHEUS_ADDR`` and fall
    back to 9100 on ``0.0.0.0``.
    """

    registry: CollectorRegistry | None = None
    metric_contract: dict[str, MetricSpec] | None = None
    start_server: bool = False
    port: int | None = None
    addr: str | None = None
    propagate_to_children: bool = False

    def with_env(self, env: dict[str, str] | None = None) -> PrometheusConfig:
        """Return a copy with missing server fields loaded from env names."""
        source = os.environ if env is None else env
        port = self.port
        if port is None:
            raw_port = source.get("ECS_AGENT_PROMETHEUS_PORT")
            if raw_port is not None:
                port = int(raw_port)
        addr = self.addr
        if addr is None:
            addr = source.get("ECS_AGENT_PROMETHEUS_ADDR")
        return replace(self, port=port, addr=addr)


class PrometheusPlugin:
    """Prometheus metrics backend mounted as an observability plugin.

    Consumes raw EventBus events (the metric contract needs low-cardinality
    raw data, not redacted telemetry records) and optionally owns an embedded
    ``/metrics`` HTTP server for the lifetime of the installation.
    """

    def __init__(
        self,
        config: PrometheusConfig | None = None,
        *,
        metrics: PrometheusMetrics | None = None,
    ) -> None:
        self.name = "prometheus"
        self.config = PrometheusConfig() if config is None else config
        self.propagate_to_children = self.config.propagate_to_children
        self._metrics = metrics
        self._server_handle: MetricsServerHandle | None = None

    @property
    def metrics(self) -> PrometheusMetrics | None:
        """Return the metrics recorder once started (or the injected one)."""
        return self._metrics

    @property
    def server_handle(self) -> MetricsServerHandle | None:
        """Return the embedded metrics server handle while it is running."""
        return self._server_handle

    def telemetry_sink(self) -> TelemetrySink | None:
        """No record-pipeline capability; metrics consume raw events."""
        return None

    def event_subscriptions(self, world: Any) -> tuple[EventSubscription, ...]:
        """Return the metric recorder's event subscriptions."""
        _ = world
        return _event_subscriptions(self._require_metrics())

    async def start(self, world: Any) -> None:
        """Create the metrics recorder and optionally the embedded server."""
        _ = world
        self.config = self.config.with_env()
        self._require_metrics()
        if self.config.start_server and self._server_handle is None:
            self._server_handle = start_metrics_server(
                self.config.port if self.config.port is not None else 9100,
                addr=self.config.addr if self.config.addr is not None else "0.0.0.0",
                metrics=self._metrics,
            )

    async def flush(self) -> None:
        """Prometheus collectors are pull-based; nothing to flush."""

    async def shutdown(self) -> None:
        """Close the embedded metrics server when one was started."""
        if self._server_handle is not None:
            self._server_handle.close(timeout=5)
            self._server_handle = None

    def _require_metrics(self) -> PrometheusMetrics:
        if self._metrics is None:
            self._metrics = PrometheusMetrics(
                registry=self.config.registry,
                metric_contract=self.config.metric_contract,
            )
        return self._metrics


def _resolve_registry(metrics: PrometheusMetrics | CollectorRegistry | None) -> CollectorRegistry:
    if metrics is None:
        return PrometheusMetrics().registry
    if isinstance(metrics, PrometheusMetrics):
        return metrics.registry
    return metrics


def render_metrics(metrics: PrometheusMetrics | CollectorRegistry | None = None) -> bytes:
    """Render metrics from an isolated registry using Prometheus text format."""
    prometheus = _prometheus_client()
    rendered: bytes = prometheus.generate_latest(_resolve_registry(metrics))
    return rendered


def make_metrics_asgi_app(metrics: PrometheusMetrics | CollectorRegistry | None = None) -> Any:
    """Create a Prometheus ASGI app bound to the provided metrics registry."""
    prometheus = _prometheus_client()
    registry = _resolve_registry(metrics)

    async def app(scope: dict[str, Any], receive: Any, send: Any) -> None:
        _ = receive
        if scope.get("type") != "http":
            raise ValueError("Prometheus metrics ASGI app only supports HTTP scopes")
        body = prometheus.generate_latest(registry)
        await send(
            {
                "type": "http.response.start",
                "status": HTTPStatus.OK.value,
                "headers": [
                    (b"content-type", prometheus.CONTENT_TYPE_LATEST.encode()),
                    (b"content-length", str(len(body)).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})

    return app


def make_metrics_wsgi_app(metrics: PrometheusMetrics | CollectorRegistry | None = None) -> Any:
    """Create a Prometheus WSGI app bound to the provided metrics registry."""
    prometheus = _prometheus_client()
    registry = _resolve_registry(metrics)

    def app(environ: dict[str, Any], start_response: Any) -> list[bytes]:
        _ = environ
        body = prometheus.generate_latest(registry)
        start_response(
            "200 OK",
            [
                ("Content-Type", prometheus.CONTENT_TYPE_LATEST),
                ("Content-Length", str(len(body))),
            ],
        )
        return [body]

    return app


def _make_metrics_handler(registry: CollectorRegistry) -> type[BaseHTTPRequestHandler]:
    prometheus = _prometheus_client()

    class MetricsHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
            body = prometheus.generate_latest(registry)
            self.send_response(HTTPStatus.OK.value)
            self.send_header("Content-Type", prometheus.CONTENT_TYPE_LATEST)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_OPTIONS(self) -> None:  # noqa: N802 - stdlib handler API
            self.send_response(HTTPStatus.OK.value)
            self.send_header("Allow", "OPTIONS, GET")
            self.end_headers()

        def log_message(self, format: str, *args: object) -> None:
            _ = (format, args)

    return MetricsHandler


def start_metrics_server(
    port: int,
    *,
    addr: str = "0.0.0.0",
    metrics: PrometheusMetrics | CollectorRegistry | None = None,
) -> MetricsServerHandle:
    """Start a standalone Prometheus metrics HTTP server with cleanup helpers."""
    server = ThreadingHTTPServer(
        (addr, port), _make_metrics_handler(_resolve_registry(metrics))
    )
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return MetricsServerHandle(server=server, thread=thread)


__all__ = [
    "ALLOWED_LABELS",
    "FORBIDDEN_LABELS",
    "EVENT_SUBSCRIPTION_MATRIX",
    "METRIC_CONTRACT",
    "MetricsServerHandle",
    "MetricSpec",
    "MetricType",
    "PROMETHEUS_IMPORT_ERROR",
    "PrometheusConfig",
    "PrometheusMetrics",
    "PrometheusPlugin",
    "make_metrics_asgi_app",
    "make_metrics_wsgi_app",
    "render_metrics",
    "start_metrics_server",
    "validate_metric_labels",
]
