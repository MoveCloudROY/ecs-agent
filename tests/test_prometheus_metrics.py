"""Prometheus metrics package surface and contract tests."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Protocol

import pytest
from prometheus_client import REGISTRY
from prometheus_client.registry import CollectorRegistry

from ecs_agent.core import World


class _ProcessSystem(Protocol):
    async def process(self, world: World) -> None: ...


EXPECTED_METRIC_NAMES = {
    "ecs_agent_runs_total",
    "ecs_agent_runner_ticks_total",
    "ecs_agent_runner_tick_duration_seconds",
    "ecs_agent_system_executions_total",
    "ecs_agent_system_execution_duration_seconds",
    "ecs_agent_active_entities",
    "ecs_agent_llm_invocations_total",
    "ecs_agent_llm_invocation_duration_seconds",
    "ecs_agent_llm_tokens_total",
    "ecs_agent_llm_retries_total",
    "ecs_agent_tool_calls_total",
    "ecs_agent_tool_call_duration_seconds",
    "ecs_agent_tool_denied_total",
    "ecs_agent_tool_approved_total",
    "ecs_agent_errors_total",
    "ecs_agent_terminals_total",
    "ecs_agent_stream_events_total",
    "ecs_agent_stream_first_delta_seconds",
    "ecs_agent_stream_duration_seconds",
    "ecs_agent_subagent_lifecycle_total",
    "ecs_agent_message_bus_events_total",
    "ecs_agent_checkpoint_operations_total",
    "ecs_agent_compaction_operations_total",
    "ecs_agent_mcts_nodes_scored_total",
    "ecs_agent_plan_steps_total",
    "ecs_agent_tool_result_cached_total",
}

EXPECTED_ALLOWED_LABELS = {
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

EXPECTED_FORBIDDEN_LABELS = {
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
    "tool_args",
    "tool_results",
    "exception",
    "artifact_path",
    "api_key",
    "token",
}


def test_metrics_public_surface_importable_from_package_and_root() -> None:
    """Prometheus API names are importable from both metrics and root packages."""
    from ecs_agent import PrometheusMetrics as RootPrometheusMetrics
    from ecs_agent import install_prometheus_metrics as root_install_prometheus_metrics
    from ecs_agent import uninstall_prometheus_metrics as root_uninstall_prometheus_metrics
    from ecs_agent.metrics import (
        PrometheusMetrics,
        install_prometheus_metrics,
        make_metrics_asgi_app,
        make_metrics_wsgi_app,
        render_metrics,
        start_metrics_server,
        uninstall_prometheus_metrics,
    )

    assert RootPrometheusMetrics is PrometheusMetrics
    assert root_install_prometheus_metrics is install_prometheus_metrics
    assert root_uninstall_prometheus_metrics is uninstall_prometheus_metrics
    assert PrometheusMetrics.__name__ == "PrometheusMetrics"
    assert callable(install_prometheus_metrics)
    assert callable(uninstall_prometheus_metrics)
    assert callable(render_metrics)
    assert callable(make_metrics_asgi_app)
    assert callable(make_metrics_wsgi_app)
    assert callable(start_metrics_server)


def test_metric_contract_freezes_names_types_and_allowed_labels() -> None:
    """Metric contract encodes exact names, types, and low-cardinality labels."""
    from ecs_agent.metrics import ALLOWED_LABELS, FORBIDDEN_LABELS, METRIC_CONTRACT

    assert set(METRIC_CONTRACT) == EXPECTED_METRIC_NAMES
    assert ALLOWED_LABELS == frozenset(EXPECTED_ALLOWED_LABELS)
    assert EXPECTED_FORBIDDEN_LABELS <= FORBIDDEN_LABELS

    for spec in METRIC_CONTRACT.values():
        assert spec.name in EXPECTED_METRIC_NAMES
        assert spec.metric_type in {"counter", "histogram", "gauge"}
        assert set(spec.labels) <= ALLOWED_LABELS
        assert set(spec.labels).isdisjoint(FORBIDDEN_LABELS)


def test_prometheus_metrics_uses_private_registry_and_renders_bytes() -> None:
    """Creating a metrics surface does not mutate the global Prometheus registry."""
    from ecs_agent.metrics import PrometheusMetrics, render_metrics

    before_global_names = {sample.name for sample in REGISTRY.collect()}

    metrics = PrometheusMetrics()
    output = render_metrics(metrics)

    after_global_names = {sample.name for sample in REGISTRY.collect()}
    assert before_global_names == after_global_names
    assert isinstance(output, bytes)
    assert b"ecs_agent_runs_total" in output
    assert b"entity_id" not in output


def _sample(
    registry: CollectorRegistry, name: str, labels: dict[str, str] | None = None
) -> float | None:
    return registry.get_sample_value(name, labels)


def _count_sample(
    registry: CollectorRegistry,
    name: str,
    labels: dict[str, str] | None = None,
) -> float | None:
    return _sample(registry, f"{name}_total", labels)


def test_forbidden_labels_are_rejected_by_contract_helper() -> None:
    """ID-class and sensitive labels cannot enter metric definitions."""
    from ecs_agent.metrics import validate_metric_labels

    validate_metric_labels(("system", "status"))


    with pytest.raises(ValueError, match="entity_id"):
        validate_metric_labels(("system", "entity_id"))

    with pytest.raises(ValueError, match="raw_prompt_text"):
        validate_metric_labels(("raw_prompt_text",))


def test_installed_metrics_can_be_rendered_without_global_registry() -> None:
    """Install helper returns the same isolated metrics surface render_metrics accepts."""
    from ecs_agent.metrics import PrometheusMetrics, install_prometheus_metrics, render_metrics

    metrics = install_prometheus_metrics()

    assert isinstance(metrics, PrometheusMetrics)
    assert b"ecs_agent_active_entities" in render_metrics(metrics)


async def test_prometheus_metrics_direct_handlers_update_core_llm_and_tool_metrics() -> None:
    """Direct recorder handlers update counters, histograms, and gauges."""
    from ecs_agent.accounting.models import UsageRecord
    from ecs_agent.metrics import PrometheusMetrics
    from ecs_agent.types import (
        EntityId,
        LLMInvocationEvent,
        LLMRetryEvent,
        RunCompletedEvent,
        RunnerTickCompletedEvent,
        SystemExecutionCompletedEvent,
        ToolApprovedEvent,
        ToolDeniedEvent,
        ToolExecutionCompletedEvent,
    )

    metrics = PrometheusMetrics()

    await metrics.handle_runner_tick_completed(
        RunnerTickCompletedEvent(
            tick=1,
            status="success",
            duration_seconds=0.25,
            active_entities=3,
        )
    )
    await metrics.handle_run_completed(
        RunCompletedEvent(
            status="terminal_component",
            reason="terminal_component",
            duration_seconds=1.0,
            ticks=1,
            active_entities=2,
        )
    )
    await metrics.handle_system_execution_completed(
        SystemExecutionCompletedEvent(
            system="ReasoningSystem",
            status="success",
            duration_seconds=0.5,
        )
    )
    await metrics.handle_llm_invocation(
        LLMInvocationEvent(
            entity_id=1,
            provider_id="openai",
            model="gpt-4o-mini",
            operation="completion",
            status="success",
            streaming=True,
            duration_seconds=0.75,
            usage=UsageRecord(prompt_tokens=11, completion_tokens=7, total_tokens=18),
        )
    )
    await metrics.handle_llm_retry(
        LLMRetryEvent(provider_id="openai", model="gpt-4o-mini", reason="http_429", attempt=2)
    )
    await metrics.handle_tool_execution_completed(
        ToolExecutionCompletedEvent(
            entity_id=EntityId(1),
            tool_call_id="call_123",
            tool_name="read_file",
            result="ignored raw text",
            success=True,
            duration_seconds=0.125,
        )
    )
    await metrics.handle_tool_approved(
        ToolApprovedEvent(
            entity_id=EntityId(1),
            tool_call_id="call_123",
            tool_name="read_file",
            policy="require_approval",
        )
    )
    await metrics.handle_tool_denied(
        ToolDeniedEvent(
            entity_id=EntityId(1),
            tool_call_id="call_456",
            tool_name="write_file",
            reason="policy_denied",
        )
    )

    registry = metrics.registry
    assert registry.get_sample_value(
        "ecs_agent_runner_ticks_total", {"status": "success"}
    ) == 1.0
    assert registry.get_sample_value("ecs_agent_active_entities") == 2.0
    assert registry.get_sample_value(
        "ecs_agent_runs_total", {"status": "terminal_component"}
    ) == 1.0
    assert registry.get_sample_value(
        "ecs_agent_terminals_total", {"reason": "terminal_component"}
    ) == 1.0
    assert registry.get_sample_value(
        "ecs_agent_system_executions_total",
        {"system": "ReasoningSystem", "status": "success"},
    ) == 1.0
    assert registry.get_sample_value(
        "ecs_agent_llm_invocations_total",
        {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "operation": "completion",
            "status": "success",
            "streaming": "true",
        },
    ) == 1.0
    assert registry.get_sample_value(
        "ecs_agent_llm_tokens_total",
        {"provider": "openai", "model": "gpt-4o-mini", "token_type": "prompt"},
    ) == 11.0
    assert registry.get_sample_value(
        "ecs_agent_llm_retries_total",
        {"provider": "openai", "model": "gpt-4o-mini", "reason": "http_429"},
    ) == 1.0
    assert registry.get_sample_value(
        "ecs_agent_tool_calls_total", {"tool": "read_file", "status": "success"}
    ) == 1.0
    assert registry.get_sample_value(
        "ecs_agent_tool_approved_total",
        {"tool": "read_file", "policy": "require_approval"},
    ) == 1.0
    assert registry.get_sample_value(
        "ecs_agent_tool_denied_total", {"tool": "write_file", "reason": "policy_denied"}
    ) == 1.0


async def test_install_prometheus_metrics_binds_world_event_bus_idempotently() -> None:
    """Installing metrics twice on a world does not double-subscribe handlers."""
    from ecs_agent.core import World
    from ecs_agent.metrics import install_prometheus_metrics
    from ecs_agent.types import RunnerTickCompletedEvent

    world = World()

    first = install_prometheus_metrics(world)
    second = install_prometheus_metrics(world)

    assert second is first

    await world.event_bus.publish(
        RunnerTickCompletedEvent(
            tick=1,
            status="success",
            duration_seconds=0.1,
            active_entities=4,
        )
    )

    assert first.registry.get_sample_value(
        "ecs_agent_runner_ticks_total", {"status": "success"}
    ) == 1.0


async def test_uninstall_prometheus_metrics_removes_subscriptions_and_is_idempotent() -> None:
    """Uninstall removes recorder subscriptions and is safe to call repeatedly."""
    from ecs_agent.core import World
    from ecs_agent.metrics import install_prometheus_metrics, uninstall_prometheus_metrics
    from ecs_agent.types import RunnerTickCompletedEvent

    world = World()
    metrics = install_prometheus_metrics(world)

    await world.event_bus.publish(
        RunnerTickCompletedEvent(
            tick=1,
            status="success",
            duration_seconds=0.1,
            active_entities=4,
        )
    )
    assert metrics.registry.get_sample_value(
        "ecs_agent_runner_ticks_total", {"status": "success"}
    ) == 1.0

    removed = uninstall_prometheus_metrics(world)
    removed_again = uninstall_prometheus_metrics(world)

    assert removed is metrics
    assert removed_again is None

    await world.event_bus.publish(
        RunnerTickCompletedEvent(
            tick=2,
            status="success",
            duration_seconds=0.1,
            active_entities=4,
        )
    )

    assert metrics.registry.get_sample_value(
        "ecs_agent_runner_ticks_total", {"status": "success"}
    ) == 1.0


async def test_reinstall_prometheus_metrics_after_uninstall_counts_once() -> None:
    """Reinstalling after uninstall creates one active subscription set."""
    from ecs_agent.core import World
    from ecs_agent.metrics import install_prometheus_metrics, uninstall_prometheus_metrics
    from ecs_agent.types import RunnerTickCompletedEvent

    world = World()
    first = install_prometheus_metrics(world)
    assert uninstall_prometheus_metrics(world) is first

    second = install_prometheus_metrics(world)
    third = install_prometheus_metrics(world)

    assert second is third
    assert second is not first

    await world.event_bus.publish(
        RunnerTickCompletedEvent(
            tick=1,
            status="success",
            duration_seconds=0.1,
            active_entities=4,
        )
    )

    assert first.registry.get_sample_value(
        "ecs_agent_runner_ticks_total", {"status": "success"}
    ) is None
    assert second.registry.get_sample_value(
        "ecs_agent_runner_ticks_total", {"status": "success"}
    ) == 1.0


async def test_metrics_subscriber_failure_is_swallowed_by_event_bus() -> None:
    """A failing metrics subscriber does not prevent other agent subscribers from running."""
    from ecs_agent.core import World
    from ecs_agent.types import RunnerTickCompletedEvent

    world = World()

    async def broken_handler(event: RunnerTickCompletedEvent) -> None:
        _ = event
        raise RuntimeError("metrics backend unavailable")

    observed: list[int] = []

    async def agent_handler(event: RunnerTickCompletedEvent) -> None:
        observed.append(event.tick)

    world.event_bus.subscribe(RunnerTickCompletedEvent, broken_handler)
    world.event_bus.subscribe(RunnerTickCompletedEvent, agent_handler)

    await world.event_bus.publish(
        RunnerTickCompletedEvent(
            tick=9,
            status="error",
            duration_seconds=0.1,
            active_entities=1,
        )
    )

    assert observed == [9]


async def test_installed_metrics_update_from_offline_reasoning_tool_run() -> None:
    """A representative offline agent run updates runtime, LLM, tool, and terminal metrics."""
    from ecs_agent.components import ConversationComponent, LLMComponent, ToolRegistryComponent
    from ecs_agent.core import Runner, World
    from ecs_agent.metrics import install_prometheus_metrics, render_metrics
    from ecs_agent.providers import FakeModel
    from ecs_agent.systems.reasoning import ReasoningSystem
    from ecs_agent.systems.tool_execution import ToolExecutionSystem
    from ecs_agent.types import CompletionResult, Message, ToolCall, ToolSchema, Usage

    world = World()
    metrics = install_prometheus_metrics(world)

    async def lookup(city: str) -> str:
        return f"weather for {city}"

    tool_call = ToolCall(id="call_raw_123", name="lookup_weather", arguments={"city": "Paris"})
    model = FakeModel(
        responses=[
            CompletionResult(
                message=Message(role="assistant", content="", tool_calls=[tool_call]),
                usage=Usage(prompt_tokens=3, completion_tokens=2, total_tokens=5),
            ),
            CompletionResult(
                message=Message(role="assistant", content="sunny"),
                usage=Usage(prompt_tokens=4, completion_tokens=1, total_tokens=5),
            ),
        ],
        model_id="offline-model",
    )
    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(model=model))
    world.add_component(entity_id, ConversationComponent(messages=[Message(role="user", content="weather?")]))
    world.add_component(
        entity_id,
        ToolRegistryComponent(
            tools={
                "lookup_weather": ToolSchema(
                    name="lookup_weather",
                    description="Lookup weather",
                    parameters={},
                )
            },
            handlers={"lookup_weather": lookup},
        ),
    )
    world.register_system(ReasoningSystem(), priority=0)
    world.register_system(ToolExecutionSystem(), priority=1)

    await Runner().run(world, max_ticks=4)

    registry = metrics.registry
    assert _count_sample(registry, "ecs_agent_runs", {"status": "terminal_component"}) == 1.0
    assert _count_sample(registry, "ecs_agent_runner_ticks", {"status": "success"}) == 1.0
    assert _count_sample(registry, "ecs_agent_runner_ticks", {"status": "terminal_component"}) == 1.0
    assert _count_sample(
        registry,
        "ecs_agent_system_executions",
        {"system": "ecs_agent.systems.reasoning.ReasoningSystem", "status": "success"},
    ) == 2.0
    assert _count_sample(
        registry,
        "ecs_agent_llm_invocations",
        {
            "provider": "FakeModel",
            "model": "offline-model",
            "operation": "reasoning",
            "status": "success",
            "streaming": "false",
        },
    ) == 2.0
    assert _count_sample(
        registry,
        "ecs_agent_tool_calls",
        {"tool": "lookup_weather", "status": "success"},
    ) == 1.0
    assert _sample(registry, "ecs_agent_active_entities") is not None

    output = render_metrics(metrics)
    assert b"call_raw_123" not in output
    assert b"weather for Paris" not in output


async def test_framework_owned_non_reasoning_llm_paths_emit_logical_invocations() -> None:
    """Planning, replanning, tree search, and compaction model calls publish invocation events."""
    from ecs_agent.components import (
        CompactionConfigComponent,
        ConversationComponent,
        LLMComponent,
        PlanComponent,
        PlanSearchComponent,
    )
    from ecs_agent.core import World
    from ecs_agent.metrics import install_prometheus_metrics
    from ecs_agent.providers import FakeModel
    from ecs_agent.systems.compaction import CompactionSystem
    from ecs_agent.systems.planning import PlanningSystem
    from ecs_agent.systems.replanning import ReplanningSystem
    from ecs_agent.systems.tree_search import TreeSearchSystem
    from ecs_agent.types import CompletionResult, Message, Usage

    async def run_with_model(
        system: _ProcessSystem, components: list[object]
    ) -> CollectorRegistry:
        world = World()
        metrics = install_prometheus_metrics(world)
        entity_id = world.create_entity()
        for component in components:
            world.add_component(entity_id, component)
        await system.process(world)
        return metrics.registry

    planning_registry = await run_with_model(
        PlanningSystem(),
        [
            LLMComponent(
                model=FakeModel(
                    [CompletionResult(Message(role="assistant", content="planned"), usage=Usage(total_tokens=1))],
                    model_id="plan-model",
                )
            ),
            ConversationComponent([Message(role="user", content="plan")]),
            PlanComponent(["step"]),
        ],
    )
    assert _count_sample(
        planning_registry,
        "ecs_agent_llm_invocations",
        {"provider": "FakeModel", "model": "plan-model", "operation": "planning", "status": "success", "streaming": "false"},
    ) == 1.0

    replanning_registry = await run_with_model(
        ReplanningSystem(),
        [
            LLMComponent(
                model=FakeModel(
                    [CompletionResult(Message(role="assistant", content='{"revised_steps": ["next"]}'))],
                    model_id="replan-model",
                )
            ),
            ConversationComponent([Message(role="user", content="goal"), Message(role="assistant", content="done")]),
            PlanComponent(["done", "old"], current_step=1),
        ],
    )
    assert _count_sample(
        replanning_registry,
        "ecs_agent_llm_invocations",
        {"provider": "FakeModel", "model": "replan-model", "operation": "replanning", "status": "success", "streaming": "false"},
    ) == 1.0

    tree_registry = await run_with_model(
        TreeSearchSystem(),
        [
            LLMComponent(
                model=FakeModel(
                    [
                        CompletionResult(Message(role="assistant", content="action")),
                        CompletionResult(Message(role="assistant", content="0.7")),
                    ],
                    model_id="tree-model",
                )
            ),
            ConversationComponent([Message(role="user", content="search")]),
            PlanSearchComponent(max_depth=1, max_branching=1),
        ],
    )
    assert _count_sample(
        tree_registry,
        "ecs_agent_llm_invocations",
        {"provider": "FakeModel", "model": "tree-model", "operation": "tree_search_expand", "status": "success", "streaming": "false"},
    ) == 1.0
    assert _count_sample(
        tree_registry,
        "ecs_agent_llm_invocations",
        {"provider": "FakeModel", "model": "tree-model", "operation": "tree_search_simulate", "status": "success", "streaming": "false"},
    ) == 1.0

    compaction_registry = await run_with_model(
        CompactionSystem(),
        [
            LLMComponent(
                model=FakeModel(
                    [CompletionResult(Message(role="assistant", content="summary"))],
                    model_id="compact-model",
                )
            ),
            ConversationComponent(
                [
                    Message(role="user", content="one two three four five"),
                    Message(role="assistant", content="six seven eight nine ten"),
                ]
            ),
            CompactionConfigComponent(threshold_tokens=1),
        ],
    )
    assert _count_sample(
        compaction_registry,
        "ecs_agent_llm_invocations",
        {"provider": "FakeModel", "model": "compact-model", "operation": "compaction", "status": "success", "streaming": "false"},
    ) == 1.0


async def test_retry_model_emits_retry_without_double_counting_logical_invocation() -> None:
    """Retry attempts increment retry metrics while the caller emits one logical invocation."""
    import httpx

    from ecs_agent.components import ConversationComponent, LLMComponent
    from ecs_agent.core import World
    from ecs_agent.metrics import install_prometheus_metrics
    from ecs_agent.providers.retry_model import RetryModel
    from ecs_agent.systems.reasoning import ReasoningSystem
    from ecs_agent.types import CompletionResult, Message, RetryConfig

    class FlakyProvider:
        model_id = "retry-model"
        provider_id = "flaky-provider"

        def __init__(self) -> None:
            self.calls = 0

        async def complete(self, messages: list[Message], tools=None, stream: bool = False, response_format=None):
            _ = (messages, tools, stream, response_format)
            self.calls += 1
            if self.calls == 1:
                request = httpx.Request("POST", "https://example.test")
                response = httpx.Response(429, request=request)
                raise httpx.HTTPStatusError("rate limited", request=request, response=response)
            return CompletionResult(Message(role="assistant", content="ok"))

    world = World()
    metrics = install_prometheus_metrics(world)
    provider = FlakyProvider()
    entity_id = world.create_entity()
    world.add_component(
        entity_id,
        LLMComponent(model=RetryModel(provider, RetryConfig(max_attempts=2, min_wait=0.0, max_wait=0.0))),
    )
    world.add_component(entity_id, ConversationComponent([Message(role="user", content="hello")]))

    await ReasoningSystem().process(world)

    registry = metrics.registry
    assert provider.calls == 2
    assert _count_sample(
        registry,
        "ecs_agent_llm_retries",
        {"provider": "flaky-provider", "model": "retry-model", "reason": "http_429"},
    ) == 1.0
    assert _count_sample(
        registry,
        "ecs_agent_llm_invocations",
        {"provider": "flaky-provider", "model": "retry-model", "operation": "reasoning", "status": "success", "streaming": "false"},
    ) == 1.0


async def test_streaming_metrics_capture_first_delta_duration_and_interruption() -> None:
    """Streaming events expose lifecycle counts plus first-delta and duration observations."""
    import asyncio
    from ecs_agent.components import ConversationComponent, InterruptionComponent, LLMComponent, StreamingComponent
    from ecs_agent.core import World
    from ecs_agent.metrics import install_prometheus_metrics
    from ecs_agent.providers import FakeModel
    from ecs_agent.systems.reasoning import ReasoningSystem
    from ecs_agent.types import (
        CompletionResult,
        InterruptionReason,
        Message,
        StreamDelta,
    )

    class TwoChunkStreamingModel(FakeModel):
        async def _stream_complete(self, result: CompletionResult) -> AsyncIterator[StreamDelta]:
            _ = result
            yield StreamDelta(content="a")
            yield StreamDelta(content="b")

    success_world = World()
    success_metrics = install_prometheus_metrics(success_world)
    success_entity = success_world.create_entity()
    success_world.add_component(
        success_entity,
        LLMComponent(model=TwoChunkStreamingModel([CompletionResult(Message(role="assistant", content="ab"))], model_id="stream-model")),
    )
    success_world.add_component(success_entity, ConversationComponent([Message(role="user", content="stream")]))
    success_world.add_component(success_entity, StreamingComponent(enabled=True))

    await ReasoningSystem().process(success_world)

    success_registry = success_metrics.registry
    stream_labels = {"provider": "TwoChunkStreamingModel", "model": "stream-model", "operation": "reasoning", "status": "success"}
    assert _count_sample(success_registry, "ecs_agent_stream_events", {"event": "start", "status": "started"}) == 1.0
    assert _count_sample(success_registry, "ecs_agent_stream_events", {"event": "delta", "status": "observed"}) == 2.0
    assert _count_sample(success_registry, "ecs_agent_stream_events", {"event": "end", "status": "success"}) == 1.0
    assert _sample(success_registry, "ecs_agent_stream_first_delta_seconds_count", stream_labels) == 1.0
    assert _sample(success_registry, "ecs_agent_stream_duration_seconds_count", stream_labels) == 1.0

    class InterruptingStreamingModel(FakeModel):
        def __init__(self, world: World, entity_id) -> None:
            super().__init__([CompletionResult(Message(role="assistant", content="partial"))], model_id="interrupt-model")
            self._world = world
            self._entity_id = entity_id

        async def _stream_complete(self, result: CompletionResult) -> AsyncIterator[StreamDelta]:
            _ = result
            yield StreamDelta(content="partial")
            self._world.add_component(
                self._entity_id,
                InterruptionComponent(reason=InterruptionReason.USER_REQUESTED, message="stop"),
            )
            await asyncio.sleep(0)
            yield StreamDelta(content="ignored")

    interrupted_world = World()
    interrupted_metrics = install_prometheus_metrics(interrupted_world)
    interrupted_entity = interrupted_world.create_entity()
    interrupted_world.add_component(interrupted_entity, ConversationComponent([Message(role="user", content="stream")]))
    interrupted_world.add_component(interrupted_entity, StreamingComponent(enabled=True))
    interrupted_world.add_component(
        interrupted_entity,
        LLMComponent(model=InterruptingStreamingModel(interrupted_world, interrupted_entity)),
    )

    with pytest.raises(asyncio.CancelledError):
        await ReasoningSystem().process(interrupted_world)

    interrupted_registry = interrupted_metrics.registry
    interrupted_labels = {"provider": "InterruptingStreamingModel", "model": "interrupt-model", "operation": "reasoning", "status": "cancelled"}
    interrupted_success_labels = {"provider": "InterruptingStreamingModel", "model": "interrupt-model", "operation": "reasoning", "status": "success"}
    assert _count_sample(interrupted_registry, "ecs_agent_stream_events", {"event": "interrupted", "status": "cancelled"}) == 1.0
    assert _sample(interrupted_registry, "ecs_agent_stream_first_delta_seconds_count", interrupted_labels) == 1.0
    assert _sample(interrupted_registry, "ecs_agent_stream_first_delta_seconds_count", interrupted_success_labels) is None
    assert _sample(interrupted_registry, "ecs_agent_stream_duration_seconds_count", interrupted_labels) == 1.0


async def test_non_blocking_stream_delta_events_carry_bounded_metadata_and_first_delta() -> None:
    """Non-blocking stream delta events carry the same bounded timing metadata as blocking events."""
    from ecs_agent.components import ConversationComponent, LLMComponent, StreamingComponent
    from ecs_agent.providers import FakeModel
    from ecs_agent.systems.reasoning import ReasoningSystem
    from ecs_agent.types import (
        CompletionResult,
        Message,
        StreamContentDeltaEvent,
        StreamDelta,
        StreamReasoningDeltaEvent,
    )

    class ReasoningThenContentStreamingModel(FakeModel):
        async def _stream_complete(
            self, result: CompletionResult
        ) -> AsyncIterator[StreamDelta]:
            _ = result
            yield StreamDelta(reasoning_content="think")
            yield StreamDelta(content="answer")

    world = World()
    reasoning_deltas: list[StreamReasoningDeltaEvent] = []
    content_deltas: list[StreamContentDeltaEvent] = []

    async def record_reasoning(event: StreamReasoningDeltaEvent) -> None:
        reasoning_deltas.append(event)

    async def record_content(event: StreamContentDeltaEvent) -> None:
        content_deltas.append(event)

    world.event_bus.subscribe(StreamReasoningDeltaEvent, record_reasoning)
    world.event_bus.subscribe(StreamContentDeltaEvent, record_content)
    entity_id = world.create_entity()
    world.add_component(
        entity_id,
        LLMComponent(
            model=ReasoningThenContentStreamingModel(
                [CompletionResult(Message(role="assistant", content="answer"))],
                model_id="metadata-stream-model",
            )
        ),
    )
    world.add_component(
        entity_id, ConversationComponent([Message(role="user", content="stream")])
    )
    world.add_component(
        entity_id,
        StreamingComponent(enabled=True, non_blocking_delta_publish=True),
    )

    await ReasoningSystem().process(world)
    for _ in range(10):
        if reasoning_deltas and content_deltas:
            break
        await asyncio.sleep(0)

    assert len(reasoning_deltas) == 1
    assert len(content_deltas) == 1
    first_reasoning_delta = reasoning_deltas[0]
    assert first_reasoning_delta.provider_id == "ReasoningThenContentStreamingModel"
    assert first_reasoning_delta.model == "metadata-stream-model"
    assert first_reasoning_delta.operation == "reasoning"
    assert first_reasoning_delta.first_delta_seconds is not None
    assert first_reasoning_delta.first_delta_seconds >= 0.0

    content_delta = content_deltas[0]
    assert content_delta.provider_id == "ReasoningThenContentStreamingModel"
    assert content_delta.model == "metadata-stream-model"
    assert content_delta.operation == "reasoning"
    assert content_delta.first_delta_seconds is None


async def test_runtime_control_handlers_use_conservative_label_safe_counts() -> None:
    """Runtime-control event handlers count bounded operations without raw IDs or text labels."""
    from ecs_agent.metrics import PrometheusMetrics, render_metrics
    from ecs_agent.types import (
        CheckpointCreatedEvent,
        CompactionCompleteEvent,
        DelegationCompletedEvent,
        DelegationStartedEvent,
        EntityId,
        MCTSNodeScoredEvent,
        MessageBusDeliveredEvent,
        MessageBusEnvelope,
        MessageBusPublishedEvent,
        MessageBusResponseEvent,
        MessageBusTimeoutEvent,
        PlanStepCompletedEvent,
        ToolResultCachedEvent,
    )

    metrics = PrometheusMetrics()
    entity_id = EntityId(1)
    envelope = MessageBusEnvelope(
        id="message-raw-id",
        source="agent-a",
        type="com.example.raw.topic",
        specversion="1.0",
        correlationid="corr-raw-id",
        traceparent="00-raw-trace-raw-parent-01",
    )

    await metrics.handle_delegation_started(
        DelegationStartedEvent(
            entity_id=entity_id,
            subagent_name="researcher",
            task="raw prompt text must not be exported",
            correlation_id="corr-raw-id",
            traceparent="trace-raw-id",
        )
    )
    await metrics.handle_delegation_completed(
        DelegationCompletedEvent(
            entity_id=entity_id,
            subagent_name="researcher",
            result="raw result text must not be exported",
            success=False,
            error="raw exception text",
        )
    )
    await metrics.handle_message_bus_published(
        MessageBusPublishedEvent(entity_id=entity_id, envelope=envelope, topic="user.raw.topic")
    )
    await metrics.handle_message_bus_delivered(
        MessageBusDeliveredEvent(entity_id=entity_id, subscriber_id=EntityId(2), envelope=envelope)
    )
    await metrics.handle_message_bus_timeout(MessageBusTimeoutEvent(entity_id=entity_id, correlation_id="corr"))
    await metrics.handle_message_bus_response(
        MessageBusResponseEvent(entity_id=entity_id, correlation_id="corr", envelope=envelope)
    )
    await metrics.handle_checkpoint_created(
        CheckpointCreatedEvent(entity_id=entity_id, checkpoint_id=99, timestamp=1.0)
    )
    await metrics.handle_compaction_complete(
        CompactionCompleteEvent(entity_id=entity_id, original_tokens=100, compacted_tokens=30)
    )
    await metrics.handle_mcts_node_scored(MCTSNodeScoredEvent(entity_id=entity_id, node_id=42, score=0.9))
    await metrics.handle_plan_step_completed(
        PlanStepCompletedEvent(
            entity_id=entity_id,
            step_index=3,
            step_description="raw step text must not be exported",
        )
    )
    await metrics.handle_tool_result_cached(
        ToolResultCachedEvent(
            entity_id=entity_id,
            tool_call_id="call-raw-id",
            artifact_path="/tmp/raw/path",
        )
    )

    registry = metrics.registry
    assert registry.get_sample_value(
        "ecs_agent_subagent_lifecycle_total", {"phase": "running", "status": "running"}
    ) == 1.0
    assert registry.get_sample_value(
        "ecs_agent_subagent_lifecycle_total", {"phase": "completed", "status": "failed"}
    ) == 1.0
    assert registry.get_sample_value(
        "ecs_agent_message_bus_events_total", {"event": "message", "operation": "publish"}
    ) == 1.0
    assert registry.get_sample_value(
        "ecs_agent_message_bus_events_total", {"event": "message", "operation": "deliver"}
    ) == 1.0
    assert registry.get_sample_value(
        "ecs_agent_checkpoint_operations_total", {"operation": "save", "status": "success"}
    ) == 1.0
    assert registry.get_sample_value(
        "ecs_agent_compaction_operations_total", {"operation": "compact", "status": "success"}
    ) == 1.0
    assert registry.get_sample_value(
        "ecs_agent_mcts_nodes_scored_total", {"phase": "score", "status": "success"}
    ) == 1.0
    assert registry.get_sample_value(
        "ecs_agent_plan_steps_total", {"operation": "execute", "status": "success"}
    ) == 1.0
    assert registry.get_sample_value(
        "ecs_agent_tool_result_cached_total", {"status": "cached"}
    ) == 1.0

    output = render_metrics(metrics)
    assert b"raw prompt text" not in output
    assert b"raw result text" not in output
    assert b"user.raw.topic" not in output
    assert b"call-raw-id" not in output
    assert b"/tmp/raw/path" not in output


async def test_runtime_control_metrics_increment_from_representative_offline_paths(
    tmp_path,
) -> None:
    """Runtime-control counters update from real offline subsystem flows."""
    from ecs_agent.components import (
        CheckpointComponent,
        CompactionConfigComponent,
        ContextTrimConfig,
        ConversationComponent,
        LLMComponent,
        PendingToolCallsComponent,
        PlanComponent,
        PlanSearchComponent,
        SubagentRegistryComponent,
        ToolRegistryComponent,
    )
    from ecs_agent.core import World
    from ecs_agent.metrics import install_prometheus_metrics, render_metrics
    from ecs_agent.providers import FakeModel
    from ecs_agent.scratchbook import ArtifactRegistry
    from ecs_agent.systems.checkpoint import CheckpointSystem
    from ecs_agent.systems.compaction import CompactionSystem
    from ecs_agent.systems.message_bus import MessageBusSystem
    from ecs_agent.systems.planning import PlanningSystem
    from ecs_agent.systems.replanning import ReplanningSystem
    from ecs_agent.systems.subagent import SubagentSystem
    from ecs_agent.systems.tool_execution import ToolExecutionSystem
    from ecs_agent.systems.tree_search import TreeSearchSystem
    from ecs_agent.types import CompletionResult, Message, SubagentConfig, ToolCall, ToolSchema

    world = World()
    metrics = install_prometheus_metrics(world)

    parent = world.create_entity()
    world.add_component(
        parent,
        SubagentRegistryComponent(
            subagents={
                "worker": SubagentConfig(
                    name="worker",
                    model=FakeModel(
                        [CompletionResult(Message(role="assistant", content="child done"))],
                        model_id="raw-subagent-model",
                    ),
                )
            }
        ),
    )
    world.add_component(parent, ToolRegistryComponent(tools={}, handlers={}))
    subagent_system = SubagentSystem()
    await subagent_system.process(world)
    subagent_tools = world.get_component(parent, ToolRegistryComponent)
    assert subagent_tools is not None
    await subagent_tools.handlers["subagent"](
        category="worker",
        prompt="raw subagent prompt must not appear",
    )

    bus = MessageBusSystem(publish_timeout=0.1, request_timeout=0.01)
    bus.subscribe(topic="raw.topic.with.ids", subscriber_id="subscriber-raw-id")
    await bus.process(world)
    await bus.publish(
        topic="raw.topic.with.ids",
        message={"payload": "raw message text must not appear"},
    )
    with pytest.raises(TimeoutError):
        await bus.request(
            topic="raw.topic.without.response",
            message={"payload": "raw request text must not appear"},
            timeout=0.01,
        )

    checkpoint_world = World()
    install_prometheus_metrics(checkpoint_world, metrics=metrics)
    checkpoint_entity = checkpoint_world.create_entity()
    checkpoint_world.add_component(checkpoint_entity, CheckpointComponent())
    checkpoint_world.add_component(
        checkpoint_entity,
        ConversationComponent([Message(role="user", content="checkpoint raw content")]),
    )
    await CheckpointSystem().process(checkpoint_world)
    await CheckpointSystem.undo(checkpoint_world, providers={}, tool_handlers={})

    async def tool_handler() -> str:
        return "tool raw result " * 30

    tool_entity = world.create_entity()
    world.add_component(
        tool_entity,
        ConversationComponent([Message(role="user", content="cache raw conversation")]),
    )
    world.add_component(
        tool_entity,
        PendingToolCallsComponent(
            [ToolCall(id="tool-call-raw-id", name="big_tool", arguments={})]
        ),
    )
    world.add_component(
        tool_entity,
        ToolRegistryComponent(
            tools={"big_tool": ToolSchema(name="big_tool", description="", parameters={})},
            handlers={"big_tool": tool_handler},
        ),
    )
    world.add_component(
        tool_entity,
        ContextTrimConfig(max_tokens=1, token_estimation_chars_per_token=1.0),
    )
    await ToolExecutionSystem(registry=ArtifactRegistry(tmp_path / "scratchbook")).process(world)

    planning_entity = world.create_entity()
    world.add_component(
        planning_entity,
        LLMComponent(
            model=FakeModel(
                [CompletionResult(Message(role="assistant", content="plan answer"))],
                model_id="planning-raw-model",
            )
        ),
    )
    world.add_component(
        planning_entity,
        ConversationComponent([Message(role="user", content="plan raw objective")]),
    )
    world.add_component(planning_entity, PlanComponent(["raw step text must not appear"]))
    await PlanningSystem().process(world)

    replanning_entity = world.create_entity()
    world.add_component(
        replanning_entity,
        LLMComponent(
            model=FakeModel(
                [CompletionResult(Message(role="assistant", content='{"revised_steps": ["raw revised step"]}'))],
                model_id="replanning-raw-model",
            )
        ),
    )
    world.add_component(
        replanning_entity,
        ConversationComponent(
            [Message(role="user", content="replan raw objective"), Message(role="assistant", content="done")]
        ),
    )
    world.add_component(replanning_entity, PlanComponent(["done", "raw old step"], current_step=1))
    await ReplanningSystem().process(world)

    tree_entity = world.create_entity()
    world.add_component(
        tree_entity,
        LLMComponent(
            model=FakeModel(
                [
                    CompletionResult(Message(role="assistant", content="raw action text")),
                    CompletionResult(Message(role="assistant", content="0.42")),
                ],
                model_id="tree-raw-model",
            )
        ),
    )
    world.add_component(
        tree_entity,
        ConversationComponent([Message(role="user", content="tree raw objective")]),
    )
    world.add_component(tree_entity, PlanSearchComponent(max_depth=1, max_branching=1))
    await TreeSearchSystem().process(world)

    compact_entity = world.create_entity()
    world.add_component(
        compact_entity,
        LLMComponent(
            model=FakeModel(
                [CompletionResult(Message(role="assistant", content="raw compact summary"))],
                model_id="compact-raw-model",
            )
        ),
    )
    world.add_component(
        compact_entity,
        ConversationComponent(
            [Message(role="user", content="one two three"), Message(role="assistant", content="four five six")]
        ),
    )
    world.add_component(compact_entity, CompactionConfigComponent(threshold_tokens=1))
    await CompactionSystem().process(world)

    registry = metrics.registry
    assert _count_sample(registry, "ecs_agent_subagent_lifecycle", {"phase": "running", "status": "running"}) == 1.0
    assert _count_sample(registry, "ecs_agent_subagent_lifecycle", {"phase": "completed", "status": "succeeded"}) == 1.0
    assert _count_sample(registry, "ecs_agent_message_bus_events", {"event": "message", "operation": "publish"}) == 2.0
    assert _count_sample(registry, "ecs_agent_message_bus_events", {"event": "message", "operation": "deliver"}) == 1.0
    assert _count_sample(registry, "ecs_agent_message_bus_events", {"event": "message", "operation": "timeout"}) == 1.0
    assert _count_sample(registry, "ecs_agent_checkpoint_operations", {"operation": "save", "status": "success"}) == 1.0
    assert _count_sample(registry, "ecs_agent_checkpoint_operations", {"operation": "restore", "status": "success"}) == 1.0
    assert _count_sample(registry, "ecs_agent_tool_result_cached", {"status": "cached"}) == 1.0
    assert _count_sample(registry, "ecs_agent_plan_steps", {"operation": "execute", "status": "success"}) == 1.0
    assert _count_sample(registry, "ecs_agent_plan_steps", {"operation": "revise", "status": "success"}) == 1.0
    assert _count_sample(registry, "ecs_agent_mcts_nodes_scored", {"phase": "score", "status": "success"}) == 1.0
    assert _count_sample(registry, "ecs_agent_compaction_operations", {"operation": "compact", "status": "success"}) == 1.0

    output = render_metrics(metrics)
    forbidden = [
        b"raw.topic.with.ids",
        b"raw.topic.without.response",
        b"subscriber-raw-id",
        b"checkpoint raw content",
        b"tool-call-raw-id",
        b"scratchbook/records/tool",
        b"raw step text",
        b"raw old step",
        b"raw revised step",
        b"raw action text",
        b"0.42",
        b"raw compact summary",
        b"raw subagent prompt",
    ]
    for raw_value in forbidden:
        assert raw_value not in output


async def test_runtime_control_events_preserve_backward_compatibility_with_safe_defaults() -> None:
    """New low-cardinality event metadata is optional for existing publishers."""
    from ecs_agent.metrics import PrometheusMetrics
    from ecs_agent.types import (
        CheckpointCreatedEvent,
        CompactionCompleteEvent,
        EntityId,
        MCTSNodeScoredEvent,
        PlanStepCompletedEvent,
        PlanRevisedEvent,
        ToolResultCachedEvent,
    )

    metrics = PrometheusMetrics()
    entity_id = EntityId(7)

    await metrics.handle_checkpoint_created(
        CheckpointCreatedEvent(entity_id=entity_id, checkpoint_id=123, timestamp=1.0)
    )
    await metrics.handle_compaction_complete(
        CompactionCompleteEvent(entity_id=entity_id, original_tokens=99, compacted_tokens=3)
    )
    await metrics.handle_mcts_node_scored(
        MCTSNodeScoredEvent(entity_id=entity_id, node_id=456, score=0.87)
    )
    await metrics.handle_plan_step_completed(
        PlanStepCompletedEvent(entity_id=entity_id, step_index=2, step_description="raw text")
    )
    await metrics.handle_plan_revised(
        PlanRevisedEvent(entity_id=entity_id, old_steps=["raw old"], new_steps=["raw new"])
    )
    await metrics.handle_tool_result_cached(
        ToolResultCachedEvent(entity_id=entity_id, tool_call_id="raw-call", artifact_path="/raw/path")
    )

    registry = metrics.registry
    assert _count_sample(registry, "ecs_agent_checkpoint_operations", {"operation": "save", "status": "success"}) == 1.0
    assert _count_sample(registry, "ecs_agent_compaction_operations", {"operation": "compact", "status": "success"}) == 1.0
    assert _count_sample(registry, "ecs_agent_mcts_nodes_scored", {"phase": "score", "status": "success"}) == 1.0
    assert _count_sample(registry, "ecs_agent_plan_steps", {"operation": "execute", "status": "success"}) == 1.0
    assert _count_sample(registry, "ecs_agent_plan_steps", {"operation": "revise", "status": "success"}) == 1.0
    assert _count_sample(registry, "ecs_agent_tool_result_cached", {"status": "cached"}) == 1.0
