"""Prometheus demo behavior and file layout tests."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

from ecs_agent.providers.config import ApiFormat
from ecs_agent.accounting.instrumentation import resolve_provider_id
from ecs_agent.components import LLMComponent
from ecs_agent.core import World
from ecs_agent.metrics import install_prometheus_metrics, render_metrics
from ecs_agent.systems.reasoning import ReasoningSystem
from ecs_agent.systems.tool_execution import ToolExecutionSystem
from ecs_agent.types import CompletionResult, Message, ToolCall


REPO_ROOT = Path(__file__).resolve().parents[1]
DEMO_ROOT = REPO_ROOT / "examples" / "prometheus"


def _load_demo_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "agent_metrics_demo", DEMO_ROOT / "agent_metrics_demo.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_demo_requires_real_llm_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """The metrics demo is a real LLM workflow and refuses missing credentials."""
    demo = _load_demo_module()
    monkeypatch.delenv("LLM_API_KEY", raising=False)

    with pytest.raises(ValueError, match="LLM_API_KEY"):
        demo.load_llm_config(demo.parse_args([]))


def test_demo_infers_anthropic_messages_from_anthropic_base_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Anthropic-compatible endpoints are selected without requiring extra flags."""
    demo = _load_demo_module()
    monkeypatch.setenv("LLM_API_KEY", "secret-test-key")
    monkeypatch.setenv("LLM_BASE_URL", "https://api.deepseek.com/anthropic")
    monkeypatch.setenv("LLM_MODEL", "deepseek-v4-flash")
    monkeypatch.delenv("LLM_API_FORMAT", raising=False)

    config = demo.load_llm_config(demo.parse_args([]))

    assert config.api_format is ApiFormat.ANTHROPIC_MESSAGES
    assert config.model_id == "deepseek-v4-flash"
    assert config.base_url == "https://api.deepseek.com/anthropic"
    assert config.provider_id == "deepseek"
    assert "secret-test-key" not in repr(config)


def test_demo_infers_openai_responses_from_dashscope_responses_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DashScope Responses-compatible URLs select the Responses adapter."""
    demo = _load_demo_module()
    monkeypatch.setenv("LLM_API_KEY", "secret-test-key")
    monkeypatch.setenv(
        "LLM_BASE_URL",
        "https://dashscope.aliyuncs.com/api/v2/apps/protocols/compatible-mode/v1",
    )
    monkeypatch.setenv("LLM_MODEL", "qwen3.5-flash")
    monkeypatch.setenv("LLM_Provider", "Aliyun")
    monkeypatch.delenv("LLM_API_FORMAT", raising=False)

    config = demo.load_llm_config(demo.parse_args([]))

    assert config.api_format is ApiFormat.OPENAI_RESPONSES
    assert config.provider_id == "Aliyun"


def test_demo_model_exposes_provider_label_for_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The configured provider label reaches LLM accounting and metrics labels."""
    demo = _load_demo_module()
    monkeypatch.setenv("LLM_API_KEY", "secret-test-key")
    monkeypatch.setenv("LLM_PROVIDER", "Aliyun")

    config = demo.load_llm_config(demo.parse_args([]))
    model = demo.create_llm_model(config)

    assert resolve_provider_id(model) == "Aliyun"


def test_demo_source_does_not_use_fake_model() -> None:
    """The Prometheus demo should exercise real provider instrumentation."""
    source = (DEMO_ROOT / "agent_metrics_demo.py").read_text(encoding="utf-8")

    assert "FakeModel" not in source
    assert "from ecs_agent.providers import Model" in source


def test_prometheus_demo_compose_uses_organized_config_paths() -> None:
    """Compose mounts Prometheus and Grafana files from tidy subdirectories."""
    compose = (DEMO_ROOT / "docker-compose.yml").read_text(encoding="utf-8")

    assert "./prometheus/prometheus.yml:/etc/prometheus/prometheus.yml:ro" in compose
    assert "./prometheus.yml:/etc/prometheus/prometheus.yml:ro" not in compose
    assert (DEMO_ROOT / "prometheus" / "prometheus.yml").is_file()
    assert (DEMO_ROOT / "grafana" / "provisioning" / "datasources" / "datasource.yml").is_file()
    assert not (DEMO_ROOT / "grafana" / "provisioning" / "datasources" / "prometheus.yml").exists()


def test_prometheus_demo_readme_documents_real_llm_workflow() -> None:
    """README explains real credentialed LLM runs instead of offline fake runs."""
    readme = (DEMO_ROOT / "README.md").read_text(encoding="utf-8")

    assert "FakeModel" not in readme
    assert "LLM_API_KEY" in readme
    assert "LLM_API_FORMAT=anthropic_messages" in readme
    assert "LLM_API_FORMAT=openai_responses" in readme


@pytest.mark.asyncio
async def test_demo_run_deletes_completed_entity_between_iterations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repeated demo iterations do not reprocess completed prior agents."""
    demo = _load_demo_module()
    world = World(name="test-prometheus-demo")
    world.register_system(ReasoningSystem(priority=0), priority=0)
    model = _CountingModel()
    await demo.run_agent_once(world, model, iteration=1)
    await demo.run_agent_once(world, model, iteration=2)

    assert model.complete_count == 2
    assert world.query(LLMComponent) == []


@pytest.mark.asyncio
async def test_demo_tool_workflow_records_tool_metrics() -> None:
    """The demo workflow exercises ToolExecutionSystem metrics with real tool schemas."""
    demo = _load_demo_module()
    world = World(name="test-prometheus-demo-tools")
    metrics = install_prometheus_metrics(world)
    world.register_system(ReasoningSystem(priority=0), priority=0)
    world.register_system(ToolExecutionSystem(priority=5), priority=5)
    model = _ToolCallingModel()

    await demo.run_agent_once(world, model, iteration=1)

    output = render_metrics(metrics)
    assert model.complete_count == 2
    assert b'ecs_agent_tool_calls_total{status="success",tool="summarize_metrics"} 1.0' in output
    assert world.query(LLMComponent) == []


class _CountingModel:
    """Minimal model stub that counts real runner invocations."""

    model_id = "counting-model"
    provider_id = "test-provider"

    def __init__(self) -> None:
        self.complete_count = 0

    async def complete(
        self,
        messages: list[Message],
        tools: list[object] | None = None,
    ) -> CompletionResult:
        _ = messages
        _ = tools
        self.complete_count += 1
        return CompletionResult(
            message=Message(role="assistant", content=f"done {self.complete_count}")
        )


class _ToolCallingModel:
    """Minimal model stub that requests one local demo tool before final answer."""

    model_id = "tool-calling-model"
    provider_id = "test-provider"

    def __init__(self) -> None:
        self.complete_count = 0

    async def complete(
        self,
        messages: list[Message],
        tools: list[object] | None = None,
    ) -> CompletionResult:
        _ = messages
        _ = tools
        self.complete_count += 1
        if self.complete_count == 1:
            return CompletionResult(
                message=Message(
                    role="assistant",
                    content="Inspecting metrics with a tool.",
                    tool_calls=[
                        ToolCall(
                            id="call-summary",
                            name="summarize_metrics",
                            arguments={
                                "metric_family": "ecs_agent_llm_invocations_total",
                                "focus": "latency",
                            },
                        )
                    ],
                )
            )
        return CompletionResult(
            message=Message(role="assistant", content="Tool metrics recorded.")
        )
