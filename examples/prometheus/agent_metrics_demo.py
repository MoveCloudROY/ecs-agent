"""Run a real ecs-agent workflow with a Prometheus metrics endpoint."""

import argparse
import asyncio
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from urllib.parse import urlparse

from ecs_agent.components import (
    ConversationComponent,
    LLMComponent,
    TerminalComponent,
    ToolRegistryComponent,
)
from ecs_agent.core import Runner, World
from ecs_agent.logging import configure_logging
from ecs_agent.metrics import install_prometheus_metrics, start_metrics_server
from ecs_agent.providers import Model
from ecs_agent.providers.config import ApiFormat
from ecs_agent.systems.error_handling import ErrorHandlingSystem
from ecs_agent.systems.memory import MemorySystem
from ecs_agent.systems.reasoning import ReasoningSystem
from ecs_agent.systems.tool_execution import ToolExecutionSystem
from ecs_agent.providers.protocol import LLMModel
from ecs_agent.types import Message, ToolSchema


@dataclass(slots=True)
class LLMDemoConfig:
    """Configuration for the credentialed Prometheus demo model."""

    api_key: str = field(repr=False)
    base_url: str
    model_id: str
    api_format: ApiFormat
    provider_id: str
    connect_timeout: float
    read_timeout: float
    write_timeout: float
    pool_timeout: float


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line options for the Prometheus demo."""
    parser = argparse.ArgumentParser(
        description="Expose ecs-agent Prometheus metrics for a real tool-using agent."
    )
    parser.add_argument(
        "--metrics-addr",
        default="0.0.0.0",
        help="Address for the standalone /metrics server.",
    )
    parser.add_argument(
        "--metrics-port",
        type=int,
        default=9100,
        help="Port for the standalone /metrics server.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=5.0,
        help="Seconds to wait between demo agent runs.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=0,
        help="Number of demo runs before exit. Use 0 to run until Ctrl+C.",
    )
    parser.add_argument(
        "--llm-base-url",
        default=None,
        help="LLM API base URL. Defaults to LLM_BASE_URL or an OpenAI-compatible URL.",
    )
    parser.add_argument(
        "--llm-model",
        default=None,
        help="LLM model id. Defaults to LLM_MODEL or qwen3.5-flash.",
    )
    parser.add_argument(
        "--llm-provider",
        default=None,
        help="Provider label for metrics. Defaults to LLM_PROVIDER or LLM_Provider.",
    )
    parser.add_argument(
        "--llm-api-format",
        choices=[api_format.value for api_format in ApiFormat],
        default=None,
        help="Provider wire format. Defaults to LLM_API_FORMAT or URL-based inference.",
    )
    parser.add_argument(
        "--llm-connect-timeout",
        type=float,
        default=None,
        help="LLM connect timeout in seconds. Defaults to LLM_CONNECT_TIMEOUT or 10.",
    )
    parser.add_argument(
        "--llm-read-timeout",
        type=float,
        default=None,
        help="LLM read timeout in seconds. Defaults to LLM_READ_TIMEOUT or 120.",
    )
    parser.add_argument(
        "--llm-write-timeout",
        type=float,
        default=None,
        help="LLM write timeout in seconds. Defaults to LLM_WRITE_TIMEOUT or 10.",
    )
    parser.add_argument(
        "--llm-pool-timeout",
        type=float,
        default=None,
        help="LLM pool timeout in seconds. Defaults to LLM_POOL_TIMEOUT or 10.",
    )
    return parser.parse_args(argv)


def load_llm_config(args: argparse.Namespace) -> LLMDemoConfig:
    """Load real LLM configuration from CLI options and environment variables."""
    api_key = os.environ.get("LLM_API_KEY", "")
    if not api_key:
        raise ValueError("LLM_API_KEY is required to run the Prometheus LLM demo.")

    base_url = _option_or_env(
        args.llm_base_url,
        "LLM_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    ).rstrip("/")
    api_format = _resolve_api_format(args.llm_api_format, base_url)
    return LLMDemoConfig(
        api_key=api_key,
        base_url=base_url,
        model_id=_option_or_env(args.llm_model, "LLM_MODEL", "qwen3.5-flash"),
        api_format=api_format,
        provider_id=_resolve_provider_id(args.llm_provider, api_format, base_url),
        connect_timeout=_float_option_or_env(
            args.llm_connect_timeout, "LLM_CONNECT_TIMEOUT", 10.0
        ),
        read_timeout=_float_option_or_env(args.llm_read_timeout, "LLM_READ_TIMEOUT", 120.0),
        write_timeout=_float_option_or_env(
            args.llm_write_timeout, "LLM_WRITE_TIMEOUT", 10.0
        ),
        pool_timeout=_float_option_or_env(args.llm_pool_timeout, "LLM_POOL_TIMEOUT", 10.0),
    )


def _option_or_env(value: str | None, env_name: str, default: str) -> str:
    """Return a CLI string, environment string, or default value."""
    if value:
        return value
    return os.environ.get(env_name, default)


def _float_option_or_env(value: float | None, env_name: str, default: float) -> float:
    """Return a CLI float, environment float, or default value."""
    if value is not None:
        return value
    raw_value = os.environ.get(env_name)
    if raw_value is None:
        return default
    return float(raw_value)


def _resolve_provider_id(
    value: str | None,
    api_format: ApiFormat,
    base_url: str,
) -> str:
    """Resolve provider id with support for the legacy mixed-case variable."""
    if value:
        return value
    env_value = os.environ.get("LLM_PROVIDER") or os.environ.get("LLM_Provider")
    if env_value:
        return env_value

    host = urlparse(base_url).hostname or ""
    if "deepseek" in host:
        return "deepseek"
    if "dashscope" in host or "aliyuncs" in host:
        return "aliyun"
    if api_format is ApiFormat.ANTHROPIC_MESSAGES:
        return "anthropic"
    return "openai"


def _resolve_api_format(value: str | None, base_url: str) -> ApiFormat:
    """Resolve explicit API format or infer it from known compatible URLs."""
    raw_value = value or os.environ.get("LLM_API_FORMAT")
    if raw_value:
        return ApiFormat(raw_value)

    normalized_url = base_url.lower().rstrip("/")
    if normalized_url.endswith("/anthropic"):
        return ApiFormat.ANTHROPIC_MESSAGES
    if "/api/v2/apps/protocols/compatible-mode/v1" in normalized_url:
        return ApiFormat.OPENAI_RESPONSES
    return ApiFormat.OPENAI_CHAT_COMPLETIONS


async def summarize_metrics(metric_family: str, focus: str) -> str:
    """Return a deterministic local summary for a metrics family."""
    return f"{metric_family}: demo sample is healthy for {focus}."


async def classify_metric(metric_family: str) -> str:
    """Return a deterministic category for a metrics family."""
    if "tool" in metric_family:
        return "tool instrumentation"
    if "llm" in metric_family:
        return "model instrumentation"
    return "runtime instrumentation"


def create_llm_model(config: LLMDemoConfig) -> LLMModel:
    """Create the configured real LLM model for the demo workflow."""
    return Model(
        config.model_id,
        base_url=config.base_url,
        api_key=config.api_key,
        api_format=config.api_format,
        provider_id=config.provider_id,
        connect_timeout=config.connect_timeout,
        read_timeout=config.read_timeout,
        write_timeout=config.write_timeout,
        pool_timeout=config.pool_timeout,
        enable_store=config.api_format is ApiFormat.OPENAI_RESPONSES,
    )


async def run_agent_once(world: World, model: LLMModel, iteration: int) -> None:
    """Create one real model-backed tool agent and run it through Runner."""
    entity_id = world.create_entity()
    world.add_component(
        entity_id,
        LLMComponent(
            model=model,
            system_prompt=(
                "You are a Prometheus metrics demo agent. Before giving the final "
                "answer, call summarize_metrics and classify_metric using the "
                "available tools. Keep the final answer concise."
            ),
        ),
    )
    world.add_component(
        entity_id,
        ConversationComponent(
            messages=[
                Message(
                    role="user",
                    content=(
                        f"Demo run #{iteration}: inspect ecs_agent_llm_invocations_total "
                        "with focus latency, then explain what the dashboard should show."
                    ),
                )
            ]
        ),
    )
    world.add_component(entity_id, _build_tool_registry())

    try:
        await Runner().run(world, max_ticks=5)
    finally:
        world.delete_entity(entity_id)
        for terminal_entity_id, _ in list(world.query(TerminalComponent)):
            world.remove_component(terminal_entity_id, TerminalComponent)


def _build_tool_registry() -> ToolRegistryComponent:
    """Build safe local tools for the Prometheus demo workflow."""
    return ToolRegistryComponent(
        tools={
            "summarize_metrics": ToolSchema(
                name="summarize_metrics",
                description="Summarize a local Prometheus metric family for the demo.",
                parameters={
                    "type": "object",
                    "properties": {
                        "metric_family": {
                            "type": "string",
                            "description": "ecs-agent metric family name to summarize.",
                        },
                        "focus": {
                            "type": "string",
                            "description": "Operational concern to focus on.",
                        },
                    },
                    "required": ["metric_family", "focus"],
                },
            ),
            "classify_metric": ToolSchema(
                name="classify_metric",
                description="Classify a local Prometheus metric family by dashboard area.",
                parameters={
                    "type": "object",
                    "properties": {
                        "metric_family": {
                            "type": "string",
                            "description": "ecs-agent metric family name to classify.",
                        },
                    },
                    "required": ["metric_family"],
                },
            ),
        },
        handlers={
            "summarize_metrics": summarize_metrics,
            "classify_metric": classify_metric,
        },
    )


async def main(argv: Sequence[str] | None = None) -> None:
    """Run the metrics endpoint and periodically generate real LLM samples."""
    args = parse_args(argv)
    config = load_llm_config(args)
    configure_logging(json_output=False)
    model = create_llm_model(config)

    world = World(name="prometheus-demo")
    metrics = install_prometheus_metrics(world)
    world.register_system(ReasoningSystem(priority=0), priority=0)
    world.register_system(ToolExecutionSystem(priority=5), priority=5)
    world.register_system(MemorySystem(), priority=10)
    world.register_system(ErrorHandlingSystem(priority=99), priority=99)

    handle = start_metrics_server(
        args.metrics_port,
        addr=args.metrics_addr,
        metrics=metrics,
    )
    print(
        f"ecs-agent metrics are available at "
        f"http://{args.metrics_addr}:{args.metrics_port}/metrics"
    )
    if args.metrics_addr == "0.0.0.0":
        print(f"Local scrape URL: http://127.0.0.1:{args.metrics_port}/metrics")
    print("Start Prometheus from this directory with: docker compose up")

    iteration = 1
    try:
        while args.iterations <= 0 or iteration <= args.iterations:
            await run_agent_once(world, model, iteration)
            print(f"Recorded real LLM demo agent run #{iteration}")
            iteration += 1
            if args.iterations > 0 and iteration > args.iterations:
                break
            await asyncio.sleep(args.interval)
    except KeyboardInterrupt:
        print("Stopping Prometheus metrics demo.")
    finally:
        handle.close(timeout=5)


if __name__ == "__main__":
    asyncio.run(main())
