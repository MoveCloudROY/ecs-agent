"""Run an ecs-agent demo with a standalone Prometheus metrics endpoint.

This example is intended to be scraped by the Prometheus configuration in this
directory. It uses ``FakeModel`` by default, so it works without live LLM
credentials while still exercising runner, system, LLM, token, and terminal
metrics.
"""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Sequence

from ecs_agent.components import ConversationComponent, LLMComponent, TerminalComponent
from ecs_agent.core import Runner, World
from ecs_agent.logging import configure_logging
from ecs_agent.metrics import install_prometheus_metrics, start_metrics_server
from ecs_agent.providers import FakeModel
from ecs_agent.systems.reasoning import ReasoningSystem
from ecs_agent.types import CompletionResult, Message, Usage


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line options for the Prometheus demo."""
    parser = argparse.ArgumentParser(
        description="Expose ecs-agent Prometheus metrics for a local Prometheus scrape."
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
    return parser.parse_args(argv)


async def run_agent_once(world: World, iteration: int) -> None:
    """Create one demo entity and run it through the normal Runner path."""
    entity_id = world.create_entity()
    world.add_component(
        entity_id,
        LLMComponent(
            model=FakeModel(
                responses=[
                    CompletionResult(
                        message=Message(
                            role="assistant",
                            content=f"Prometheus demo response #{iteration}",
                        ),
                        usage=Usage(prompt_tokens=8, completion_tokens=6, total_tokens=14),
                    )
                ],
                model_id="prometheus-demo-fake-model",
            ),
            system_prompt="You are a Prometheus metrics demo agent.",
        ),
    )
    world.add_component(
        entity_id,
        ConversationComponent(
            messages=[Message(role="user", content=f"Generate demo metrics #{iteration}")]
        ),
    )

    await Runner().run(world, max_ticks=3)

    for terminal_entity_id, _ in list(world.query(TerminalComponent)):
        world.remove_component(terminal_entity_id, TerminalComponent)


async def main(argv: Sequence[str] | None = None) -> None:
    """Run the metrics endpoint and periodically generate demo samples."""
    args = parse_args(argv)
    configure_logging(json_output=False)

    world = World(name="prometheus-demo")
    metrics = install_prometheus_metrics(world)
    world.register_system(ReasoningSystem(priority=0), priority=0)

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
            await run_agent_once(world, iteration)
            print(f"Recorded demo agent run #{iteration}")
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
