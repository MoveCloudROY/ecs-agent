"""TUI entrypoint for the plan-and-task example.

Usage (same environment variables as ``main.py``)::

    LLM_API_KEY=sk-... uv run python -m examples.e2e.plan_and_task.tui

Runs ``Runner.run`` and the Textual app concurrently on one asyncio loop:
the app renders event-bus traffic, the bridge feeds submitted input back
into the world's pending ``UserInputRequestedEvent`` future.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from ecs_agent.accounting import AccountingSubscriber
from ecs_agent.core import Runner
from ecs_agent.logging import configure_logging, get_logger
from examples.e2e.plan_and_task.billing import BillingSubscriber
from examples.e2e.plan_and_task.main import (
    build_model_from_env,
    build_plan_task_world,
    install_plan_task_langfuse_observability,
)
from examples.e2e.plan_and_task.tui.session import create_tui_session

logger = get_logger(__name__)

_WORKFLOW_BASE_DIR = Path(__file__).resolve().parents[1]


async def main() -> None:
    debug_mode = os.environ.get("DEBUG", "").lower() in ("1", "true")
    configure_logging(json_output=False, level="DEBUG" if debug_mode else None)

    llm_model = build_model_from_env()

    world, agent_id, _adapter_ref, runtime_state = await build_plan_task_world(
        model=llm_model,
        base_dir=_WORKFLOW_BASE_DIR,
        enable_tool_sink=True,
    )
    langfuse_handle = await install_plan_task_langfuse_observability(world)

    billing = BillingSubscriber()
    billing.subscribe(world.event_bus)
    accounting = AccountingSubscriber()
    accounting.subscribe(world.event_bus)

    session = create_tui_session(world, agent_id, runtime_state)

    max_ticks_env = os.environ.get("PLAN_TASK_MAX_AGENT_TICKS")
    max_ticks: int | None = int(max_ticks_env) if max_ticks_env else None

    async def run_world() -> None:
        runner = Runner()
        try:
            await runner.run(world, max_ticks=max_ticks)
        finally:
            if session.app.is_running:
                session.app.exit()

    try:
        async with asyncio.TaskGroup() as task_group:
            task_group.create_task(session.app.run_async())
            task_group.create_task(run_world())
    finally:
        if langfuse_handle is not None:
            await langfuse_handle.flush()
            await langfuse_handle.shutdown()
        billing.log_session_summary()


if __name__ == "__main__":
    asyncio.run(main())
