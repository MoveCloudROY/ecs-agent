"""Plan-and-task E2E example entrypoint."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from ecs_agent.components import (
    ConversationComponent,
    LLMComponent,
    UserPromptConfigComponent,
)
from ecs_agent.core import Runner, World
from ecs_agent.logging import configure_logging, get_logger
from ecs_agent.providers import FakeProvider, OpenAIProvider
from ecs_agent.providers.config import ApiFormat, ProviderConfig
from ecs_agent.providers.protocol import LLMProvider
from ecs_agent.systems.error_handling import ErrorHandlingSystem
from ecs_agent.systems.memory import MemorySystem
from ecs_agent.systems.reasoning import ReasoningSystem
from ecs_agent.systems.tool_execution import ToolExecutionSystem
from ecs_agent.types import CompletionResult, Message

from examples.e2e.plan_and_task.artifacts import ArtifactAdapter
from examples.e2e.plan_and_task.commands import parse_command
from examples.e2e.plan_and_task.controller import PlanController
from examples.e2e.plan_and_task.runtime import (
    build_runtime_config,
    setup_interactive_input,
)
from examples.e2e.plan_and_task.state_models import RuntimeState
from examples.e2e.plan_and_task.state_machine import WorkflowStateMachine
from examples.e2e.plan_and_task.task_exec import TaskExec

logger = get_logger(__name__)

_WORKFLOW_BASE_DIR = Path(__file__).parent


def _format_status(payload: dict[str, object]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _require_state(state: RuntimeState | None) -> RuntimeState:
    if state is None:
        raise ValueError(
            "No active workflow state. Start with /plan:start <description>."
        )
    return state


def _handle_command(
    command_text: str,
    state: RuntimeState | None,
    controller: PlanController,
    adapter: ArtifactAdapter,
) -> RuntimeState | None:
    command = parse_command(command_text.strip())

    if command.name == "/plan:start":
        description = " ".join(command.args).strip()
        if not description:
            raise ValueError("/plan:start requires a non-empty description")
        next_state = controller.handle_plan_start(adapter, description)
        print(
            "Plan started:\n" + _format_status(controller.get_plan_status(next_state))
        )
        return next_state

    current_state = _require_state(state)

    if command.name == "/plan:status":
        print(_format_status(controller.get_plan_status(current_state)))
        return current_state

    if command.name == "/plan:finalize":
        next_state = controller.handle_plan_finalize(current_state, adapter)
        print(
            "Plan finalized:\n" + _format_status(controller.get_plan_status(next_state))
        )
        return next_state

    if command.name == "/task:start":
        task_exec = TaskExec(state=current_state)
        next_state = task_exec.initialize_task_queue(current_state, adapter)
        print(
            "Task queue initialized:\n"
            + _format_status(
                {
                    "workflow_id": next_state.workflow_id,
                    "phase": next_state.phase,
                    "status": next_state.status,
                    "current_task_id": next_state.current_task_id,
                    "task_count": len(next_state.tasks),
                }
            )
        )
        return next_state

    if command.name == "/task:status":
        print(
            _format_status(
                {
                    "workflow_id": current_state.workflow_id,
                    "phase": current_state.phase,
                    "status": current_state.status,
                    "current_task_id": current_state.current_task_id,
                    "tasks": [
                        {
                            "task_id": task.task_id,
                            "status": task.status,
                            "retry_count": task.retry_count,
                        }
                        for task in current_state.tasks
                    ],
                    "active_subagents": [
                        {
                            "session_id": record.session_id,
                            "task_id": record.task_id,
                            "status": record.status,
                        }
                        for record in current_state.active_subagents
                    ],
                }
            )
        )
        return current_state

    if command.name == "/task:resume":
        next_state = controller.handle_task_resume(current_state, adapter)
        print(
            "Task resumed:\n" + _format_status(controller.get_plan_status(next_state))
        )
        return next_state

    if command.name == "/task:replan":
        reason = " ".join(command.args).strip()
        if not reason:
            raise ValueError("/task:replan requires a non-empty reason")
        next_state = controller.handle_task_replan(current_state, adapter, reason)
        print(
            "Task replanned:\n" + _format_status(controller.get_plan_status(next_state))
        )
        return next_state

    if command.name == "/task:abort":
        next_state = controller.handle_task_abort(
            current_state,
            adapter,
            reason="user abort",
        )
        print(
            "Task aborted:\n" + _format_status(controller.get_plan_status(next_state))
        )
        return next_state

    raise ValueError(f"Unsupported command: {command.name}")


async def main() -> None:
    debug_mode = os.environ.get("DEBUG", "").lower() in ("1", "true")
    configure_logging(json_output=False, level="DEBUG" if debug_mode else None)

    if debug_mode:
        logger.info("debug_mode_enabled")

    world = World()

    api_key: str = os.environ.get("LLM_API_KEY", "")
    base_url: str = os.environ.get(
        "LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    model: str = os.environ.get("LLM_MODEL", "qwen3.5-flash")

    provider: LLMProvider
    if api_key:
        logger.info("using_provider", provider="OpenAIProvider", model=model)
        print(f"Using OpenAIProvider with model: {model}")
        provider = OpenAIProvider(
            config=ProviderConfig(
                provider_id="openai",
                base_url=base_url,
                api_key=api_key,
                api_format=ApiFormat.OPENAI_CHAT_COMPLETIONS,
            ),
            model=model,
        )
    else:
        logger.info("using_provider", provider="FakeProvider")
        print("No LLM_API_KEY set. Using FakeProvider for demonstration.")
        provider = FakeProvider(
            responses=[
                CompletionResult(
                    message=Message(
                        role="assistant",
                        content=(
                            "Plan-and-task example scaffold is active. "
                            "Use one of the supported slash commands to continue."
                        ),
                    )
                )
            ]
        )

    agent_id = world.create_entity()
    world.add_component(
        agent_id,
        LLMComponent(provider=provider, model=model if api_key else "fake"),
    )
    world.add_component(agent_id, ConversationComponent(messages=[]))
    world.add_component(agent_id, UserPromptConfigComponent(triggers=[]))

    world.register_system(ReasoningSystem(priority=0), priority=0)
    world.register_system(ToolExecutionSystem(priority=5), priority=5)
    world.register_system(MemorySystem(), priority=10)
    world.register_system(ErrorHandlingSystem(priority=99), priority=99)

    runtime_config = build_runtime_config()
    adapter = ArtifactAdapter(
        base_dir=_WORKFLOW_BASE_DIR,
        workflow_id=runtime_config.workflow_id,
    )
    controller = PlanController()
    state_machine = WorkflowStateMachine()
    runtime_state: list[RuntimeState | None] = [None]

    state_path = adapter.state_dir / "runtime_state.json"
    if state_path.exists():
        try:
            restored_state = adapter.read_state()
            runtime_state[0] = state_machine.handle_restart(restored_state, adapter)
        except ValueError as exc:
            logger.warning(
                "plan_task_restore_failed",
                workflow_id=runtime_config.workflow_id,
                exception=str(exc),
            )
            print(f"Error: {exc}")
        else:
            restored_runtime_state = _require_state(runtime_state[0])
            print(
                "Restored workflow state:\n"
                + _format_status(controller.get_plan_status(restored_runtime_state))
            )

    def handle_command(text: str) -> bool:
        try:
            runtime_state[0] = _handle_command(
                text,
                runtime_state[0],
                controller,
                adapter,
            )
            return True
        except ValueError as exc:
            print(f"Error: {exc}")
            return True

    interactive_mode_str = os.environ.get("PLAN_TASK_INTERACTIVE", "1")
    if interactive_mode_str.lower() in ("0", "false"):
        if debug_mode:
            logger.info("interactive_input_disabled", reason="env_var_set")
    else:
        if debug_mode:
            logger.info("interactive_input_enabled", reason="default_or_env_var_set")
        await setup_interactive_input(
            world,
            agent_id,
            workflow_id=runtime_config.workflow_id,
            command_handler=handle_command,
        )

    runner = Runner()
    await runner.run(world, max_ticks=None)

    conv = world.get_component(agent_id, ConversationComponent)
    if conv is not None:
        logger.info("conversation_complete", message_count=len(conv.messages))
        print("\nConversation:")
        for msg in conv.messages:
            print(f"  {msg.role}: {msg.content}")
    else:
        logger.warning("no_conversation_found")
        print("No conversation found")


if __name__ == "__main__":
    asyncio.run(main())
