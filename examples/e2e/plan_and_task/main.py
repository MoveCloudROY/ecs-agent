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
    SubagentRegistryComponent,
    SubagentSessionTableComponent,
    ToolRegistryComponent,
    UserPromptConfigComponent,
)
from ecs_agent.core import Runner, World
from ecs_agent.logging import configure_logging, get_logger
from ecs_agent.prompts.contracts import PromptTemplateSource, SystemPromptConfigSpec
from ecs_agent.providers import OpenAIProvider
from ecs_agent.providers.config import ApiFormat, ProviderConfig
from ecs_agent.providers.protocol import LLMProvider
from ecs_agent.systems.error_handling import ErrorHandlingSystem
from ecs_agent.systems.memory import MemorySystem
from ecs_agent.systems.reasoning import ReasoningSystem
from ecs_agent.systems.subagent import SubagentSystem
from ecs_agent.systems.system_prompt_render_system import SystemPromptRenderSystem
from ecs_agent.systems.tool_execution import ToolExecutionSystem
from ecs_agent.systems.user_prompt_normalization_system import (
    UserPromptNormalizationSystem,
)
from ecs_agent.types import EntityId, InheritancePolicy, SubagentConfig, ToolSchema

from examples.e2e.plan_and_task.artifacts import ArtifactAdapter
from examples.e2e.plan_and_task.commands import parse_command
from examples.e2e.plan_and_task.controller import PlanController
from examples.e2e.plan_and_task.prompts import (
    PLAN_INTERVIEW_SYSTEM_PROMPT,
    build_advisor_prompt,
    build_qa_prompt,
)
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


def build_plan_task_world(
    provider: LLMProvider,
    model: str,
    workflow_id: str,
    base_dir: Path | None = None,
) -> tuple[World, EntityId, ArtifactAdapter, list[RuntimeState | None]]:
    world = World()

    agent_id = world.create_entity()
    world.add_component(
        agent_id,
        LLMComponent(provider=provider, model=model),
    )
    world.add_component(agent_id, ConversationComponent(messages=[]))
    world.add_component(
        agent_id,
        SystemPromptConfigSpec(
            template_source=PromptTemplateSource(inline=PLAN_INTERVIEW_SYSTEM_PROMPT)
        ),
    )
    world.add_component(agent_id, UserPromptConfigComponent(triggers=[]))
    world.add_component(agent_id, ToolRegistryComponent(tools={}, handlers={}))
    world.add_component(agent_id, SubagentSessionTableComponent(sessions={}))
    world.add_component(
        agent_id,
        SubagentRegistryComponent(
            subagents={
                "advisor": SubagentConfig(
                    name="advisor",
                    provider=provider,
                    model=model,
                    description="Reviews workflow drafts as an advisor.",
                    system_prompt=build_advisor_prompt("<current draft content>"),
                    max_ticks=5,
                    inheritance_policy=InheritancePolicy(
                        inherit_system_prompt=False,
                        inherit_tools=[],
                    ),
                ),
                "qa": SubagentConfig(
                    name="qa",
                    provider=provider,
                    model=model,
                    description="Performs QA review of workflow drafts.",
                    system_prompt=build_qa_prompt(
                        "<current draft content>",
                        "<advisor verdict>",
                    ),
                    max_ticks=5,
                    inheritance_policy=InheritancePolicy(
                        inherit_system_prompt=False,
                        inherit_tools=[],
                    ),
                ),
            }
        ),
    )

    adapter = ArtifactAdapter(
        base_dir=base_dir or _WORKFLOW_BASE_DIR,
        workflow_id=workflow_id,
    )
    controller = PlanController()
    runtime_state: list[RuntimeState | None] = [None]

    tool_registry = world.get_component(agent_id, ToolRegistryComponent)
    if tool_registry is None:
        raise ValueError("ToolRegistryComponent is required for plan-and-task world")

    tool_registry.tools["record_advisor_verdict"] = ToolSchema(
        name="record_advisor_verdict",
        description="Record the advisor review verdict for the current workflow draft.",
        parameters={
            "verdict": {
                "type": "string",
                "description": "Advisor verdict: approved, revise, or blocked.",
            },
            "notes": {
                "type": "string",
                "description": "Optional notes from the advisor review.",
                "required": False,
            },
        },
    )

    async def record_advisor_verdict(verdict: str, notes: str = "") -> str:
        runtime_state[0] = controller.handle_advisor_review(
            _require_state(runtime_state[0]),
            adapter,
            verdict,
            notes=notes or None,
        )
        return f"Advisor verdict '{verdict}' recorded."

    tool_registry.handlers["record_advisor_verdict"] = record_advisor_verdict

    tool_registry.tools["record_qa_verdict"] = ToolSchema(
        name="record_qa_verdict",
        description="Record the QA review verdict for the current workflow draft.",
        parameters={
            "verdict": {
                "type": "string",
                "description": "QA verdict: approved, revise, or blocked.",
            },
            "notes": {
                "type": "string",
                "description": "Optional notes from the QA review.",
                "required": False,
            },
        },
    )

    async def record_qa_verdict(verdict: str, notes: str = "") -> str:
        runtime_state[0] = controller.handle_qa_review(
            _require_state(runtime_state[0]),
            adapter,
            verdict,
            notes=notes or None,
        )
        return f"QA verdict '{verdict}' recorded."

    tool_registry.handlers["record_qa_verdict"] = record_qa_verdict

    world.register_system(SystemPromptRenderSystem(priority=-20), priority=-20)
    world.register_system(UserPromptNormalizationSystem(priority=-10), priority=-10)
    subagent_system = SubagentSystem(priority=-1)
    world.register_system(subagent_system, priority=-1)
    subagent_system.install_subagent_tool(world, agent_id, tool_name="subagent")
    subagent_system.install_subagent_control_tools(world, agent_id)
    world.register_system(ReasoningSystem(priority=0), priority=0)
    world.register_system(ToolExecutionSystem(priority=5), priority=5)
    world.register_system(MemorySystem(), priority=10)
    world.register_system(ErrorHandlingSystem(priority=99), priority=99)

    return world, agent_id, adapter, runtime_state


async def main() -> None:
    debug_mode = os.environ.get("DEBUG", "").lower() in ("1", "true")
    configure_logging(json_output=False, level="DEBUG" if debug_mode else None)

    if debug_mode:
        logger.info("debug_mode_enabled")

    api_key: str = os.environ.get("LLM_API_KEY", "")
    base_url: str = os.environ.get(
        "LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    model: str = os.environ.get("LLM_MODEL", "qwen3.5-flash")

    if not api_key:
        print("Error: LLM_API_KEY environment variable is required.")
        print(
            "Example: LLM_API_KEY=sk-... uv run python examples/e2e/plan_and_task/main.py"
        )
        sys.exit(1)

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

    runtime_config = build_runtime_config()
    world, agent_id, adapter, runtime_state = build_plan_task_world(
        provider=provider,
        model=model,
        workflow_id=runtime_config.workflow_id,
        base_dir=_WORKFLOW_BASE_DIR,
    )
    controller = PlanController()
    state_machine = WorkflowStateMachine()

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
