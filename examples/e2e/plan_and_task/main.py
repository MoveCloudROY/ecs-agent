"""Plan-and-task E2E example entrypoint."""

from __future__ import annotations

import asyncio
import json
import os
import re as _re
import sys
from pathlib import Path
from typing import cast

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
from ecs_agent.components.definitions import ScriptHandler
from ecs_agent.core import Runner, World
from ecs_agent.logging import configure_logging, get_logger
from ecs_agent.prompts.contracts import (
    PromptTemplateSource,
    SystemPromptConfigSpec,
    TriggerSpec,
)
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
from ecs_agent.types import (
    DelegationCompletedEvent,
    EntityId,
    InheritancePolicy,
    SubagentConfig,
)

from examples.e2e.plan_and_task.artifacts import (
    ArtifactAdapter as LegacyArtifactAdapter,
)
from examples.e2e.plan_and_task.scratchbook_adapter import (
    PlanTaskScratchbookAdapter as ArtifactAdapter,
)
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

logger = get_logger(__name__)

_VERDICT_PATTERN = _re.compile(r"\b(approved|revise|blocked)\b", _re.IGNORECASE)

_WORKFLOW_BASE_DIR = Path(__file__).parent


def _extract_verdict_from_result(result: str) -> str:
    """Extract verdict from subagent result text. Defaults to 'revise' on no match."""
    match = _VERDICT_PATTERN.search(result)
    if match is None:
        logger.warning(
            "plan_task_verdict_extraction_failed", result_preview=result[:120]
        )
        return "revise"
    return match.group(1).lower()


def _format_status(payload: dict[str, object]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _require_state(state: RuntimeState | None) -> RuntimeState:
    if state is None:
        raise ValueError(
            "No active workflow state. Start with /plan:start <description>."
        )
    return state


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

    async def _on_delegation_completed(event: DelegationCompletedEvent) -> None:
        if event.entity_id != agent_id:
            return
        if not event.success:
            logger.warning(
                "plan_task_subagent_failed",
                subagent_name=event.subagent_name,
                error=getattr(event, "error", None),
            )
            return
        verdict_str = _extract_verdict_from_result(event.result)
        current = runtime_state[0]
        if current is None:
            logger.warning(
                "plan_task_delegation_completed_no_state",
                subagent_name=event.subagent_name,
            )
            return
        try:
            if event.subagent_name == "advisor":
                runtime_state[0] = controller.handle_advisor_review(
                    current, adapter, verdict_str, notes=event.result[:500]
                )
            elif event.subagent_name == "qa":
                runtime_state[0] = controller.handle_qa_review(
                    current, adapter, verdict_str, notes=event.result[:500]
                )
        except ValueError as exc:
            logger.error(
                "plan_task_verdict_recording_failed",
                subagent_name=event.subagent_name,
                exception=str(exc),
            )

    world.event_bus.subscribe(DelegationCompletedEvent, _on_delegation_completed)

    tool_registry = world.get_component(agent_id, ToolRegistryComponent)
    if tool_registry is None:
        raise ValueError("ToolRegistryComponent is required for plan-and-task world")

    async def _handle_plan_start(
        _world: World, _entity_id: EntityId, user_text: str
    ) -> str | None:
        parts = user_text.strip().split(None, 1)
        description = parts[1].strip() if len(parts) > 1 else ""
        if not description:
            return "Error: /plan:start requires a non-empty description."
        try:
            runtime_state[0] = controller.handle_plan_start(adapter, description)
            status = controller.get_plan_status(_require_state(runtime_state[0]))
            return f"Plan started:\n{_format_status(status)}"
        except ValueError as exc:
            return f"Error: {exc}"

    async def _handle_plan_status(
        _world: World, _entity_id: EntityId, _user_text: str
    ) -> str | None:
        if runtime_state[0] is None:
            return "No active workflow. Use /plan:start <description> to begin."
        return _format_status(
            controller.get_plan_status(_require_state(runtime_state[0]))
        )

    async def _handle_plan_finalize(
        _world: World, _entity_id: EntityId, _user_text: str
    ) -> str | None:
        try:
            runtime_state[0] = controller.handle_plan_finalize(
                _require_state(runtime_state[0]), adapter
            )
            return f"Plan finalized:\n{_format_status(controller.get_plan_status(_require_state(runtime_state[0])))}"
        except ValueError as exc:
            return f"Error: {exc}"

    async def _handle_task_start(
        _world: World, _entity_id: EntityId, _user_text: str
    ) -> str | None:
        try:
            from examples.e2e.plan_and_task.task_exec import TaskExec

            current = _require_state(runtime_state[0])
            task_exec = TaskExec(state=current)
            runtime_state[0] = task_exec.initialize_task_queue(current, adapter)
            s = _require_state(runtime_state[0])
            return _format_status(
                {
                    "workflow_id": s.workflow_id,
                    "phase": s.phase,
                    "status": s.status,
                    "current_task_id": s.current_task_id,
                    "task_count": len(s.tasks),
                }
            )
        except ValueError as exc:
            return f"Error: {exc}"

    async def _handle_task_status(
        _world: World, _entity_id: EntityId, _user_text: str
    ) -> str | None:
        if runtime_state[0] is None:
            return "No active workflow."
        s = runtime_state[0]
        return _format_status(
            {
                "workflow_id": s.workflow_id,
                "phase": s.phase,
                "status": s.status,
                "current_task_id": s.current_task_id,
                "tasks": [
                    {
                        "task_id": t.task_id,
                        "status": t.status,
                        "retry_count": t.retry_count,
                    }
                    for t in s.tasks
                ],
            }
        )

    async def _handle_task_resume(
        _world: World, _entity_id: EntityId, _user_text: str
    ) -> str | None:
        try:
            runtime_state[0] = controller.handle_task_resume(
                _require_state(runtime_state[0]), adapter
            )
            return f"Task resumed:\n{_format_status(controller.get_plan_status(_require_state(runtime_state[0])))}"
        except ValueError as exc:
            return f"Error: {exc}"

    async def _handle_task_replan(
        _world: World, _entity_id: EntityId, user_text: str
    ) -> str | None:
        parts = user_text.strip().split(None, 1)
        reason = parts[1].strip() if len(parts) > 1 else ""
        if not reason:
            return "Error: /task:replan requires a non-empty reason."
        try:
            runtime_state[0] = controller.handle_task_replan(
                _require_state(runtime_state[0]), adapter, reason
            )
            return f"Task replanned:\n{_format_status(controller.get_plan_status(_require_state(runtime_state[0])))}"
        except ValueError as exc:
            return f"Error: {exc}"

    async def _handle_task_abort(
        _world: World, _entity_id: EntityId, _user_text: str
    ) -> str | None:
        try:
            runtime_state[0] = controller.handle_task_abort(
                _require_state(runtime_state[0]), adapter, reason="user abort"
            )
            return f"Task aborted:\n{_format_status(controller.get_plan_status(_require_state(runtime_state[0])))}"
        except ValueError as exc:
            return f"Error: {exc}"

    script_handlers: dict[str, ScriptHandler] = {
        "plan_start": _handle_plan_start,
        "plan_status": _handle_plan_status,
        "plan_finalize": _handle_plan_finalize,
        "task_start": _handle_task_start,
        "task_status": _handle_task_status,
        "task_resume": _handle_task_resume,
        "task_replan": _handle_task_replan,
        "task_abort": _handle_task_abort,
    }
    triggers = [
        TriggerSpec(
            pattern="/plan:start",
            match_mode="prefix",
            action="script",
            content="plan_start",
        ),
        TriggerSpec(
            pattern="/plan:status",
            match_mode="prefix",
            action="script",
            content="plan_status",
        ),
        TriggerSpec(
            pattern="/plan:finalize",
            match_mode="prefix",
            action="script",
            content="plan_finalize",
        ),
        TriggerSpec(
            pattern="/task:start",
            match_mode="prefix",
            action="script",
            content="task_start",
        ),
        TriggerSpec(
            pattern="/task:status",
            match_mode="prefix",
            action="script",
            content="task_status",
        ),
        TriggerSpec(
            pattern="/task:resume",
            match_mode="prefix",
            action="script",
            content="task_resume",
        ),
        TriggerSpec(
            pattern="/task:replan",
            match_mode="prefix",
            action="script",
            content="task_replan",
        ),
        TriggerSpec(
            pattern="/task:abort",
            match_mode="prefix",
            action="script",
            content="task_abort",
        ),
    ]
    world.add_component(
        agent_id,
        UserPromptConfigComponent(triggers=triggers, script_handlers=script_handlers),
    )

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
            runtime_state[0] = state_machine.handle_restart(
                restored_state,
                cast(LegacyArtifactAdapter, adapter),
            )
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
