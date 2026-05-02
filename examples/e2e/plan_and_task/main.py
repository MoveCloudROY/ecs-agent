"""Plan-and-task E2E example entrypoint."""

from __future__ import annotations

import asyncio
import json
import os
import re as _re
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
    WorkflowRuntimeComponent,
)
from ecs_agent.components.definitions import ScriptHandler
from ecs_agent.core import Runner, World
from ecs_agent.logging import configure_logging, get_logger
from ecs_agent.prompts.contracts import (
    PromptTemplateSource,
    SystemPromptConfigSpec,
    TriggerSpec,
)
from ecs_agent.providers import Model
from ecs_agent.providers.config import ApiFormat
from ecs_agent.providers.protocol import LLMModel
from ecs_agent.systems import WorkflowStateSystem
from ecs_agent.systems.error_handling import ErrorHandlingSystem
from ecs_agent.tools import BuiltinToolsSkill
from ecs_agent.skills.manager import SkillManager
from ecs_agent.skills.discovery import discover_skills
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
    Message,
    SubagentConfig,
)

from ecs_agent.accounting import AccountingSubscriber
from ecs_agent.workflows import install_workflow
from examples.e2e.plan_and_task.billing import BillingSubscriber
from examples.e2e.plan_and_task.scratchbook_adapter import (
    PlanTaskScratchbookAdapter as ArtifactAdapter,
    build_scratchbook_prompt_config,
)
from examples.e2e.plan_and_task.controller import PlanController, ResumeAction
from examples.e2e.plan_and_task.prompts import (
    ADVISOR_SYSTEM_PROMPT,
    PLAN_QA_REVIEW_SYSTEM_PROMPT,
    QA_SYSTEM_PROMPT,
    WRITE_PLAN_SYSTEM_PROMPT,
    build_write_plan_prompt,
)
from examples.e2e.plan_and_task.runtime import (
    setup_interactive_input,
    derive_workflow_id_from_llm,
)
from examples.e2e.plan_and_task.state_machine import WorkflowStateMachine
from examples.e2e.plan_and_task.state_models import RuntimeState
from examples.e2e.plan_and_task.workflow_spec import PLAN_TASK_WORKFLOW_SPEC

logger = get_logger(__name__)

_VERDICT_PATTERN = _re.compile(r"\b(approved|revise|blocked)\b", _re.IGNORECASE)

_WORKFLOW_BASE_DIR = Path(__file__).parent
_SKILLS_DIR = Path(__file__).parent / ".claude" / "skills"


def _extract_verdict_from_result(result: str) -> str:
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


def _require_adapter(adapter: ArtifactAdapter | None) -> ArtifactAdapter:
    if adapter is None:
        raise ValueError(
            "No active workflow adapter. Start with /plan:start <description>."
        )
    return adapter


def build_plan_task_world(
    model: LLMModel,
    base_dir: Path | None = None,
) -> tuple[World, EntityId, list[ArtifactAdapter | None], list[RuntimeState | None]]:
    discover_skills([_SKILLS_DIR])

    world = World()

    agent_id = world.create_entity()
    world.add_component(
        agent_id,
        LLMComponent(model=model),
    )
    world.add_component(agent_id, ConversationComponent(messages=[]))
    world.add_component(
        agent_id,
        SystemPromptConfigSpec(
            template_source=PromptTemplateSource(inline="${_workflow_state_prompt}")
        ),
    )
    world.add_component(agent_id, ToolRegistryComponent(tools={}, handlers={}))

    _builtin_skill = BuiltinToolsSkill().bind_workspace(
        str(base_dir or _WORKFLOW_BASE_DIR)
    )
    SkillManager().install(world, agent_id, _builtin_skill)

    world.add_component(agent_id, SubagentSessionTableComponent(sessions={}))
    world.add_component(
        agent_id,
        SubagentRegistryComponent(
            subagents={
                "advisor": SubagentConfig(
                    name="advisor",
                    model=model,
                    description="Reviews workflow drafts as an advisor.",
                    system_prompt=ADVISOR_SYSTEM_PROMPT,
                    max_ticks=30,
                    inheritance_policy=InheritancePolicy(
                        inherit_system_prompt=False,
                        inherit_tools=["read_file", "glob"],
                        inherit_permissions=True,
                    ),
                ),
                "qa": SubagentConfig(
                    name="qa",
                    model=model,
                    description="Performs QA review of workflow drafts.",
                    system_prompt=QA_SYSTEM_PROMPT,
                    max_ticks=30,
                    inheritance_policy=InheritancePolicy(
                        inherit_system_prompt=False,
                        inherit_tools=["read_file", "glob"],
                        inherit_permissions=True,
                    ),
                ),
                "plan_qa": SubagentConfig(
                    name="plan_qa",
                    model=model,
                    description="Performs QA review of the finalized workflow_plan.md.",
                    system_prompt=PLAN_QA_REVIEW_SYSTEM_PROMPT,
                    max_ticks=30,
                    inheritance_policy=InheritancePolicy(
                        inherit_system_prompt=False,
                        inherit_tools=["read_file", "glob"],
                        inherit_permissions=True,
                    ),
                ),
                "plan_writer": SubagentConfig(
                    name="plan_writer",
                    model=model,
                    description="Converts an approved draft into a structured workflow_plan.md using the writing-plans skill.",
                    system_prompt=WRITE_PLAN_SYSTEM_PROMPT,
                    skills=["writing-plans"],
                    max_ticks=None,
                    inheritance_policy=InheritancePolicy(
                        inherit_system_prompt=False,
                        inherit_tools=["read_file", "write_file", "edit_file", "glob"],
                    ),
                ),
            }
        ),
    )

    adapter_ref: list[ArtifactAdapter | None] = [None]
    controller = PlanController()
    runtime_state: list[RuntimeState | None] = [None]
    _base_dir = base_dir or _WORKFLOW_BASE_DIR

    def _sync_workflow_state(w: World, eid: EntityId, phase: str) -> None:
        runtime = w.get_component(eid, WorkflowRuntimeComponent)
        if runtime is not None:
            runtime.current_state_id = phase

    def _activate_task_phase(
        w: World, eid: EntityId, phase: str, trigger_text: str
    ) -> None:
        _sync_workflow_state(w, eid, phase)
        conv = w.get_component(eid, ConversationComponent)
        if conv is not None:
            conv.messages.clear()
            conv.messages.append(Message(role="user", content=trigger_text))


    def _load_workflow(w: World, eid: EntityId, workflow_id: str) -> RuntimeState:
        new_adapter = ArtifactAdapter(base_dir=_base_dir, workflow_id=workflow_id)
        state = new_adapter.read_state()
        state = WorkflowStateMachine().handle_restart(state, new_adapter)
        adapter_ref[0] = new_adapter
        runtime_state[0] = state
        w.add_component(eid, build_scratchbook_prompt_config(workflow_id))
        _sync_workflow_state(w, eid, state.phase)
        return state

    def _workflow_id_from_command(user_text: str) -> str:
        parts = user_text.strip().split(None, 1)
        return parts[1].strip() if len(parts) > 1 else ""

    def _ensure_task_workflow_loaded(
        w: World,
        eid: EntityId,
        user_text: str,
        *,
        command_name: str,
    ) -> RuntimeState:
        if runtime_state[0] is not None:
            return runtime_state[0]

        workflow_id = _workflow_id_from_command(user_text)
        if not workflow_id:
            raise ValueError(
                "No active workflow state. "
                f"Provide a workflow_id: {command_name} <workflow_id>, "
                "or start a new workflow with /plan:start <description>."
            )
        state = _load_workflow(w, eid, workflow_id)
        logger.info(
            "plan_task_task_command_auto_loaded_state",
            command=command_name,
            workflow_id=workflow_id,
            phase=state.phase,
        )
        return state

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
        adapter = adapter_ref[0]
        if adapter is None:
            logger.warning(
                "plan_task_delegation_completed_no_adapter",
                subagent_name=event.subagent_name,
            )
            return
        try:
            if event.subagent_name == "advisor":
                runtime_state[0] = controller.handle_advisor_review(
                    current, adapter, verdict_str, notes=event.result[:500]
                )
                _sync_workflow_state(world, agent_id, _require_state(runtime_state[0]).phase)
            elif event.subagent_name == "qa":
                new_state = controller.handle_qa_review(
                    current, adapter, verdict_str, notes=event.result[:500]
                )
                runtime_state[0] = new_state
                _sync_workflow_state(world, agent_id, new_state.phase)
                if new_state.phase == "WRITE_PLAN":
                    conv = world.get_component(agent_id, ConversationComponent)
                    if conv is not None:
                        draft_path = str(
                            (adapter.plan_dir / "draft.md").relative_to(adapter.base_dir)
                        )
                        plan_path = str(
                            (adapter.plan_dir / "workflow_plan.md").relative_to(adapter.base_dir)
                        )
                        trigger_msg = build_write_plan_prompt(draft_path, plan_path)
                        conv.messages.append(
                            Message(role="user", content=trigger_msg)
                        )
                        logger.info(
                            "plan_task_auto_trigger_plan_writer",
                            workflow_id=new_state.workflow_id,
                        )
            elif event.subagent_name == "plan_qa":
                runtime_state[0] = controller.handle_plan_qa_review(
                    current, adapter, verdict_str, notes=event.result[:500]
                )
                _sync_workflow_state(world, agent_id, _require_state(runtime_state[0]).phase)
            elif event.subagent_name == "plan_writer":
                runtime_state[0] = controller.handle_write_plan_completed(
                    current, adapter
                )
                _sync_workflow_state(world, agent_id, _require_state(runtime_state[0]).phase)
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
            derived_id = (
                await derive_workflow_id_from_llm(description, model)
                or description[:40].strip()
            )
            adapter = ArtifactAdapter(
                base_dir=_base_dir,
                workflow_id=derived_id,
            )
            adapter_ref[0] = adapter
            _world.add_component(
                _entity_id, build_scratchbook_prompt_config(derived_id)
            )
            runtime_state[0] = controller.handle_plan_start(adapter, description)
            _sync_workflow_state(_world, _entity_id, _require_state(runtime_state[0]).phase)
            status = controller.get_plan_status(_require_state(runtime_state[0]))
            logger.info(
                "plan_task_command_plan_start",
                workflow_id=derived_id,
                description_len=len(description),
            )
            return (
                f"Plan started (workflow_id={derived_id!r}):\n{_format_status(status)}"
            )
        except ValueError as exc:
            logger.warning("plan_task_command_error", command="plan_start", exception=str(exc))
            return f"Error: {exc}"

    async def _handle_plan_status(
        _world: World, _entity_id: EntityId, _user_text: str
    ) -> str | None:
        if runtime_state[0] is None:
            return "No active workflow. Use /plan:start <description> to begin."
        logger.debug("plan_task_command_plan_status", workflow_id=runtime_state[0].workflow_id)
        return _format_status(
            controller.get_plan_status(_require_state(runtime_state[0]))
        )

    async def _handle_plan_finalize(
        _world: World, _entity_id: EntityId, _user_text: str
    ) -> str | None:
        try:
            runtime_state[0] = controller.handle_plan_finalize(
                _require_state(runtime_state[0]), _require_adapter(adapter_ref[0])
            )
            _sync_workflow_state(_world, _entity_id, _require_state(runtime_state[0]).phase)
            logger.info(
                "plan_task_command_plan_finalize",
                workflow_id=_require_state(runtime_state[0]).workflow_id,
            )
            return f"Plan finalized:\n{_format_status(controller.get_plan_status(_require_state(runtime_state[0])))}"
        except ValueError as exc:
            logger.warning("plan_task_command_error", command="plan_finalize", exception=str(exc))
            return f"Error: {exc}"

    async def _handle_task_start(
        _world: World, _entity_id: EntityId, _user_text: str
    ) -> str | None:
        try:
            from examples.e2e.plan_and_task.task_exec import TaskExec

            # Guard against re-triggering: the /task:start message stays as the last
            # role="user" entry (tool results use role="tool"), so the trigger would
            # fire on every subsequent tick. Skip re-initialization once task execution
            # is already active.
            workflow_runtime = _world.get_component(_entity_id, WorkflowRuntimeComponent)
            if (
                workflow_runtime is not None
                and workflow_runtime.current_state_id == "TASK_RUNNING"
            ):
                return None

            if runtime_state[0] is None:
                loaded_state = _ensure_task_workflow_loaded(
                    _world,
                    _entity_id,
                    _user_text,
                    command_name="/task:start",
                )
                logger.info(
                    "plan_task_task_start_auto_loaded_state",
                    workflow_id=loaded_state.workflow_id,
                    phase=loaded_state.phase,
                )

            current = _require_state(runtime_state[0])
            task_exec = TaskExec(state=current)
            runtime_state[0] = task_exec.initialize_task_queue(
                current, _require_adapter(adapter_ref[0])
            )
            _activate_task_phase(
                _world,
                _entity_id,
                _require_state(runtime_state[0]).phase,
                _user_text,
            )
            s = _require_state(runtime_state[0])
            logger.info(
                "plan_task_command_task_start",
                workflow_id=s.workflow_id,
                task_count=len(s.tasks),
                current_task_id=s.current_task_id,
            )
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
            logger.warning("plan_task_command_error", command="task_start", exception=str(exc))
            return f"Error: {exc}"

    async def _handle_task_status(
        _world: World, _entity_id: EntityId, _user_text: str
    ) -> str | None:
        if runtime_state[0] is None:
            return "No active workflow."
        s = runtime_state[0]
        logger.debug("plan_task_command_task_status", workflow_id=s.workflow_id, phase=s.phase)
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
            if runtime_state[0] is not None and runtime_state[0].phase == "TASK_RUNNING":
                return None
            _ensure_task_workflow_loaded(
                _world,
                _entity_id,
                _user_text,
                command_name="/task:resume",
            )
            runtime_state[0] = controller.handle_task_resume(
                _require_state(runtime_state[0]), _require_adapter(adapter_ref[0])
            )
            _activate_task_phase(
                _world,
                _entity_id,
                _require_state(runtime_state[0]).phase,
                _user_text,
            )
            logger.info(
                "plan_task_command_task_resume",
                workflow_id=_require_state(runtime_state[0]).workflow_id,
            )
            return f"Task resumed:\n{_format_status(controller.get_plan_status(_require_state(runtime_state[0])))}"
        except ValueError as exc:
            logger.warning("plan_task_command_error", command="task_resume", exception=str(exc))
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
                _require_state(runtime_state[0]),
                _require_adapter(adapter_ref[0]),
                reason,
            )
            _sync_workflow_state(_world, _entity_id, _require_state(runtime_state[0]).phase)
            s = _require_state(runtime_state[0])
            logger.info(
                "plan_task_command_task_replan",
                workflow_id=s.workflow_id,
                task_id=s.current_task_id,
            )
            return f"Task replanned:\n{_format_status(controller.get_plan_status(_require_state(runtime_state[0])))}"
        except ValueError as exc:
            logger.warning("plan_task_command_error", command="task_replan", exception=str(exc))
            return f"Error: {exc}"

    async def _handle_task_abort(
        _world: World, _entity_id: EntityId, _user_text: str
    ) -> str | None:
        try:
            runtime_state[0] = controller.handle_task_abort(
                _require_state(runtime_state[0]),
                _require_adapter(adapter_ref[0]),
                reason="user abort",
            )
            _sync_workflow_state(_world, _entity_id, _require_state(runtime_state[0]).phase)
            s = _require_state(runtime_state[0])
            logger.info(
                "plan_task_command_task_abort",
                workflow_id=s.workflow_id,
                task_id=s.current_task_id,
            )
            return f"Task aborted:\n{_format_status(controller.get_plan_status(_require_state(runtime_state[0])))}"
        except ValueError as exc:
            logger.warning("plan_task_command_error", command="task_abort", exception=str(exc))
            return f"Error: {exc}"

    async def _handle_plan_resume(
        _world: World, _entity_id: EntityId, user_text: str
    ) -> str | None:
        parts = user_text.strip().split(None, 1)
        workflow_id = parts[1].strip() if len(parts) > 1 else ""
        if not workflow_id:
            return "Error: /plan:resume requires a non-empty workflow_id."
        try:
            state = _load_workflow(_world, _entity_id, workflow_id)
            actions = controller.reconcile_after_resume(state, _require_adapter(adapter_ref[0]))
            _sync_workflow_state(_world, _entity_id, state.phase)
            for action in actions:
                if action == ResumeAction.TRIGGER_PLAN_WRITER:
                    conv = _world.get_component(_entity_id, ConversationComponent)
                    if conv is not None:
                        loaded_adapter = _require_adapter(adapter_ref[0])
                        draft_path = str(
                            (loaded_adapter.plan_dir / "draft.md").relative_to(loaded_adapter.base_dir)
                        )
                        plan_path = str(
                            (loaded_adapter.plan_dir / "workflow_plan.md").relative_to(loaded_adapter.base_dir)
                        )
                        conv.messages.append(
                            Message(role="user", content=build_write_plan_prompt(draft_path, plan_path))
                        )
                        logger.info(
                            "plan_task_auto_trigger_plan_writer",
                            workflow_id=workflow_id,
                            source="reconcile_after_resume",
                        )
            logger.info(
                "plan_task_command_plan_resume",
                workflow_id=workflow_id,
                phase=state.phase,
            )
            return (
                f"Workflow resumed (workflow_id={workflow_id!r}):\n"
                f"{_format_status(controller.get_plan_status(state))}"
            )
        except ValueError as exc:
            logger.warning("plan_task_command_error", command="plan_resume", exception=str(exc))
            return f"Error: {exc}"

    async def _handle_plan_write(
        _world: World, _entity_id: EntityId, _user_text: str
    ) -> str | None:
        try:
            adapter = _require_adapter(adapter_ref[0])
            runtime_state[0] = controller.handle_write_plan(
                _require_state(runtime_state[0]), adapter
            )
            _sync_workflow_state(_world, _entity_id, _require_state(runtime_state[0]).phase)
            s = _require_state(runtime_state[0])
            logger.info("plan_task_command_plan_write", workflow_id=s.workflow_id)
            draft_path = str(
                (adapter.plan_dir / "draft.md").relative_to(adapter.base_dir)
            )
            plan_path = str(
                (adapter.plan_dir / "workflow_plan.md").relative_to(adapter.base_dir)
            )
            return build_write_plan_prompt(draft_path, plan_path)
        except ValueError as exc:
            logger.warning("plan_task_command_error", command="plan_write", exception=str(exc))
            return f"Error: {exc}"

    async def _handle_plan_qa_review(
        _world: World, _entity_id: EntityId, user_text: str
    ) -> str | None:
        parts = user_text.strip().split(None, 2)
        verdict = parts[1].strip() if len(parts) > 1 else ""
        notes = parts[2].strip() if len(parts) > 2 else None
        if verdict not in {"approved", "revise", "blocked"}:
            return "Error: /plan:qa_review requires verdict: approved | revise | blocked"
        try:
            runtime_state[0] = controller.handle_plan_qa_review(
                _require_state(runtime_state[0]),
                _require_adapter(adapter_ref[0]),
                verdict,
                notes=notes,
            )
            _sync_workflow_state(_world, _entity_id, _require_state(runtime_state[0]).phase)
            s = _require_state(runtime_state[0])
            logger.info(
                "plan_task_command_plan_qa_review",
                workflow_id=s.workflow_id,
                verdict=verdict,
            )
            return f"Plan QA review recorded ({verdict}):\n{_format_status(controller.get_plan_status(s))}"
        except ValueError as exc:
            logger.warning("plan_task_command_error", command="plan_qa_review", exception=str(exc))
            return f"Error: {exc}"

    script_handlers: dict[str, ScriptHandler] = {
        "plan_start": _handle_plan_start,
        "plan_resume": _handle_plan_resume,
        "plan_status": _handle_plan_status,
        "plan_finalize": _handle_plan_finalize,
        "plan_write": _handle_plan_write,
        "plan_qa_review": _handle_plan_qa_review,
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
            pattern="/plan:resume",
            match_mode="prefix",
            action="script",
            content="plan_resume",
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
            pattern="/plan:write",
            match_mode="prefix",
            action="script",
            content="plan_write",
        ),
        TriggerSpec(
            pattern="/plan:qa_review",
            match_mode="prefix",
            action="script",
            content="plan_qa_review",
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

    install_workflow(world, agent_id, PLAN_TASK_WORKFLOW_SPEC, agent_key="main")

    world.register_system(WorkflowStateSystem(priority=-25), priority=-25)
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

    return world, agent_id, adapter_ref, runtime_state


async def main() -> None:
    debug_mode = os.environ.get("DEBUG", "").lower() in ("1", "true")
    configure_logging(json_output=False, level="DEBUG" if debug_mode else None)

    if debug_mode:
        logger.info("debug_mode_enabled")

    api_key: str = os.environ.get("LLM_API_KEY", "")
    base_url: str = os.environ.get(
        "LLM_BASE_URL",
        "https://dashscope.aliyuncs.com/api/v2/apps/protocols/compatible-mode/v1",
    )
    model_name: str = os.environ.get("LLM_MODEL", "qwen3.6-flash")
    api_format_str: str = os.environ.get("LLM_API_FORMAT", "openai_responses")

    if not api_key:
        print("Error: LLM_API_KEY environment variable is required.")
        print(
            "Example: LLM_API_KEY=sk-... uv run python examples/e2e/plan_and_task/main.py"
        )
        sys.exit(1)

    if api_format_str == ApiFormat.ANTHROPIC_MESSAGES:
        logger.info("using_model", model_name=model_name, api_format="anthropic_messages")
        print(f"Using Anthropic Messages API with model: {model_name}")
        llm_model: LLMModel = Model(
            model_name,
            base_url=base_url,
            api_key=api_key,
            api_format=ApiFormat.ANTHROPIC_MESSAGES,
        )
    else:
        api_format = ApiFormat.OPENAI_RESPONSES
        if api_format_str == ApiFormat.OPENAI_CHAT_COMPLETIONS:
            api_format = ApiFormat.OPENAI_CHAT_COMPLETIONS
        logger.info("using_model", model_name=model_name, api_format=api_format)
        print(f"Using model: {model_name}")
        llm_model = Model(
            model_name,
            base_url=base_url,
            api_key=api_key,
            api_format=api_format,
            enable_store=api_format == ApiFormat.OPENAI_RESPONSES,
        )

    world, agent_id, _, _ = build_plan_task_world(
        model=llm_model,
        base_dir=_WORKFLOW_BASE_DIR,
    )

    billing = BillingSubscriber()
    billing.subscribe(world.event_bus)

    accounting = AccountingSubscriber()
    accounting.subscribe(world.event_bus)

    interactive_mode_str = os.environ.get("PLAN_TASK_INTERACTIVE", "1")
    if interactive_mode_str.lower() in ("0", "false"):
        if debug_mode:
            logger.info("interactive_input_disabled", reason="env_var_set")
    else:
        if debug_mode:
            logger.info("interactive_input_enabled", reason="default_or_env_var_set")
        await setup_interactive_input(world, agent_id)

    max_ticks_env = os.environ.get("PLAN_TASK_MAX_AGENT_TICKS")
    max_ticks: int | None = int(max_ticks_env) if max_ticks_env else None

    runner = Runner()
    await runner.run(world, max_ticks=max_ticks)
    billing.log_session_summary()

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
