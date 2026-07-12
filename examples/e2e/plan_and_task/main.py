"""Plan-and-task E2E example entrypoint."""

from __future__ import annotations

import asyncio
import datetime
import json
import os
import re as _re
import sys
from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from ecs_agent.components import (
    CompactionConfigComponent,
    ConversationComponent,
    ConversationArchiveComponent,
    CurrentCompactionSummaryComponent,
    LLMComponent,
    PhaseApprovalsComponent,
    PhaseComponent,
    RenderedSystemPromptComponent,
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
from ecs_agent.phases import bind_phase_graph, resume_phase_graph
from ecs_agent.providers import Model
from ecs_agent.providers.config import ApiFormat
from ecs_agent.providers.protocol import LLMModel
from ecs_agent.systems.compaction import CompactionSystem
from ecs_agent.systems.error_handling import ErrorHandlingSystem
from ecs_agent.tools import BuiltinToolsSkill
from ecs_agent.skills.manager import SkillManager
from ecs_agent.skills.discovery import discover_skills
from ecs_agent.systems.reasoning import ReasoningSystem
from ecs_agent.systems.subagent import SubagentSystem
from ecs_agent.systems.system_prompt_render_system import SystemPromptRenderSystem
from ecs_agent.systems.tool_execution import ToolExecutionSystem
from ecs_agent.scratchbook import ArtifactRegistry
from ecs_agent.systems.user_prompt_normalization_system import (
    UserPromptNormalizationSystem,
)
from ecs_agent.types import (
    DelegationCompletedEvent,
    EntityId,
    InheritancePolicy,
    Message,
    PhaseChangedEvent,
    SubagentConfig,
    CompactionMethod,
    ToolSchema,
)

from ecs_agent.accounting import AccountingSubscriber
from examples.e2e.plan_and_task.ask_tool import install_ask_question_tool
from examples.e2e.plan_and_task.billing import BillingSubscriber
from examples.e2e.plan_and_task.scratchbook_adapter import (
    PlanTaskScratchbookAdapter as ArtifactAdapter,
    build_scratchbook_prompt_config,
)
from examples.e2e.plan_and_task.controller import PlanController, ResumeAction
from examples.e2e.plan_and_task.phase_graph import (
    PLAN_TASK_PHASE_GRAPH,
    REVIEW_VERDICTS,
)
from examples.e2e.plan_and_task.phase_sync import save_state
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
from examples.e2e.plan_and_task.state_models import RuntimeState

if TYPE_CHECKING:
    from ecs_agent.observability.sinks import TelemetrySink
    from ecs_agent.plugins import PluginsHandle

logger = get_logger(__name__)

_VERDICT_PATTERN = _re.compile(
    r"\b(" + "|".join(REVIEW_VERDICTS) + r")\b", _re.IGNORECASE
)
# Reviewer prompts end replies with a `VERDICT: <token>` line. Prose may
# legitimately contain verdict words (e.g. "blocked by a missing mitigation"
# in a FAIL reason), so the marker line takes precedence over a bare-word scan.
_VERDICT_LINE_PATTERN = _re.compile(
    r"^[ \t]*verdict[ \t]*:[ \t]*(" + "|".join(REVIEW_VERDICTS) + r")\b",
    _re.IGNORECASE | _re.MULTILINE,
)

_WORKFLOW_BASE_DIR = Path(__file__).parent
_SKILLS_DIR = Path(__file__).parent / ".claude" / "skills"
_PLAN_TASK_COMPACTION_PRIORITY = -30
_DEFAULT_COMPACTION_THRESHOLD_TOKENS = 300_000
_DEFAULT_COMPACTION_METHOD: CompactionMethod = "predrop_then_compact"


class _FilteredBuiltinToolsSkill:
    """Builtin tool bundle wrapper that omits tools before installation."""

    name = BuiltinToolsSkill.name
    description = BuiltinToolsSkill.description
    is_tool_bundle = True

    def __init__(self, skill: BuiltinToolsSkill, excluded_tools: set[str]) -> None:
        self._skill = skill
        self._excluded_tools = excluded_tools

    def tools(self) -> dict[str, tuple[ToolSchema, Callable[..., Awaitable[str]]]]:
        return {
            name: tool
            for name, tool in self._skill.tools().items()
            if name not in self._excluded_tools
        }

    def system_prompt(self) -> str:
        return self._skill.system_prompt()

    def install(self, world: World, entity_id: EntityId) -> None:
        self._skill.install(world, entity_id)

    def uninstall(self, world: World, entity_id: EntityId) -> None:
        self._skill.uninstall(world, entity_id)


def _env_flag_enabled(value: str | None) -> bool:
    return value is not None and value.lower() in {"1", "true", "yes", "on"}


async def install_plan_task_langfuse_observability(
    world: World,
    *,
    env: Mapping[str, str] | None = None,
    sink: TelemetrySink | None = None,
) -> PluginsHandle | None:
    """Install optional Langfuse observability for the plan-and-task example."""
    source = os.environ if env is None else env
    if not _env_flag_enabled(source.get("PLAN_TASK_LANGFUSE")):
        return None

    from ecs_agent.plugins import install_plugins
    from ecs_agent.plugins.langfuse import LangfuseConfig, LangfusePlugin

    config = LangfuseConfig(
        environment=source.get("PLAN_TASK_LANGFUSE_ENVIRONMENT", "plan-and-task"),
        release=source.get("PLAN_TASK_LANGFUSE_RELEASE"),
        session_id=source.get("PLAN_TASK_LANGFUSE_SESSION_ID"),
        tags=["plan-and-task"],
        metadata={"source": "examples/e2e/plan_and_task"},
    )
    return await install_plugins(world, [LangfusePlugin(config, sink=sink)])


def _extract_verdict_from_result(result: str) -> str:
    marker_matches: list[str] = _VERDICT_LINE_PATTERN.findall(result)
    if marker_matches:
        return marker_matches[-1].lower()
    match = _VERDICT_PATTERN.search(result)
    if match is None:
        logger.warning(
            "plan_task_verdict_extraction_failed", result_preview=result[:120]
        )
        return "revise"
    return match.group(1).lower()


def _format_status(payload: dict[str, object]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _install_auto_compaction(
    world: World,
    entity_id: EntityId,
    *,
    threshold_tokens: int,
    compaction_method: CompactionMethod,
) -> None:
    world.add_component(
        entity_id,
        CompactionConfigComponent(
            threshold_tokens=threshold_tokens,
            compaction_method=compaction_method,
        ),
    )
    world.add_component(entity_id, ConversationArchiveComponent())


def _reset_compaction_state(world: World, entity_id: EntityId) -> None:
    world.remove_component(entity_id, CurrentCompactionSummaryComponent)
    world.add_component(entity_id, ConversationArchiveComponent())
    world.remove_component(entity_id, RenderedSystemPromptComponent)


def _reset_workflow_boundary_state(
    world: World, entity_id: EntityId, *, preserve_user_text: str | None = None
) -> None:
    _reset_compaction_state(world, entity_id)
    conversation = world.get_component(entity_id, ConversationComponent)
    if conversation is not None:
        conversation.messages.clear()
        if preserve_user_text is not None:
            conversation.messages.append(
                Message(role="user", content=preserve_user_text)
            )


async def resume_workflow(
    world: World,
    entity_id: EntityId,
    workflow_id: str,
    *,
    base_dir: Path,
    preserve_user_text: str | None = None,
    adapter_ref: list[ArtifactAdapter | None] | None = None,
    runtime_state_ref: list[RuntimeState | None] | None = None,
) -> tuple[RuntimeState, ArtifactAdapter, list[ResumeAction]]:
    """Single restore path shared by /plan:resume, task-command auto-load, and tests.

    Steps, in order: read persisted state, mark stale subagents, hand the
    session refs over (so PhaseChangedEvent subscribers journal the restore's
    own transitions into THIS workflow, not a previously active one), restore
    the phase graph via resume_phase_graph() — validation, structural-drift
    detection, the on_resume TASK_RUNNING→TASK_BLOCKED demotion, and the
    approval-gate replay against the persisted verdicts — then persist the
    snapshot and reset the conversation boundary.
    """
    adapter = ArtifactAdapter(base_dir=base_dir, workflow_id=workflow_id)
    state = adapter.read_state()
    stale_task_ids = adapter.mark_stale_subagents(state)
    if adapter_ref is not None:
        adapter_ref[0] = adapter
    if runtime_state_ref is not None:
        runtime_state_ref[0] = state
    # Rehydrate the framework audit ledger from the persisted verdicts BEFORE
    # binding, so latest_verdicts() is truthful for any subscriber observing
    # the restore's own transitions. Deliberately NOT via record_approval():
    # the rehydrated record IS the original entry — no duplicate ledger rows.
    world.add_component(
        entity_id,
        PhaseApprovalsComponent(
            records=[
                {
                    "phase": v.phase,
                    "verdict": v.verdict,
                    "notes": v.notes,
                    "decided_at": v.decided_at,
                }
                for v in state.review_verdicts
            ]
        ),
    )
    report = await resume_phase_graph(
        world,
        entity_id,
        PLAN_TASK_PHASE_GRAPH,
        phase=state.phase,
        graph_hash=state.graph_hash,
        agent_key="main",
        approvals={v.phase: v.verdict for v in state.review_verdicts},
    )
    if report.demoted_from is not None:
        logger.info(
            "plan_task_restart_blocked",
            workflow_id=state.workflow_id,
            stale_task_ids=stale_task_ids,
        )
    if report.replayed is not None:
        logger.info(
            "plan_task_reconcile_advanced",
            workflow_id=state.workflow_id,
            to_phase=report.replayed[1],
            source="reconcile_after_resume",
        )
    state.updated_at = datetime.datetime.now(datetime.UTC).isoformat()
    # Persist-time snapshot: stamps phase, graph hash, and derived status.
    save_state(world, entity_id, state, adapter)
    _reset_workflow_boundary_state(
        world, entity_id, preserve_user_text=preserve_user_text
    )
    world.add_component(entity_id, build_scratchbook_prompt_config(workflow_id))
    actions = (
        [ResumeAction.TRIGGER_PLAN_WRITER]
        if report.component.phase == "WRITE_PLAN"
        else []
    )
    return state, adapter, actions


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


async def build_plan_task_world(
    model: LLMModel,
    base_dir: Path | None = None,
    *,
    compaction_threshold_tokens: int = _DEFAULT_COMPACTION_THRESHOLD_TOKENS,
    compaction_method: CompactionMethod = _DEFAULT_COMPACTION_METHOD,
    enable_tool_sink: bool = False,
) -> tuple[World, EntityId, list[ArtifactAdapter | None], list[RuntimeState | None]]:
    discover_skills([_SKILLS_DIR])

    world = World()

    agent_id = world.create_entity()
    world.add_component(
        agent_id,
        LLMComponent(model=model),
    )
    world.add_component(agent_id, ConversationComponent(messages=[]))
    _install_auto_compaction(
        world,
        agent_id,
        threshold_tokens=compaction_threshold_tokens,
        compaction_method=compaction_method,
    )
    world.add_component(
        agent_id,
        SystemPromptConfigSpec(
            template_source=PromptTemplateSource(inline="${_phase_prompt}")
        ),
    )
    world.add_component(agent_id, ToolRegistryComponent(tools={}, handlers={}))

    _builtin_skill = BuiltinToolsSkill().bind_workspace(
        str(base_dir or _WORKFLOW_BASE_DIR)
    )
    SkillManager().install(
        world,
        agent_id,
        _FilteredBuiltinToolsSkill(_builtin_skill, excluded_tools={"explore"}),
    )

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
                    max_ticks=None,
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
                    max_ticks=None,
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
                    max_ticks=None,
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
    controller = PlanController(world, agent_id)
    runtime_state: list[RuntimeState | None] = [None]
    _base_dir = base_dir or _WORKFLOW_BASE_DIR

    def _reset_task_conversation(w: World, eid: EntityId, trigger_text: str) -> None:
        conv = w.get_component(eid, ConversationComponent)
        if conv is not None:
            conv.messages.clear()
            conv.messages.append(Message(role="user", content=trigger_text))

    async def _load_workflow(
        w: World,
        eid: EntityId,
        workflow_id: str,
        *,
        preserve_user_text: str | None = None,
    ) -> tuple[RuntimeState, list[ResumeAction]]:
        state, _new_adapter, actions = await resume_workflow(
            w,
            eid,
            workflow_id,
            base_dir=_base_dir,
            preserve_user_text=preserve_user_text,
            adapter_ref=adapter_ref,
            runtime_state_ref=runtime_state,
        )
        return state, actions

    def _workflow_id_from_command(user_text: str) -> str:
        parts = user_text.strip().split(None, 1)
        return parts[1].strip() if len(parts) > 1 else ""

    async def _ensure_task_workflow_loaded(
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
        # resume_workflow replays pending approval gates exactly like
        # /plan:resume (e.g. an approved PLAN_QA_REVIEW advances to
        # PLAN_FINALIZED before task init). Returned actions are discarded:
        # TRIGGER_PLAN_WRITER is a planning-flow concern; a workflow still in
        # WRITE_PLAN proceeds to initialize_task_queue's existing clear error.
        state, _actions = await _load_workflow(
            w,
            eid,
            workflow_id,
            preserve_user_text=user_text,
        )
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
                runtime_state[0] = await controller.handle_advisor_review(
                    current, adapter, verdict_str, notes=event.result[:500]
                )
            elif event.subagent_name == "qa":
                new_state = await controller.handle_qa_review(
                    current, adapter, verdict_str, notes=event.result[:500]
                )
                runtime_state[0] = new_state
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
                runtime_state[0] = await controller.handle_plan_qa_review(
                    current, adapter, verdict_str, notes=event.result[:500]
                )
            elif event.subagent_name == "plan_writer":
                runtime_state[0] = await controller.handle_write_plan_completed(
                    current, adapter
                )
        except ValueError as exc:
            logger.error(
                "plan_task_verdict_recording_failed",
                subagent_name=event.subagent_name,
                exception=str(exc),
            )

    world.event_bus.subscribe(DelegationCompletedEvent, _on_delegation_completed)

    async def _on_phase_changed(event: PhaseChangedEvent) -> None:
        """Journal every committed transition into the active workflow's events.jsonl."""
        if event.entity_id != agent_id:
            return
        journal_adapter = adapter_ref[0]
        if journal_adapter is None:
            return
        current_state = runtime_state[0]
        journal_adapter.append_event(
            {
                "type": "phase_transition",
                "workflow_id": journal_adapter.workflow_id,
                "from": event.from_phase,
                "to": event.to_phase,
                "reason": event.reason,
                "forced": event.forced,
                "tick": event.tick,
                "task_id": (
                    current_state.current_task_id
                    if current_state is not None
                    else None
                ),
                "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
            }
        )

    world.event_bus.subscribe(PhaseChangedEvent, _on_phase_changed)

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
            _reset_workflow_boundary_state(
                _world,
                _entity_id,
                preserve_user_text=user_text,
            )
            _world.add_component(
                _entity_id, build_scratchbook_prompt_config(derived_id)
            )
            runtime_state[0] = await controller.handle_plan_start(adapter, description)
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
            runtime_state[0] = await controller.handle_plan_finalize(
                _require_state(runtime_state[0]), _require_adapter(adapter_ref[0])
            )
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
            phase_component = _world.get_component(_entity_id, PhaseComponent)
            if (
                phase_component is not None
                and phase_component.phase == "TASK_RUNNING"
            ):
                return None

            if runtime_state[0] is None:
                loaded_state = await _ensure_task_workflow_loaded(
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
            task_exec = TaskExec(state=current, world=_world, entity_id=_entity_id)
            runtime_state[0] = await task_exec.initialize_task_queue(
                current, _require_adapter(adapter_ref[0])
            )
            _reset_task_conversation(_world, _entity_id, _user_text)
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
            # Same re-trigger guard as _handle_task_start: consult the runtime
            # authority (PhaseComponent) and skip only when task execution is
            # already active. An unloaded workflow leaves the component at IDLE,
            # so the load path below still runs.
            phase_component = _world.get_component(_entity_id, PhaseComponent)
            if (
                phase_component is not None
                and phase_component.phase == "TASK_RUNNING"
            ):
                return None
            await _ensure_task_workflow_loaded(
                _world,
                _entity_id,
                _user_text,
                command_name="/task:resume",
            )
            runtime_state[0] = await controller.handle_task_resume(
                _require_state(runtime_state[0]), _require_adapter(adapter_ref[0])
            )
            _reset_task_conversation(_world, _entity_id, _user_text)
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
            runtime_state[0] = await controller.handle_task_replan(
                _require_state(runtime_state[0]),
                _require_adapter(adapter_ref[0]),
                reason,
            )
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
            runtime_state[0] = await controller.handle_task_abort(
                _require_state(runtime_state[0]),
                _require_adapter(adapter_ref[0]),
                reason="user abort",
            )
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
            state, actions = await _load_workflow(
                _world,
                _entity_id,
                workflow_id,
                preserve_user_text=user_text,
            )
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
            runtime_state[0] = await controller.handle_write_plan(
                _require_state(runtime_state[0]), adapter
            )
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
        if verdict not in REVIEW_VERDICTS:
            return (
                "Error: /plan:qa_review requires verdict: "
                + " | ".join(REVIEW_VERDICTS)
            )
        try:
            runtime_state[0] = await controller.handle_plan_qa_review(
                _require_state(runtime_state[0]),
                _require_adapter(adapter_ref[0]),
                verdict,
                notes=notes,
            )
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
    command_hints: dict[str, str] = {
        "plan_start": "<description> — start a new planning workflow",
        "plan_resume": "<workflow_id> — resume a persisted workflow",
        "plan_status": "show current plan status",
        "plan_finalize": "finalize the reviewed plan",
        "plan_write": "convert the approved draft into workflow_plan.md",
        "plan_qa_review": "<approved|revise|blocked> [notes] — record plan QA verdict",
        "task_start": "[workflow_id] — initialize the task queue and run",
        "task_status": "show task queue status",
        "task_resume": "[workflow_id] — resume task execution",
        "task_replan": "<reason> — replan the remaining tasks",
        "task_abort": "abort task execution",
    }
    # Each command's slash pattern derives from its handler key by turning the
    # FIRST underscore into a colon: plan_start -> /plan:start, plan_qa_review
    # -> /plan:qa_review. Keep new commands in script_handlers above only, with
    # a matching completion hint in command_hints.
    triggers = [
        TriggerSpec(
            pattern="/" + name.replace("_", ":", 1),
            match_mode="prefix",
            action="script",
            content=name,
            description=command_hints.get(name, ""),
        )
        for name in script_handlers
    ]
    world.add_component(
        agent_id,
        UserPromptConfigComponent(triggers=triggers, script_handlers=script_handlers),
    )

    await bind_phase_graph(world, agent_id, PLAN_TASK_PHASE_GRAPH, agent_key="main")

    world.register_system(
        CompactionSystem(), priority=_PLAN_TASK_COMPACTION_PRIORITY
    )
    world.register_system(UserPromptNormalizationSystem(priority=-10), priority=-10)
    world.register_system(SystemPromptRenderSystem(priority=-5), priority=-5)
    subagent_system = SubagentSystem(priority=-1)
    world.register_system(subagent_system, priority=-1)
    subagent_system.install_subagent_tool(world, agent_id, tool_name="subagent")
    subagent_system.install_subagent_control_tools(world, agent_id)
    # Interactive clarification tool: pauses the turn to put structured
    # questions to the user (surfaced by the TUI modal or the stdin runtime).
    install_ask_question_tool(world, agent_id)
    world.register_system(ReasoningSystem(priority=0), priority=0)
    # ISSUE-3: with the tool sink on, large tool outputs are written to
    # scratchbook/records/tool/<id> and only the record_path is kept inline, so
    # they are not resent verbatim every turn. The agent reads the artifact via
    # its file tools when it needs the content.
    tool_sink_registry = ArtifactRegistry(root=_base_dir) if enable_tool_sink else None
    world.register_system(
        ToolExecutionSystem(priority=5, registry=tool_sink_registry), priority=5
    )
    world.register_system(ErrorHandlingSystem(priority=99), priority=99)

    return world, agent_id, adapter_ref, runtime_state


def build_model_from_env() -> LLMModel:
    """Construct the LLM model from ``LLM_*`` environment variables.

    Shared by the stdin REPL entrypoint (this module) and the TUI entrypoint
    (``examples.e2e.plan_and_task.tui``). Exits the process when
    ``LLM_API_KEY`` is missing.
    """
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
        return Model(
            model_name,
            base_url=base_url,
            api_key=api_key,
            api_format=ApiFormat.ANTHROPIC_MESSAGES,
        )

    api_format = ApiFormat.OPENAI_RESPONSES
    if api_format_str == ApiFormat.OPENAI_CHAT_COMPLETIONS:
        api_format = ApiFormat.OPENAI_CHAT_COMPLETIONS
    logger.info("using_model", model_name=model_name, api_format=api_format)
    print(f"Using model: {model_name}")
    # LLM_ENABLE_STORE=0 disables stored-response chaining (previous_response_id)
    # for gateways that reject it over plain HTTP Responses API.
    store_enabled = os.environ.get("LLM_ENABLE_STORE", "1").lower() not in (
        "0",
        "false",
    )
    return Model(
        model_name,
        base_url=base_url,
        api_key=api_key,
        api_format=api_format,
        enable_store=api_format == ApiFormat.OPENAI_RESPONSES and store_enabled,
    )


async def main() -> None:
    debug_mode = os.environ.get("DEBUG", "").lower() in ("1", "true")
    configure_logging(json_output=False, level="DEBUG" if debug_mode else None)

    if debug_mode:
        logger.info("debug_mode_enabled")

    llm_model = build_model_from_env()

    world, agent_id, _, _ = await build_plan_task_world(
        model=llm_model,
        base_dir=_WORKFLOW_BASE_DIR,
        enable_tool_sink=True,
    )
    langfuse_handle = await install_plan_task_langfuse_observability(world)

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
    try:
        await runner.run(world, max_ticks=max_ticks)
    finally:
        if langfuse_handle is not None:
            await langfuse_handle.flush()
            await langfuse_handle.shutdown()
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
