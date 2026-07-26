"""Task-execution tool for the plan-and-task example.

During ``TASK_RUNNING`` the main agent executes each task itself (with the
built-in file/bash tools) and verifies its acceptance criteria. This module
installs the missing piece: a ``complete_task`` tool the agent calls to *record*
a task as done — which marks it complete, writes an evidence record, and advances
the live queue to the next task (or to ``TASK_COMPLETED`` when none remain).

Without it the queue never advances: the agent can do the work but has no way to
commit completion, so the workflow sits on the first task forever. The tool wraps
``TaskExec.record_task_completion`` and mutates the same ``runtime_state`` /
``adapter`` refs the slash-command handlers use.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable

from ecs_agent.components import PhaseComponent, ToolRegistryComponent
from ecs_agent.core.world import World
from ecs_agent.logging import get_logger
from ecs_agent.types import EntityId, ToolSchema

from examples.e2e.plan_and_task.scratchbook_adapter import (
    PlanTaskScratchbookAdapter as ArtifactAdapter,
)
from examples.e2e.plan_and_task.state_models import RuntimeState

logger = get_logger(__name__)


def build_complete_task_schema(tool_name: str = "complete_task") -> ToolSchema:
    """Declarative schema for the ``complete_task`` tool."""
    return ToolSchema(
        name=tool_name,
        # The result is a short confirmation the model reads to pick up the next
        # task; keep it inline rather than externalizing to a scratchbook path.
        inline_result=True,
        description=(
            "Record the ACTIVE task as complete and advance the queue. Call this "
            "ONLY after you have executed the active task and verified EVERY one "
            "of its acceptance_criteria. It marks the task done, writes an "
            "evidence record under evidence/, appends a memory entry, and moves "
            "the live queue to the next task — or finishes the workflow "
            "(TASK_COMPLETED) when no tasks remain.\n\n"
            "Do NOT call it for a task you have not finished, and do not skip "
            "ahead: only the task marked active in state/runtime_state.json can "
            "be completed.\n\n"
            "INTERFACE:\n"
            "  summary (required) — what you did and how each acceptance "
            "criterion was verified (commands run, files changed, expected vs "
            "actual).\n"
            "  evidence_refs (optional) — list of file paths / commands that "
            "prove completion.\n"
            "  task_id (optional) — defaults to the active task; if given it must "
            "match the active task.\n\n"
            "RETURNS: JSON {completed_task, next_task, phase, workflow_done}."
        ),
        parameters={
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": (
                        "What was done and how each acceptance criterion was "
                        "verified."
                    ),
                },
                "evidence_refs": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "File paths or commands proving completion.",
                },
                "task_id": {
                    "type": "string",
                    "description": "Defaults to the active task; must match it if given.",
                },
            },
            "required": ["summary"],
        },
    )


def make_complete_task_handler(
    world: World,
    entity_id: EntityId,
    runtime_state_ref: list[RuntimeState | None],
    adapter_ref: list[ArtifactAdapter | None],
) -> Callable[..., Awaitable[str]]:
    """Build the ``complete_task`` handler bound to the workflow refs.

    Validates the phase and active task, delegates to
    ``TaskExec.record_task_completion``, updates ``runtime_state_ref`` in place,
    and returns a JSON confirmation naming the next task.
    """

    async def complete_task(
        summary: object = None,
        evidence_refs: object = None,
        task_id: object = None,
    ) -> str:
        from examples.e2e.plan_and_task.task_exec import TaskExec

        state = runtime_state_ref[0]
        if state is None:
            return "Error: no active workflow; start one before completing tasks."
        phase = world.get_component(entity_id, PhaseComponent)
        if phase is None or phase.phase != "TASK_RUNNING":
            current = phase.phase if phase is not None else "none"
            return (
                "Error: complete_task is only valid during TASK_RUNNING "
                f"(current phase: {current})."
            )
        adapter = adapter_ref[0]
        if adapter is None:
            return "Error: no active workflow adapter."
        if not isinstance(summary, str) or not summary.strip():
            return (
                "Error: summary is required — describe what you did and how each "
                "acceptance criterion was verified."
            )
        active = task_id if isinstance(task_id, str) and task_id else state.current_task_id
        if active is None:
            return "Error: no active task to complete."
        if state.current_task_id is not None and active != state.current_task_id:
            return (
                f"Error: only the active task ({state.current_task_id}) can be "
                f"completed; got {active!r}."
            )
        refs = (
            [str(ref) for ref in evidence_refs]
            if isinstance(evidence_refs, list)
            else None
        )
        try:
            task_exec = TaskExec(state=state, world=world, entity_id=entity_id)
            new_state = await task_exec.record_task_completion(
                state, adapter, active, evidence_refs=refs, summary=summary.strip()
            )
        except ValueError as exc:
            logger.warning(
                "plan_task_complete_task_failed",
                entity_id=int(entity_id),
                task_id=active,
                exception=str(exc),
            )
            return f"Error: {exc}"
        runtime_state_ref[0] = new_state
        workflow_done = new_state.phase == "TASK_COMPLETED"
        logger.info(
            "plan_task_complete_task_tool",
            entity_id=int(entity_id),
            completed_task=active,
            next_task=new_state.current_task_id,
            workflow_done=workflow_done,
        )
        return json.dumps(
            {
                "completed_task": active,
                "next_task": new_state.current_task_id,
                "phase": new_state.phase,
                "workflow_done": workflow_done,
            },
            ensure_ascii=False,
        )

    return complete_task


def install_complete_task_tool(
    world: World,
    entity_id: EntityId,
    runtime_state_ref: list[RuntimeState | None],
    adapter_ref: list[ArtifactAdapter | None],
    tool_name: str = "complete_task",
) -> None:
    """Register the ``complete_task`` tool on ``entity_id``'s tool registry."""
    registry = world.get_component(entity_id, ToolRegistryComponent)
    if registry is None:
        raise ValueError(f"Entity {entity_id} missing ToolRegistryComponent")
    registry.tools[tool_name] = build_complete_task_schema(tool_name)
    registry.handlers[tool_name] = make_complete_task_handler(
        world, entity_id, runtime_state_ref, adapter_ref
    )
    logger.info(
        "plan_task_complete_task_tool_installed",
        entity_id=int(entity_id),
        tool_name=tool_name,
    )


__all__ = [
    "build_complete_task_schema",
    "make_complete_task_handler",
    "install_complete_task_tool",
]
