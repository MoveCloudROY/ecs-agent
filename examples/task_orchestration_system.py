"""Runnable task orchestration example using dependency waves, mixed backends, persistence, and serialization."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Awaitable, Callable

from ecs_agent.components import (
    ConversationComponent,
    LLMComponent,
    ScratchbookIndexComponent,
    ScratchbookRefComponent,
    SubagentRegistryComponent,
    TaskComponent,
    ToolRegistryComponent,
)
from ecs_agent.core import World
from ecs_agent.providers import OpenAIProvider
from ecs_agent.providers.fake_provider import FakeProvider
from ecs_agent.providers.protocol import LLMProvider
from ecs_agent.serialization import WorldSerializer
from ecs_agent.scratchbook.service import ScratchbookService
from ecs_agent.task import (
    TaskExecutor,
    TaskFetchingUnit,
    TaskPersistenceService,
    TaskState,
    TransitionRequest,
    WavePlanner,
    analyze_task_dependencies,
    manual_unblock_task,
    transition_task_state,
)
from ecs_agent.types import (
    CompletionResult,
    EntityId,
    Message,
    ScratchbookRef,
    SubagentConfig,
    TaskBlockedEvent,
    TaskCompletedEvent,
    TaskCreatedEvent,
    TaskStateChangedEvent,
    TaskStatus,
    ToolCall,
    ToolSchema,
)


DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen3.5-flash"


class _SerializationSafeProvider:
    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        stream: bool = False,
        response_format: dict[str, Any] | None = None,
    ) -> CompletionResult:
        del messages, tools, stream, response_format
        raise RuntimeError("serialization-safe provider should not execute")


def _build_manager_provider() -> FakeProvider:
    return FakeProvider(
        responses=[
            CompletionResult(
                message=Message(
                    role="assistant",
                    content="I will collect the execution constraints first.",
                    tool_calls=[
                        ToolCall(
                            id="tool_collect_requirements",
                            name="collect_constraints",
                            arguments={"scope": "launch readiness"},
                        )
                    ],
                )
            ),
            CompletionResult(
                message=Message(
                    role="assistant",
                    content="I will turn the gathered inputs into a concrete plan.",
                    tool_calls=[
                        ToolCall(
                            id="tool_draft_execution_plan",
                            name="synthesize_plan",
                            arguments={"inputs": "requirements + research"},
                        )
                    ],
                )
            ),
            CompletionResult(
                message=Message(
                    role="assistant",
                    content="I will package the final rollout brief.",
                    tool_calls=[
                        ToolCall(
                            id="tool_publish_brief",
                            name="write_brief",
                            arguments={
                                "plan": "phased rollout",
                                "risks": "staffing and data quality",
                            },
                        )
                    ],
                )
            ),
        ]
    )


def _build_subagent_provider(result_text: str) -> FakeProvider:
    return FakeProvider(
        responses=[
            CompletionResult(message=Message(role="assistant", content=result_text))
        ]
    )


def _load_runtime_providers() -> tuple[
    LLMProvider, LLMProvider, LLMProvider, str, bool
]:
    api_key = os.environ.get("LLM_API_KEY", "")
    base_url = os.environ.get("LLM_BASE_URL", DEFAULT_BASE_URL)
    model = os.environ.get("LLM_MODEL", DEFAULT_MODEL)

    if api_key:
        return (
            OpenAIProvider(api_key=api_key, base_url=base_url, model=model),
            OpenAIProvider(api_key=api_key, base_url=base_url, model=model),
            OpenAIProvider(api_key=api_key, base_url=base_url, model=model),
            model,
            True,
        )

    return (
        _build_manager_provider(),
        _build_subagent_provider(
            "background_research: users need sequencing, auditability, and retry-safe dispatch"
        ),
        _build_subagent_provider(
            "risk_review: main risks are staffing gaps, missing checkpoints, and stale dependencies"
        ),
        "fake-orchestrator",
        False,
    )


def _build_tool_registry(subagents: SubagentRegistryComponent) -> ToolRegistryComponent:
    async def collect_constraints(scope: str) -> str:
        return (
            f"collect_constraints: scope={scope}; owners=product, design, qa; "
            "success=clear milestones, launch checklist, rollback path"
        )

    async def synthesize_plan(inputs: str) -> str:
        return (
            f"synthesize_plan: built plan from {inputs}; "
            "waves=discovery, implementation, validation"
        )

    async def write_brief(plan: str, risks: str) -> str:
        return f"write_brief: packaged final brief with {plan}; risk summary={risks}"

    async def delegate(subagent_name: str, task: str) -> str:
        config = subagents.subagents[subagent_name]
        result = await config.provider.complete([Message(role="user", content=task)])
        if not isinstance(result, CompletionResult):
            return f"Error: unexpected streaming result for subagent {subagent_name}"
        return result.message.content

    tools = {
        "collect_constraints": ToolSchema(
            name="collect_constraints",
            description="Collect delivery constraints and success criteria.",
            parameters={
                "type": "object",
                "properties": {
                    "scope": {"type": "string", "description": "Planning scope."}
                },
                "required": ["scope"],
            },
        ),
        "synthesize_plan": ToolSchema(
            name="synthesize_plan",
            description="Turn task inputs into a phased execution plan.",
            parameters={
                "type": "object",
                "properties": {
                    "inputs": {"type": "string", "description": "Combined inputs."}
                },
                "required": ["inputs"],
            },
        ),
        "write_brief": ToolSchema(
            name="write_brief",
            description="Package a final brief for stakeholders.",
            parameters={
                "type": "object",
                "properties": {
                    "plan": {"type": "string", "description": "Execution plan."},
                    "risks": {"type": "string", "description": "Risk summary."},
                },
                "required": ["plan", "risks"],
            },
        ),
        "delegate": ToolSchema(
            name="delegate",
            description="Delegate a task to a named subagent.",
            parameters={
                "type": "object",
                "properties": {
                    "subagent_name": {
                        "type": "string",
                        "description": "Registered subagent name.",
                    },
                    "task": {"type": "string", "description": "Delegated task text."},
                },
                "required": ["subagent_name", "task"],
            },
        ),
    }
    handlers: dict[str, Callable[..., Awaitable[str]]] = {
        "collect_constraints": collect_constraints,
        "synthesize_plan": synthesize_plan,
        "write_brief": write_brief,
        "delegate": delegate,
    }
    return ToolRegistryComponent(tools=tools, handlers=handlers)


def _build_tasks() -> list[TaskComponent]:
    return [
        TaskComponent(
            description="Use the collect_constraints tool to collect requirements for $initiative",
            expected_output="A concise constraint and owner summary",
            assigned_agent=EntityId(1),
            tools=["collect_constraints"],
            context_dependencies=[],
            task_id="collect_requirements",
            status=TaskStatus.PENDING,
            priority=10,
        ),
        TaskComponent(
            description="Run background research for $initiative",
            expected_output="Research notes with rollout signals",
            assigned_agent="researcher",
            tools=["delegate"],
            context_dependencies=[],
            task_id="background_research",
            status=TaskStatus.PENDING,
            priority=9,
        ),
        TaskComponent(
            description="Use the synthesize_plan tool to draft an execution plan for $initiative",
            expected_output="A phased execution plan",
            assigned_agent=None,
            tools=["synthesize_plan"],
            context_dependencies=["collect_requirements", "background_research"],
            task_id="draft_execution_plan",
            status=TaskStatus.PENDING,
            priority=8,
        ),
        TaskComponent(
            description="Review risks for $initiative",
            expected_output="A focused risk assessment",
            assigned_agent="reviewer",
            tools=["delegate"],
            context_dependencies=["draft_execution_plan"],
            task_id="risk_review",
            status=TaskStatus.PENDING,
            priority=7,
        ),
        TaskComponent(
            description="Use the write_brief tool to publish the final brief for $initiative",
            expected_output="A stakeholder-ready rollout brief",
            assigned_agent=None,
            tools=["write_brief"],
            context_dependencies=["draft_execution_plan"],
            task_id="publish_brief",
            status=TaskStatus.PENDING,
            priority=6,
        ),
    ]


def _set_status(
    task: TaskComponent, states: dict[str, TaskState], status: TaskStatus
) -> None:
    task.status = status
    current_state = states[task.task_id]
    states[task.task_id] = TaskState(
        task_id=current_state.task_id,
        status=status,
        retry_count=current_state.retry_count,
        max_retries=current_state.max_retries,
        blocked_until_manual=current_state.blocked_until_manual,
        blocked_reason=current_state.blocked_reason,
    )


def _current_status(task: TaskComponent) -> TaskStatus:
    return task.status


async def _record_transition(
    world: World,
    manager_id: EntityId,
    persistence: TaskPersistenceService,
    task: TaskComponent,
    old_status: TaskStatus,
    new_status: TaskStatus,
) -> None:
    event = TaskStateChangedEvent(
        entity_id=manager_id,
        task_id=task.task_id,
        old_status=old_status,
        new_status=new_status,
    )
    persistence.append_task_event(task.task_id, event)
    await world.event_bus.publish(event)


async def _refresh_ready_states(
    world: World,
    manager_id: EntityId,
    tasks: list[TaskComponent],
    states: dict[str, TaskState],
    persistence: TaskPersistenceService,
    blocked_transitions: dict[str, int],
) -> None:
    analysis = analyze_task_dependencies(tasks)

    for task in tasks:
        current_state = states[task.task_id]

        if task.task_id in analysis.ready_task_ids:
            if task.status is TaskStatus.PENDING:
                next_state = transition_task_state(
                    current_state,
                    TransitionRequest(target_status=TaskStatus.READY),
                )
                states[task.task_id] = next_state
                old_status = current_state.status
                _set_status(task, states, next_state.status)
                await _record_transition(
                    world,
                    manager_id,
                    persistence,
                    task,
                    old_status,
                    next_state.status,
                )
                persistence.persist_task_snapshot(task.task_id, task)
            elif task.status is TaskStatus.BLOCKED:
                next_state = manual_unblock_task(
                    current_state, reason="dependencies resolved"
                )
                states[task.task_id] = next_state
                old_status = current_state.status
                _set_status(task, states, next_state.status)
                await _record_transition(
                    world,
                    manager_id,
                    persistence,
                    task,
                    old_status,
                    next_state.status,
                )
                persistence.persist_task_snapshot(task.task_id, task)
            continue

        if task.status is TaskStatus.PENDING:
            reason = analysis.blocked_reasons[task.task_id][0]
            next_state = transition_task_state(
                current_state,
                TransitionRequest(target_status=TaskStatus.BLOCKED, reason=reason),
            )
            states[task.task_id] = next_state
            old_status = current_state.status
            _set_status(task, states, next_state.status)
            await _record_transition(
                world,
                manager_id,
                persistence,
                task,
                old_status,
                next_state.status,
            )
            blocked_event = TaskBlockedEvent(
                entity_id=manager_id,
                task_id=task.task_id,
                reason=reason,
                blocked_on=list(task.context_dependencies),
            )
            persistence.append_task_event(task.task_id, blocked_event)
            await world.event_bus.publish(blocked_event)
            blocked_transitions[task.task_id] = (
                blocked_transitions.get(task.task_id, 0) + 1
            )
            persistence.persist_task_snapshot(task.task_id, task)


def _collect_tasks(
    world: World, task_entities: dict[str, EntityId]
) -> list[TaskComponent]:
    ordered_ids = sorted(task_entities, key=lambda task_id: task_entities[task_id])
    tasks: list[TaskComponent] = []
    for task_id in ordered_ids:
        task = world.get_component(task_entities[task_id], TaskComponent)
        if task is not None:
            tasks.append(task)
    return tasks


async def run_demo() -> dict[str, Any]:
    with TemporaryDirectory() as temp_dir:
        scratchbook_root = Path(temp_dir) / ".scratchbook"
        scratchbook = ScratchbookService(scratchbook_root)
        persistence = TaskPersistenceService(scratchbook)
        (
            manager_provider,
            researcher_provider,
            reviewer_provider,
            model,
            is_real_mode,
        ) = _load_runtime_providers()

        world = World()
        manager_id = world.create_entity()
        world.add_component(
            manager_id,
            LLMComponent(
                provider=manager_provider,
                model=model,
                system_prompt=(
                    "You are an orchestration manager. "
                    "When a request matches an available local tool, call exactly one tool and do not answer from memory. "
                    "Use collect_constraints for requirements gathering, synthesize_plan for planning, "
                    "and write_brief for final publication. Use delegated subagents for research or risk review tasks."
                ),
            ),
        )
        world.add_component(
            manager_id,
            ConversationComponent(
                messages=[Message(role="user", content="Run the orchestration demo.")]
            ),
        )
        world.add_component(manager_id, ScratchbookIndexComponent())

        subagents = SubagentRegistryComponent(
            subagents={
                "researcher": SubagentConfig(
                    name="researcher",
                    provider=researcher_provider,
                    model=model if is_real_mode else "fake-researcher",
                    description="Provides market and delivery research.",
                    system_prompt="Return crisp research findings.",
                ),
                "reviewer": SubagentConfig(
                    name="reviewer",
                    provider=reviewer_provider,
                    model=model if is_real_mode else "fake-reviewer",
                    description="Reviews rollout risks.",
                    system_prompt="Return a concise risk review.",
                ),
            }
        )
        world.add_component(manager_id, subagents)
        world.add_component(manager_id, _build_tool_registry(subagents))

        completed_from_bus: list[str] = []

        async def on_task_completed(event: TaskCompletedEvent) -> None:
            completed_from_bus.append(event.task_id)

        world.event_bus.subscribe(TaskCompletedEvent, on_task_completed)

        task_entities: dict[str, EntityId] = {}
        states: dict[str, TaskState] = {}

        for task in _build_tasks():
            entity_id = world.create_entity()
            task_entities[task.task_id] = entity_id
            world.add_component(entity_id, task)
            states[task.task_id] = TaskState(
                task_id=task.task_id,
                status=task.status,
                max_retries=task.max_retries,
            )
            persistence.persist_task_snapshot(task.task_id, task)
            created_event = TaskCreatedEvent(
                entity_id=manager_id,
                task_id=task.task_id,
                description=task.description,
            )
            persistence.append_task_event(task.task_id, created_event)

        executor = TaskExecutor()
        fetching_unit = TaskFetchingUnit()
        wave_planner = WavePlanner()
        placeholder_snapshot = {"initiative": "Task Orchestration System"}

        waves: list[list[str]] = []
        completed_tasks: list[str] = []
        backend_types: dict[str, str] = {}
        blocked_transitions: dict[str, int] = {}
        final_brief = ""

        while True:
            tasks = _collect_tasks(world, task_entities)
            await _refresh_ready_states(
                world,
                manager_id,
                tasks,
                states,
                persistence,
                blocked_transitions,
            )
            analysis = analyze_task_dependencies(tasks)
            wave_plan = wave_planner.compute_waves(analysis)
            requests = fetching_unit.generate_dispatch_requests(
                wave_plan=wave_plan,
                tasks=tasks,
                snapshot=placeholder_snapshot,
                writer_id="task_fetching_unit",
            )
            if not requests:
                break

            current_wave = [request.task_id for request in requests]
            waves.append(current_wave)

            for request in requests:
                task_component = world.get_component(
                    task_entities[request.task_id], TaskComponent
                )
                if task_component is None:
                    continue

                running_state = transition_task_state(
                    states[task_component.task_id],
                    TransitionRequest(target_status=TaskStatus.RUNNING),
                )
                states[task_component.task_id] = running_state
                old_status = _current_status(task_component)
                _set_status(task_component, states, running_state.status)
                await _record_transition(
                    world,
                    manager_id,
                    persistence,
                    task_component,
                    old_status,
                    running_state.status,
                )
                persistence.persist_task_snapshot(
                    task_component.task_id, task_component
                )

                result = await executor.execute_dispatch_request(
                    world, manager_id, request
                )
                backend_types[task_component.task_id] = result.backend_type

                completed_state = transition_task_state(
                    states[task_component.task_id],
                    TransitionRequest(target_status=TaskStatus.COMPLETED),
                )
                states[task_component.task_id] = completed_state
                old_status = _current_status(task_component)
                _set_status(task_component, states, completed_state.status)
                await _record_transition(
                    world,
                    manager_id,
                    persistence,
                    task_component,
                    old_status,
                    completed_state.status,
                )
                completed_event = TaskCompletedEvent(
                    entity_id=manager_id,
                    task_id=task_component.task_id,
                    result=result.result_content,
                )
                persistence.append_task_event(task_component.task_id, completed_event)
                await world.event_bus.publish(completed_event)
                persistence.persist_task_snapshot(
                    task_component.task_id, task_component
                )

                scratchbook.write_artifact(
                    f"{task_component.task_id}_result",
                    "tasks/results",
                    {
                        "task_id": task_component.task_id,
                        "backend": result.backend_type,
                        "result": result.result_content,
                    },
                )
                result_ref = scratchbook.read_artifact(
                    f"{task_component.task_id}_result", "tasks/results"
                )
                if result_ref is not None:
                    ref = ScratchbookRef(
                        artifact_id=f"{task_component.task_id}_result",
                        category="tasks/results",
                        content_hash=str(len(result.result_content)),
                        timestamp="persisted",
                    )
                    index = world.get_component(manager_id, ScratchbookIndexComponent)
                    if index is not None:
                        index.artifacts[ref.artifact_id] = ref
                    existing_ref = world.get_component(
                        task_entities[task_component.task_id], ScratchbookRefComponent
                    )
                    if existing_ref is None:
                        world.add_component(
                            task_entities[task_component.task_id],
                            ScratchbookRefComponent(
                                artifact_id=ref.artifact_id,
                                category=ref.category,
                                content_hash=ref.content_hash,
                                timestamp=ref.timestamp,
                            ),
                        )
                    else:
                        existing_ref.artifact_id = ref.artifact_id
                        existing_ref.category = ref.category
                        existing_ref.content_hash = ref.content_hash
                        existing_ref.timestamp = ref.timestamp

                completed_tasks.append(task_component.task_id)
                if task_component.task_id == "publish_brief":
                    final_brief = result.result_content

        checkpoint_path = Path(temp_dir) / "task_orchestration_world.json"
        tool_registry = world.get_component(manager_id, ToolRegistryComponent)
        index = world.get_component(manager_id, ScratchbookIndexComponent)
        llm_component = world.get_component(manager_id, LLMComponent)
        if llm_component is not None and is_real_mode:
            llm_component.provider = _SerializationSafeProvider()
        world.remove_component(manager_id, ToolRegistryComponent)
        world.remove_component(manager_id, SubagentRegistryComponent)
        WorldSerializer.save(world, checkpoint_path)
        restored_world = WorldSerializer.load(
            checkpoint_path,
            providers={
                model: manager_provider,
                "default": manager_provider,
                "fake-orchestrator": manager_provider,
                "fake-researcher": researcher_provider,
                "fake-reviewer": reviewer_provider,
            },
            tool_handlers=tool_registry.handlers if tool_registry is not None else {},
        )

        restored_task_statuses: dict[str, str] = {}
        for task_id, entity_id in task_entities.items():
            restored_task = restored_world.get_component(entity_id, TaskComponent)
            if restored_task is not None:
                restored_task_statuses[task_id] = restored_task.status.value

        artifact_ids = []
        restored_index = restored_world.get_component(
            manager_id, ScratchbookIndexComponent
        )
        if restored_index is not None:
            artifact_ids = sorted(restored_index.artifacts)
        elif index is not None:
            artifact_ids = sorted(index.artifacts)

        event_log_lengths = {
            task_id: len(persistence.read_task_events(task_id))
            for task_id in task_entities
        }

        return {
            "waves": waves,
            "backend_types": backend_types,
            "completed_tasks": completed_tasks,
            "completed_from_bus": completed_from_bus,
            "provider_mode": "real" if is_real_mode else "fake",
            "blocked_transitions": blocked_transitions,
            "snapshot_count": len(scratchbook.list_artifacts("tasks/snapshots")),
            "event_log_lengths": event_log_lengths,
            "serialization_roundtrip": {
                "restored_task_statuses": restored_task_statuses,
                "artifact_ids": artifact_ids,
            },
            "final_brief": final_brief,
        }


async def main() -> None:
    report = await run_demo()
    print("Task orchestration waves:")
    for index, wave in enumerate(report["waves"], start=1):
        print(f"  Wave {index}: {', '.join(wave)}")
    print("Completed tasks:", ", ".join(report["completed_tasks"]))
    print("Final brief:", report["final_brief"])


if __name__ == "__main__":
    asyncio.run(main())
