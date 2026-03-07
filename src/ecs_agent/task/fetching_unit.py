"""Task fetching unit that emits dispatch requests from ready waves."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from ecs_agent.components import TaskComponent
from ecs_agent.logging import get_logger
from ecs_agent.placeholder.renderer import StrictPlaceholderRenderer
from ecs_agent.task.wave_planner import WavePlanResult
from ecs_agent.types import EntityId, TaskStatus

logger = get_logger(__name__)


class PlaceholderRenderer(Protocol):
    def substitute(self, template: str, snapshot: dict[str, Any]) -> str: ...


@dataclass(slots=True, frozen=True)
class DispatchRequest:
    task_id: str
    wave_number: int
    sequence_number: int
    description: str
    expected_output: str
    assigned_agent: EntityId | str | None
    tools: tuple[str, ...]
    context_dependencies: tuple[str, ...]
    priority: int


class TaskFetchingUnit:
    def __init__(
        self,
        renderer: PlaceholderRenderer | None = None,
        state_owner: str = "task_fetching_unit",
    ) -> None:
        self._renderer = renderer or StrictPlaceholderRenderer()
        self._state_owner = state_owner

    def generate_dispatch_requests(
        self,
        *,
        wave_plan: WavePlanResult,
        tasks: list[TaskComponent],
        snapshot: Mapping[str, Any],
        writer_id: str,
    ) -> tuple[DispatchRequest, ...]:
        if writer_id != self._state_owner:
            raise ValueError(
                "single writer violation: "
                f"expected '{self._state_owner}', got '{writer_id}'"
            )

        tasks_by_id = self._build_task_index(tasks)
        frozen_snapshot = dict(snapshot)

        requests: list[DispatchRequest] = []
        sequence_number = 0

        for wave in sorted(
            wave_plan.waves, key=lambda current_wave: current_wave.wave_number
        ):
            for task_id in wave.task_ids:
                task = tasks_by_id.get(task_id)
                if task is None:
                    raise ValueError(f"wave references unknown task_id '{task_id}'")
                if task.status is not TaskStatus.READY:
                    continue

                requests.append(
                    DispatchRequest(
                        task_id=task.task_id,
                        wave_number=wave.wave_number,
                        sequence_number=sequence_number,
                        description=self._renderer.substitute(
                            task.description,
                            frozen_snapshot,
                        ),
                        expected_output=self._renderer.substitute(
                            task.expected_output,
                            frozen_snapshot,
                        ),
                        assigned_agent=task.assigned_agent,
                        tools=tuple(task.tools),
                        context_dependencies=tuple(task.context_dependencies),
                        priority=task.priority,
                    )
                )
                sequence_number += 1

        logger.info(
            "task_fetching_dispatch_requests_generated",
            request_count=len(requests),
            wave_count=len(wave_plan.waves),
        )
        return tuple(requests)

    def _build_task_index(self, tasks: list[TaskComponent]) -> dict[str, TaskComponent]:
        tasks_by_id: dict[str, TaskComponent] = {}
        for task in tasks:
            if task.task_id in tasks_by_id:
                raise ValueError(f"duplicate task_id '{task.task_id}'")
            tasks_by_id[task.task_id] = task
        return tasks_by_id


__all__ = ["DispatchRequest", "TaskFetchingUnit"]
