from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from ecs_agent.core.world import World


class System(Protocol):
    async def process(self, world: World) -> None: ...


class SystemExecutor:
    def __init__(self) -> None:
        self._systems: list[tuple[System, int]] = []

    def register(self, system: System, priority: int) -> None:
        self._systems.append((system, priority))

    async def execute(self, world: World) -> None:
        if not self._systems:
            return

        systems_by_priority: dict[int, list[System]] = {}
        for system, priority in self._systems:
            priority_systems = systems_by_priority.setdefault(priority, [])
            priority_systems.append(system)

        for priority in sorted(systems_by_priority):
            async with asyncio.TaskGroup() as task_group:
                for system in systems_by_priority[priority]:
                    task_group.create_task(system.process(world))
