from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Protocol

from ecs_agent.logging import STANDARD_EVENT_NAMES, get_logger

if TYPE_CHECKING:
    from ecs_agent.core.world import World

logger = get_logger(__name__)

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
                    task_group.create_task(self._execute_system(system, world))

    async def _execute_system(self, system: System, world: World) -> None:
        """Execute a single system with logging and error handling."""
        system_name = f"{system.__class__.__module__}.{system.__class__.__name__}"
        logger.info(STANDARD_EVENT_NAMES["SYSTEM_START"], system=system_name)

        start_time = time.monotonic()
        try:
            await system.process(world)
            duration_ms = (time.monotonic() - start_time) * 1000
            logger.info(
                STANDARD_EVENT_NAMES["SYSTEM_COMPLETE"],
                system=system_name,
                duration_ms=duration_ms,
            )
        except Exception as exc:
            duration_ms = (time.monotonic() - start_time) * 1000
            logger.error(
                STANDARD_EVENT_NAMES["SYSTEM_ERROR"],
                system=system_name,
                exception=str(exc),
                duration_ms=duration_ms,
            )
            raise
