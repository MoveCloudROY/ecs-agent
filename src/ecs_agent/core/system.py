from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol

from ecs_agent.logging import STANDARD_EVENT_NAMES, get_logger

if TYPE_CHECKING:
    from ecs_agent.core.world import World

logger = get_logger(__name__)


class System(Protocol):
    async def process(self, world: World) -> None: ...


class SystemExecutor:
    def __init__(self) -> None:
        self._systems: list[tuple[System, int, str]] = []
        self._pending_ops: list[Callable[[], None]] = []

    def register(self, system: System, priority: int, handle: str | None = None) -> str:
        actual_handle = handle or str(uuid.uuid4())
        self._pending_ops.append(
            lambda: self._systems.append((system, priority, actual_handle))
        )
        return actual_handle

    def remove(self, handle: str) -> None:
        self._pending_ops.append(lambda: self._remove_by_handle(handle))

    def replace(self, handle: str, new_system: System) -> None:
        self._pending_ops.append(lambda: self._replace_by_handle(handle, new_system))

    def _remove_by_handle(self, handle: str) -> None:
        self._systems = [s for s in self._systems if s[2] != handle]

    def _replace_by_handle(self, handle: str, new_system: System) -> None:
        for i, (system, priority, h) in enumerate(self._systems):
            if h == handle:
                self._systems[i] = (new_system, priority, handle)
                break

    def apply_pending(self) -> None:
        for op in self._pending_ops:
            op()
        self._pending_ops.clear()

    async def execute(self, world: World) -> None:
        if not self._systems:
            return

        systems_by_priority: dict[int, list[System]] = {}
        for system, priority, _ in self._systems:
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
