from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from ecs_agent.logging import STANDARD_EVENT_NAMES, get_logger
from ecs_agent.types import (
    SystemExecutionCompletedEvent,
    SystemExecutionStartedEvent,
    SystemHandle,
)

if TYPE_CHECKING:
    from ecs_agent.core.world import World

logger = get_logger(__name__)


class System(Protocol):
    async def process(self, world: World) -> None: ...


@dataclass(slots=True)
class _SystemEntry:
    handle: SystemHandle
    system: System
    priority: int
    order: int


@dataclass(slots=True)
class _RegisterOperation:
    handle: SystemHandle
    system: System
    priority: int


@dataclass(slots=True)
class _RemoveOperation:
    handle: SystemHandle


@dataclass(slots=True)
class _ReplaceOperation:
    handle: SystemHandle
    system: System
    priority: int | None


class SystemExecutor:
    def __init__(self) -> None:
        self._systems: list[_SystemEntry] = []
        self._pending_operations: list[
            _RegisterOperation | _RemoveOperation | _ReplaceOperation
        ] = []
        self._next_handle_id = 1
        self._next_order = 0

    def register(self, system: System, priority: int) -> SystemHandle:
        handle = SystemHandle(f"system_{self._next_handle_id}")
        self._next_handle_id += 1
        self._pending_operations.append(
            _RegisterOperation(handle=handle, system=system, priority=priority)
        )
        return handle

    def remove(self, handle: SystemHandle) -> None:
        self._pending_operations.append(_RemoveOperation(handle=handle))

    def replace(
        self, handle: SystemHandle, system: System, priority: int | None = None
    ) -> None:
        self._pending_operations.append(
            _ReplaceOperation(handle=handle, system=system, priority=priority)
        )

    def apply_queued_operations(self) -> None:
        for operation in self._pending_operations:
            if isinstance(operation, _RegisterOperation):
                self._systems.append(
                    _SystemEntry(
                        handle=operation.handle,
                        system=operation.system,
                        priority=operation.priority,
                        order=self._next_order,
                    )
                )
                self._next_order += 1
            elif isinstance(operation, _RemoveOperation):
                self._systems = [
                    entry for entry in self._systems if entry.handle != operation.handle
                ]
            else:
                for index, entry in enumerate(self._systems):
                    if entry.handle == operation.handle:
                        replacement_priority = (
                            operation.priority
                            if operation.priority is not None
                            else entry.priority
                        )
                        self._systems[index] = _SystemEntry(
                            handle=entry.handle,
                            system=operation.system,
                            priority=replacement_priority,
                            order=entry.order,
                        )
                        break

        self._pending_operations.clear()

    async def execute(self, world: World) -> None:
        self.apply_queued_operations()

        if not self._systems:
            return

        systems_by_priority: dict[int, list[System]] = {}
        ordered_systems = sorted(
            self._systems, key=lambda entry: (entry.priority, entry.order)
        )
        for entry in ordered_systems:
            priority_systems = systems_by_priority.setdefault(entry.priority, [])
            priority_systems.append(entry.system)

        for priority in sorted(systems_by_priority):
            async with asyncio.TaskGroup() as task_group:
                for system in systems_by_priority[priority]:
                    task_group.create_task(self._execute_system(system, world))

    async def _execute_system(self, system: System, world: World) -> None:
        """Execute a single system with logging and error handling."""
        system_name = f"{system.__class__.__module__}.{system.__class__.__name__}"
        logger.info(STANDARD_EVENT_NAMES["SYSTEM_START"], system=system_name)
        await world.event_bus.publish(SystemExecutionStartedEvent(system=system_name))

        start_time = time.monotonic()
        try:
            await system.process(world)
            duration_seconds = time.monotonic() - start_time
            duration_ms = duration_seconds * 1000
            logger.info(
                STANDARD_EVENT_NAMES["SYSTEM_COMPLETE"],
                system=system_name,
                duration_ms=duration_ms,
            )
            await world.event_bus.publish(
                SystemExecutionCompletedEvent(
                    system=system_name,
                    status="success",
                    duration_seconds=duration_seconds,
                )
            )
        except Exception as exc:
            duration_seconds = time.monotonic() - start_time
            duration_ms = duration_seconds * 1000
            logger.error(
                STANDARD_EVENT_NAMES["SYSTEM_ERROR"],
                system=system_name,
                exception=str(exc),
                duration_ms=duration_ms,
            )
            await world.event_bus.publish(
                SystemExecutionCompletedEvent(
                    system=system_name,
                    status="error",
                    duration_seconds=duration_seconds,
                )
            )
            raise
