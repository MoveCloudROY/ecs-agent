"""Tier-1 hang-family regressions.

Three permanent-hang classes from the 2026-07-18 framework review:
1. Cancelled sessions never woke `subagent_wait` futures (`is_wake_worthy`
   excluded "cancelled"; the CancelledError path enqueued no notification).
2. Exceptions escaping the background finalization code (persist, timeout
   status update race) left sessions parked in "running" with the task's
   exception unobserved.
3. `SubagentWaitSystem` had no defensive re-check: a session cancelled
   directly on the runtime manager (e.g. from a TUI inspector) hung the
   world's tick forever when `timeout=None`.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, cast

import pytest

from ecs_agent.components import (
    SubagentNotificationQueueComponent,
    SubagentSessionTableComponent,
    SubagentWaitComponent,
)
from ecs_agent.core import World
from ecs_agent.systems.subagent.background import BackgroundSessionRunner
from ecs_agent.systems.subagent.notifications import NotificationCoordinator
from ecs_agent.systems.subagent.tools import make_cancel_handler
from ecs_agent.systems.subagent_runtime import (
    SubagentRuntimeManager,
    reset_global_scheduler,
)
from ecs_agent.systems.subagent_wait import SubagentWaitSystem
from ecs_agent.types import EntityId, SubagentConfig, SubagentSessionRecord

ISO = "2026-07-19T00:00:00Z"


@pytest.fixture(autouse=True)
def _reset_scheduler() -> Any:
    reset_global_scheduler()
    yield
    reset_global_scheduler()


def _record(
    session_id: str,
    parent: EntityId,
    status: str = "running",
) -> SubagentSessionRecord:
    return SubagentSessionRecord(
        session_id=session_id,
        category="researcher",
        prompt="do it",
        parent_entity_id=parent,
        created_at=ISO,
        updated_at=ISO,
        background=True,
        status=status,  # type: ignore[arg-type]
    )


def _attach_wait(
    world: World,
    parent: EntityId,
    record: SubagentSessionRecord,
) -> SubagentWaitComponent:
    world.add_component(
        parent,
        SubagentSessionTableComponent(sessions={record.session_id: record}),
    )
    wait = SubagentWaitComponent(
        session_ids=[record.session_id],
        resolved_session_ids=[record.session_id],
        future=asyncio.get_running_loop().create_future(),
    )
    world.add_component(parent, wait)
    return wait


async def _noop_publish(*args: Any, **kwargs: Any) -> None:
    return None


def _build_background(
    manager: SubagentRuntimeManager,
    coordinator: NotificationCoordinator,
    world: World,
    parent: EntityId,
    record: SubagentSessionRecord,
    *,
    execute_core: Any,
    persist_result: Any = lambda result: None,
    publish_events: Any = _noop_publish,
) -> Any:
    runner = BackgroundSessionRunner(manager, coordinator)
    return runner.build_coroutine(
        world,
        parent,
        record.category,
        record.prompt,
        record.session_id,
        record,
        SubagentConfig(name=record.category, model=object()),
        resolved_timeout=None,
        execute_core=execute_core,
        persist_result=persist_result,
        publish_events=publish_events,
    )


# --- 1. cancelled sessions wake waiters ------------------------------------------


async def test_cancelled_session_notification_wakes_wait_future() -> None:
    """A cancelled background session is wake-worthy for subagent_wait."""
    world = World()
    parent = world.create_entity()
    record = _record("ses_cancel_wake", parent, status="running")
    wait = _attach_wait(world, parent, record)
    record.status = "cancelled"

    NotificationCoordinator().enqueue_parent_notification(world, record)

    assert isinstance(wait.future, asyncio.Future)
    assert wait.future.done()
    queue = world.get_component(parent, SubagentNotificationQueueComponent)
    assert queue is not None
    assert [n.terminal_status for n in queue.notifications] == ["cancelled"]


async def test_cancel_session_of_running_task_resolves_wait_future() -> None:
    """cancel_session on a running task must wake a pending subagent_wait."""
    world = World()
    parent = world.create_entity()
    record = _record("ses_cancel_running", parent, status="queued")
    wait = _attach_wait(world, parent, record)

    async def hang_forever(*args: Any, **kwargs: Any) -> tuple[str, bool, str | None]:
        await asyncio.Event().wait()
        return ("", True, None)

    manager = SubagentRuntimeManager()
    coroutine = _build_background(
        manager,
        NotificationCoordinator(),
        world,
        parent,
        record,
        execute_core=hang_forever,
    )
    await manager.enqueue_session(record.session_id, record, coroutine)
    # Let the admitted task start running before cancelling it.
    for _ in range(5):
        await asyncio.sleep(0)

    await manager.cancel_session(record.session_id)

    assert isinstance(wait.future, asyncio.Future)
    await asyncio.wait_for(asyncio.shield(wait.future), timeout=1.0)
    queue = world.get_component(parent, SubagentNotificationQueueComponent)
    assert queue is not None
    assert [n.terminal_status for n in queue.notifications] == ["cancelled"]


async def test_cancel_tool_on_queued_session_resolves_wait_future() -> None:
    """Cancelling a still-queued session via the tool must wake the waiter."""
    world = World()
    parent = world.create_entity()

    manager = SubagentRuntimeManager(max_background_concurrency=1)
    coordinator = NotificationCoordinator()

    # Hog the single slot so the target session stays queued.
    hog = _record("ses_hog", parent, status="queued")

    async def hog_forever() -> None:
        await asyncio.Event().wait()

    await manager.enqueue_session(hog.session_id, hog, hog_forever)

    target = _record("ses_queued_target", parent, status="queued")
    wait = _attach_wait(world, parent, target)

    async def never_runs() -> None:
        return None

    await manager.enqueue_session(target.session_id, target, never_runs)
    assert await manager.get_queue_position(target.session_id) is not None

    class _FakeSystem:
        _runtime_manager = manager
        _notification_coordinator = coordinator

    handler = make_cancel_handler(cast(Any, _FakeSystem()), world, parent)
    payload = json.loads(await handler(target.session_id))

    assert payload["lifecycle_status"] == "cancelled"
    assert isinstance(wait.future, asyncio.Future)
    await asyncio.wait_for(asyncio.shield(wait.future), timeout=1.0)

    await manager.cancel_session(hog.session_id)


# --- 2. escaped finalization exceptions must not park sessions in running --------


async def test_escaped_session_exception_forces_failed_status() -> None:
    """An exception escaping the session coroutine marks the session failed."""
    manager = SubagentRuntimeManager()
    parent = EntityId(1)
    record = _record("ses_escape", parent, status="queued")

    async def boom() -> None:
        raise RuntimeError("boom")

    await manager.enqueue_session(record.session_id, record, boom)
    task = await manager.get_task(record.session_id)
    assert task is not None
    await asyncio.wait_for(task, timeout=1.0)

    session = await manager.get_session(record.session_id)
    assert session is not None
    assert session.status == "failed"
    assert "boom" in (session.error or "")
    assert manager.get_or_create_session_event(record.session_id).is_set()
    assert manager._scheduler.active_count == 0


async def test_success_finalization_failure_marks_failed_and_wakes_waiter() -> None:
    """persist_result blowing up must finalize as failed, not park as running."""
    world = World()
    parent = world.create_entity()
    record = _record("ses_persist_fail", parent, status="queued")
    wait = _attach_wait(world, parent, record)

    async def succeed(*args: Any, **kwargs: Any) -> tuple[str, bool, str | None]:
        return ("payload", True, None)

    def exploding_persist(result: str) -> None:
        raise OSError("disk full")

    published: list[dict[str, Any]] = []

    async def record_publish(*args: Any, **kwargs: Any) -> None:
        published.append(kwargs)

    manager = SubagentRuntimeManager()
    coroutine = _build_background(
        manager,
        NotificationCoordinator(),
        world,
        parent,
        record,
        execute_core=succeed,
        persist_result=exploding_persist,
        publish_events=record_publish,
    )
    await manager.enqueue_session(record.session_id, record, coroutine)
    task = await manager.get_task(record.session_id)
    assert task is not None
    await asyncio.wait_for(task, timeout=1.0)

    session = await manager.get_session(record.session_id)
    assert session is not None
    assert session.status == "failed"
    assert "disk full" in (session.error or "")
    assert manager.get_or_create_session_event(record.session_id).is_set()
    assert isinstance(wait.future, asyncio.Future)
    assert wait.future.done()
    assert published
    assert published[-1]["success"] is False


async def test_update_timeout_tolerates_concurrently_cancelled_session() -> None:
    """The timeout-vs-cancel race must not raise out of the timeout path."""
    manager = SubagentRuntimeManager()
    record = _record("ses_race", EntityId(1), status="running")
    await manager.restore_session_metadata(record)
    await manager.update_status(record.session_id, "cancelled")

    await manager.update_timeout(record.session_id, "Error: timeout")

    session = await manager.get_session(record.session_id)
    assert session is not None
    assert session.status == "cancelled"
    assert manager.get_or_create_session_event(record.session_id).is_set()


# --- 3. defensive wait re-check ---------------------------------------------------


async def test_wait_recheck_unblocks_on_untracked_terminal_transition() -> None:
    """A terminal transition with no notification must not hang the wait forever.

    Simulates cancellation performed directly on the runtime state (e.g. a TUI
    inspector) that never goes through the notification coordinator.
    """
    world = World()
    parent = world.create_entity()
    record = _record("ses_direct_cancel", parent, status="running")
    world.add_component(
        parent,
        SubagentSessionTableComponent(sessions={record.session_id: record}),
    )
    world.add_component(
        parent,
        SubagentWaitComponent(session_ids=[record.session_id], timeout=None),
    )

    system = SubagentWaitSystem(defensive_recheck_interval=0.05)
    process_task = asyncio.create_task(system.process(world))
    await asyncio.sleep(0.02)

    record.status = "cancelled"  # direct mutation: no notification, no future

    await asyncio.wait_for(process_task, timeout=2.0)
    assert world.get_component(parent, SubagentWaitComponent) is None
