"""TDD tests for event-driven subagent_result waiting (Task 13).

subagent_result no longer polls the session table every 0.1s; it awaits a per-session
asyncio.Event in SubagentRuntimeManager that is set on any terminal transition. The
Event is sticky, so a terminal transition that happens before the waiter starts still
wakes it. All existing JSON response shapes are preserved.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from ecs_agent.components import (
    SubagentSessionTableComponent,
    ToolRegistryComponent,
)
from ecs_agent.core import World
from ecs_agent.systems.subagent import SubagentSystem
from ecs_agent.systems.subagent_runtime import (
    SubagentRuntimeManager,
    reset_global_scheduler,
)
from ecs_agent.types import EntityId, SubagentSessionRecord


ISO = "2026-07-03T00:00:00Z"


def _record(session_id: str, status: str = "running") -> SubagentSessionRecord:
    return SubagentSessionRecord(
        session_id=session_id,
        category="researcher",
        prompt="do it",
        parent_entity_id=EntityId(1),
        created_at=ISO,
        updated_at=ISO,
        background=True,
        status=status,  # type: ignore[arg-type]
    )


@pytest.fixture(autouse=True)
def _reset_scheduler() -> None:
    reset_global_scheduler()
    yield
    reset_global_scheduler()


# --- runtime manager event contract ---------------------------------------------


async def test_manager_session_event_is_stable_and_initially_unset() -> None:
    mgr = SubagentRuntimeManager()
    ev1 = mgr.get_or_create_session_event("s1")
    ev2 = mgr.get_or_create_session_event("s1")
    assert ev1 is ev2
    assert not ev1.is_set()


async def test_update_status_to_terminal_signals_event() -> None:
    mgr = SubagentRuntimeManager()
    await mgr.restore_session_metadata(_record("s1", status="running"))
    ev = mgr.get_or_create_session_event("s1")
    await mgr.update_status("s1", "succeeded")
    assert ev.is_set()


async def test_update_status_nonterminal_does_not_signal_event() -> None:
    mgr = SubagentRuntimeManager()
    await mgr.restore_session_metadata(_record("s1", status="queued"))
    ev = mgr.get_or_create_session_event("s1")
    await mgr.update_status("s1", "running")
    assert not ev.is_set()


async def test_update_timeout_signals_event() -> None:
    mgr = SubagentRuntimeManager()
    await mgr.restore_session_metadata(_record("s1", status="running"))
    ev = mgr.get_or_create_session_event("s1")
    await mgr.update_timeout("s1", "Error: timed out")
    assert ev.is_set()


async def test_terminal_before_waiter_is_sticky() -> None:
    # Signal happens BEFORE anyone gets the event: a later waiter must still wake.
    mgr = SubagentRuntimeManager()
    await mgr.restore_session_metadata(_record("s1", status="running"))
    await mgr.update_status("s1", "succeeded")
    ev = mgr.get_or_create_session_event("s1")
    await asyncio.wait_for(ev.wait(), timeout=1.0)  # must not hang
    assert ev.is_set()


# --- result handler behavior ----------------------------------------------------


def _system_with_session(record: SubagentSessionRecord) -> tuple[SubagentSystem, object]:
    system = SubagentSystem()
    world = World()
    parent = world.create_entity()
    world.add_component(parent, SubagentSessionTableComponent())
    world.add_component(parent, ToolRegistryComponent(tools={}, handlers={}))
    system.install_subagent_control_tools(world, parent)
    handler = world.get_component(parent, ToolRegistryComponent).handlers[
        "subagent_result"
    ]
    return system, handler


async def test_result_handler_returns_immediately_when_terminal() -> None:
    system, handler = _system_with_session(_record("s1", status="succeeded"))
    await system._runtime_manager.restore_session_metadata(_record("s1", "succeeded"))
    payload = json.loads(await asyncio.wait_for(handler(session_id="s1"), timeout=1.0))
    assert payload["lifecycle_status"] == "succeeded"
    assert payload["session_id"] == "s1"


async def test_result_handler_wakes_when_session_completes() -> None:
    system, handler = _system_with_session(_record("s1", status="running"))
    await system._runtime_manager.restore_session_metadata(_record("s1", "running"))

    waiter = asyncio.create_task(handler(session_id="s1"))
    await asyncio.sleep(0.05)  # let the waiter reach the event
    await system._runtime_manager.update_status("s1", "succeeded")

    payload = json.loads(await asyncio.wait_for(waiter, timeout=1.0))
    assert payload["lifecycle_status"] == "succeeded"


async def test_result_handler_timeout_returns_timeout_json() -> None:
    system, handler = _system_with_session(_record("s1", status="running"))
    await system._runtime_manager.restore_session_metadata(_record("s1", "running"))
    payload = json.loads(
        await asyncio.wait_for(handler(session_id="s1", timeout=0.1), timeout=2.0)
    )
    assert "Timeout waiting for session result" in payload["error"]
    assert payload["session_id"] == "s1"


async def test_result_handler_cancel_while_waiting_returns_terminal() -> None:
    system, handler = _system_with_session(_record("s1", status="running"))
    await system._runtime_manager.restore_session_metadata(_record("s1", "running"))

    waiter = asyncio.create_task(handler(session_id="s1"))
    await asyncio.sleep(0.05)
    await system._runtime_manager.cancel_session("s1")

    payload = json.loads(await asyncio.wait_for(waiter, timeout=1.0))
    assert payload["lifecycle_status"] == "cancelled"


async def test_result_handler_concurrent_waiters_both_resolve() -> None:
    system, handler = _system_with_session(_record("s1", status="running"))
    await system._runtime_manager.restore_session_metadata(_record("s1", "running"))

    w1 = asyncio.create_task(handler(session_id="s1"))
    w2 = asyncio.create_task(handler(session_id="s1"))
    await asyncio.sleep(0.05)
    await system._runtime_manager.update_status("s1", "succeeded")

    r1, r2 = await asyncio.wait_for(asyncio.gather(w1, w2), timeout=1.0)
    assert json.loads(r1)["lifecycle_status"] == "succeeded"
    assert json.loads(r2)["lifecycle_status"] == "succeeded"
