"""Tool installers and control-tool handlers for subagent delegation (Task 9).

Wires the ``subagent`` tool and the five control tools (status/wait/result/cancel/
resume) onto an entity's ``ToolRegistryComponent``. Schemas come from
``tool_schemas``; the handlers are thin adapters that parse args, call into the
``SubagentSystem`` (runtime manager, service, resume), and format JSON.

Handlers take the owning ``SubagentSystem`` and resolve ``system._...`` seams at call
time so white-box monkeypatches remain effective. ``SubagentSystem`` keeps thin
``install_subagent_tool`` / ``install_subagent_control_tools`` methods delegating here.
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any

from ecs_agent.components import (
    SubagentRegistryComponent,
    SubagentWaitComponent,
    ToolRegistryComponent,
)
from ecs_agent.components.definitions import SubagentSessionTableComponent
from ecs_agent.core.world import World
from ecs_agent.logging import get_logger
from ecs_agent.systems.subagent.tool_schemas import (
    build_cancel_schema,
    build_result_schema,
    build_resume_schema,
    build_status_schema,
    build_subagent_schema,
    build_wait_schema,
)
from ecs_agent.types import EntityId, render_subagent_session_reminder_table

if TYPE_CHECKING:
    from ecs_agent.systems.subagent import SubagentSystem

logger = get_logger(__name__)


def install_subagent_tool(
    system: SubagentSystem,
    world: World,
    entity_id: EntityId,
    tool_name: str = "subagent",
    override: bool = False,
) -> None:
    registry = world.get_component(entity_id, SubagentRegistryComponent)
    if registry is None:
        raise ValueError(f"Entity {entity_id} missing SubagentRegistryComponent")

    tool_registry = world.get_component(entity_id, ToolRegistryComponent)
    if tool_registry is None:
        raise ValueError(f"Entity {entity_id} missing ToolRegistryComponent")

    free_mode_enabled = registry.free_subagent_config.enabled
    tool_registry.tools[tool_name] = build_subagent_schema(
        tool_name, free_mode_enabled=free_mode_enabled
    )

    if tool_name in tool_registry.handlers and not override:
        return

    tool_registry.handlers[tool_name] = system._make_subagent_handler(world, entity_id)

    logger.info(
        "subagent_tool_installed",
        entity_id=entity_id,
        tool_name=tool_name,
        available_subagents=list(registry.subagents.keys()),
    )


def install_subagent_control_tools(
    system: SubagentSystem,
    world: World,
    entity_id: EntityId,
) -> None:
    table = world.get_component(entity_id, SubagentSessionTableComponent)
    if table is None:
        raise ValueError(f"Entity {entity_id} missing SubagentSessionTableComponent")

    tool_registry = world.get_component(entity_id, ToolRegistryComponent)
    if tool_registry is None:
        raise ValueError(f"Entity {entity_id} missing ToolRegistryComponent")

    # Install subagent_status tool
    tool_registry.tools["subagent_status"] = build_status_schema()
    tool_registry.handlers["subagent_status"] = make_status_handler(
        system, world, entity_id
    )

    tool_registry.tools["subagent_wait"] = build_wait_schema()
    tool_registry.handlers["subagent_wait"] = make_wait_handler(system, world, entity_id)

    # Install subagent_result tool
    tool_registry.tools["subagent_result"] = build_result_schema()
    tool_registry.handlers["subagent_result"] = make_result_handler(
        system, world, entity_id
    )

    # Install subagent_cancel tool
    tool_registry.tools["subagent_cancel"] = build_cancel_schema()
    tool_registry.handlers["subagent_cancel"] = make_cancel_handler(
        system, world, entity_id
    )

    # Install subagent_resume tool
    tool_registry.tools["subagent_resume"] = build_resume_schema()
    tool_registry.handlers["subagent_resume"] = make_resume_handler(
        system, world, entity_id
    )

    logger.info("subagent_control_tools_installed", entity_id=entity_id)


def make_status_handler(
    system: SubagentSystem, world: World, parent_entity_id: EntityId
) -> Any:
    """Create handler for subagent_status tool."""

    async def status_handler(session_id: str | None = None) -> str:
        logger.debug(
            "subagent_status_queried",
            parent_entity=parent_entity_id,
            session_id=session_id,
        )

        table = world.get_component(parent_entity_id, SubagentSessionTableComponent)
        if table is None:
            return json.dumps({"error": "SubagentSessionTableComponent not found"})

        if session_id is None:
            table_text = render_subagent_session_reminder_table(table.sessions)
            return json.dumps(
                {
                    "status": "ok",
                    "session_count": len(table.sessions),
                    "summary_table": table_text,
                }
            )

        session = table.sessions.get(session_id)
        if session is None:
            return json.dumps(
                {
                    "error": f"Session not found: {session_id}",
                    "session_id": session_id,
                }
            )

        logger.debug(
            "subagent_status_found",
            parent_entity=parent_entity_id,
            session_id=session_id,
            lifecycle_status=session.status,
        )
        queue_position: int | None = None
        if session.status == "queued":
            queue_position = await system._runtime_manager.get_queue_position(
                session.session_id
            )

        return json.dumps(
            system._session_payload(
                session,
                status="ok",
                queue_position=queue_position,
            )
        )

    return status_handler


def make_wait_handler(
    system: SubagentSystem, world: World, parent_entity_id: EntityId
) -> Any:
    async def wait_handler(
        session_ids: list[str] | None = None,
        timeout: float | str | None = None,
        auto_restart_budget: int = 0,
    ) -> str:
        if isinstance(timeout, str):
            timeout = float(timeout)

        wait_component = SubagentWaitComponent(
            session_ids=session_ids,
            timeout=timeout,
            future=None,
            started_at=system._utc_now_iso(),
            auto_restart_budget=auto_restart_budget,
        )
        world.add_component(parent_entity_id, wait_component)
        logger.info(
            "subagent_wait_requested",
            parent_entity=parent_entity_id,
            session_ids=session_ids,
            timeout=timeout,
            auto_restart_budget=auto_restart_budget,
        )
        return (
            "Waiting for background subagents. "
            "Will be notified when all sessions complete."
        )

    return wait_handler


def make_result_handler(
    system: SubagentSystem, world: World, parent_entity_id: EntityId
) -> Any:
    """Create handler for subagent_result tool."""

    async def result_handler(
        session_id: str,
        read_method: str = "full",
        timeout: float | str | None = None,
    ) -> str:
        if read_method not in {"full", "summary"}:
            return json.dumps(
                {
                    "error": (
                        f"Invalid read_method '{read_method}'. "
                        "Expected one of: full, summary"
                    ),
                    "read_method": read_method,
                    "session_id": session_id,
                }
            )

        if isinstance(timeout, str):
            timeout = float(timeout)

        logger.info(
            "subagent_result_requested",
            parent_entity=parent_entity_id,
            session_id=session_id,
            read_method=read_method,
            timeout=timeout,
        )
        session = await system._runtime_manager.get_session(session_id)
        if session is None:
            return json.dumps(
                {
                    "error": f"Session not found: {session_id}",
                    "session_id": session_id,
                }
            )

        if session.status in ("failed", "timed_out", "cancelled"):
            if read_method == "summary":
                if session.result_summary is None:
                    return json.dumps(
                        {
                            "error": 'Summary not available for this session. Retry with read_method="full".',
                            "read_method": "summary",
                            "session_id": session_id,
                        }
                    )
                return json.dumps(system._summary_payload(session, status="terminal"))
            return json.dumps(system._session_payload(session, status="terminal"))

        if session.status == "succeeded":
            logger.info(
                "subagent_result_ready",
                parent_entity=parent_entity_id,
                session_id=session_id,
                result_length=len(session.result_excerpt or ""),
            )
            if read_method == "summary":
                if session.result_summary is None:
                    return json.dumps(
                        {
                            "error": 'Summary not available for this session. Retry with read_method="full".',
                            "read_method": "summary",
                            "session_id": session_id,
                        }
                    )
                return json.dumps(system._summary_payload(session, status="success"))
            return json.dumps(system._session_payload(session, status="success"))

        logger.info(
            "subagent_result_awaiting",
            parent_entity=parent_entity_id,
            session_id=session_id,
            timeout=timeout,
        )
        # Event-driven wait (no polling): the runtime manager sets a sticky
        # per-session Event on any terminal transition. Awaiting it wakes the
        # caller immediately on completion; the Event's stickiness means a
        # transition that lands before this await still resolves it.
        event = system._runtime_manager.get_or_create_session_event(session_id)
        try:
            if timeout is None:
                await event.wait()
            else:
                await asyncio.wait_for(event.wait(), timeout)
        except asyncio.TimeoutError:
            logger.warning(
                "subagent_result_timeout",
                parent_entity=parent_entity_id,
                session_id=session_id,
                timeout=timeout,
            )
            return json.dumps(
                {
                    "error": f"Timeout waiting for session result after {timeout}s",
                    "session_id": session_id,
                    "timeout": timeout,
                }
            )
        except asyncio.CancelledError:
            logger.warning(
                "subagent_result_cancelled",
                parent_entity=parent_entity_id,
                session_id=session_id,
            )
            return json.dumps(
                {
                    "error": f"Session was cancelled: {session_id}",
                    "session_id": session_id,
                }
            )

        session = await system._runtime_manager.get_session(session_id)
        if session is None:
            return json.dumps(
                {
                    "error": f"Session disappeared while waiting: {session_id}",
                    "session_id": session_id,
                }
            )
        logger.info(
            "subagent_result_collected",
            parent_entity=parent_entity_id,
            session_id=session_id,
            lifecycle_status=session.status,
        )
        return system._terminal_result_payload(session, read_method, session_id)

    return result_handler


def make_cancel_handler(
    system: SubagentSystem, world: World, parent_entity_id: EntityId
) -> Any:
    """Create handler for subagent_cancel tool."""

    async def cancel_handler(session_id: str) -> str:
        logger.info(
            "subagent_cancel_requested",
            parent_entity=parent_entity_id,
            session_id=session_id,
        )
        session = await system._runtime_manager.get_session(session_id)
        if session is None:
            return json.dumps(
                {
                    "error": f"Session not found: {session_id}",
                    "session_id": session_id,
                }
            )

        if session.status in ("failed", "timed_out", "cancelled", "succeeded"):
            return json.dumps(
                {
                    "error": f"Session already terminal: {session.status}",
                    "session_id": session_id,
                    "lifecycle_status": session.status,
                }
            )

        await system._runtime_manager.cancel_session(session_id)
        await system._runtime_manager.sync_to_component(world, parent_entity_id)

        session = await system._runtime_manager.get_session(session_id)

        logger.info(
            "subagent_cancel_completed",
            parent_entity=parent_entity_id,
            session_id=session_id,
            final_status=session.status if session else "cancelled",
        )
        return json.dumps(
            {
                "status": "cancelled",
                "session_id": session_id,
                "lifecycle_status": session.status if session else "cancelled",
            }
        )

    return cancel_handler


def make_resume_handler(
    system: SubagentSystem, world: World, parent_entity_id: EntityId
) -> Any:
    async def resume_handler(session_id: str) -> str:
        logger.info(
            "subagent_resume_requested",
            parent_entity=parent_entity_id,
            session_id=session_id,
        )
        try:
            new_session_id = await system._resume_session(
                world, parent_entity_id, session_id
            )
        except ValueError as exc:
            return json.dumps({"error": str(exc), "session_id": session_id})

        new_session = await system._runtime_manager.get_session(new_session_id)
        return json.dumps(
            {
                "status": "resumed",
                "original_session_id": session_id,
                "new_session_id": new_session_id,
                "session_id": new_session_id,
                "category": new_session.category if new_session else "",
                "lifecycle_status": new_session.status if new_session else "queued",
            }
        )

    return resume_handler


__all__ = [
    "install_subagent_tool",
    "install_subagent_control_tools",
    "make_status_handler",
    "make_wait_handler",
    "make_result_handler",
    "make_cancel_handler",
    "make_resume_handler",
]
