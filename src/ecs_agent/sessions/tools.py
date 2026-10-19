"""Model-visible session discovery and search using the actual tool caller."""

from dataclasses import asdict
import json
from typing import Annotated

from ecs_agent.components import SessionPolicyComponent, SessionRefComponent
from ecs_agent.sessions.service import SessionService
from ecs_agent.tools.context import current_tool_context
from ecs_agent.tools.discovery import tool


def session_tool_caller() -> tuple[SessionService, str, SessionPolicyComponent]:
    """Resolve identity on each invocation; inherited handlers carry no parent grant."""
    context = current_tool_context()
    service = context.world.session_service
    if service is None:
        raise RuntimeError("Session recording is not enabled in this World")
    ref = context.world.get_component(context.entity_id, SessionRefComponent)
    if ref is None:
        raise RuntimeError("Session identity is not installed for this entity")
    policy = context.world.get_component(context.entity_id, SessionPolicyComponent)
    if policy is None:
        raise RuntimeError("Session policy is not installed for this entity")
    return service, ref.session_id, policy


@tool(description="Discover permitted agent sessions. Pass read_uri to read_file to inspect a fixed snapshot; repeat discovery to refresh.", concurrency_safe=True, inline_result=True)
async def sessions_list(
    parent_session_id: Annotated[str | None, "Optional parent session filter."] = None,
    agent_name: Annotated[str | None, "Optional exact agent name filter."] = None,
    cursor: Annotated[str | None, "Opaque next_cursor from the previous page with the same filters."] = None,
    limit: Annotated[int, "Maximum results, default 20, bounded to 100."] = 20,
) -> str:
    service, caller_id, policy = session_tool_caller()
    page = service.list_sessions(caller_id, policy, parent_session_id=parent_session_id, agent_name=agent_name, cursor=cursor, limit=limit)
    return json.dumps(asdict(page), ensure_ascii=False)


@tool(description="Search literal text in permitted sessions, including archived tool results. Use each hit's read_uri with read_file, and cite session_id and record_id.", concurrency_safe=True, inline_result=True)
async def sessions_search(
    query: Annotated[str, "Literal text to find, including Chinese; not a regular expression."],
    session_id: Annotated[str | None, "Optional exact session filter."] = None,
    agent_name: Annotated[str | None, "Optional exact agent name filter."] = None,
    cursor: Annotated[str | None, "Opaque next_cursor with unchanged query and filters."] = None,
    limit: Annotated[int, "Maximum results, default 20, bounded to 100."] = 20,
) -> str:
    service, caller_id, policy = session_tool_caller()
    page = service.search(caller_id, policy, query, session_id=session_id, agent_name=agent_name, cursor=cursor, limit=limit)
    return json.dumps(asdict(page), ensure_ascii=False)
