"""Declarative ToolSchema builders for subagent delegation + control tools.

Data-only (Task 3 of the subagent package refactor): each function returns a
``ToolSchema``. No logic, no ECS/world coupling. The descriptions and parameter
schemas are moved verbatim from ``SubagentSystem.install_*`` and are asserted by the
existing test suite — keep them byte-for-byte.
"""

from __future__ import annotations

from ecs_agent.types import ToolSchema


def build_subagent_schema(tool_name: str, *, free_mode_enabled: bool) -> ToolSchema:
    """Schema for the unified ``subagent`` delegation tool.

    The ``category`` description and the INTERFACE free-mode line vary with whether
    free-form (unregistered) subagents are enabled on the entity's registry.
    """
    category_description = (
        'Registered subagent type/name (e.g. "researcher", "coder") or, because free-form subagents are enabled, any unregistered descriptive category name.'
        if free_mode_enabled
        else 'Registered subagent type/name (e.g. "researcher", "coder"). Must match a key in SubagentRegistryComponent.'
    )
    free_mode_description = (
        "  - Free-form subagents are enabled: category may be any descriptive category name, including unregistered names.\n"
        if free_mode_enabled
        else "  - category must match a subagent registered in SubagentRegistryComponent.\n"
    )

    return ToolSchema(
        name=tool_name,
        description=(
            "Spawn a subagent to handle a self-contained task. The subagent runs in its own "
            "isolated World with inherited tools and skills, then returns its final answer.\n\n"
            "WHEN TO CALL:\n"
            "  - Use when a subtask is independent and can be fully delegated (research, "
            "    code generation, analysis, tool-heavy work).\n"
            "  - Use background=True when you want to launch multiple subagents in parallel "
            "    and collect results later via subagent_result.\n"
            "  - Use background=False (default) when you need the result before continuing.\n\n"
            "INTERFACE:\n"
            f"{free_mode_description}"
            "  prompt   (required) — full task instruction for the subagent; be specific.\n"
            "  load_skills        — extra skill names to inject on top of category defaults.\n"
            "  background         — if True, returns a JSON payload with session_id immediately; "
            "                       use subagent_result(session_id) to retrieve the answer later.\n"
            "  stream             — when background=True, mirror child-world streaming events onto the "
            "                       parent EventBus as session-scoped SubagentStream* telemetry.\n"
            "  timeout            — max seconds to wait before aborting (null = no limit).\n\n"
            "RETURNS (sync): final answer string from the subagent.\n"
            "RETURNS (background): JSON {session_id, status, category, lifecycle_status}.\n\n"
            "EXAMPLES:\n"
            "  // Synchronous — block until done\n"
            '  subagent(category="researcher", prompt="Summarize the latest papers on RAG.")\n\n'
            "  // Parallel — launch two subagents, collect later\n"
            '  subagent(category="coder", prompt="Write unit tests for auth.py.", background=True)\n'
            '  subagent(category="reviewer", prompt="Review auth.py for security issues.", background=True)\n\n'
            "  // With extra skill and timeout\n"
            '  subagent(category="analyst", prompt="Analyze Q1 sales data.", load_skills=["sql"], timeout=120)'
        ),
        parameters={
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": category_description,
                },
                "prompt": {
                    "type": "string",
                    "description": "Full task instruction for the subagent. Be explicit: include goal, context, expected output format, and any constraints.",
                },
                "load_skills": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Extra skill names to load on top of this category's defaults. Use when the task requires capabilities not in the base skill set.",
                },
                "background": {
                    "type": "boolean",
                    "description": "If true, launch the subagent asynchronously and return immediately with a session_id. Use subagent_result(session_id) to collect the answer later. Default false (synchronous — blocks until done).",
                },
                "stream": {
                    "type": "boolean",
                    "description": "If true, this session bridges streaming token events to the parent entity's EventBus as SubagentStreamDeltaEvent. Only meaningful when background=True. Default false.",
                },
                "timeout": {
                    "type": ["number", "null"],
                    "description": "Maximum seconds to wait before aborting. null means no timeout. Only respected in sync mode and for background collection in subagent_result.",
                },
            },
            "required": ["category", "prompt"],
        },
    )


def build_status_schema() -> ToolSchema:
    return ToolSchema(
        name="subagent_status",
        description=(
            "Check the status of background subagent sessions. Use this to decide when "
            "to call subagent_result.\n\n"
            "WHEN TO CALL:\n"
            "  - After launching one or more background subagents (background=True) to see "
            "    which have succeeded and which are still running.\n"
            "  - Without arguments to get a summary table of all active sessions.\n"
            "  - With a specific session_id to get detailed info on one session.\n\n"
            "INTERFACE:\n"
            "  session_id (optional) — omit to list all sessions; provide to inspect one.\n\n"
            "RETURNS (no session_id): JSON {status, session_count, summary_table}.\n"
            "RETURNS (with session_id): JSON {session_id, status, category, lifecycle_status, ...}.\n\n"
            "EXAMPLES:\n"
            "  // List all running background sessions\n"
            "  subagent_status()\n\n"
            "  // Inspect a specific session\n"
            '  subagent_status(session_id="ses_abc123")'
        ),
        parameters={
            "type": "object",
            "properties": {
                "session_id": {
                    "type": ["string", "null"],
                    "description": "Session ID returned by a background subagent call. Omit to list all active sessions.",
                }
            },
            "required": [],
        },
    )


def build_wait_schema() -> ToolSchema:
    return ToolSchema(
        name="subagent_wait",
        description=(
            "Wait as a barrier until ALL background subagent sessions in the scope "
            "reach a terminal state. Use this after launching all useful background "
            "subagents so the parent can stop polling and resume only when every "
            "session has completed (succeeded, failed, timed_out, or cancelled).\n\n"
            "INTERFACE:\n"
            "  session_ids (optional) — the wait scope. Omit to snapshot all currently "
            "    active sessions at wait-start. Explicit IDs define a fixed scope.\n"
            "  timeout     (optional) — per-period check interval in seconds. If running "
            "    sessions remain at timeout, the deadline is extended automatically. "
            "    If any sessions have failed or are missing, a role=\"user\" notification "
            "    with subagent_resume instructions is injected. null = wait indefinitely.\n\n"
            "RETURNS: acknowledgment string immediately; waiting happens in "
            "SubagentWaitSystem."
        ),
        parameters={
            "type": "object",
            "properties": {
                "session_ids": {
                    "type": ["array", "null"],
                    "items": {"type": "string"},
                    "description": "Optional session IDs to wait for. Omit or null to wait for all currently active background sessions. The wait resolves only when every session in the scope reaches a terminal status.",
                },
                "timeout": {
                    "type": ["number", "null"],
                    "description": "Max seconds per wait period. If sessions are still running when the timeout fires, the deadline is extended automatically. If any sessions have failed, a failure notification is injected for the LLM to act on. null = wait indefinitely.",
                },
                "auto_restart_budget": {
                    "type": "integer",
                    "description": "Max automatic restarts per failed session before surfacing failures to the LLM. 0 (default) = disabled; failed sessions are immediately surfaced for LLM-driven subagent_resume. >0 = the wait system auto-restarts failed sessions up to this budget before surfacing.",
                    "default": 0,
                },
            },
            "required": [],
        },
    )


def build_result_schema() -> ToolSchema:
    return ToolSchema(
        name="subagent_result",
        description=(
            "Block until a background subagent session finishes, then return its result.\n\n"
            "WHEN TO CALL:\n"
            "  - After launching a subagent with background=True and you are ready to use "
            "    its output.\n"
            "  - You may call subagent_status first to check if the session is already succeeded "
            "    (avoiding unnecessary blocking).\n\n"
            "INTERFACE:\n"
            "  session_id (required) — the session_id from the background subagent response.\n"
            "  timeout    (optional) — max seconds to wait; null = wait indefinitely.\n"
            "  read_method (optional) — 'full' (default) returns the complete result; "
            "'summary' returns the cached summary captured by the background "
            "subagent (cheaper). If no summary is cached, returns an error.\n\n"
            "RETURNS: final answer string from the subagent, or an error/timeout message.\n\n"
            "EXAMPLES:\n"
            "  // Wait for a previously launched background subagent\n"
            '  subagent_result(session_id="ses_abc123")\n\n'
            "  // Wait with a 60-second timeout\n"
            '  subagent_result(session_id="ses_abc123", timeout=60)\n\n'
            "  // Fetch only the cached summary (background sessions that used the result envelope)\n"
            '  subagent_result(session_id="ses_abc123", read_method="summary")'
        ),
        parameters={
            "type": "object",
            "properties": {
                "session_id": {
                    "type": "string",
                    "description": "Session ID of the background subagent to wait for. Obtained from the subagent(background=True) response.",
                },
                "timeout": {
                    "type": ["number", "null"],
                    "description": "Max seconds to wait before returning a timeout error. null = wait indefinitely until the session finishes.",
                },
                "read_method": {
                    "type": ["string", "null"],
                    "enum": ["full", "summary"],
                    "description": (
                        "How to read the result. "
                        "'full' (default) returns the complete subagent output. "
                        "'summary' returns the cached summary captured by the subagent "
                        "via the <subagent_background_result> envelope — much cheaper "
                        "than fetching the full result. "
                        "If summary is not available, an error payload is returned."
                    ),
                    "default": "full",
                },
            },
            "required": ["session_id"],
        },
    )


def build_cancel_schema() -> ToolSchema:
    return ToolSchema(
        name="subagent_cancel",
        description=(
            "Abort a running background subagent session and free its resources.\n\n"
            "WHEN TO CALL:\n"
            "  - When a background subagent is no longer needed (e.g. another session "
            "    already produced the answer, or the task was superseded).\n"
            "  - After a timeout or error, to clean up a stuck session.\n\n"
            "INTERFACE:\n"
            "  session_id (required) — the session_id to cancel.\n\n"
            "RETURNS: JSON {status, session_id, lifecycle_status}.\n\n"
            "EXAMPLES:\n"
            "  // Cancel a session that is no longer needed\n"
            '  subagent_cancel(session_id="ses_abc123")'
        ),
        parameters={
            "type": "object",
            "properties": {
                "session_id": {
                    "type": "string",
                    "description": "Session ID of the background subagent to abort. Obtained from the subagent(background=True) response.",
                }
            },
            "required": ["session_id"],
        },
    )


def build_resume_schema() -> ToolSchema:
    return ToolSchema(
        name="subagent_resume",
        description=(
            "Restart a failed, timed-out, or cancelled background subagent session.\n\n"
            "WHEN TO CALL:\n"
            "  - After a background subagent has failed, timed out, or been cancelled.\n"
            "  - When you want to retry a failed subtask with the same configuration.\n"
            "  - After subagent_wait surfaces timeout failures, call this for each\n"
            "    failed session_id to restart it, then call subagent_wait again.\n\n"
            "INTERFACE:\n"
            "  session_id (required) — the session_id of the failed/timed_out/cancelled session.\n\n"
            "RETURNS: JSON {status, original_session_id, new_session_id, category, lifecycle_status}.\n\n"
            "EXAMPLES:\n"
            '  subagent_resume(session_id="ses_abc123")'
        ),
        parameters={
            "type": "object",
            "properties": {
                "session_id": {
                    "type": "string",
                    "description": (
                        "Session ID of the failed/timed_out/cancelled "
                        "background subagent to restart. The new session "
                        "inherits the original category, prompt, skills, "
                        "and timeout."
                    ),
                }
            },
            "required": ["session_id"],
        },
    )


__all__ = [
    "build_subagent_schema",
    "build_status_schema",
    "build_wait_schema",
    "build_result_schema",
    "build_cancel_schema",
    "build_resume_schema",
]
