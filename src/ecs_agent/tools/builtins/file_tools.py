"""Built-in file manipulation tools."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Annotated

from ecs_agent.logging import get_logger
from ecs_agent.tools.builtins._numeric import parse_bounded_integer
from ecs_agent.tools.builtins.file_snapshot import (
    compute_content_digest,
    current_file_snapshot_state,
)
from ecs_agent.tools.discovery import get_tool_definition, tool

logger = get_logger(__name__)


def _validate_path(file_path: str, workspace_root: str) -> Path:
    if re.match(r"^[A-Za-z][A-Za-z0-9+.-]*://", file_path) or file_path.startswith("session:"):
        raise ValueError("URI resources are read-only and are not filesystem paths")
    workspace = Path(workspace_root).resolve()
    target = (workspace / file_path).resolve()

    if not target.is_relative_to(workspace):
        raise ValueError(f"Path outside workspace: {file_path}")

    return target


@tool(
    description=(
        "Read clean UTF-8 file content from workspace, or a session:// read_uri from session discovery/search. "
        "Session resources are read-only; limit=0 means 80 lines, capped at 12000 characters. Follow next_offset with the same URI. "
        "Always specify offset and limit to avoid reading oversized files. "
        "Use line_numbers=true before edit_file to see exact absolute line positions, including blank lines."
    ),
    concurrency_safe=True,
    inline_result=True,
)
async def read_file(
    file_path: Annotated[str, "Workspace-relative file path or session:// read_uri."],
    workspace_root: str,
    offset: Annotated[
        int | str,
        "1-based line number to start reading from. Accepts an integer or numeric string.",
    ] = 1,
    limit: Annotated[
        int | str,
        "Maximum lines. 0 means all lines for files, or a bounded 80-line page for session resources. Accepts an integer or numeric string.",
    ] = 0,
    line_numbers: Annotated[
        bool,
        "Prefix each returned line with its absolute 1-based line number. Use true when preparing an edit_file operation.",
    ] = False,
) -> str:
    normalized_offset = _parse_non_negative_numeric_input(
        offset,
        field_name="offset",
        minimum=1,
    )
    normalized_limit = _parse_non_negative_numeric_input(
        limit,
        field_name="limit",
        minimum=0,
    )
    if file_path.startswith("session:"):
        from ecs_agent.sessions.resources import read_session_resource
        from ecs_agent.sessions.tools import session_tool_caller

        service, caller_id, policy = session_tool_caller()
        return read_session_resource(service, caller_id, policy, file_path, offset=normalized_offset, limit=normalized_limit, line_numbers=line_numbers)
    target = _validate_path(file_path, workspace_root)
    logger.info(
        "read_file",
        file_path=file_path,
        offset=normalized_offset,
        limit=normalized_limit,
    )
    content = target.read_text(encoding="utf-8")
    current_file_snapshot_state().record_read(
        file_path,
        target,
        content,
        offset=normalized_offset,
        limit=normalized_limit,
    )
    if not content:
        return ""
    all_lines = content.splitlines()
    start = max(0, normalized_offset - 1)
    if start >= len(all_lines):
        return ""
    selected = (
        all_lines[start:]
        if normalized_limit <= 0
        else all_lines[start : start + normalized_limit]
    )
    if line_numbers:
        return "\n".join(f"{start + index + 1}: {line}" for index, line in enumerate(selected))
    return "\n".join(selected)


# Capability metadata survives the built-in workspace wrapper. Installers verify
# this before registering query tools and never replace an application's reader.
setattr(read_file, "supports_session_resources", True)
setattr(get_tool_definition(read_file).handler, "supports_session_resources", True)


def _parse_non_negative_numeric_input(
    value: int | str,
    *,
    field_name: str,
    minimum: int,
) -> int:
    message = f"read_file {field_name} must be an integer >= {minimum}: {value}"
    return parse_bounded_integer(
        value,
        minimum=minimum,
        invalid_message=message,
        range_message=message,
    )


@tool(description="Write UTF-8 content to file in workspace.")
async def write_file(
    file_path: Annotated[
        str, "Workspace-relative path to the file to write or overwrite."
    ],
    content: Annotated[str, "Full UTF-8 text content to write to the file."],
    workspace_root: str,
) -> str:
    target = _validate_path(file_path, workspace_root)
    if target.exists():
        original = target.read_text(encoding="utf-8")
        snapshot = current_file_snapshot_state().latest_for(target)
        if snapshot is None:
            raise ValueError("Use read_file before overwriting an existing file")
        if compute_content_digest(original) != snapshot.content_digest:
            raise ValueError("File changed since it was last read; call read_file again")

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    current_file_snapshot_state().record_read(
        file_path,
        target,
        content,
        offset=1,
        limit=0,
    )
    logger.info("write_file", file_path=file_path, bytes_written=len(content))
    return f"Wrote {len(content)} bytes to {file_path}"
