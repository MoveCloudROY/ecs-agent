"""Built-in file manipulation tools."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from ecs_agent.logging import get_logger
from ecs_agent.tools.builtins.file_snapshot import (
    compute_content_digest,
    current_file_snapshot_state,
)
from ecs_agent.tools.discovery import tool

logger = get_logger(__name__)


def _validate_path(file_path: str, workspace_root: str) -> Path:
    workspace = Path(workspace_root).resolve()
    target = (workspace / file_path).resolve()

    if not target.is_relative_to(workspace):
        raise ValueError(f"Path outside workspace: {file_path}")

    return target


@tool(
    description=(
        "Read clean UTF-8 file content from workspace. "
        "Always specify offset and limit to avoid reading oversized files."
    )
)
async def read_file(
    file_path: Annotated[str, "Workspace-relative path to the file to read."],
    workspace_root: str,
    offset: Annotated[
        int | str,
        "1-based line number to start reading from. Accepts an integer or numeric string.",
    ] = 1,
    limit: Annotated[
        int | str,
        "Maximum number of lines to return. Use 0 to read all lines. Accepts an integer or numeric string.",
    ] = 0,
) -> str:
    target = _validate_path(file_path, workspace_root)
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
    return "\n".join(selected)


def _parse_non_negative_numeric_input(
    value: int | str,
    *,
    field_name: str,
    minimum: int,
) -> int:
    if isinstance(value, bool):
        raise ValueError(f"read_file {field_name} must be a numeric value: {value}")
    if isinstance(value, int):
        parsed = value
    elif value.isdecimal():
        parsed = int(value)
    else:
        raise ValueError(f"read_file {field_name} must be a numeric value: {value}")

    if parsed < minimum:
        raise ValueError(f"read_file {field_name} must be >= {minimum}: {value}")
    return parsed


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
