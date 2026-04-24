"""Built-in file manipulation tools."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from ecs_agent.logging import get_logger
from ecs_agent.tools.discovery import tool
from ecs_agent.tools.builtins.edit_tool import compute_line_hash, format_file_with_hashes  # noqa: F401

logger = get_logger(__name__)


def format_file_with_hashes_range(lines: list[str], start_line: int) -> str:
    rendered = []
    for i, content in enumerate(lines, start=start_line):
        line_hash = compute_line_hash(i, content)
        rendered.append(f"{i}#{line_hash}|{content}")
    return "\n".join(rendered)


def _validate_path(file_path: str, workspace_root: str) -> Path:
    workspace = Path(workspace_root).resolve()
    target = (workspace / file_path).resolve()

    if not target.is_relative_to(workspace):
        raise ValueError(f"Path outside workspace: {file_path}")

    return target


@tool(
    description=(
        "Read UTF-8 file content from workspace with hash-anchored line numbers. "
        "Always specify offset and limit to avoid reading oversized files."
    )
)
async def read_file(
    file_path: Annotated[str, "Workspace-relative path to the file to read."],
    workspace_root: str,
    offset: Annotated[int, "1-based line number to start reading from."] = 1,
    limit: Annotated[
        int, "Maximum number of lines to return. Use 0 to read all lines."
    ] = 0,
) -> str:
    target = _validate_path(file_path, workspace_root)
    logger.info("read_file", file_path=file_path, offset=offset, limit=limit)
    content = target.read_text(encoding="utf-8")
    if not content:
        return ""
    all_lines = content.splitlines()
    start = max(0, offset - 1)
    if start >= len(all_lines):
        return ""
    selected = all_lines[start:] if limit <= 0 else all_lines[start : start + limit]
    return format_file_with_hashes_range(selected, start_line=start + 1)


@tool(description="Write UTF-8 content to file in workspace.")
async def write_file(
    file_path: Annotated[
        str, "Workspace-relative path to the file to write or overwrite."
    ],
    content: Annotated[str, "Full UTF-8 text content to write to the file."],
    workspace_root: str,
) -> str:
    target = _validate_path(file_path, workspace_root)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    logger.info("write_file", file_path=file_path, bytes_written=len(content))
    return f"Wrote {len(content)} bytes to {file_path}"
