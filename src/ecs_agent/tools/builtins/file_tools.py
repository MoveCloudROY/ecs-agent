"""Built-in file manipulation tools."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from ecs_agent.logging import get_logger
from ecs_agent.tools.discovery import tool
from ecs_agent.tools.builtins.edit_tool import format_file_with_hashes

logger = get_logger(__name__)


def _validate_path(file_path: str, workspace_root: str) -> Path:
    workspace = Path(workspace_root).resolve()
    target = (workspace / file_path).resolve()

    if not target.is_relative_to(workspace):
        raise ValueError(f"Path outside workspace: {file_path}")

    return target


@tool(description="Read UTF-8 file content from workspace.")
async def read_file(
    file_path: Annotated[str, "Workspace-relative path to the file to read."],
    workspace_root: str,
) -> str:
    target = _validate_path(file_path, workspace_root)
    logger.info("read_file", file_path=file_path)
    content = target.read_text(encoding="utf-8")
    return format_file_with_hashes(content)


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
