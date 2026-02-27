"""Built-in file manipulation tools."""

from __future__ import annotations

from pathlib import Path

from ecs_agent.logging import get_logger
from ecs_agent.tools.discovery import tool

logger = get_logger(__name__)


def _validate_path(file_path: str, workspace_root: str) -> Path:
    """Validate target path is contained within workspace root."""
    workspace = Path(workspace_root).resolve()
    target = (workspace / file_path).resolve()

    if not target.is_relative_to(workspace):
        raise ValueError(f"Path outside workspace: {file_path}")

    return target


@tool(description="Read UTF-8 file content from workspace.")
async def read_file(file_path: str, workspace_root: str) -> str:
    """Read UTF-8 file content from workspace."""
    target = _validate_path(file_path, workspace_root)
    logger.info("read_file", file_path=file_path)
    return target.read_text(encoding="utf-8")


@tool(description="Write UTF-8 content to file in workspace.")
async def write_file(file_path: str, content: str, workspace_root: str) -> str:
    """Write UTF-8 content to file in workspace."""
    target = _validate_path(file_path, workspace_root)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    logger.info("write_file", file_path=file_path, bytes_written=len(content))
    return f"Wrote {len(content)} bytes to {file_path}"
