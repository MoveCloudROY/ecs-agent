"""Built-in glob tool for workspace file discovery."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from ecs_agent.logging import get_logger
from ecs_agent.tools.discovery import tool

logger = get_logger(__name__)


@tool(
    description="Find files matching a glob pattern in a workspace subtree.",
    concurrency_safe=True,
)
async def glob(
    pattern: Annotated[
        str, "Glob expression to match files (e.g. '**/*.py', '*.txt')."
    ],
    base_path: Annotated[
        str,
        "Workspace-relative subdirectory to search in. Use '.' for the workspace root.",
    ],
    workspace_root: str,
) -> str:
    workspace = Path(workspace_root).resolve()

    if base_path == ".":
        base_dir = workspace
    else:
        base_dir = (workspace / base_path).resolve()
        if not base_dir.is_relative_to(workspace):
            raise ValueError(f"Path outside workspace: {base_path}")

    matches = sorted(base_dir.glob(pattern))
    relative_paths = []
    for match in matches:
        if match.is_file():
            try:
                rel = match.relative_to(workspace)
                relative_paths.append(str(rel))
            except ValueError:
                continue

    logger.info(
        "glob", pattern=pattern, base_path=base_path, matches=len(relative_paths)
    )
    return "\n".join(relative_paths)
