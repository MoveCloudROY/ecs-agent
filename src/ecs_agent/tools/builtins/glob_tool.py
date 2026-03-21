"""Built-in glob tool for workspace file discovery."""

from __future__ import annotations

from pathlib import Path

from ecs_agent.logging import get_logger
from ecs_agent.tools.discovery import tool

logger = get_logger(__name__)


@tool(description="Find files matching a glob pattern in a workspace subtree.")
async def glob(pattern: str, base_path: str, workspace_root: str) -> str:
    """Find files matching a glob pattern in a workspace subtree.

    Args:
        pattern: Glob expression (e.g. "**/*.py", "*.txt")
        base_path: Workspace-relative subdirectory to search in (e.g. "src" or ".")
        workspace_root: Absolute path to the workspace root (injected automatically)

    Returns:
        Newline-delimited sorted relative paths (relative to workspace_root).
        Returns empty string if no matches.

    Raises:
        ValueError: If base_path resolves outside workspace_root.
    """
    workspace = Path(workspace_root).resolve()

    # Handle base_path == "." as the workspace itself
    if base_path == ".":
        base_dir = workspace
    else:
        # Resolve the base_path and check it's within workspace
        base_dir = (workspace / base_path).resolve()
        if not base_dir.is_relative_to(workspace):
            raise ValueError(f"Path outside workspace: {base_path}")

    # Find all matching files
    matches = sorted(base_dir.glob(pattern))
    relative_paths = []
    for match in matches:
        if match.is_file():
            try:
                rel = match.relative_to(workspace)
                relative_paths.append(str(rel))
            except ValueError:
                # Skip if somehow outside workspace (shouldn't happen)
                continue

    logger.info(
        "glob", pattern=pattern, base_path=base_path, matches=len(relative_paths)
    )
    return "\n".join(relative_paths)
