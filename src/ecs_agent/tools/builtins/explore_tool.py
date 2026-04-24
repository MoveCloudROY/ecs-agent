"""Built-in explore tool for directory tree visualization."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from ecs_agent.logging import get_logger
from ecs_agent.tools.discovery import tool

logger = get_logger(__name__)


def _build_tree(
    path: Path, workspace: Path, max_depth: int, depth: int, prefix: str
) -> list[str]:
    if depth > max_depth:
        return []

    try:
        children = sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
    except PermissionError:
        return []

    lines: list[str] = []
    count = len(children)
    for idx, child in enumerate(children):
        is_last = idx == count - 1
        connector = "└── " if is_last else "├── "
        suffix = "/" if child.is_dir() else ""
        lines.append(f"{prefix}{connector}{child.name}{suffix}")
        if child.is_dir() and depth < max_depth:
            extension = "    " if is_last else "│   "
            lines.extend(
                _build_tree(child, workspace, max_depth, depth + 1, prefix + extension)
            )

    return lines


@tool(description="Display the directory tree of a workspace path up to a given depth.")
async def explore(
    path: Annotated[
        str, "Workspace-relative path to explore. Use '.' for workspace root."
    ],
    max_depth: Annotated[int, "Maximum depth of directory tree to display (1–5)."],
    workspace_root: str,
) -> str:
    workspace = Path(workspace_root).resolve()

    if path == ".":
        target = workspace
    else:
        target = (workspace / path).resolve()
        if not target.is_relative_to(workspace):
            raise ValueError(f"Path outside workspace: {path}")

    if not target.exists():
        raise FileNotFoundError(f"Path not found: {path}")

    if not target.is_dir():
        raise ValueError(f"Path is not a directory: {path}")

    label = "." if path == "." else path
    logger.info("explore", path=path, max_depth=max_depth)
    lines = [label] + _build_tree(target, workspace, max_depth, depth=1, prefix="")
    return "\n".join(lines)
