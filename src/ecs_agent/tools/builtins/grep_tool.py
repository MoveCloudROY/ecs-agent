"""Built-in grep tool — uses ripgrep (rg) when available, falls back to Python re."""

from __future__ import annotations

import asyncio
import re
import shutil
from pathlib import Path
from typing import Annotated

from ecs_agent.logging import get_logger
from ecs_agent.tools.discovery import tool

logger = get_logger(__name__)

# Resolved once at import time; None means rg is not on PATH.
_RG_BIN: str | None = shutil.which("rg")


def _validate_path(file_path: str, workspace_root: str) -> Path:
    workspace = Path(workspace_root).resolve()
    target = (workspace / file_path).resolve()
    if not target.is_relative_to(workspace):
        raise ValueError(f"Path outside workspace: {file_path}")
    return target


async def _grep_rg(pattern: str, target: Path) -> str:
    """Search using ripgrep. Returns 'LINE: content' lines or raises on error."""
    process = await asyncio.create_subprocess_exec(
        _RG_BIN,  # type: ignore[arg-type]
        "--line-number",
        "--no-filename",
        "--color=never",
        "--",
        pattern,
        str(target),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()

    # rg exit codes: 0 = matches, 1 = no matches, 2 = error
    if process.returncode == 2:
        raise RuntimeError(stderr.decode("utf-8", errors="replace").strip())

    output = stdout.decode("utf-8", errors="replace")
    if not output.strip():
        return ""

    # rg outputs "LINE_NUM:content"; reformat to "LINE_NUM: content"
    lines = []
    for line in output.splitlines():
        num, _, content = line.partition(":")
        lines.append(f"{num}: {content}")
    return "\n".join(lines)


def _grep_python(pattern: str, target: Path) -> str:
    """Fallback: search using Python re module."""
    content = target.read_text(encoding="utf-8")
    compiled = re.compile(pattern)
    matches = [
        f"{i}: {line}"
        for i, line in enumerate(content.splitlines(), start=1)
        if compiled.search(line)
    ]
    return "\n".join(matches)


@tool(
    description=(
        "Search a file for lines matching a regex pattern. "
        "Returns matching lines with line numbers in 'LINE: content' format."
    )
)
async def grep(
    pattern: Annotated[str, "Regular expression pattern to search for."],
    file_path: Annotated[str, "Workspace-relative path to the file to search."],
    workspace_root: str,
) -> str:
    target = _validate_path(file_path, workspace_root)
    backend = "rg" if _RG_BIN else "python"
    logger.info("grep", pattern=pattern, file_path=file_path, backend=backend)

    if _RG_BIN:
        try:
            return await _grep_rg(pattern, target)
        except Exception as exc:
            logger.warning("grep_rg_failed", error=str(exc), fallback="python")

    return _grep_python(pattern, target)
