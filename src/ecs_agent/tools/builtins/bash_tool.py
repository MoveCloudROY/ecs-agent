"""Built-in bash execution tool."""

from __future__ import annotations

import asyncio
from pathlib import Path

from ecs_agent.logging import get_logger
from ecs_agent.tools.discovery import tool

logger = get_logger(__name__)


@tool(description="Execute shell command in workspace with timeout.")
async def bash(command: str, timeout: float, workspace_root: str) -> str:
    """Execute shell command in workspace with timeout."""
    workspace = Path(workspace_root).resolve()

    process = await asyncio.create_subprocess_shell(
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(workspace),
    )

    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
    except asyncio.TimeoutError as exc:
        process.kill()
        await process.wait()
        raise ValueError(f"Command timed out after {timeout}s") from exc

    stdout_text = stdout.decode("utf-8", errors="replace")
    stderr_text = stderr.decode("utf-8", errors="replace")
    logger.info("bash", command=command, returncode=process.returncode)

    if process.returncode != 0:
        return (
            f"Exit code {process.returncode}\n"
            f"STDOUT:\n{stdout_text}\n"
            f"STDERR:\n{stderr_text}"
        )

    return stdout_text
