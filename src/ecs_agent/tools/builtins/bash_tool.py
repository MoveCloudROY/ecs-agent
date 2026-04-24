"""Built-in bash execution tools."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated

from ecs_agent.logging import get_logger
from ecs_agent.tools.discovery import tool

logger = get_logger(__name__)


@tool(description="Execute shell command in workspace with timeout.")
async def bash(
    command: Annotated[str, "Shell command to execute."],
    timeout: Annotated[
        float, "Maximum execution time in seconds before the process is killed."
    ],
    workspace_root: str,
) -> str:
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


@tool(
    description="Execute a tmux subcommand for interactive terminal session management."
)
async def interactive_bash(
    tmux_command: Annotated[
        str,
        "tmux subcommand and arguments without the leading 'tmux' prefix. "
        "Example: 'new-session -d -s myapp' or 'send-keys -t myapp \"ls\" Enter'.",
    ],
) -> str:
    process = await asyncio.create_subprocess_exec(
        "tmux",
        *_split_tmux_args(tmux_command),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    stdout, stderr = await process.communicate()
    stdout_text = stdout.decode("utf-8", errors="replace")
    stderr_text = stderr.decode("utf-8", errors="replace")
    logger.info(
        "interactive_bash", tmux_command=tmux_command, returncode=process.returncode
    )

    if process.returncode != 0:
        return (
            f"Exit code {process.returncode}\n"
            f"STDOUT:\n{stdout_text}\n"
            f"STDERR:\n{stderr_text}"
        )

    return stdout_text


def _split_tmux_args(tmux_command: str) -> list[str]:
    import shlex

    return shlex.split(tmux_command)
