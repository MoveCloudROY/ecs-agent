"""bwrap-backed sandbox execution wrappers."""

from __future__ import annotations

import asyncio
import shutil
from collections.abc import Awaitable, Callable
from typing import Any

from ecs_agent.components import SandboxConfigComponent
from ecs_agent.types import ToolSchema

_BWRAP_AVAILABLE: bool | None = None


def _has_bwrap() -> bool:
    global _BWRAP_AVAILABLE
    if _BWRAP_AVAILABLE is None:
        _BWRAP_AVAILABLE = shutil.which("bwrap") is not None
    return _BWRAP_AVAILABLE


async def bwrap_execute(command: str, timeout: float = 30.0) -> str:
    if _has_bwrap():
        process = await asyncio.create_subprocess_exec(
            "bwrap",
            "--ro-bind",
            "/",
            "/",
            "--dev",
            "/dev",
            "--proc",
            "/proc",
            "--tmpfs",
            "/tmp",
            "--unshare-all",
            "--die-with-parent",
            "--",
            "sh",
            "-c",
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    else:
        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        return f"Error: command timed out after {timeout}s"

    output = stdout.decode().strip()
    error_output = stderr.decode().strip()
    if process.returncode != 0:
        return error_output or output or f"Error: command failed ({process.returncode})"
    return output


def wrap_sandbox_handler(
    handler: Callable[..., Awaitable[str]],
    schema: ToolSchema,
    config: SandboxConfigComponent,
) -> Callable[..., Awaitable[str]]:
    if not schema.sandbox_compatible or config.sandbox_mode != "bwrap":
        return handler

    async def wrapped_handler(**kwargs: Any) -> str:
        command = kwargs.get("command")
        if isinstance(command, str):
            return await bwrap_execute(command, timeout=config.timeout)
        return await handler(**kwargs)

    return wrapped_handler


__all__ = ["bwrap_execute", "wrap_sandbox_handler"]
