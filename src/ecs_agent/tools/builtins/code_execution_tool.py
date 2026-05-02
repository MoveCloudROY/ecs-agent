"""Built-in code_execution tool for running code snippets programmatically."""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import sys
import tempfile
from pathlib import Path
from typing import Annotated

from ecs_agent.logging import get_logger
from ecs_agent.tools.discovery import tool

logger = get_logger(__name__)

_SUPPORTED_LANGUAGES = frozenset({"python"})


async def _terminate_process_group(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return

    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGTERM)

    try:
        await asyncio.wait_for(process.wait(), timeout=1.0)
        return
    except asyncio.TimeoutError:
        pass

    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGKILL)

    with contextlib.suppress(Exception):
        await asyncio.wait_for(process.wait(), timeout=1.0)


@tool(
    description=(
        "Execute a code snippet and return its stdout/stderr output. "
        "Supported languages: python."
    )
)
async def code_execution(
    code: Annotated[str, "The source code to execute."],
    language: Annotated[
        str, "Programming language. Supported: 'python'."
    ],
    timeout: Annotated[float, "Maximum execution time in seconds."] = 30.0,
) -> str:
    lang = language.lower().strip()
    if lang not in _SUPPORTED_LANGUAGES:
        raise ValueError(
            f"Unsupported language: {language!r}. Supported: {sorted(_SUPPORTED_LANGUAGES)}"
        )

    logger.info("code_execution", language=lang, code_length=len(code))

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(code)
        tmp_path = Path(f.name)

    try:
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            str(tmp_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        try:
            stdout_b, stderr_b = await asyncio.wait_for(
                process.communicate(), timeout=timeout
            )
        except asyncio.TimeoutError as exc:
            await _terminate_process_group(process)
            raise ValueError(
                f"Code execution timed out after {timeout}s"
            ) from exc
        except asyncio.CancelledError:
            await _terminate_process_group(process)
            raise

        stdout = stdout_b.decode("utf-8", errors="replace")
        stderr = stderr_b.decode("utf-8", errors="replace")

        if process.returncode != 0:
            return (
                f"Exit code {process.returncode}\n"
                f"STDOUT:\n{stdout}\n"
                f"STDERR:\n{stderr}"
            )
        return stdout
    finally:
        tmp_path.unlink(missing_ok=True)
