"""Grep skill for pattern searching in files."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from ecs_agent.core.world import World
from ecs_agent.logging import get_logger
from ecs_agent.skills.protocol import Skill
from ecs_agent.tools.discovery import tool
from ecs_agent.types import EntityId, ToolSchema

logger = get_logger(__name__)


@tool(description="Search file contents using pattern.")
async def grep(pattern: str, path: str, recursive: bool = False) -> str:
    """Search file contents using pattern."""
    try:
        # Build grep command
        cmd = ["grep"]
        if recursive:
            cmd.append("-r")
        cmd.extend([pattern, path])

        # Run grep via asyncio subprocess
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=5.0)
        stdout_text = stdout.decode("utf-8", errors="replace")
        stderr_text = stderr.decode("utf-8", errors="replace")

        if process.returncode == 0:
            logger.info("grep_success", pattern=pattern, path=path)
            return stdout_text if stdout_text else "(no output)"
        elif process.returncode == 1:
            # No matches found
            logger.info("grep_no_match", pattern=pattern, path=path)
            return "(no matches found)"
        else:
            # Error (file not found, etc.)
            logger.warning("grep_error", pattern=pattern, path=path, stderr=stderr_text)
            return f"error: {stderr_text}" if stderr_text else "error: command failed"

    except asyncio.TimeoutError:
        logger.error("grep_timeout", pattern=pattern, path=path)
        return "error: grep timed out"
    except Exception as exc:
        logger.error("grep_exception", pattern=pattern, path=path, exception=str(exc))
        return f"error: {str(exc)}"


class GrepSkill(Skill):
    """Skill providing grep tool for pattern searching."""

    name = "grep"
    description = "Search file contents using patterns."

    def tools(self) -> dict[str, tuple[ToolSchema, Callable[..., Awaitable[str]]]]:
        """Return grep tool schema and handler."""
        from ecs_agent.tools.discovery import (
            _build_parameters_schema,
            _create_async_handler,
        )

        schema = ToolSchema(
            name="grep",
            description="Search file contents using pattern.",
            parameters=_build_parameters_schema(grep),
        )
        handler = _create_async_handler(grep)
        return {"grep": (schema, handler)}

    def system_prompt(self) -> str:
        """Return usage guidance for the LLM."""
        return "You have access to a grep tool for searching file contents. Use it to find patterns in files."

    def install(self, world: World, entity_id: EntityId) -> None:
        """Install hook for grep skill."""
        _ = world
        _ = entity_id
        logger.info("grep_skill_install")

    def uninstall(self, world: World, entity_id: EntityId) -> None:
        """Uninstall hook for grep skill."""
        _ = world
        _ = entity_id
        logger.info("grep_skill_uninstall")


__all__ = ["GrepSkill"]
