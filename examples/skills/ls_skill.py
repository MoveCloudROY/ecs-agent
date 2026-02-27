"""Ls skill for listing directory contents."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path

from ecs_agent.core.world import World
from ecs_agent.logging import get_logger
from ecs_agent.skills.protocol import Skill
from ecs_agent.tools.discovery import tool
from ecs_agent.types import EntityId, ToolSchema

logger = get_logger(__name__)


@tool(description="List directory contents.")
async def ls(
    path: str = ".", all_files: bool = False, long_format: bool = False
) -> str:
    """List directory contents using pathlib."""
    try:
        target = Path(path).resolve()

        if not target.exists():
            logger.warning("ls_path_not_found", path=path)
            return f"error: path not found: {path}"

        if not target.is_dir():
            logger.warning("ls_not_directory", path=path)
            return f"error: not a directory: {path}"

        # List directory contents
        entries = []
        try:
            for item in sorted(target.iterdir()):
                # Skip hidden files unless all_files=True
                if item.name.startswith(".") and not all_files:
                    continue

                if long_format:
                    # Simple long format with size and type indicator
                    size = item.stat().st_size if item.is_file() else "-"
                    type_indicator = "/" if item.is_dir() else ""
                    entries.append(f"{size:10} {item.name}{type_indicator}")
                else:
                    # Simple listing with directory indicator
                    type_indicator = "/" if item.is_dir() else ""
                    entries.append(f"{item.name}{type_indicator}")
        except PermissionError:
            logger.warning("ls_permission_denied", path=path)
            return f"error: permission denied: {path}"

        if not entries:
            logger.info("ls_empty_directory", path=path)
            return "(empty directory)"

        result = "\n".join(entries)
        logger.info("ls_success", path=path, count=len(entries))
        return result

    except Exception as exc:
        logger.error("ls_exception", path=path, exception=str(exc))
        return f"error: {str(exc)}"


class LsSkill(Skill):
    """Skill providing ls tool for listing directories."""

    name = "ls"
    description = "List directory contents."

    def tools(self) -> dict[str, tuple[ToolSchema, Callable[..., Awaitable[str]]]]:
        """Return ls tool schema and handler."""
        from ecs_agent.tools.discovery import (
            _build_parameters_schema,
            _create_async_handler,
        )

        schema = ToolSchema(
            name="ls",
            description="List directory contents.",
            parameters=_build_parameters_schema(ls),
        )
        handler = _create_async_handler(ls)
        return {"ls": (schema, handler)}

    def system_prompt(self) -> str:
        """Return usage guidance for the LLM."""
        return "You have access to an ls tool for listing directory contents. Use it to explore filesystem structure."

    def install(self, world: World, entity_id: EntityId) -> None:
        """Install hook for ls skill."""
        _ = world
        _ = entity_id
        logger.info("ls_skill_install")

    def uninstall(self, world: World, entity_id: EntityId) -> None:
        """Uninstall hook for ls skill."""
        _ = world
        _ = entity_id
        logger.info("ls_skill_uninstall")


__all__ = ["LsSkill"]
