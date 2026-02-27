"""Built-in tools skill."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from ecs_agent.core.world import World
from ecs_agent.logging import get_logger
from ecs_agent.skills.protocol import Skill
from ecs_agent.tools.discovery import scan_module
from ecs_agent.types import EntityId, ToolSchema

from ecs_agent.tools.builtins import bash_tool, edit_tool, file_tools

logger = get_logger(__name__)


class BuiltinToolsSkill(Skill):
    """Skill providing read_file, write_file, edit_file, bash."""

    name = "builtin-tools"
    description = "Basic file manipulation and bash execution tools."

    def tools(self) -> dict[str, tuple[ToolSchema, Callable[..., Awaitable[str]]]]:
        discovered: dict[str, tuple[ToolSchema, Callable[..., Awaitable[str]]]] = {}
        for module in (file_tools, bash_tool, edit_tool):
            discovered.update(scan_module(module))
        return discovered

    def system_prompt(self) -> str:
        return ""

    def install(self, world: World, entity_id: EntityId) -> None:
        _ = world
        _ = entity_id
        logger.info("builtin_tools_skill_install")

    def uninstall(self, world: World, entity_id: EntityId) -> None:
        _ = world
        _ = entity_id
        logger.info("builtin_tools_skill_uninstall")


__all__ = ["BuiltinToolsSkill"]
