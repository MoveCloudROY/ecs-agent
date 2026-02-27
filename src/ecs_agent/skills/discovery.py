"""Skill discovery from filesystem."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ecs_agent.core.world import World
from ecs_agent.logging import get_logger
from ecs_agent.skills.protocol import Skill
from ecs_agent.types import EntityId, SkillDiscoveryEvent

try:
    from ecs_agent.mcp.adapter import MCPSkillAdapter as _MCPSkillAdapter
    from ecs_agent.mcp.client import MCPClient as _MCPClient

    MCPSkillAdapter: Any = _MCPSkillAdapter
    MCPClient: Any = _MCPClient
except ImportError:
    MCPClient = None
    MCPSkillAdapter = None

if TYPE_CHECKING:
    from ecs_agent.skills.manager import SkillManager

logger = get_logger(__name__)


@dataclass(slots=True)
class DiscoveryReport:
    installed_skills: list[str]
    failed_sources: list[tuple[str, str]]
    skipped_mcp: list[str]


@dataclass(slots=True)
class _MCPClientConfig:
    server_name: str
    transport_type: str
    config: dict[str, Any]


class SkillDiscovery:
    """Discover and load Skill implementations from filesystem paths.

    Args:
        skill_paths: List of directory paths to scan for Python modules containing Skill classes.
    """

    def __init__(self, skill_paths: list[str | Path]) -> None:
        self.skill_paths = skill_paths

    def discover(self) -> list[Skill]:
        """Scan configured paths and return all discovered Skill instances.

        Returns:
            List of Skill instances found in the configured paths.
        """
        skills: list[Skill] = []

        for base_path in self.skill_paths:
            path = Path(base_path)
            if not path.exists():
                logger.warning("skill_path_not_found", path=str(path))
                continue

            for file_path in path.glob("*.py"):
                if file_path.name == "__init__.py":
                    continue

                try:
                    spec = importlib.util.spec_from_file_location(
                        file_path.stem, file_path
                    )
                    if spec is None or spec.loader is None:
                        logger.warning(
                            "skill_load_failed",
                            path=str(file_path),
                            error="spec or loader is None",
                        )
                        continue

                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)

                    for attr_name in dir(module):
                        obj = getattr(module, attr_name)
                        # Skip non-classes
                        if not isinstance(obj, type):
                            continue
                        # Skip the Skill protocol itself
                        if obj.__name__ == "Skill":
                            continue
                        # Try to instantiate and check if it's a Skill
                        try:
                            skill_instance = obj()
                            # Verify it's actually a Skill instance
                            if isinstance(skill_instance, Skill):
                                skills.append(skill_instance)
                                logger.info(
                                    "skill_discovered",
                                    path=str(file_path),
                                    skill_name=skill_instance.name,
                                )
                        except Exception:
                            # Not a Skill class or instantiation failed
                            # This is normal for non-Skill classes, don't log
                            continue

                except Exception as exc:
                    logger.warning(
                        "skill_load_failed", path=str(file_path), error=str(exc)
                    )

        return skills

    def discover_and_install(
        self, world: World, entity_id: EntityId, manager: "SkillManager"
    ) -> list[str]:
        """Discover skills and install them via SkillManager.

        Args:
            world: World instance for installation.
            entity_id: Entity to install skills on.
            manager: SkillManager to handle installation.

        Returns:
            List of installed skill names.
        """

        skills = self.discover()
        skill_names: list[str] = []

        for skill in skills:
            manager.install(world, entity_id, skill)
            skill_names.append(skill.name)

        return skill_names


class DiscoveryManager:
    def __init__(
        self,
        skill_paths: list[str | Path] | None = None,
        mcp_configs: list[dict[str, Any]] | None = None,
    ) -> None:
        self.skill_paths = skill_paths or []
        self.mcp_configs = mcp_configs or []

    async def auto_discover_and_install(
        self, world: World, entity_id: EntityId, manager: "SkillManager"
    ) -> DiscoveryReport:
        report = DiscoveryReport(installed_skills=[], failed_sources=[], skipped_mcp=[])

        for base_path in self.skill_paths:
            path = Path(base_path)
            source = str(path)

            if not path.exists():
                error = "path not found"
                report.failed_sources.append((source, error))
                await world.event_bus.publish(
                    SkillDiscoveryEvent(source=source, skills_found=[], errors=[error])
                )
                continue

            discovered = SkillDiscovery([path]).discover()
            source_installed: list[str] = []
            source_errors: list[str] = []

            for skill in discovered:
                try:
                    manager.install(world, entity_id, skill)
                    report.installed_skills.append(skill.name)
                    source_installed.append(skill.name)
                except Exception as exc:
                    error = str(exc)
                    report.failed_sources.append((source, error))
                    source_errors.append(error)

            await world.event_bus.publish(
                SkillDiscoveryEvent(
                    source=source, skills_found=source_installed, errors=source_errors
                )
            )

        for mcp_config in self.mcp_configs:
            server_name = self._server_name_from_config(mcp_config)

            try:
                if MCPClient is None or MCPSkillAdapter is None:
                    raise RuntimeError("MCP dependencies are not available")

                client = MCPClient(self._to_mcp_component(server_name, mcp_config))
                await client.connect()
                skill = MCPSkillAdapter(client, server_name)
                manager.install(world, entity_id, skill)
                report.installed_skills.append(skill.name)

                await world.event_bus.publish(
                    SkillDiscoveryEvent(
                        source=server_name, skills_found=[skill.name], errors=[]
                    )
                )
            except Exception as exc:
                error = str(exc)
                logger.warning("mcp_unavailable", server=server_name, error=error)
                report.skipped_mcp.append(server_name)
                report.failed_sources.append((server_name, error))
                await world.event_bus.publish(
                    SkillDiscoveryEvent(
                        source=server_name, skills_found=[], errors=[error]
                    )
                )

        return report

    def _server_name_from_config(self, config: dict[str, Any]) -> str:
        raw_name = config.get("server_name", config.get("name", "mcp"))
        if isinstance(raw_name, str) and raw_name:
            return raw_name
        return "mcp"

    def _to_mcp_component(
        self, server_name: str, mcp_config: dict[str, Any]
    ) -> _MCPClientConfig:
        transport_raw = mcp_config.get("transport_type", "stdio")
        transport_type = (
            transport_raw if transport_raw in {"stdio", "sse", "http"} else "stdio"
        )
        config_data = mcp_config.get("config")
        if not isinstance(config_data, dict):
            config_data = {}

        if "command" in mcp_config and "command" not in config_data:
            command = mcp_config.get("command")
            if isinstance(command, str):
                config_data = {**config_data, "command": command}

        return _MCPClientConfig(
            server_name=server_name,
            transport_type=transport_type,
            config=config_data,
        )
