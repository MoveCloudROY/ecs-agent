"""Skill discovery from filesystem."""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from ecs_agent.core.world import World
from ecs_agent.logging import get_logger
from ecs_agent.skills.catalog import SkillDescriptor, SkillType, register as _catalog_register
from ecs_agent.skills.script_skill import ScriptSkill
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
    """Discover and load script-skill descriptors from filesystem paths."""

    def __init__(self, skill_paths: list[str | Path]) -> None:
        self.skill_paths = skill_paths

    def discover(self) -> list[SkillDescriptor]:
        """Scan configured paths and return discovered script-skill descriptors."""

        descriptors: list[SkillDescriptor] = []

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
                        if not isinstance(obj, type):
                            continue
                        if obj.__name__ == "ScriptSkill":
                            continue

                        try:
                            skill_instance = obj()
                            if isinstance(skill_instance, ScriptSkill):
                                descriptors.append(
                                    SkillDescriptor(
                                        name=skill_instance.name,
                                        skill_type=SkillType.SCRIPT,
                                        source_path=file_path.resolve(),
                                        metadata={},
                                        _materializer=_build_script_materializer(obj),
                                    )
                                )
                                logger.info(
                                    "skill_discovered",
                                    path=str(file_path),
                                    skill_name=skill_instance.name,
                                )
                        except Exception:
                            continue

                except Exception as exc:
                    logger.warning(
                        "skill_load_failed",
                        path=str(file_path),
                        error=str(exc),
                    )

        return descriptors

    def discover_and_install(
        self,
        world: World,
        entity_id: EntityId,
        manager: "SkillManager",
    ) -> list[str]:
        """Discover script skills and install them via SkillManager."""

        descriptors = self.discover()
        skill_names: list[str] = []

        for descriptor in descriptors:
            runtime_skill = descriptor.materialize()
            if not isinstance(runtime_skill, ScriptSkill):
                raise TypeError(
                    f"Descriptor '{descriptor.name}' materialized a non-ScriptSkill object"
                )
            manager.install(world, entity_id, runtime_skill)
            skill_names.append(descriptor.name)

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
        self,
        world: World,
        entity_id: EntityId,
        manager: "SkillManager",
        directories: list[Path] | None = None,
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

            for descriptor in discovered:
                try:
                    runtime_skill = descriptor.materialize()
                    if not isinstance(runtime_skill, ScriptSkill):
                        raise TypeError(
                            f"Descriptor '{descriptor.name}' materialized a non-ScriptSkill object"
                        )

                    manager.install(world, entity_id, runtime_skill)
                    report.installed_skills.append(descriptor.name)
                    source_installed.append(descriptor.name)
                except Exception as exc:
                    error = str(exc)
                    report.failed_sources.append((source, error))
                    source_errors.append(error)

            await world.event_bus.publish(
                SkillDiscoveryEvent(
                    source=source,
                    skills_found=source_installed,
                    errors=source_errors,
                )
            )

        if directories:
            for base_dir in directories:
                source = str(base_dir)
                md_source_installed: list[str] = []
                md_source_errors: list[str] = []

                try:
                    markdown_descriptors = discover_skills([base_dir])
                    for descriptor in markdown_descriptors:
                        try:
                            runtime_skill = descriptor.materialize()
                            if not isinstance(runtime_skill, ScriptSkill):
                                raise TypeError(
                                    f"Descriptor '{descriptor.name}' materialized a non-ScriptSkill object"
                                )

                            manager.index(world, entity_id, runtime_skill)
                            report.installed_skills.append(descriptor.name)
                            md_source_installed.append(descriptor.name)
                        except Exception as exc:
                            error = str(exc)
                            report.failed_sources.append((source, error))
                            md_source_errors.append(error)

                    await world.event_bus.publish(
                        SkillDiscoveryEvent(
                            source=source,
                            skills_found=md_source_installed,
                            errors=md_source_errors,
                        )
                    )
                except Exception as exc:
                    error = str(exc)
                    report.failed_sources.append((source, error))
                    await world.event_bus.publish(
                        SkillDiscoveryEvent(
                            source=source, skills_found=[], errors=[error]
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
                        source=server_name,
                        skills_found=[skill.name],
                        errors=[],
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
        self,
        server_name: str,
        mcp_config: dict[str, Any],
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


def discover_skills(directories: list[Path]) -> list[SkillDescriptor]:
    """Discover markdown skill descriptors from SKILL.md files."""

    from ecs_agent.skills.skill import Skill

    discovered_by_name: dict[str, SkillDescriptor] = {}
    discovered_path_by_name: dict[str, str] = {}

    for base_dir in directories:
        if not base_dir.exists():
            logger.warning("skill_path_not_found", path=str(base_dir))
            continue

        for skill_file in sorted(base_dir.rglob("SKILL.md")):
            try:
                skill = Skill(skill_file)
                if not skill.valid:
                    logger.warning(
                        "skill_invalid",
                        path=str(skill_file),
                        skill_name=skill.name,
                    )
                    continue

                skill_name = skill.name
                if skill_name in discovered_by_name:
                    raise ValueError(
                        "Skill name collision: "
                        f"'{skill_name}' found at both "
                        f"'{discovered_path_by_name[skill_name]}' and "
                        f"'{str(skill_file)}'. "
                        "Remove one SKILL.md or rename the skill."
                    )

                resolved_skill_file = skill_file.resolve()
                discovered_by_name[skill_name] = SkillDescriptor(
                    name=skill_name,
                    skill_type=SkillType.MARKDOWN,
                    source_path=resolved_skill_file,
                    metadata=_read_frontmatter_metadata(skill_file),
                    _materializer=_build_markdown_materializer(resolved_skill_file),
                )
                _catalog_register(discovered_by_name[skill_name])
                discovered_path_by_name[skill_name] = str(skill_file)
                logger.info(
                    "skill_discovered",
                    path=str(skill_file),
                    skill_name=skill_name,
                )
            except ValueError:
                raise
            except Exception as exc:
                logger.warning(
                    "skill_load_failed",
                    path=str(skill_file),
                    error=str(exc),
                )
    return list(discovered_by_name.values())


def _build_script_materializer(skill_class: type[Any]) -> Callable[[], Any]:
    def _materialize() -> Any:
        return skill_class()

    return _materialize


def _build_markdown_materializer(skill_path: Path) -> Callable[[], Any]:
    from ecs_agent.skills.skill import Skill

    def _materialize() -> Any:
        return Skill(skill_path)

    return _materialize


def _read_frontmatter_metadata(skill_file: Path) -> dict[str, Any]:
    try:
        raw = skill_file.read_bytes()
    except Exception:
        return {}

    lines = [line.rstrip(b"\r") for line in raw.split(b"\n")]
    first_non_empty_idx: int | None = None
    for idx, line in enumerate(lines):
        if line.strip():
            first_non_empty_idx = idx
            break

    if first_non_empty_idx is None or lines[first_non_empty_idx] != b"---":
        return {}

    closing_idx: int | None = None
    for idx in range(first_non_empty_idx + 1, len(lines)):
        if lines[idx] == b"---":
            closing_idx = idx
            break

    if closing_idx is None:
        return {}

    frontmatter_bytes = b"\n".join(lines[first_non_empty_idx + 1 : closing_idx])
    try:
        frontmatter_text = frontmatter_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return {}

    parsed = yaml.safe_load(frontmatter_text)
    if isinstance(parsed, dict):
        return dict(parsed)
    return {}
