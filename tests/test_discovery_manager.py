from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import pytest

from ecs_agent.core import World
from ecs_agent.skills.discovery import DiscoveryManager, DiscoveryReport
from ecs_agent.skills.manager import SkillManager
from ecs_agent.types import EntityId, SkillDiscoveryEvent, ToolSchema


def _write_skill_file(path: Path, class_name: str, skill_name: str) -> None:
    path.write_text(
        f'''
from collections.abc import Awaitable, Callable
from ecs_agent.core.world import World
from ecs_agent.skills.protocol import Skill
from ecs_agent.types import EntityId, ToolSchema


class {class_name}(Skill):
    name = "{skill_name}"
    description = "{skill_name} skill"

    def tools(self) -> dict[str, tuple[ToolSchema, Callable[..., Awaitable[str]]]]:
        return {{}}

    def system_prompt(self) -> str:
        return ""

    def install(self, world: World, entity_id: EntityId) -> None:
        pass

    def uninstall(self, world: World, entity_id: EntityId) -> None:
        pass
'''
    )


class _FakeMCPClient:
    _offline_servers: set[str] = set()

    def __init__(self, config: Any) -> None:
        self.server_name = str(getattr(config, "server_name", ""))

    async def connect(self) -> None:
        if self.server_name in self._offline_servers:
            raise RuntimeError(f"{self.server_name} offline")


class _FakeMCPSkillAdapter:
    def __init__(self, mcp_client: _FakeMCPClient, server_name: str) -> None:
        self.name = f"mcp-{server_name}"
        self.description = f"MCP skill for {server_name}"

    def tools(self) -> dict[str, tuple[ToolSchema, Callable[..., Awaitable[str]]]]:
        return {}

    def system_prompt(self) -> str:
        return ""

    def install(self, world: World, entity_id: EntityId) -> None:
        return None

    def uninstall(self, world: World, entity_id: EntityId) -> None:
        return None


def _patch_mcp(monkeypatch: pytest.MonkeyPatch, offline_servers: set[str]) -> None:
    _FakeMCPClient._offline_servers = offline_servers
    monkeypatch.setattr("ecs_agent.skills.discovery.MCPClient", _FakeMCPClient)
    monkeypatch.setattr(
        "ecs_agent.skills.discovery.MCPSkillAdapter", _FakeMCPSkillAdapter
    )


@pytest.mark.asyncio
async def test_discovery_manager_auto_discovers_file_and_skips_unavailable_mcp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_mcp(monkeypatch, offline_servers={"offline"})
    _write_skill_file(tmp_path / "local_skill.py", "LocalSkill", "local")

    world = World()
    entity_id = world.create_entity()
    manager = SkillManager()
    discovery = DiscoveryManager(
        skill_paths=[tmp_path],
        mcp_configs=[
            {"server_name": "online", "transport_type": "stdio", "config": {}},
            {"server_name": "offline", "transport_type": "stdio", "config": {}},
        ],
    )

    report = await discovery.auto_discover_and_install(world, entity_id, manager)

    assert "local" in report.installed_skills
    assert "mcp-online" in report.installed_skills
    assert "offline" in report.skipped_mcp


@pytest.mark.asyncio
async def test_discovery_manager_auto_disables_failed_mcp_connect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_mcp(monkeypatch, offline_servers={"fs"})

    world = World()
    entity_id = world.create_entity()
    manager = SkillManager()
    discovery = DiscoveryManager(
        mcp_configs=[{"server_name": "fs", "transport_type": "stdio", "config": {}}]
    )

    report = await discovery.auto_discover_and_install(world, entity_id, manager)

    assert report.installed_skills == []
    assert report.skipped_mcp == ["fs"]
    assert report.failed_sources == [("fs", "fs offline")]


@pytest.mark.asyncio
async def test_discovery_manager_report_contains_expected_fields_for_mixed_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    missing_path = tmp_path / "missing"
    _patch_mcp(monkeypatch, offline_servers={"bad-server"})
    _write_skill_file(tmp_path / "ok_skill.py", "OkSkill", "ok")

    world = World()
    entity_id = world.create_entity()
    manager = SkillManager()
    discovery = DiscoveryManager(
        skill_paths=[tmp_path, missing_path],
        mcp_configs=[
            {"server_name": "good-server", "transport_type": "stdio", "config": {}},
            {"server_name": "bad-server", "transport_type": "stdio", "config": {}},
        ],
    )

    report = await discovery.auto_discover_and_install(world, entity_id, manager)

    assert isinstance(report, DiscoveryReport)
    assert set(report.installed_skills) == {"ok", "mcp-good-server"}
    assert (str(missing_path), "path not found") in report.failed_sources
    assert ("bad-server", "bad-server offline") in report.failed_sources
    assert report.skipped_mcp == ["bad-server"]


@pytest.mark.asyncio
async def test_discovery_manager_publishes_skill_discovery_event_per_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_mcp(monkeypatch, offline_servers={"offline"})
    _write_skill_file(tmp_path / "event_skill.py", "EventSkill", "event-skill")

    world = World()
    entity_id = world.create_entity()
    manager = SkillManager()
    events: list[SkillDiscoveryEvent] = []

    async def on_discovery(event: SkillDiscoveryEvent) -> None:
        events.append(event)

    world.event_bus.subscribe(SkillDiscoveryEvent, on_discovery)

    discovery = DiscoveryManager(
        skill_paths=[tmp_path],
        mcp_configs=[
            {"server_name": "online", "transport_type": "stdio", "config": {}},
            {"server_name": "offline", "transport_type": "stdio", "config": {}},
        ],
    )
    await discovery.auto_discover_and_install(world, entity_id, manager)

    assert len(events) == 3
    assert {event.source for event in events} == {str(tmp_path), "online", "offline"}
    by_source = {event.source: event for event in events}
    assert by_source[str(tmp_path)].skills_found == ["event-skill"]
    assert by_source["online"].skills_found == ["mcp-online"]
    assert by_source["offline"].skills_found == []
    assert by_source["offline"].errors == ["offline offline"]


@pytest.mark.asyncio
async def test_discovery_manager_empty_config_returns_empty_report() -> None:
    world = World()
    entity_id = world.create_entity()
    manager = SkillManager()
    discovery = DiscoveryManager()

    report = await discovery.auto_discover_and_install(world, entity_id, manager)

    assert report == DiscoveryReport(
        installed_skills=[], failed_sources=[], skipped_mcp=[]
    )


@pytest.mark.asyncio
async def test_discovery_manager_mixed_success_failure_installs_available_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_mcp(monkeypatch, offline_servers={"offline"})
    _write_skill_file(tmp_path / "a_skill.py", "ASkill", "a")
    _write_skill_file(tmp_path / "b_skill.py", "BSkill", "b")
    missing_path = tmp_path / "not_there"

    world = World()
    entity_id = world.create_entity()
    manager = SkillManager()
    discovery = DiscoveryManager(
        skill_paths=[tmp_path, missing_path],
        mcp_configs=[
            {"server_name": "online", "transport_type": "stdio", "config": {}},
            {"server_name": "offline", "transport_type": "stdio", "config": {}},
        ],
    )

    report = await discovery.auto_discover_and_install(world, entity_id, manager)

    assert set(report.installed_skills) == {"a", "b", "mcp-online"}
    assert (str(missing_path), "path not found") in report.failed_sources
    assert ("offline", "offline offline") in report.failed_sources
    assert report.skipped_mcp == ["offline"]
