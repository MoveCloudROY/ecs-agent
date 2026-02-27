"""Tests for skill discovery system."""

from pathlib import Path

import pytest

from ecs_agent.core.world import World
from ecs_agent.skills.discovery import SkillDiscovery
from ecs_agent.skills.manager import SkillManager
from ecs_agent.skills.protocol import Skill
from ecs_agent.components import SkillComponent


def test_skill_discovery_finds_valid_skill(tmp_path: Path) -> None:
    """Test discover() finds Skill classes in temp directory."""
    skill_file = tmp_path / "demo_skill.py"
    skill_file.write_text(
        """
from collections.abc import Awaitable, Callable
from ecs_agent.core.world import World
from ecs_agent.skills.protocol import Skill
from ecs_agent.types import EntityId, ToolSchema


class DemoSkill(Skill):
    name = "demo"
    description = "Demo skill"

    def tools(self) -> dict[str, tuple[ToolSchema, Callable[..., Awaitable[str]]]]:
        return {}

    def system_prompt(self) -> str:
        return ""

    def install(self, world: World, entity_id: EntityId) -> None:
        pass

    def uninstall(self, world: World, entity_id: EntityId) -> None:
        pass
"""
    )

    discovery = SkillDiscovery(skill_paths=[tmp_path])
    skills = discovery.discover()

    assert len(skills) == 1
    assert skills[0].name == "demo"
    assert skills[0].description == "Demo skill"


def test_skill_discovery_skips_non_skill_classes(tmp_path: Path) -> None:
    """Test discover() skips non-Skill classes and non-.py files."""
    # Valid skill file
    valid_skill = tmp_path / "valid_skill.py"
    valid_skill.write_text(
        """
from collections.abc import Awaitable, Callable
from ecs_agent.core.world import World
from ecs_agent.skills.protocol import Skill
from ecs_agent.types import EntityId, ToolSchema


class ValidSkill(Skill):
    name = "valid"
    description = "Valid skill"

    def tools(self) -> dict[str, tuple[ToolSchema, Callable[..., Awaitable[str]]]]:
        return {}

    def system_prompt(self) -> str:
        return ""

    def install(self, world: World, entity_id: EntityId) -> None:
        pass

    def uninstall(self, world: World, entity_id: EntityId) -> None:
        pass
"""
    )

    # Non-Skill class
    plain_class = tmp_path / "plain_class.py"
    plain_class.write_text(
        """
class PlainClass:
    pass
"""
    )

    # Non-.py file
    txt_file = tmp_path / "readme.txt"
    txt_file.write_text("Just text")

    discovery = SkillDiscovery(skill_paths=[tmp_path])
    skills = discovery.discover()

    assert len(skills) == 1
    assert skills[0].name == "valid"


def test_skill_discovery_handles_empty_directory(tmp_path: Path) -> None:
    """Test discover() returns empty list for empty directory."""
    discovery = SkillDiscovery(skill_paths=[tmp_path])
    skills = discovery.discover()

    assert skills == []


def test_skill_discovery_handles_non_existent_directory(tmp_path: Path) -> None:
    """Test discover() logs warning and continues for non-existent directory."""
    non_existent = tmp_path / "does_not_exist"

    discovery = SkillDiscovery(skill_paths=[non_existent])
    skills = discovery.discover()

    assert skills == []


def test_skill_discovery_handles_malformed_python_file(tmp_path: Path) -> None:
    """Test discover() logs error and continues for malformed Python files."""
    # Valid skill
    valid_skill = tmp_path / "valid_skill.py"
    valid_skill.write_text(
        """
from collections.abc import Awaitable, Callable
from ecs_agent.core.world import World
from ecs_agent.skills.protocol import Skill
from ecs_agent.types import EntityId, ToolSchema


class ValidSkill(Skill):
    name = "valid"
    description = "Valid skill"

    def tools(self) -> dict[str, tuple[ToolSchema, Callable[..., Awaitable[str]]]]:
        return {}

    def system_prompt(self) -> str:
        return ""

    def install(self, world: World, entity_id: EntityId) -> None:
        pass

    def uninstall(self, world: World, entity_id: EntityId) -> None:
        pass
"""
    )

    # Malformed Python file
    broken_file = tmp_path / "broken.py"
    broken_file.write_text("this is not valid python code!!! @#$%^&*()")

    discovery = SkillDiscovery(skill_paths=[tmp_path])
    skills = discovery.discover()

    # Should only find the valid skill, ignore the broken one
    assert len(skills) == 1
    assert skills[0].name == "valid"


def test_skill_discovery_and_install(tmp_path: Path) -> None:
    """Test discover_and_install() installs via SkillManager."""
    skill_file = tmp_path / "install_skill.py"
    skill_file.write_text(
        """
from collections.abc import Awaitable, Callable
from ecs_agent.core.world import World
from ecs_agent.skills.protocol import Skill
from ecs_agent.types import EntityId, ToolSchema


class InstallSkill(Skill):
    name = "install-test"
    description = "Install test skill"

    def tools(self) -> dict[str, tuple[ToolSchema, Callable[..., Awaitable[str]]]]:
        return {}

    def system_prompt(self) -> str:
        return ""

    def install(self, world: World, entity_id: EntityId) -> None:
        pass

    def uninstall(self, world: World, entity_id: EntityId) -> None:
        pass
"""
    )

    world = World()
    entity = world.create_entity()
    manager = SkillManager()
    discovery = SkillDiscovery(skill_paths=[tmp_path])

    skill_names = discovery.discover_and_install(world, entity, manager)

    assert skill_names == ["install-test"]

    # Verify installed
    skill_component = world.get_component(entity, SkillComponent)
    assert skill_component is not None
    assert "install-test" in skill_component.skills


def test_skill_discovery_multiple_paths(tmp_path: Path) -> None:
    """Test multiple skill_paths merge results."""
    dir1 = tmp_path / "dir1"
    dir1.mkdir()
    dir2 = tmp_path / "dir2"
    dir2.mkdir()

    skill1 = dir1 / "skill1.py"
    skill1.write_text(
        """
from collections.abc import Awaitable, Callable
from ecs_agent.core.world import World
from ecs_agent.skills.protocol import Skill
from ecs_agent.types import EntityId, ToolSchema


class Skill1(Skill):
    name = "skill1"
    description = "First skill"

    def tools(self) -> dict[str, tuple[ToolSchema, Callable[..., Awaitable[str]]]]:
        return {}

    def system_prompt(self) -> str:
        return ""

    def install(self, world: World, entity_id: EntityId) -> None:
        pass

    def uninstall(self, world: World, entity_id: EntityId) -> None:
        pass
"""
    )

    skill2 = dir2 / "skill2.py"
    skill2.write_text(
        """
from collections.abc import Awaitable, Callable
from ecs_agent.core.world import World
from ecs_agent.skills.protocol import Skill
from ecs_agent.types import EntityId, ToolSchema


class Skill2(Skill):
    name = "skill2"
    description = "Second skill"

    def tools(self) -> dict[str, tuple[ToolSchema, Callable[..., Awaitable[str]]]]:
        return {}

    def system_prompt(self) -> str:
        return ""

    def install(self, world: World, entity_id: EntityId) -> None:
        pass

    def uninstall(self, world: World, entity_id: EntityId) -> None:
        pass
"""
    )

    discovery = SkillDiscovery(skill_paths=[dir1, dir2])
    skills = discovery.discover()

    assert len(skills) == 2
    skill_names = {s.name for s in skills}
    assert skill_names == {"skill1", "skill2"}


def test_skill_discovery_skips_init_py(tmp_path: Path) -> None:
    """Test discover() skips __init__.py files."""
    init_file = tmp_path / "__init__.py"
    init_file.write_text(
        """
# This file should be skipped
from collections.abc import Awaitable, Callable
from ecs_agent.core.world import World
from ecs_agent.skills.protocol import Skill
from ecs_agent.types import EntityId, ToolSchema


class InitSkill(Skill):
    name = "init"
    description = "Init skill"

    def tools(self) -> dict[str, tuple[ToolSchema, Callable[..., Awaitable[str]]]]:
        return {}

    def system_prompt(self) -> str:
        return ""

    def install(self, world: World, entity_id: EntityId) -> None:
        pass

    def uninstall(self, world: World, entity_id: EntityId) -> None:
        pass
"""
    )

    discovery = SkillDiscovery(skill_paths=[tmp_path])
    skills = discovery.discover()

    assert skills == []
