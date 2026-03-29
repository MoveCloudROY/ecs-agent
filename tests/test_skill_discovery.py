"""Tests for skill discovery system."""

from pathlib import Path

import pytest

from ecs_agent.core.world import World
from ecs_agent.skills.catalog import SkillType
from ecs_agent.skills.discovery import SkillDiscovery, discover_skills
from ecs_agent.skills.manager import SkillManager
from ecs_agent.skills.skill import Skill
from ecs_agent.components import SkillComponent


def test_skill_discovery_finds_valid_skill(tmp_path: Path) -> None:
    """Test discover() finds Skill classes in temp directory."""
    skill_file = tmp_path / "demo_skill.py"
    skill_file.write_text(
        """
from collections.abc import Awaitable, Callable
from ecs_agent.core.world import World
from ecs_agent.skills.script_skill import ScriptSkill
from ecs_agent.types import EntityId, ToolSchema


class DemoSkill(ScriptSkill):
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
    assert skills[0].materialize().description == "Demo skill"


def test_skill_discovery_skips_non_skill_classes(tmp_path: Path) -> None:
    """Test discover() skips non-Skill classes and non-.py files."""
    # Valid skill file
    valid_skill = tmp_path / "valid_skill.py"
    valid_skill.write_text(
        """
from collections.abc import Awaitable, Callable
from ecs_agent.core.world import World
from ecs_agent.skills.script_skill import ScriptSkill
from ecs_agent.types import EntityId, ToolSchema


class ValidSkill(ScriptSkill):
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
from ecs_agent.skills.script_skill import ScriptSkill
from ecs_agent.types import EntityId, ToolSchema


class ValidSkill(ScriptSkill):
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
from ecs_agent.skills.script_skill import ScriptSkill
from ecs_agent.types import EntityId, ToolSchema


class InstallSkill(ScriptSkill):
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
from ecs_agent.skills.script_skill import ScriptSkill
from ecs_agent.types import EntityId, ToolSchema


class Skill1(ScriptSkill):
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
from ecs_agent.skills.script_skill import ScriptSkill
from ecs_agent.types import EntityId, ToolSchema


class Skill2(ScriptSkill):
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
from ecs_agent.skills.script_skill import ScriptSkill
from ecs_agent.types import EntityId, ToolSchema


class InitSkill(ScriptSkill):
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


def test_discover_skills_returns_metadata_without_eager_system_prompt_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    skill_dir = tmp_path / ".claude" / "skills" / "lazy"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_bytes(
        (b"---\nname: lazy\ndescription: metadata only\n---\n# Body\n\xff\xfe\x00\x00")
    )

    def _raise_if_called(_: Skill) -> str:
        raise AssertionError("system_prompt() must not be called during discovery")

    monkeypatch.setattr(Skill, "system_prompt", _raise_if_called)

    skills = discover_skills([tmp_path])

    assert len(skills) == 1
    assert skills[0].name == "lazy"
    assert skills[0].metadata["description"] == "metadata only"


def test_discover_skills_skips_invalid_file_and_keeps_valid_skills(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    valid_dir = tmp_path / ".claude" / "skills" / "valid"
    valid_dir.mkdir(parents=True)
    (valid_dir / "SKILL.md").write_text(
        "---\nname: valid\ndescription: valid description\n---\nValid body"
    )

    invalid_dir = tmp_path / ".claude" / "skills" / "broken"
    invalid_dir.mkdir(parents=True)
    (invalid_dir / "SKILL.md").write_text(
        "---\nname: broken\ndescription: [unterminated\n---\nBroken body"
    )

    caplog.set_level("WARNING")
    skills = discover_skills([tmp_path])

    assert [skill.name for skill in skills] == ["valid"]
    assert any(
        "skill_invalid" in record.getMessage() or "invalid_yaml" in record.getMessage()
        for record in caplog.records
    )


def test_discover_skills_duplicate_name_conflict_raises_value_error(
    tmp_path: Path,
) -> None:
    first = tmp_path / ".claude" / "skills" / "a-first"
    first.mkdir(parents=True)
    (first / "SKILL.md").write_text(
        "---\nname: duplicate\ndescription: first\n---\nBody"
    )

    second = tmp_path / ".claude" / "skills" / "z-second"
    second.mkdir(parents=True)
    (second / "SKILL.md").write_text(
        "---\nname: duplicate\ndescription: second\n---\nBody"
    )

    with pytest.raises(ValueError) as exc_info:
        discover_skills([tmp_path])

    assert str(exc_info.value) == (
        "Skill name collision: 'duplicate' found at both "
        f"'{first / 'SKILL.md'}' and '{second / 'SKILL.md'}'. "
        "Remove one SKILL.md or rename the skill."
    )


def test_discover_skills_returns_markdown_descriptors(tmp_path: Path) -> None:
    skill_dir = tmp_path / ".claude" / "skills" / "writer"
    skill_dir.mkdir(parents=True)
    skill_path = skill_dir / "SKILL.md"
    skill_path.write_text(
        "---\nname: writer\ndescription: Compose docs\nversion: 1.2.0\ncategory: docs\n---\nBody"
    )

    descriptors = discover_skills([tmp_path])

    assert len(descriptors) == 1
    descriptor = descriptors[0]
    assert descriptor.name == "writer"
    assert descriptor.skill_type is SkillType.MARKDOWN
    assert descriptor.source_path == skill_path
    assert descriptor.metadata["description"] == "Compose docs"
    assert descriptor.metadata["version"] == "1.2.0"
    assert descriptor.metadata["category"] == "docs"

    runtime_skill = descriptor.materialize()
    assert isinstance(runtime_skill, Skill)
    assert runtime_skill.name == "writer"


def test_script_discovery_returns_descriptors(tmp_path: Path) -> None:
    skill_file = tmp_path / "demo_skill.py"
    skill_file.write_text(
        """
from collections.abc import Awaitable, Callable
from ecs_agent.core.world import World
from ecs_agent.skills.script_skill import ScriptSkill
from ecs_agent.types import EntityId, ToolSchema


class DemoSkill(ScriptSkill):
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

    descriptors = SkillDiscovery(skill_paths=[tmp_path]).discover()

    assert len(descriptors) == 1
    descriptor = descriptors[0]
    assert descriptor.name == "demo"
    assert descriptor.skill_type is SkillType.SCRIPT
    assert descriptor.source_path == skill_file
    runtime_skill = descriptor.materialize()
    assert runtime_skill.name == "demo"
