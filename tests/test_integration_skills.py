"""Integration tests for Skills + MCP + Built-in Tools feature set."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import pytest

from ecs_agent.components.definitions import (
    SkillComponent,
    SkillMetadata,
    ToolRegistryComponent,
)
from ecs_agent.core.world import World
from ecs_agent.types import ToolSchema

if TYPE_CHECKING:
    from ecs_agent.skills.protocol import Skill


# Helper to create test skill
class _TestSkill:
    """Custom test skill for integration tests."""

    def __init__(self) -> None:
        self.name = "test_skill"
        self.description = "A test skill for integration testing"

    def tools(self) -> dict[str, tuple[ToolSchema, object]]:
        """Return test tool schemas."""

        async def test_tool_handler(input: str) -> str:
            return input.upper()

        return {
            "test_tool": (
                ToolSchema(
                    name="test_tool",
                    description="A test tool",
                    parameters={
                        "type": "object",
                        "properties": {"input": {"type": "string"}},
                        "required": ["input"],
                    },
                ),
                test_tool_handler,
            )
        }

    def system_prompt(self) -> str:
        """Return empty system prompt."""
        return ""

    def install(self, world: World, entity_id: int) -> None:
        """No-op install hook."""
        pass

    def uninstall(self, world: World, entity_id: int) -> None:
        """No-op uninstall hook."""
        pass


def test_import_skill_protocol() -> None:
    """Verify Skill protocol can be imported from main package."""
    from ecs_agent import Skill

    assert Skill is not None


def test_import_skill_manager() -> None:
    """Verify SkillManager can be imported from main package."""
    from ecs_agent import SkillManager

    assert SkillManager is not None


def test_import_builtin_tools_skill_from_main() -> None:
    """Verify BuiltinToolsSkill can be imported from main package."""
    from ecs_agent import BuiltinToolsSkill

    assert BuiltinToolsSkill is not None


def test_import_builtin_tools_skill_from_tools() -> None:
    """Verify BuiltinToolsSkill can be imported from tools subpackage."""
    from ecs_agent.tools.builtins import BuiltinToolsSkill

    assert BuiltinToolsSkill is not None


def test_import_skill_component() -> None:
    """Verify SkillComponent can be imported from main package."""
    from ecs_agent import SkillComponent

    assert SkillComponent is not None


def test_import_skill_metadata() -> None:
    """Verify SkillMetadata can be imported from main package."""
    from ecs_agent import SkillMetadata

    assert SkillMetadata is not None


def test_mcp_import_without_package_raises_helpful_error() -> None:
    """Verify MCPClient import raises ImportError when mcp package not installed."""
    # Remove any mock mcp modules from previous tests
    mcp_modules_to_remove = [k for k in sys.modules if k.startswith("mcp")]
    for mod in mcp_modules_to_remove:
        del sys.modules[mod]

    # Ensure mcp is not available
    try:
        import mcp  # noqa: F401

        pytest.skip("mcp package is installed, cannot test ImportError behavior")
    except ImportError:
        pass

    # Now verify that importing MCPClient from ecs_agent doesn't crash
    # but simply doesn't export it
    try:
        from ecs_agent import MCPClient  # type: ignore[attr-defined]

        # If we get here, mcp WAS available (shouldn't happen with guard above)
        pytest.fail("MCPClient should not be importable without mcp package")
    except ImportError as exc:
        # This is expected - the symbol doesn't exist in __all__
        assert "MCPClient" in str(exc) or "cannot import" in str(exc)


def test_full_skill_lifecycle() -> None:
    """Test full skill lifecycle: install → verify tools → uninstall → verify removed."""
    from ecs_agent import BuiltinToolsSkill, SkillManager

    world = World()
    entity = world.create_entity()
    manager = SkillManager()

    # Install BuiltinToolsSkill
    skill = BuiltinToolsSkill()
    manager.install(world, entity, skill)

    # Verify tools are registered
    registry = world.get_component(entity, ToolRegistryComponent)
    assert registry is not None
    assert len(registry.tools) == 5  # 4 builtin tools + 1 meta-tool
    tool_names = set(registry.tools.keys())
    assert "read_file" in tool_names
    assert "write_file" in tool_names
    assert "edit_file" in tool_names
    assert "bash" in tool_names
    assert "load_skill_details" in tool_names

    # Verify SkillComponent metadata
    skill_comp = world.get_component(entity, SkillComponent)
    assert skill_comp is not None
    assert len(skill_comp.skills) == 1
    metadata = skill_comp.skills["builtin-tools"]
    assert metadata.name == "builtin-tools"
    assert set(metadata.tool_names) == {"read_file", "write_file", "edit_file", "bash"}

    # Uninstall skill
    manager.uninstall(world, entity, "builtin-tools")

    # Verify tools are removed (meta-tool is also cleaned up when no skills remain)
    registry = world.get_component(entity, ToolRegistryComponent)
    assert registry is not None
    assert len(registry.tools) == 0  # All tools removed including meta-tool

    # Verify metadata removed
    skill_comp = world.get_component(entity, SkillComponent)
    assert skill_comp is not None
    assert len(skill_comp.skills) == 0

def test_multiple_skills_on_same_entity() -> None:
    """Test multiple skills can be installed on the same entity."""
    from ecs_agent import BuiltinToolsSkill, SkillManager

    world = World()
    entity = world.create_entity()
    manager = SkillManager()

    # Install BuiltinToolsSkill
    builtin = BuiltinToolsSkill()
    manager.install(world, entity, builtin)

    # Install custom test skill
    test_skill = _TestSkill()
    manager.install(world, entity, test_skill)

    # Verify both skills' tools are present
    registry = world.get_component(entity, ToolRegistryComponent)
    assert registry is not None
    assert len(registry.tools) == 6  # 4 builtin + 1 test + 1 meta-tool
    tool_names = set(registry.tools.keys())
    assert "read_file" in tool_names
    assert "write_file" in tool_names
    assert "edit_file" in tool_names
    assert "bash" in tool_names
    assert "test_tool" in tool_names
    assert "load_skill_details" in tool_names

    # Verify metadata for both skills
    skill_comp = world.get_component(entity, SkillComponent)
    assert skill_comp is not None
    assert len(skill_comp.skills) == 2
    skill_names = set(skill_comp.skills.keys())
    assert "builtin-tools" in skill_names
    assert "test_skill" in skill_names


async def test_load_skill_details_meta_tool() -> None:
    """Test load_skill_details meta-tool provides Tier 2 full schema output."""
    from ecs_agent import BuiltinToolsSkill, SkillManager

    world = World()
    entity = world.create_entity()
    manager = SkillManager()

    # Install BuiltinToolsSkill
    skill = BuiltinToolsSkill()
    manager.install(world, entity, skill)

    # Verify load_skill_details tool exists
    registry = world.get_component(entity, ToolRegistryComponent)
    assert registry is not None
    details_tool = registry.tools.get("load_skill_details")
    assert details_tool is not None
    assert "Tier 2" in details_tool.description  # Check for Tier 2 description

    # Call load_skill_details for builtin-tools
    handler = registry.handlers.get("load_skill_details")
    assert handler is not None
    result = await handler(skill_name="builtin-tools")

    # Verify Tier 2 output (markdown with full schemas)
    assert "Skill: builtin-tools" in result
    assert "Description: Basic file manipulation" in result
    assert "read_file" in result
    assert "write_file" in result
    assert "edit_file" in result
    assert "bash" in result
    # Verify parameters are included
    assert "parameters" in result
    assert "workspace_root" in result

def test_skill_uninstall_removes_only_owned_tools() -> None:
    """Test that uninstalling a skill only removes tools it owns."""
    from ecs_agent import BuiltinToolsSkill, SkillManager

    world = World()
    entity = world.create_entity()
    manager = SkillManager()

    # Install both skills
    builtin = BuiltinToolsSkill()
    test_skill = _TestSkill()
    manager.install(world, entity, builtin)
    manager.install(world, entity, test_skill)

    # Uninstall only builtin skill
    manager.uninstall(world, entity, "builtin-tools")

    # Verify only builtin tools removed, test_tool remains
    registry = world.get_component(entity, ToolRegistryComponent)
    assert registry is not None
    assert len(registry.tools) == 2  # test_tool + load_skill_details meta-tool
    tool_names = set(registry.tools.keys())
    assert "test_tool" in tool_names
    assert "load_skill_details" in tool_names
    assert "read_file" not in tool_names
    assert "write_file" not in tool_names


def test_skill_manager_duplicate_installation_raises_error() -> None:
    """Test that installing the same skill twice raises ValueError."""
    from ecs_agent import BuiltinToolsSkill, SkillManager

    world = World()
    entity = world.create_entity()
    manager = SkillManager()

    # Install skill once
    skill = BuiltinToolsSkill()
    manager.install(world, entity, skill)

    # Try to install same skill again - should raise ValueError
    with pytest.raises(ValueError, match="Tool name collision"):
        manager.install(world, entity, skill)

    # Verify only one instance in metadata
    skill_comp = world.get_component(entity, SkillComponent)
    assert skill_comp is not None
    assert len(skill_comp.skills) == 1

    # Verify tools are not duplicated
    registry = world.get_component(entity, ToolRegistryComponent)
    assert registry is not None
    assert len(registry.tools) == 5  # 4 builtin + 1 meta-tool
