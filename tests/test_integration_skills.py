"""Integration tests for Skills + MCP + Built-in Tools feature set."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from ecs_agent.components.definitions import (
    SkillComponent,
    ToolRegistryComponent,
)
from ecs_agent.core.world import World
from ecs_agent.types import ToolSchema


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
        from ecs_agent import MCPClient as _MCPClient  # type: ignore[attr-defined]

        assert _MCPClient is not None

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


def _write_markdown_skill_fixture(base_dir: Path) -> str:
    skill_name = "lazy-skill"
    skill_dir = base_dir / ".claude" / "skills" / skill_name
    scripts_dir = skill_dir / "scripts"
    scripts_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        "name: lazy-skill\n"
        "description: lazily activated skill\n"
        "user-invocable: false\n"
        "disable-model-invocation: true\n"
        "---\n"
        "Do not eagerly activate this skill.",
        encoding="utf-8",
    )
    (scripts_dir / "hello.py").write_text(
        "import json\n"
        "import sys\n"
        "payload = json.loads(sys.stdin.read() or '{}')\n"
        "print(payload.get('name', 'world'))\n",
        encoding="utf-8",
    )
    return skill_name


@pytest.mark.asyncio
async def test_lazy_discovery_manager_indexes_markdown_skill_without_activation(
    tmp_path: Path,
) -> None:
    from ecs_agent import SkillManager
    from ecs_agent.skills.discovery import DiscoveryManager

    skill_name = _write_markdown_skill_fixture(tmp_path)
    world = World()
    entity = world.create_entity()
    manager = SkillManager()

    await DiscoveryManager().auto_discover_and_install(
        world,
        entity,
        manager,
        directories=[tmp_path],
    )

    skill_comp = world.get_component(entity, SkillComponent)
    assert skill_comp is not None
    metadata = skill_comp.skills[skill_name]
    assert metadata.activated is False
    assert metadata.tool_names == []

    registry = world.get_component(entity, ToolRegistryComponent)
    assert registry is not None
    assert set(registry.tools) == {"load_skill_details"}
    assert "hello" not in registry.tools

    manager_for_policy_checks = SkillManager()
    assert (
        manager_for_policy_checks.can_invoke_via_slash(world, entity, "/lazy-skill")
        is False
    )
    assert (
        manager_for_policy_checks.can_model_auto_invoke_skill(
            world, entity, "lazy-skill"
        )
        is False
    )


@pytest.mark.asyncio
async def test_lazy_manager_activate_registers_markdown_prompt_and_tools_after_index(
    tmp_path: Path,
) -> None:
    from ecs_agent import SkillManager
    from ecs_agent.components import SystemPromptComponent
    from ecs_agent.skills.discovery import DiscoveryManager

    skill_name = _write_markdown_skill_fixture(tmp_path)
    world = World()
    entity = world.create_entity()
    manager = SkillManager()

    await DiscoveryManager().auto_discover_and_install(
        world,
        entity,
        manager,
        directories=[tmp_path],
    )

    manager.activate(world, entity, skill_name)

    skill_comp = world.get_component(entity, SkillComponent)
    assert skill_comp is not None
    metadata = skill_comp.skills[skill_name]
    assert metadata.activated is True

    registry = world.get_component(entity, ToolRegistryComponent)
    assert registry is not None
    assert "hello" in registry.tools
    assert "load_skill_details" in registry.tools

    prompt = world.get_component(entity, SystemPromptComponent)
    assert prompt is not None
    assert "Do not eagerly activate this skill." in prompt.content


# ---------------------------------------------------------------------------
# Naming Contract Tests (skills-refactor-v2 hard switch)
# These tests MUST FAIL until implementation tasks rename the symbols.
# ---------------------------------------------------------------------------


def test_import_skill_protocol_is_markdown_class() -> None:
    """After hard switch: `Skill` from ecs_agent must be the markdown class, not Protocol."""
    import ecs_agent
    from ecs_agent.skills.markdown_skill import Skill as _MarkdownSkill

    # After renaming, `Skill` must be the concrete markdown class
    assert ecs_agent.Skill is _MarkdownSkill, (
        "Naming contract violated: `Skill` must be the markdown skill class (formerly `MarkdownSkill`). "
        "renamed to ScriptSkill — use ScriptSkill for the Python protocol interface."
    )


def test_script_skill_is_exported_as_protocol_interface() -> None:
    """After hard switch: `ScriptSkill` must exist in ecs_agent and be the Protocol."""
    import ecs_agent

    # ScriptSkill must be exported from the package
    assert hasattr(ecs_agent, "ScriptSkill"), (
        "Naming contract violated: `ScriptSkill` not found in ecs_agent. "
        "The Python protocol interface (formerly named `Skill`) must be exported as `ScriptSkill`."
    )
    script_skill = getattr(ecs_agent, "ScriptSkill")
    assert script_skill is not None


def test_markdown_skill_name_not_exported_from_package() -> None:
    """After hard switch: `MarkdownSkill` must NOT exist in ecs_agent top-level exports."""
    import ecs_agent

    # Hard switch: no MarkdownSkill alias, code using MarkdownSkill must migrate to Skill
    assert not hasattr(ecs_agent, "MarkdownSkill"), (
        "Naming contract violated: `MarkdownSkill` is still exported from ecs_agent. "
        "Hard switch complete — remove the MarkdownSkill export. "
        "Migration: use `Skill` instead (renamed to Skill in skills-refactor-v2)."
    )


def test_legacy_skill_protocol_name_not_in_skills_init() -> None:
    """After hard switch: `Skill` in ecs_agent.skills must be the markdown class, not Protocol."""
    import ecs_agent.skills as skills_module
    from ecs_agent.skills.markdown_skill import Skill as _MarkdownSkill

    # The `Skill` export from ecs_agent.skills must now be the markdown class
    skill_export = getattr(skills_module, "Skill", None)
    assert skill_export is not None, "Skill must be exported from ecs_agent.skills"
    assert skill_export is _MarkdownSkill, (
        "Naming contract violated: ecs_agent.skills.Skill must be the markdown class. "
        "The old Skill Protocol is now ScriptSkill — use ScriptSkill for duck-typing checks. "
        "renamed to ScriptSkill — use ScriptSkill for the protocol interface."
    )


def test_script_skill_in_skills_module_is_the_protocol() -> None:
    """After hard switch: `ScriptSkill` in ecs_agent.skills must be the python Protocol."""
    import ecs_agent.skills as skills_module
    from ecs_agent.skills.protocol import ScriptSkill as _OldSkillProtocol

    # ScriptSkill must exist in the skills module
    assert hasattr(skills_module, "ScriptSkill"), (
        "Naming contract violated: `ScriptSkill` not found in ecs_agent.skills. "
        "The Python protocol (formerly `Skill`) must be re-exported as `ScriptSkill`."
    )
    script_skill = getattr(skills_module, "ScriptSkill")
    # ScriptSkill must be the protocol class (what was Skill before)
    assert script_skill is _OldSkillProtocol, (
        "Naming contract violated: ScriptSkill in ecs_agent.skills must be the Protocol class. "
        "After rename: protocol.py's Skill class becomes ScriptSkill. "
        "renamed to ScriptSkill — use ScriptSkill for protocol-based isinstance checks."
    )
