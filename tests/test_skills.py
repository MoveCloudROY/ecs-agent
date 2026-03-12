from __future__ import annotations

from dataclasses import asdict

import pytest

from ecs_agent.components import SystemPromptComponent, ToolRegistryComponent
from ecs_agent.components.definitions import SkillComponent, SkillMetadata
from ecs_agent.core import World
from ecs_agent.skills import ScriptSkill, SkillManager
from ecs_agent.types import EntityId, ToolSchema


async def _noop_handler(**_: object) -> str:
    return "ok"


async def _sum_handler(a: int, b: int) -> str:
    return str(a + b)


def _tool(name: str) -> ToolSchema:
    return ToolSchema(
        name=name,
        description=f"{name} tool",
        parameters={"type": "object", "properties": {}},
    )


class DummySkill:
    def __init__(
        self,
        name: str,
        description: str,
        tool_bundle: dict[str, tuple[ToolSchema, object]],
        prompt: str = "",
    ) -> None:
        self.name = name
        self.description = description
        self._tool_bundle = tool_bundle
        self._prompt = prompt
        self.install_calls = 0
        self.uninstall_calls = 0

    def tools(self) -> dict[str, tuple[ToolSchema, object]]:
        return self._tool_bundle

    def system_prompt(self) -> str:
        return self._prompt

    def install(self, world: World, entity_id: EntityId) -> None:
        _ = world
        _ = entity_id
        self.install_calls += 1

    def uninstall(self, world: World, entity_id: EntityId) -> None:
        _ = world
        _ = entity_id
        self.uninstall_calls += 1


def test_skill_protocol_duck_typing_compliance() -> None:
    skill = DummySkill("math", "math helpers", {"sum": (_tool("sum"), _sum_handler)})
    assert isinstance(skill, ScriptSkill)


def test_skill_component_dataclass_structure() -> None:
    metadata = SkillMetadata(
        name="math",
        description="math helpers",
        tool_names=["sum", "subtract"],
        has_system_prompt=True,
    )
    component = SkillComponent(skills={"math": metadata})

    assert component.skills["math"] == metadata
    assert asdict(metadata) == {
        "name": "math",
        "description": "math helpers",
        "tool_names": ["sum", "subtract"],
        "has_system_prompt": True,
        "activated": False,
        # Extended fields with defaults (Task 3 metadata manifest expansion):
        "user_invocable": True,
        "disable_model_invocation": False,
        "argument_hint": "",
        "allowed_tools": [],
        "context": None,
        "agent": None,
        "model": None,
        "hooks": {},
        "skill_dir_path": None,
        "slash_command": "",
        "substitution_variables": [
            "$ARGUMENTS",
            "$ARGUMENTS[0]",
            "$1",
            "${CLAUDE_SESSION_ID}",
            "${CLAUDE_SKILL_DIR}",
        ],
    }


def test_skill_index_registers_only_metadata() -> None:
    world = World()
    manager = SkillManager()
    entity_id = world.create_entity()
    world.add_component(
        entity_id,
        ToolRegistryComponent(
            tools={"existing": _tool("existing")},
            handlers={"existing": _noop_handler},
        ),
    )

    skill = DummySkill(
        name="math",
        description="math helpers",
        tool_bundle={"sum": (_tool("sum"), _sum_handler)},
        prompt="Use careful arithmetic.",
    )

    manager.index(world, entity_id, skill)

    registry = world.get_component(entity_id, ToolRegistryComponent)
    prompts = world.get_component(entity_id, SystemPromptComponent)
    metadata = manager.get_skill_metadata(world, entity_id, "math")

    assert registry is not None
    assert set(registry.tools) == {"existing", "load_skill_details"}
    assert set(registry.handlers) == {"existing", "load_skill_details"}
    assert prompts is None
    assert metadata is not None
    assert metadata.activated is False
    assert skill.install_calls == 0


def test_skill_activate_loads_prompt_and_tools_after_index() -> None:
    world = World()
    manager = SkillManager()
    entity_id = world.create_entity()
    world.add_component(entity_id, SystemPromptComponent(content="base prompt"))

    skill = DummySkill(
        name="math",
        description="math helpers",
        tool_bundle={"sum": (_tool("sum"), _sum_handler)},
        prompt="Use careful arithmetic.",
    )
    manager.index(world, entity_id, skill)

    manager.activate(world, entity_id, "math")

    registry = world.get_component(entity_id, ToolRegistryComponent)
    prompts = world.get_component(entity_id, SystemPromptComponent)
    metadata = manager.get_skill_metadata(world, entity_id, "math")

    assert registry is not None
    assert set(registry.tools) == {"sum", "load_skill_details"}
    assert set(registry.handlers) == {"sum", "load_skill_details"}
    assert prompts is not None
    assert "base prompt" in prompts.content
    assert "Use careful arithmetic." in prompts.content
    assert metadata is not None
    assert metadata.activated is True
    assert skill.install_calls == 1


def test_skill_activate_is_idempotent() -> None:
    world = World()
    manager = SkillManager()
    entity_id = world.create_entity()

    skill = DummySkill(
        name="math",
        description="math helpers",
        tool_bundle={"sum": (_tool("sum"), _sum_handler)},
        prompt="Use careful arithmetic.",
    )
    manager.index(world, entity_id, skill)

    manager.activate(world, entity_id, "math")
    manager.activate(world, entity_id, "math")

    prompts = world.get_component(entity_id, SystemPromptComponent)
    metadata = manager.get_skill_metadata(world, entity_id, "math")

    assert prompts is not None
    assert prompts.content.count("Use careful arithmetic.") == 1
    assert metadata is not None
    assert metadata.activated is True
    assert skill.install_calls == 1


def test_skill_activate_raises_for_non_indexed_skill() -> None:
    world = World()
    manager = SkillManager()
    entity_id = world.create_entity()

    with pytest.raises(ValueError, match="not indexed"):
        manager.activate(world, entity_id, "missing")


def test_skill_uninstall_removes_indexed_and_activated_artifacts() -> None:
    world = World()
    manager = SkillManager()
    entity_id = world.create_entity()
    world.add_component(entity_id, SystemPromptComponent(content="base prompt"))

    indexed_only_skill = DummySkill(
        name="indexed",
        description="indexed skill",
        tool_bundle={"idx": (_tool("idx"), _noop_handler)},
        prompt="Indexed prompt",
    )
    active_skill = DummySkill(
        name="active",
        description="active skill",
        tool_bundle={"act": (_tool("act"), _noop_handler)},
        prompt="Active prompt",
    )

    manager.index(world, entity_id, indexed_only_skill)
    manager.index(world, entity_id, active_skill)
    manager.activate(world, entity_id, "active")

    manager.uninstall(world, entity_id, "indexed")
    manager.uninstall(world, entity_id, "active")

    registry = world.get_component(entity_id, ToolRegistryComponent)
    metadata_active = manager.get_skill_metadata(world, entity_id, "active")
    metadata_indexed = manager.get_skill_metadata(world, entity_id, "indexed")
    prompt_component = world.get_component(entity_id, SystemPromptComponent)

    assert registry is not None
    assert "idx" not in registry.tools
    assert "act" not in registry.tools
    assert "load_skill_details" not in registry.tools
    assert metadata_active is None
    assert metadata_indexed is None
    assert prompt_component is not None
    assert indexed_only_skill.uninstall_calls == 1
    assert active_skill.uninstall_calls == 1


def test_skill_install_merges_tools() -> None:
    world = World()
    manager = SkillManager()
    entity_id = world.create_entity()
    world.add_component(
        entity_id,
        ToolRegistryComponent(
            tools={"existing": _tool("existing")},
            handlers={"existing": _noop_handler},
        ),
    )
    world.add_component(entity_id, SystemPromptComponent(content="base prompt"))

    skill = DummySkill(
        name="math",
        description="math helpers",
        tool_bundle={
            "sum": (_tool("sum"), _sum_handler),
            "multiply": (_tool("multiply"), _noop_handler),
        },
        prompt="Use careful arithmetic.",
    )

    manager.install(world, entity_id, skill)

    registry = world.get_component(entity_id, ToolRegistryComponent)
    prompts = world.get_component(entity_id, SystemPromptComponent)
    skills = world.get_component(entity_id, SkillComponent)

    assert registry is not None
    assert set(registry.tools) == {
        "existing",
        "sum",
        "multiply",
        "load_skill_details",
    }
    assert set(registry.handlers) == {
        "existing",
        "sum",
        "multiply",
        "load_skill_details",
    }
    assert callable(registry.handlers["existing"])
    assert callable(registry.handlers["sum"])
    assert prompts is not None
    assert "base prompt" in prompts.content
    assert "Use careful arithmetic." in prompts.content
    assert skills is not None
    assert "math" in skills.skills
    assert skills.skills["math"].tool_names == ["sum", "multiply"]
    assert skill.install_calls == 1


def test_skill_install_get_or_create_components() -> None:
    world = World()
    manager = SkillManager()
    entity_id = world.create_entity()

    skill = DummySkill(
        name="math",
        description="math helpers",
        tool_bundle={"sum": (_tool("sum"), _sum_handler)},
        prompt="Respond with equations.",
    )

    manager.install(world, entity_id, skill)

    registry = world.get_component(entity_id, ToolRegistryComponent)
    prompts = world.get_component(entity_id, SystemPromptComponent)
    skills = world.get_component(entity_id, SkillComponent)
    assert registry is not None
    assert set(registry.tools) == {"sum", "load_skill_details"}
    assert prompts is not None
    assert prompts.content == "Respond with equations."
    assert skills is not None
    assert list(skills.skills) == ["math"]


def test_skill_install_collision_raises() -> None:
    world = World()
    manager = SkillManager()
    entity_id = world.create_entity()
    world.add_component(
        entity_id,
        ToolRegistryComponent(
            tools={"read_file": _tool("read_file")},
            handlers={"read_file": _noop_handler},
        ),
    )

    skill = DummySkill(
        name="filesystem",
        description="filesystem tools",
        tool_bundle={"read_file": (_tool("read_file"), _noop_handler)},
    )

    with pytest.raises(ValueError, match="read_file"):
        manager.install(world, entity_id, skill)

    registry = world.get_component(entity_id, ToolRegistryComponent)
    skills = world.get_component(entity_id, SkillComponent)
    assert registry is not None
    assert set(registry.tools) == {"read_file", "load_skill_details"}
    assert skills is None


def test_skill_uninstall_removes_only_skill_tools() -> None:
    world = World()
    manager = SkillManager()
    entity_id = world.create_entity()
    world.add_component(
        entity_id,
        ToolRegistryComponent(
            tools={"existing": _tool("existing")},
            handlers={"existing": _noop_handler},
        ),
    )

    skill = DummySkill(
        name="math",
        description="math helpers",
        tool_bundle={"sum": (_tool("sum"), _sum_handler)},
    )
    manager.install(world, entity_id, skill)

    manager.uninstall(world, entity_id, "math")

    registry = world.get_component(entity_id, ToolRegistryComponent)
    skills = world.get_component(entity_id, SkillComponent)
    assert registry is not None
    assert set(registry.tools) == {"existing"}
    assert set(registry.handlers) == {"existing"}
    assert skills is not None
    assert skills.skills == {}
    assert skill.uninstall_calls == 1


def test_skill_multiple_skills_can_be_installed_and_listed() -> None:
    world = World()
    manager = SkillManager()
    entity_id = world.create_entity()

    skill_one = DummySkill(
        name="math",
        description="math helpers",
        tool_bundle={"sum": (_tool("sum"), _sum_handler)},
    )
    skill_two = DummySkill(
        name="text",
        description="text helpers",
        tool_bundle={"title": (_tool("title"), _noop_handler)},
    )

    manager.install(world, entity_id, skill_one)
    manager.install(world, entity_id, skill_two)

    listed = manager.list_skills(world, entity_id)
    assert {skill.name for skill in listed} == {"math", "text"}


def test_skill_uninstall_one_does_not_affect_another() -> None:
    world = World()
    manager = SkillManager()
    entity_id = world.create_entity()

    skill_one = DummySkill(
        name="math",
        description="math helpers",
        tool_bundle={"sum": (_tool("sum"), _sum_handler)},
    )
    skill_two = DummySkill(
        name="text",
        description="text helpers",
        tool_bundle={"title": (_tool("title"), _noop_handler)},
    )
    manager.install(world, entity_id, skill_one)
    manager.install(world, entity_id, skill_two)

    manager.uninstall(world, entity_id, "math")

    registry = world.get_component(entity_id, ToolRegistryComponent)
    text_meta = manager.get_skill_metadata(world, entity_id, "text")
    math_meta = manager.get_skill_metadata(world, entity_id, "math")
    assert registry is not None
    assert set(registry.tools) == {"title", "load_skill_details"}
    assert text_meta is not None
    assert text_meta.tool_names == ["title"]
    assert math_meta is None


# ---------------------------------------------------------------------------
# Naming Contract Tests (skills-refactor-v2 hard switch)
# These tests MUST FAIL until implementation tasks rename the symbols.
# ---------------------------------------------------------------------------


def test_skill_protocol_duck_typing_uses_script_skill_name() -> None:
    """After hard switch: duck-typing against protocol must use ScriptSkill, not Skill."""
    from ecs_agent.skills import ScriptSkill

    # ScriptSkill is the protocol interface for Python-based skills
    skill = DummySkill("math", "math helpers", {"sum": (_tool("sum"), _sum_handler)})
    assert isinstance(skill, ScriptSkill), (
        "Naming contract violated: DummySkill must satisfy ScriptSkill protocol. "
        "renamed to ScriptSkill — use ScriptSkill for duck-typing checks."
    )


def test_skill_export_from_skills_init_is_not_protocol_class() -> None:
    """After hard switch: `Skill` from ecs_agent.skills is the markdown class, not Protocol."""
    import tempfile
    from pathlib import Path
    from ecs_agent.skills import Skill
    from ecs_agent.skills.markdown_skill import Skill as _MarkdownSkill

    # After rename: Skill must be MarkdownSkill (concrete markdown implementation)
    assert Skill is _MarkdownSkill, (
        "Naming contract violated: `Skill` from ecs_agent.skills must be the markdown skill class. "
        "renamed to ScriptSkill — use ScriptSkill for the protocol interface, "
        "`Skill` now refers to the markdown-based skill class (formerly MarkdownSkill)."
    )

    # Also verify we can instantiate it as the markdown class
    content = "---\nname: contract-test\ndescription: contract test\n---\n# body"
    with tempfile.TemporaryDirectory() as tmpdir:
        skill_path = Path(tmpdir) / "SKILL.md"
        skill_path.write_text(content)
        instance = Skill(skill_path)
        assert instance.name == "contract-test"


# ---------------------------------------------------------------------------
# Metadata alignment: uniform invocation controls for ScriptSkill and Skill
# ---------------------------------------------------------------------------


def test_metadata_user_invocable_defaults_true_for_script_skill() -> None:
    """ScriptSkill without user_invocable attr gets user_invocable=True in SkillMetadata.

    manager.index() uses getattr(skill, 'user_invocable', True) so a plain Python
    ScriptSkill object that does not declare user_invocable defaults to True — meaning
    the skill is user-invocable unless explicitly opted-out.
    """
    world = World()
    manager = SkillManager()
    entity_id = world.create_entity()

    # DummySkill has NO user_invocable attribute — exercises the getattr default path
    skill = DummySkill(
        name="no-invocable-attr",
        description="ScriptSkill without user_invocable",
        tool_bundle={"t": (_tool("t"), _noop_handler)},
    )
    assert not hasattr(skill, "user_invocable"), (
        "DummySkill must not have user_invocable for this test"
    )

    manager.index(world, entity_id, skill)

    metadata = manager.get_skill_metadata(world, entity_id, "no-invocable-attr")
    assert metadata is not None
    assert metadata.user_invocable is True, (
        "ScriptSkill without user_invocable attr must default to user_invocable=True"
    )


def test_metadata_disable_model_invocation_defaults_false_for_script_skill() -> None:
    """ScriptSkill without disable_model_invocation attr gets disable_model_invocation=False.

    manager.index() uses getattr(skill, 'disable_model_invocation', False) so a plain Python
    ScriptSkill object that does not declare disable_model_invocation defaults to False —
    meaning model auto-invocation is enabled unless explicitly disabled.
    """
    world = World()
    manager = SkillManager()
    entity_id = world.create_entity()

    # DummySkill has NO disable_model_invocation attribute — exercises the getattr default path
    skill = DummySkill(
        name="no-disable-attr",
        description="ScriptSkill without disable_model_invocation",
        tool_bundle={"t": (_tool("t"), _noop_handler)},
    )
    assert not hasattr(skill, "disable_model_invocation"), (
        "DummySkill must not have disable_model_invocation for this test"
    )

    manager.index(world, entity_id, skill)

    metadata = manager.get_skill_metadata(world, entity_id, "no-disable-attr")
    assert metadata is not None
    assert metadata.disable_model_invocation is False, (
        "ScriptSkill without disable_model_invocation attr must default to disable_model_invocation=False"
    )

    # Also verify: can_model_auto_invoke_skill returns True (model CAN invoke by default)
    assert manager.can_model_auto_invoke_skill(world, entity_id, "no-disable-attr") is True, (
        "can_model_auto_invoke_skill must return True when disable_model_invocation=False"
    )


def test_metadata_invalid_frontmatter_gets_graceful_defaults() -> None:
    """Markdown skill with malformed/missing frontmatter produces safe metadata defaults.

    When a SKILL.md has invalid YAML (e.g. unterminated bracket), the Skill object has
    valid=False and the invocation control attributes are NOT set. The properties fall back
    to safe defaults via getattr: user_invocable=True, disable_model_invocation=False.
    """
    import tempfile
    from pathlib import Path
    from ecs_agent.skills.markdown_skill import Skill

    malformed = "---\nname: bad-yaml\ndescription: [unterminated\n---\nBody"

    with tempfile.TemporaryDirectory() as tmpdir:
        skill_path = Path(tmpdir) / "SKILL.md"
        skill_path.write_text(malformed)

        skill = Skill(skill_path)

        # Confirm the skill is invalid due to malformed frontmatter
        assert skill.valid is False, "Expected invalid skill for malformed YAML"

        # Invocation control properties must still return safe defaults
        assert skill.user_invocable is True, (
            "user_invocable must default to True even for invalid Skill (safe default)"
        )
        assert skill.disable_model_invocation is False, (
            "disable_model_invocation must default to False even for invalid Skill (safe default)"
        )
