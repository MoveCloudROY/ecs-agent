"""Tests for MarkdownSkill parser."""

import logging
import tempfile
from pathlib import Path

import pytest

from ecs_agent.components.definitions import (
    SystemPromptComponent,
    ToolRegistryComponent,
)
from ecs_agent.core.world import World
from ecs_agent.skills.markdown_skill import MarkdownSkill
from ecs_agent.skills.protocol import Skill


def test_markdown_skill_parses_yaml_frontmatter() -> None:
    """parse_skill_file extracts YAML frontmatter."""
    content = """---
name: test-skill
description: A test skill
---
# Test Skill Body
This is the content."""

    with tempfile.TemporaryDirectory() as tmpdir:
        skill_path = Path(tmpdir) / "SKILL.md"
        skill_path.write_text(content)

        skill = MarkdownSkill(skill_path)

        assert skill.name == "test-skill"
        assert skill.description == "A test skill"


def test_markdown_skill_extracts_body_as_system_prompt() -> None:
    """system_prompt() returns markdown body after frontmatter."""
    content = """---
name: my-skill
description: Testing
---
# Skill Content

This is the system prompt content.
It has multiple lines."""

    with tempfile.TemporaryDirectory() as tmpdir:
        skill_path = Path(tmpdir) / "SKILL.md"
        skill_path.write_text(content)

        skill = MarkdownSkill(skill_path)
        prompt = skill.system_prompt()

        assert "# Skill Content" in prompt
        assert "This is the system prompt content." in prompt
        assert "It has multiple lines." in prompt
        # Frontmatter should NOT be in the prompt
        assert "name: my-skill" not in prompt
        assert "---" not in prompt


def test_markdown_skill_implements_skill_protocol() -> None:
    """MarkdownSkill implements Skill protocol."""
    content = """---
name: protocol-test
description: Test protocol implementation
---
# Content"""

    with tempfile.TemporaryDirectory() as tmpdir:
        skill_path = Path(tmpdir) / "SKILL.md"
        skill_path.write_text(content)

        skill = MarkdownSkill(skill_path)

        assert isinstance(skill, Skill)


def test_markdown_skill_tools_empty_for_prompt_only() -> None:
    """Pure prompt skill with no scripts returns empty tools dict."""
    content = """---
name: prompt-only
description: Prompt-only skill
---
# Just a prompt"""

    with tempfile.TemporaryDirectory() as tmpdir:
        skill_path = Path(tmpdir) / "SKILL.md"
        skill_path.write_text(content)

        skill = MarkdownSkill(skill_path)
        tools = skill.tools()

        assert tools == {}


def test_markdown_skill_tools_from_scripts_directory() -> None:
    """Discover scripts/ directory and register tool schemas."""
    content = """---
name: script-skill
description: Skill with scripts
---
# Skill with scripts"""

    with tempfile.TemporaryDirectory() as tmpdir:
        skill_dir = Path(tmpdir)
        skill_path = skill_dir / "SKILL.md"
        skill_path.write_text(content)

        # Create scripts directory with a Python script
        scripts_dir = skill_dir / "scripts"
        scripts_dir.mkdir()
        (scripts_dir / "my_tool.py").write_text(
            "#!/usr/bin/env python3\nprint('hello')"
        )

        skill = MarkdownSkill(skill_path)
        tools = skill.tools()

        assert "my_tool" in tools
        schema, handler = tools["my_tool"]
        assert schema.name == "my_tool"
        assert callable(handler)


def test_markdown_skill_install_adds_system_prompt() -> None:
    """install() adds system prompt to SystemPromptComponent."""
    content = """---
name: install-test
description: Test install
---
# System Prompt Content"""

    with tempfile.TemporaryDirectory() as tmpdir:
        skill_path = Path(tmpdir) / "SKILL.md"
        skill_path.write_text(content)

        world = World()
        entity = world.create_entity()
        skill = MarkdownSkill(skill_path)

        skill.install(world, entity)

        prompt_comp = world.get_component(entity, SystemPromptComponent)
        assert prompt_comp is not None
        assert "# System Prompt Content" in prompt_comp.content


def test_markdown_skill_uninstall_removes_prompt() -> None:
    """uninstall() removes system prompt from SystemPromptComponent."""
    content = """---
name: uninstall-test
description: Test uninstall
---
# Content"""

    with tempfile.TemporaryDirectory() as tmpdir:
        skill_path = Path(tmpdir) / "SKILL.md"
        skill_path.write_text(content)

        world = World()
        entity = world.create_entity()
        skill = MarkdownSkill(skill_path)

        skill.install(world, entity)
        assert world.get_component(entity, SystemPromptComponent) is not None

        skill.uninstall(world, entity)
        # Should remove the system prompt added by this skill
        # (In practice, uninstall should remove ONLY this skill's prompt)
        prompt_comp = world.get_component(entity, SystemPromptComponent)
        if prompt_comp:
            assert "# Content" not in prompt_comp.content


def test_markdown_skill_handles_missing_frontmatter() -> None:
    """Skill without YAML frontmatter uses sensible defaults."""
    content = """# No Frontmatter Skill

Just markdown content without frontmatter."""

    with tempfile.TemporaryDirectory() as tmpdir:
        skill_path = Path(tmpdir) / "SKILL.md"
        skill_path.write_text(content)

        skill = MarkdownSkill(skill_path)

        # Should use filename or default name
        assert skill.name != ""
        assert skill.description != ""
        assert "# No Frontmatter Skill" in skill.system_prompt()


@pytest.mark.asyncio
async def test_markdown_skill_script_tool_executes_subprocess() -> None:
    """Script tool handler runs Python subprocess with JSON stdin."""
    content = """---
name: subprocess-test
description: Test subprocess execution
---
# Subprocess test"""

    with tempfile.TemporaryDirectory() as tmpdir:
        skill_dir = Path(tmpdir)
        skill_path = skill_dir / "SKILL.md"
        skill_path.write_text(content)

        # Create script that echoes input arguments
        scripts_dir = skill_dir / "scripts"
        scripts_dir.mkdir()
        script_content = """#!/usr/bin/env python3
import json
import sys
args = json.load(sys.stdin)
print(f"Received: {args['message']}")
"""
        (scripts_dir / "echo_tool.py").write_text(script_content)

        skill = MarkdownSkill(skill_path)
        tools = skill.tools()

        assert "echo_tool" in tools
        _, handler = tools["echo_tool"]

        result = await handler(message="hello world")

        assert "Received: hello world" in result


def test_markdown_skill_install_registers_tool_handlers() -> None:
    """install() registers tool schemas and handlers in ToolRegistryComponent."""
    content = """---
name: tool-install-test
description: Test tool registration
---
# Tools test"""

    with tempfile.TemporaryDirectory() as tmpdir:
        skill_dir = Path(tmpdir)
        skill_path = skill_dir / "SKILL.md"
        skill_path.write_text(content)

        scripts_dir = skill_dir / "scripts"
        scripts_dir.mkdir()
        (scripts_dir / "example_tool.py").write_text(
            "#!/usr/bin/env python3\nprint('test')"
        )

        world = World()
        entity = world.create_entity()
        skill = MarkdownSkill(skill_path)

        skill.install(world, entity)

        tool_reg = world.get_component(entity, ToolRegistryComponent)
        assert tool_reg is not None
        assert "example_tool" in tool_reg.tools
        assert "example_tool" in tool_reg.handlers


def test_markdown_skill_multiple_scripts() -> None:
    """Discover multiple scripts in scripts/ directory."""
    content = """---
name: multi-script
description: Multiple scripts
---
# Multi-script skill"""

    with tempfile.TemporaryDirectory() as tmpdir:
        skill_dir = Path(tmpdir)
        skill_path = skill_dir / "SKILL.md"
        skill_path.write_text(content)

        scripts_dir = skill_dir / "scripts"
        scripts_dir.mkdir()
        (scripts_dir / "tool_a.py").write_text("#!/usr/bin/env python3")
        (scripts_dir / "tool_b.py").write_text("#!/usr/bin/env python3")
        (scripts_dir / "tool_c.py").write_text("#!/usr/bin/env python3")

        skill = MarkdownSkill(skill_path)
        tools = skill.tools()

        assert len(tools) == 3
        assert "tool_a" in tools
        assert "tool_b" in tools
        assert "tool_c" in tools


# --- Discovery Integration Tests (Task 10) ---


def test_skill_discovery_finds_markdown_skills() -> None:
    """discover_markdown_skills finds SKILL.md files in .claude/skills/ directories."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        skill_dir = base / ".claude" / "skills" / "my-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\nname: my-skill\ndescription: Test skill\n---\nPrompt content"
        )

        from ecs_agent.skills.discovery import discover_markdown_skills

        skills = discover_markdown_skills([base])
        assert len(skills) == 1
        assert skills[0].name == "my-skill"
        assert skills[0].description == "Test skill"


def test_skill_discovery_ignores_non_skill_md_files() -> None:
    """discover_markdown_skills ignores random .md files not named SKILL.md."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        # Create a random .md file (not SKILL.md)
        (base / "README.md").write_text("# Random README")
        # Create a .md file in a subdirectory (but not SKILL.md)
        subdir = base / "docs"
        subdir.mkdir()
        (subdir / "guide.md").write_text("# Guide")

        from ecs_agent.skills.discovery import discover_markdown_skills

        skills = discover_markdown_skills([base])
        assert len(skills) == 0


async def test_discovery_manager_auto_discovers_markdown_skills() -> None:
    """DiscoveryManager.auto_discover_and_install finds and installs MarkdownSkills."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        skill_dir = base / ".claude" / "skills" / "auto-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\nname: auto-skill\ndescription: Auto-discovered\n---\nAuto prompt"
        )

        from ecs_agent.skills.discovery import DiscoveryManager

        world = World()
        entity = world.create_entity()
        manager = DiscoveryManager()

        from ecs_agent.skills.manager import SkillManager

        skill_manager = SkillManager()

        # Auto-discover from base directory
        await manager.auto_discover_and_install(
            world, entity, skill_manager, directories=[base]
        )

        # Verify skill was installed
        from ecs_agent.components.definitions import SkillComponent

        skill_comp = world.get_component(entity, SkillComponent)
        assert skill_comp is not None
        assert "auto-skill" in skill_comp.skills
        assert skill_comp.skills["auto-skill"].name == "auto-skill"


@pytest.mark.parametrize(
    ("frontmatter", "expected_name", "expected_description", "expected_body"),
    [
        (
            "---\nname: ui-design\ndescription: Build UI\n---\n# Prompt\nUse a design system.",
            "ui-design",
            "Build UI",
            "# Prompt\nUse a design system.",
        )
    ],
)
def test_markdown_skill_contract_frontmatter_structure_extracts_body_after_closing_delimiter(
    frontmatter: str,
    expected_name: str,
    expected_description: str,
    expected_body: str,
) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        skill_path = Path(tmpdir) / "SKILL.md"
        skill_path.write_text(frontmatter)

        skill = MarkdownSkill(skill_path)

        assert skill.name == expected_name
        assert skill.description == expected_description
        assert skill.system_prompt() == expected_body


@pytest.mark.parametrize(
    "content",
    [
        "---\nname: only-name\n---\nbody",
        "---\ndescription: only description\n---\nbody",
    ],
)
def test_markdown_skill_contract_required_frontmatter_fields_missing_skips_discovery(
    content: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from ecs_agent.skills.discovery import discover_markdown_skills

    caplog.set_level(logging.WARNING)

    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        skill_dir = base / ".claude" / "skills" / "missing-required"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(content)

        skills = discover_markdown_skills([base])

        assert skills == []
        assert any(
            "required" in record.getMessage().lower() for record in caplog.records
        )


@pytest.mark.parametrize(
    "invalid_name",
    [
        "Uppercase",
        "snake_case",
        "with space",
        "a" * 65,
    ],
)
def test_markdown_skill_contract_invalid_name_format_skips_discovery(
    invalid_name: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from ecs_agent.skills.discovery import discover_markdown_skills

    caplog.set_level(logging.WARNING)

    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        skill_dir = base / ".claude" / "skills" / "invalid-name"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            f"---\nname: {invalid_name}\ndescription: bad\n---\nbody"
        )

        skills = discover_markdown_skills([base])

        assert skills == []
        assert any("name" in record.getMessage().lower() for record in caplog.records)


def test_markdown_skill_contract_invalid_yaml_skips_skill_without_raising(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from ecs_agent.skills.discovery import discover_markdown_skills

    caplog.set_level(logging.WARNING)

    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        skill_dir = base / ".claude" / "skills" / "broken-yaml"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\nname: broken\ndescription: [unterminated\n---\nBody"
        )

        skills = discover_markdown_skills([base])

        assert skills == []
        assert any("yaml" in record.getMessage().lower() for record in caplog.records)


def test_markdown_skill_contract_optional_frontmatter_defaults_persist_in_metadata() -> (
    None
):
    from ecs_agent.components.definitions import SkillComponent
    from ecs_agent.skills.discovery import DiscoveryManager
    from ecs_agent.skills.manager import SkillManager

    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        skill_dir = base / ".claude" / "skills" / "defaults"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\nname: defaults\ndescription: Defaults contract\n---\nContract body"
        )

        world = World()
        entity = world.create_entity()
        manager = SkillManager()
        discovery = DiscoveryManager()

        import asyncio

        asyncio.run(
            discovery.auto_discover_and_install(
                world,
                entity,
                manager,
                directories=[base],
            )
        )

        skill_comp = world.get_component(entity, SkillComponent)
        assert skill_comp is not None
        metadata = skill_comp.skills["defaults"]

        assert getattr(metadata, "user_invocable", None) is True
        assert getattr(metadata, "disable_model_invocation", None) is False
        assert getattr(metadata, "argument_hint", None) == ""
        assert getattr(metadata, "allowed_tools", None) == []
        assert getattr(metadata, "context", None) is None
        assert getattr(metadata, "agent", None) is None
        assert getattr(metadata, "model", None) is None
        assert getattr(metadata, "hooks", None) == {}


def test_markdown_skill_contract_slash_command_identity_maps_from_name() -> None:
    from ecs_agent.components.definitions import SkillComponent
    from ecs_agent.skills.discovery import DiscoveryManager
    from ecs_agent.skills.manager import SkillManager

    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        skill_dir = base / ".claude" / "skills" / "ui-design"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\nname: ui-design\ndescription: Slash contract\n---\nPrompt"
        )

        world = World()
        entity = world.create_entity()

        import asyncio

        asyncio.run(
            DiscoveryManager().auto_discover_and_install(
                world,
                entity,
                SkillManager(),
                directories=[base],
            )
        )

        skill_comp = world.get_component(entity, SkillComponent)
        assert skill_comp is not None
        metadata = skill_comp.skills["ui-design"]
        assert getattr(metadata, "slash_command", None) == "/ui-design"


def test_markdown_skill_contract_lazy_discovery_does_not_read_markdown_body() -> None:
    from ecs_agent.skills.discovery import discover_markdown_skills

    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        skill_dir = base / ".claude" / "skills" / "lazy"
        skill_dir.mkdir(parents=True)

        skill_file = skill_dir / "SKILL.md"
        skill_file.write_bytes(
            (
                b"---\n"
                b"name: lazy\n"
                b"description: Metadata only\n"
                b"---\n"
                b"# body starts\n"
                b"\xff\xfe\x00\x00"
            )
        )

        skills = discover_markdown_skills([base])

        assert len(skills) == 1
        assert skills[0].name == "lazy"


def test_markdown_skill_contract_substitutions_available_in_metadata_contract() -> None:
    from ecs_agent.components.definitions import SkillComponent
    from ecs_agent.skills.discovery import DiscoveryManager
    from ecs_agent.skills.manager import SkillManager

    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        skill_dir = base / ".claude" / "skills" / "subs"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\nname: subs\ndescription: substitutions\n---\nUse $ARGUMENTS"
        )

        world = World()
        entity = world.create_entity()

        import asyncio

        asyncio.run(
            DiscoveryManager().auto_discover_and_install(
                world,
                entity,
                SkillManager(),
                directories=[base],
            )
        )

        skill_comp = world.get_component(entity, SkillComponent)
        assert skill_comp is not None
        metadata = skill_comp.skills["subs"]
        substitutions = set(getattr(metadata, "substitution_variables", []))
        assert {
            "$ARGUMENTS",
            "$ARGUMENTS[0]",
            "$1",
            "${CLAUDE_SESSION_ID}",
            "${CLAUDE_SKILL_DIR}",
        }.issubset(substitutions)


def test_markdown_skill_contract_invocation_controls_are_parsed() -> None:
    from ecs_agent.components.definitions import SkillComponent
    from ecs_agent.skills.discovery import DiscoveryManager
    from ecs_agent.skills.manager import SkillManager

    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        skill_dir = base / ".claude" / "skills" / "controls"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\n"
            "name: controls\n"
            "description: controls\n"
            "user-invocable: false\n"
            "disable-model-invocation: true\n"
            "---\n"
            "Prompt"
        )

        world = World()
        entity = world.create_entity()

        import asyncio

        asyncio.run(
            DiscoveryManager().auto_discover_and_install(
                world,
                entity,
                SkillManager(),
                directories=[base],
            )
        )

        skill_comp = world.get_component(entity, SkillComponent)
        assert skill_comp is not None
        metadata = skill_comp.skills["controls"]
        assert getattr(metadata, "user_invocable", None) is False
        assert getattr(metadata, "disable_model_invocation", None) is True


def test_markdown_skill_resolve_supporting_path_resolves_relative_path(
    tmp_path: Path,
) -> None:
    skill_file = tmp_path / "SKILL.md"
    skill_file.write_text("---\nname: resolver\ndescription: resolver\n---\nPrompt")

    skill = MarkdownSkill(skill_file)
    resolved = skill.resolve_supporting_path("data.json")

    assert resolved == (tmp_path / "data.json").resolve()
    assert str(resolved).endswith("data.json")
    assert str(resolved).startswith(str(tmp_path.resolve()))


def test_markdown_skill_resolve_supporting_path_blocks_traversal(
    tmp_path: Path,
) -> None:
    skill_file = tmp_path / "SKILL.md"
    skill_file.write_text("---\nname: traversal\ndescription: traversal\n---\nPrompt")

    skill = MarkdownSkill(skill_file)

    with pytest.raises(ValueError, match="Path traversal"):
        skill.resolve_supporting_path("../etc/passwd")


def test_markdown_skill_resolve_supporting_path_blocks_absolute_path(
    tmp_path: Path,
) -> None:
    skill_file = tmp_path / "SKILL.md"
    skill_file.write_text("---\nname: absolute\ndescription: absolute\n---\nPrompt")

    skill = MarkdownSkill(skill_file)

    with pytest.raises(ValueError):
        skill.resolve_supporting_path("/etc/passwd")


def test_markdown_skill_resolve_supporting_path_allows_nested_relative(
    tmp_path: Path,
) -> None:
    skill_file = tmp_path / "SKILL.md"
    skill_file.write_text("---\nname: nested\ndescription: nested\n---\nPrompt")

    skill = MarkdownSkill(skill_file)
    resolved = skill.resolve_supporting_path("assets/image.png")

    assert resolved == (tmp_path / "assets" / "image.png").resolve()
    assert str(resolved).startswith(str(tmp_path.resolve()))


def test_markdown_skill_contract_path_traversal_is_blocked_for_supporting_files() -> (
    None
):
    with tempfile.TemporaryDirectory() as tmpdir:
        skill_path = Path(tmpdir) / "SKILL.md"
        skill_path.write_text(
            "---\nname: traversal\ndescription: traversal\n---\nPrompt"
        )

        skill = MarkdownSkill(skill_path)
        resolver = getattr(skill, "resolve_supporting_path", None)

        assert callable(resolver)
        with pytest.raises(ValueError):
            resolver("../secrets.txt")


# ---------------------------------------------------------------------------
# MarkdownSkill advanced injection guard tests
# ---------------------------------------------------------------------------


def test_markdown_skill_advanced_injection_policy_defaults_to_deny() -> None:
    """injection_policy defaults to deny-by-default."""
    with tempfile.TemporaryDirectory() as tmpdir:
        skill_path = Path(tmpdir) / "SKILL.md"
        skill_path.write_text(
            "---\nname: injection-guard\ndescription: injection guard\n---\nPrompt"
        )

        skill = MarkdownSkill(skill_path)

        assert skill.injection_policy == "deny"


@pytest.mark.parametrize(
    "content",
    [
        "safe content",
        "no backtick here",
        "",
        "regular `backtick` without bang",
    ],
)
def test_markdown_skill_injection_safe_allows_non_shell_patterns(content: str) -> None:
    """is_dynamic_injection_safe returns True for safe content."""
    with tempfile.TemporaryDirectory() as tmpdir:
        skill_path = Path(tmpdir) / "SKILL.md"
        skill_path.write_text("---\nname: safe\ndescription: safe\n---\nPrompt")

        skill = MarkdownSkill(skill_path)

        assert skill.is_dynamic_injection_safe(content) is True


@pytest.mark.parametrize(
    "content",
    [
        "use !`ls -la` here",
        "run !`echo hello`",
    ],
)
def test_markdown_skill_injection_blocked_for_shell_backtick_patterns(
    content: str,
) -> None:
    """is_dynamic_injection_safe returns False for !`...` shell execution forms."""
    with tempfile.TemporaryDirectory() as tmpdir:
        skill_path = Path(tmpdir) / "SKILL.md"
        skill_path.write_text("---\nname: blocked\ndescription: blocked\n---\nPrompt")

        skill = MarkdownSkill(skill_path)

        assert skill.is_dynamic_injection_safe(content) is False


# ---------------------------------------------------------------------------
# render_skill_content / render_with_arguments — substitution engine tests
# ---------------------------------------------------------------------------


def test_render_skill_content_substitutes_arguments_placeholder(
    tmp_path: Path,
) -> None:
    """$ARGUMENTS is replaced by the entire arguments string."""
    from ecs_agent.skills.markdown_skill import render_skill_content

    result = render_skill_content(
        template="Use this: $ARGUMENTS",
        arguments="foo bar baz",
        skill_dir=tmp_path,
    )
    assert result == "Use this: foo bar baz"


def test_render_skill_content_substitutes_arguments_indexed(
    tmp_path: Path,
) -> None:
    """$ARGUMENTS[N] is replaced by the Nth word (0-indexed)."""
    from ecs_agent.skills.markdown_skill import render_skill_content

    result = render_skill_content(
        template="first=$ARGUMENTS[0] second=$ARGUMENTS[1]",
        arguments="hello world",
        skill_dir=tmp_path,
    )
    assert result == "first=hello second=world"


def test_render_skill_content_substitutes_dollar_n_shorthand(
    tmp_path: Path,
) -> None:
    """$1 and $2 are replaced by first and second words."""
    from ecs_agent.skills.markdown_skill import render_skill_content

    result = render_skill_content(
        template="arg1=$1 arg2=$2",
        arguments="alpha beta gamma",
        skill_dir=tmp_path,
    )
    assert result == "arg1=alpha arg2=beta"


def test_render_skill_content_substitutes_session_id(
    tmp_path: Path,
) -> None:
    """${CLAUDE_SESSION_ID} is replaced by the literal string '<session-id>'."""
    from ecs_agent.skills.markdown_skill import render_skill_content

    result = render_skill_content(
        template="session=${CLAUDE_SESSION_ID}",
        arguments="",
        skill_dir=tmp_path,
    )
    assert result == "session=<session-id>"


def test_render_skill_content_substitutes_skill_dir(
    tmp_path: Path,
) -> None:
    """${CLAUDE_SKILL_DIR} is replaced by str(skill_dir)."""
    from ecs_agent.skills.markdown_skill import render_skill_content

    result = render_skill_content(
        template="dir=${CLAUDE_SKILL_DIR}",
        arguments="",
        skill_dir=tmp_path,
    )
    assert result == f"dir={tmp_path}"


def test_render_skill_content_out_of_bounds_arguments_index_returns_empty(
    tmp_path: Path,
) -> None:
    """$ARGUMENTS[99] returns empty string when there are fewer than 100 words."""
    from ecs_agent.skills.markdown_skill import render_skill_content

    result = render_skill_content(
        template="word=$ARGUMENTS[99]",
        arguments="one two three",
        skill_dir=tmp_path,
    )
    assert result == "word="


def test_render_skill_content_out_of_bounds_dollar_n_returns_empty(
    tmp_path: Path,
) -> None:
    """$9 returns empty string when fewer than 9 words provided."""
    from ecs_agent.skills.markdown_skill import render_skill_content

    result = render_skill_content(
        template="ninth=$9",
        arguments="only one",
        skill_dir=tmp_path,
    )
    assert result == "ninth="


def test_render_skill_content_unknown_variable_left_unchanged(
    tmp_path: Path,
) -> None:
    """${UNKNOWN_VAR} is left as-is (no modification)."""
    from ecs_agent.skills.markdown_skill import render_skill_content

    result = render_skill_content(
        template="x=${UNKNOWN_VAR}",
        arguments="",
        skill_dir=tmp_path,
    )
    assert result == "x=${UNKNOWN_VAR}"


def test_render_skill_content_empty_arguments_string(
    tmp_path: Path,
) -> None:
    """Empty arguments string: $ARGUMENTS becomes empty, $1 becomes empty."""
    from ecs_agent.skills.markdown_skill import render_skill_content

    result = render_skill_content(
        template="args='$ARGUMENTS' first='$1'",
        arguments="",
        skill_dir=tmp_path,
    )
    assert result == "args='' first=''"


def test_render_skill_content_arguments_index_before_full_arguments(
    tmp_path: Path,
) -> None:
    """$ARGUMENTS[N] is processed before $ARGUMENTS to avoid partial conflicts."""
    from ecs_agent.skills.markdown_skill import render_skill_content

    # If $ARGUMENTS replaced first, '$ARGUMENTS[0]' becomes 'foo bar[0]' — wrong.
    # Correct order: $ARGUMENTS[0] → 'foo', then $ARGUMENTS → 'foo bar'
    result = render_skill_content(
        template="indexed=$ARGUMENTS[0] full=$ARGUMENTS",
        arguments="foo bar",
        skill_dir=tmp_path,
    )
    assert result == "indexed=foo full=foo bar"


def test_render_with_arguments_uses_skill_dir_path(
    tmp_path: Path,
) -> None:
    """render_with_arguments() uses self.skill_dir_path for ${CLAUDE_SKILL_DIR}."""
    skill_file = tmp_path / "SKILL.md"
    skill_file.write_text(
        "---\nname: render-test\ndescription: render test\n---\nPrompt"
    )

    skill = MarkdownSkill(skill_file)
    result = skill.render_with_arguments(
        template="dir=${CLAUDE_SKILL_DIR}",
        arguments="",
    )
    assert result == f"dir={tmp_path}"


def test_render_with_arguments_full_substitution_round_trip(
    tmp_path: Path,
) -> None:
    """render_with_arguments() resolves all substitution forms in one call."""
    skill_file = tmp_path / "SKILL.md"
    skill_file.write_text("---\nname: round-trip\ndescription: round trip\n---\nPrompt")

    skill = MarkdownSkill(skill_file)
    result = skill.render_with_arguments(
        template=(
            "full=$ARGUMENTS indexed=$ARGUMENTS[0] "
            "short=$1 session=${CLAUDE_SESSION_ID} dir=${CLAUDE_SKILL_DIR}"
        ),
        arguments="hello world",
    )
    assert result == (
        f"full=hello world indexed=hello "
        f"short=hello session=<session-id> dir={tmp_path}"
    )
