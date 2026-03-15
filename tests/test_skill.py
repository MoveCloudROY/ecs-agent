"""Tests for Skill (markdown-based skill) parser."""

import logging
import tempfile
from pathlib import Path

import pytest

from ecs_agent.components.definitions import (
    SystemPromptComponent,
    ToolRegistryComponent,
)
from ecs_agent.core.world import World
from ecs_agent.skills.skill import Skill
from ecs_agent.skills.script_skill import ScriptSkill


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

        skill = Skill(skill_path)

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

        skill = Skill(skill_path)
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

        skill = Skill(skill_path)

        assert isinstance(skill, ScriptSkill)


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

        skill = Skill(skill_path)
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

        skill = Skill(skill_path)
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
        skill = Skill(skill_path)

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
        skill = Skill(skill_path)

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

        skill = Skill(skill_path)

        # Per new spec: no frontmatter → invalid
        assert skill.valid is False
        assert skill.name == ""
        assert skill.description == ""


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

        skill = Skill(skill_path)
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
        skill = Skill(skill_path)

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

        skill = Skill(skill_path)
        tools = skill.tools()

        assert len(tools) == 3
        assert "tool_a" in tools
        assert "tool_b" in tools
        assert "tool_c" in tools


# --- Contract Tests (Task 8+) ---


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

        skill = Skill(skill_path)

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
    from ecs_agent.skills.discovery import discover_skills

    caplog.set_level(logging.WARNING)

    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        skill_dir = base / ".claude" / "skills" / "missing-required"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(content)

        skills = discover_skills([base])

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
    from ecs_agent.skills.discovery import discover_skills

    caplog.set_level(logging.WARNING)

    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        skill_dir = base / ".claude" / "skills" / "invalid-name"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            f"---\nname: {invalid_name}\ndescription: bad\n---\nbody"
        )

        skills = discover_skills([base])

        assert skills == []
        assert any("name" in record.getMessage().lower() for record in caplog.records)


def test_markdown_skill_contract_invalid_yaml_skips_skill_without_raising(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from ecs_agent.skills.discovery import discover_skills

    caplog.set_level(logging.WARNING)

    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        skill_dir = base / ".claude" / "skills" / "broken-yaml"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\nname: broken\ndescription: [unterminated\n---\nBody"
        )

        skills = discover_skills([base])

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
    from ecs_agent.skills.discovery import discover_skills

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

        skills = discover_skills([base])

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

    skill = Skill(skill_file)
    resolved = skill.resolve_supporting_path("data.json")

    assert resolved == (tmp_path / "data.json").resolve()
    assert str(resolved).endswith("data.json")
    assert str(resolved).startswith(str(tmp_path.resolve()))


def test_markdown_skill_resolve_supporting_path_blocks_traversal(
    tmp_path: Path,
) -> None:
    skill_file = tmp_path / "SKILL.md"
    skill_file.write_text("---\nname: traversal\ndescription: traversal\n---\nPrompt")

    skill = Skill(skill_file)

    with pytest.raises(ValueError, match="Path traversal"):
        skill.resolve_supporting_path("../etc/passwd")


def test_markdown_skill_resolve_supporting_path_blocks_absolute_path(
    tmp_path: Path,
) -> None:
    skill_file = tmp_path / "SKILL.md"
    skill_file.write_text("---\nname: absolute\ndescription: absolute\n---\nPrompt")

    skill = Skill(skill_file)

    with pytest.raises(ValueError):
        skill.resolve_supporting_path("/etc/passwd")


def test_markdown_skill_resolve_supporting_path_allows_nested_relative(
    tmp_path: Path,
) -> None:
    skill_file = tmp_path / "SKILL.md"
    skill_file.write_text("---\nname: nested\ndescription: nested\n---\nPrompt")

    skill = Skill(skill_file)
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

        skill = Skill(skill_path)
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

        skill = Skill(skill_path)

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

        skill = Skill(skill_path)

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

        skill = Skill(skill_path)

        assert skill.is_dynamic_injection_safe(content) is False


# ---------------------------------------------------------------------------
# render_skill_content / render_with_arguments — substitution engine tests
# ---------------------------------------------------------------------------


def test_render_skill_content_substitutes_arguments_placeholder(
    tmp_path: Path,
) -> None:
    """$ARGUMENTS is replaced by the entire arguments string."""
    from ecs_agent.skills.skill import render_skill_content

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
    from ecs_agent.skills.skill import render_skill_content

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
    from ecs_agent.skills.skill import render_skill_content

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
    from ecs_agent.skills.skill import render_skill_content

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
    from ecs_agent.skills.skill import render_skill_content

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
    from ecs_agent.skills.skill import render_skill_content

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
    from ecs_agent.skills.skill import render_skill_content

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
    from ecs_agent.skills.skill import render_skill_content

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
    from ecs_agent.skills.skill import render_skill_content

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
    from ecs_agent.skills.skill import render_skill_content

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

    skill = Skill(skill_file)
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

    skill = Skill(skill_file)
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


# ---------------------------------------------------------------------------
# Advanced frontmatter field tests (Task 10)
# ---------------------------------------------------------------------------


def _install_skill_from_content(content: str, skill_name: str) -> object:
    """Helper: write SKILL.md, install via DiscoveryManager, return SkillMetadata."""
    import asyncio
    import tempfile
    from ecs_agent.components.definitions import SkillComponent
    from ecs_agent.skills.discovery import DiscoveryManager
    from ecs_agent.skills.manager import SkillManager

    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        skill_dir = base / ".claude" / "skills" / skill_name
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(content)

        world = World()
        entity = world.create_entity()

        asyncio.run(
            DiscoveryManager().auto_discover_and_install(
                world,
                entity,
                SkillManager(),
                directories=[base],
            )
        )

        skill_comp = world.get_component(entity, SkillComponent)
        assert skill_comp is not None, "SkillComponent missing after install"
        return skill_comp.skills[skill_name]


def test_markdown_skill_argument_hint_parses_to_metadata() -> None:
    """argument-hint in frontmatter maps to argument_hint field on SkillMetadata."""
    content = (
        "---\n"
        "name: arg-hint-skill\n"
        "description: Tests argument-hint\n"
        "argument-hint: '<query>'\n"
        "---\n"
        "Skill body"
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        skill_dir = base / ".claude" / "skills" / "arg-hint-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(content)

        import asyncio
        from ecs_agent.components.definitions import SkillComponent
        from ecs_agent.skills.discovery import DiscoveryManager
        from ecs_agent.skills.manager import SkillManager

        world = World()
        entity = world.create_entity()
        asyncio.run(
            DiscoveryManager().auto_discover_and_install(
                world, entity, SkillManager(), directories=[base]
            )
        )

        skill_comp = world.get_component(entity, SkillComponent)
        assert skill_comp is not None
        metadata = skill_comp.skills["arg-hint-skill"]
        assert metadata.argument_hint == "<query>"


def test_markdown_skill_allowed_tools_list_parses_to_metadata() -> None:
    """allowed-tools list in frontmatter maps to allowed_tools field on SkillMetadata."""
    content = (
        "---\n"
        "name: allowed-tools-skill\n"
        "description: Tests allowed-tools\n"
        "allowed-tools:\n"
        "  - bash_tool\n"
        "  - read_file\n"
        "---\n"
        "Skill body"
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        skill_dir = base / ".claude" / "skills" / "allowed-tools-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(content)

        import asyncio
        from ecs_agent.components.definitions import SkillComponent
        from ecs_agent.skills.discovery import DiscoveryManager
        from ecs_agent.skills.manager import SkillManager

        world = World()
        entity = world.create_entity()
        asyncio.run(
            DiscoveryManager().auto_discover_and_install(
                world, entity, SkillManager(), directories=[base]
            )
        )

        skill_comp = world.get_component(entity, SkillComponent)
        assert skill_comp is not None
        metadata = skill_comp.skills["allowed-tools-skill"]
        assert metadata.allowed_tools == ["bash_tool", "read_file"]


def test_markdown_skill_context_field_parses_to_metadata() -> None:
    """context string in frontmatter maps to context field on SkillMetadata."""
    content = (
        "---\n"
        "name: context-skill\n"
        "description: Tests context\n"
        "context: 'code-review'\n"
        "---\n"
        "Skill body"
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        skill_dir = base / ".claude" / "skills" / "context-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(content)

        import asyncio
        from ecs_agent.components.definitions import SkillComponent
        from ecs_agent.skills.discovery import DiscoveryManager
        from ecs_agent.skills.manager import SkillManager

        world = World()
        entity = world.create_entity()
        asyncio.run(
            DiscoveryManager().auto_discover_and_install(
                world, entity, SkillManager(), directories=[base]
            )
        )

        skill_comp = world.get_component(entity, SkillComponent)
        assert skill_comp is not None
        metadata = skill_comp.skills["context-skill"]
        assert metadata.context == "code-review"


def test_markdown_skill_agent_field_parses_to_metadata() -> None:
    """agent string in frontmatter maps to agent field on SkillMetadata."""
    content = (
        "---\n"
        "name: agent-skill\n"
        "description: Tests agent\n"
        "agent: 'reviewer-agent'\n"
        "---\n"
        "Skill body"
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        skill_dir = base / ".claude" / "skills" / "agent-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(content)

        import asyncio
        from ecs_agent.components.definitions import SkillComponent
        from ecs_agent.skills.discovery import DiscoveryManager
        from ecs_agent.skills.manager import SkillManager

        world = World()
        entity = world.create_entity()
        asyncio.run(
            DiscoveryManager().auto_discover_and_install(
                world, entity, SkillManager(), directories=[base]
            )
        )

        skill_comp = world.get_component(entity, SkillComponent)
        assert skill_comp is not None
        metadata = skill_comp.skills["agent-skill"]
        assert metadata.agent == "reviewer-agent"


def test_markdown_skill_model_field_parses_to_metadata() -> None:
    """model string in frontmatter maps to model field on SkillMetadata."""
    content = (
        "---\n"
        "name: model-skill\n"
        "description: Tests model\n"
        "model: 'claude-3-5-sonnet'\n"
        "---\n"
        "Skill body"
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        skill_dir = base / ".claude" / "skills" / "model-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(content)

        import asyncio
        from ecs_agent.components.definitions import SkillComponent
        from ecs_agent.skills.discovery import DiscoveryManager
        from ecs_agent.skills.manager import SkillManager

        world = World()
        entity = world.create_entity()
        asyncio.run(
            DiscoveryManager().auto_discover_and_install(
                world, entity, SkillManager(), directories=[base]
            )
        )

        skill_comp = world.get_component(entity, SkillComponent)
        assert skill_comp is not None
        metadata = skill_comp.skills["model-skill"]
        assert metadata.model == "claude-3-5-sonnet"


def test_markdown_skill_hooks_dict_parses_to_metadata() -> None:
    """hooks dict in frontmatter maps to hooks field on SkillMetadata."""
    content = (
        "---\n"
        "name: hooks-skill\n"
        "description: Tests hooks\n"
        "hooks:\n"
        "  pre_run: 'setup.sh'\n"
        "  post_run: 'cleanup.sh'\n"
        "---\n"
        "Skill body"
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        skill_dir = base / ".claude" / "skills" / "hooks-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(content)

        import asyncio
        from ecs_agent.components.definitions import SkillComponent
        from ecs_agent.skills.discovery import DiscoveryManager
        from ecs_agent.skills.manager import SkillManager

        world = World()
        entity = world.create_entity()
        asyncio.run(
            DiscoveryManager().auto_discover_and_install(
                world, entity, SkillManager(), directories=[base]
            )
        )

        skill_comp = world.get_component(entity, SkillComponent)
        assert skill_comp is not None
        metadata = skill_comp.skills["hooks-skill"]
        assert metadata.hooks == {"pre_run": "setup.sh", "post_run": "cleanup.sh"}


def test_markdown_skill_allowed_tools_invalid_type_produces_empty_list() -> None:
    """allowed-tools: 'not-a-list' (invalid type) → allowed_tools == [] safe default."""
    content = (
        "---\n"
        "name: bad-tools-skill\n"
        "description: Tests invalid allowed-tools\n"
        "allowed-tools: 'not-a-list'\n"
        "---\n"
        "Skill body"
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        skill_dir = base / ".claude" / "skills" / "bad-tools-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(content)

        import asyncio
        from ecs_agent.components.definitions import SkillComponent
        from ecs_agent.skills.discovery import DiscoveryManager
        from ecs_agent.skills.manager import SkillManager

        world = World()
        entity = world.create_entity()
        asyncio.run(
            DiscoveryManager().auto_discover_and_install(
                world, entity, SkillManager(), directories=[base]
            )
        )

        skill_comp = world.get_component(entity, SkillComponent)
        assert skill_comp is not None
        metadata = skill_comp.skills["bad-tools-skill"]
        assert metadata.allowed_tools == []


def test_markdown_skill_hooks_invalid_type_produces_empty_dict() -> None:
    """hooks: 'not-a-dict' (invalid type) → hooks == {} safe default."""
    content = (
        "---\n"
        "name: bad-hooks-skill\n"
        "description: Tests invalid hooks\n"
        "hooks: 'not-a-dict'\n"
        "---\n"
        "Skill body"
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        skill_dir = base / ".claude" / "skills" / "bad-hooks-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(content)

        import asyncio
        from ecs_agent.components.definitions import SkillComponent
        from ecs_agent.skills.discovery import DiscoveryManager
        from ecs_agent.skills.manager import SkillManager

        world = World()
        entity = world.create_entity()
        asyncio.run(
            DiscoveryManager().auto_discover_and_install(
                world, entity, SkillManager(), directories=[base]
            )
        )

        skill_comp = world.get_component(entity, SkillComponent)
        assert skill_comp is not None
        metadata = skill_comp.skills["bad-hooks-skill"]
        assert metadata.hooks == {}


def test_markdown_skill_user_invocable_alias_maps_to_user_invocable_field() -> None:
    """user-invocable: false in frontmatter maps to user_invocable=False on SkillMetadata."""
    content = (
        "---\n"
        "name: no-user-invoke\n"
        "description: Tests user-invocable alias\n"
        "user-invocable: false\n"
        "---\n"
        "Skill body"
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        skill_dir = base / ".claude" / "skills" / "no-user-invoke"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(content)

        import asyncio
        from ecs_agent.components.definitions import SkillComponent
        from ecs_agent.skills.discovery import DiscoveryManager
        from ecs_agent.skills.manager import SkillManager

        world = World()
        entity = world.create_entity()
        asyncio.run(
            DiscoveryManager().auto_discover_and_install(
                world, entity, SkillManager(), directories=[base]
            )
        )

        skill_comp = world.get_component(entity, SkillComponent)
        assert skill_comp is not None
        metadata = skill_comp.skills["no-user-invoke"]
        assert metadata.user_invocable is False


# ---------------------------------------------------------------------------
# Naming Contract Tests (skills-refactor-v2 hard switch)
# These tests MUST FAIL until implementation tasks rename the symbols.
# ---------------------------------------------------------------------------


def test_markdown_skill_name_replaced_by_skill_in_exports() -> None:
    """After hard switch: `Skill` from ecs_agent.skills.skill must be importable.

    Previously the class was called MarkdownSkill. After the rename it is called Skill.
    """
    import importlib

    module = importlib.import_module("ecs_agent.skills.skill")

    # After rename: the module must expose `Skill` (not `MarkdownSkill`)
    assert hasattr(module, "Skill"), (
        "Naming contract violated: ecs_agent.skills.skill must export `Skill`. "
        "renamed to Skill — the class formerly known as MarkdownSkill is now called Skill."
    )


def test_markdown_skill_class_name_is_skill_after_hard_switch() -> None:
    """After hard switch: class within skill.py must be named `Skill`, not `MarkdownSkill`.

    Hard switch: NO alias, the class must be renamed. Code using MarkdownSkill must migrate.
    """
    import importlib

    module = importlib.import_module("ecs_agent.skills.skill")

    # After rename: MarkdownSkill class must not exist (hard switch, no alias)
    assert not hasattr(module, "MarkdownSkill"), (
        "Naming contract violated: `MarkdownSkill` class still exists in skill.py. "
        "Hard switch complete — the class is now named `Skill`. "
        "Migration: replace all usages of `MarkdownSkill` with `Skill`. "
        "renamed to Skill — no compatibility alias is provided."
    )


def test_script_skill_protocol_is_importable_from_script_skill_module() -> None:
    """After hard switch: script_skill.py must export `ScriptSkill`, not `Skill`.

    The current `Skill` Protocol in protocol.py is renamed to `ScriptSkill` and moved to script_skill.py.
    """
    import importlib

    module = importlib.import_module("ecs_agent.skills.script_skill")

    # After rename: ScriptSkill must exist in script_skill.py
    assert hasattr(module, "ScriptSkill"), (
        "Naming contract violated: ecs_agent.skills.script_skill must export `ScriptSkill`. "
        "renamed to ScriptSkill — the Protocol class formerly named `Skill` is now `ScriptSkill`. "
        "Migration: replace all `from ecs_agent.skills.script_skill import Skill` with `ScriptSkill`."
    )


def test_protocol_module_skill_name_is_script_skill_no_legacy_alias() -> None:
    """After hard switch: `Skill` must NOT exist in script_skill.py (hard switch, no alias).

    The protocol class is renamed to ScriptSkill in script_skill.py. No compatibility alias is provided.
    """
    import importlib

    module = importlib.import_module("ecs_agent.skills.script_skill")

    # Hard switch: the name `Skill` must not exist in script_skill.py anymore
    assert not hasattr(module, "Skill"), (
        "Naming contract violated: `Skill` Protocol class still exists in script_skill.py. "
        "Hard switch complete — the Protocol is now named `ScriptSkill`. "
        "Migration: update all isinstance(x, Skill) checks to isinstance(x, ScriptSkill). "
        "renamed to ScriptSkill — no backward-compatible Skill alias is allowed."
    )


# ── resolve_path_references tests ──


def test_resolve_path_references_rewrites_relative_at_path() -> None:
    """@../../../file.md is resolved relative to SKILL.md and re-expressed relative to workspace."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir) / "project"
        skill_dir = workspace / "a" / "b" / "c"
        skill_dir.mkdir(parents=True)
        skill_path = skill_dir / "SKILL.md"
        skill_path.write_text("""---
name: nav
description: nav skill
---
Write to @../../../output.md please.
""")

        skill = Skill(skill_path)
        skill.resolve_path_references(str(workspace))
        prompt = skill.system_prompt()

        assert "output.md" in prompt
        # The @../../.. prefix should be gone
        assert "@../../../output.md" not in prompt


def test_resolve_path_references_preserves_non_relative_at() -> None:
    """@ signs not followed by ./ or ../ are left untouched."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir) / "project"
        skill_dir = workspace / "skills" / "s1"
        skill_dir.mkdir(parents=True)
        skill_path = skill_dir / "SKILL.md"
        skill_path.write_text("""---
name: plain
description: test
---
Contact @admin for help.
""")

        skill = Skill(skill_path)
        skill.resolve_path_references(str(workspace))
        prompt = skill.system_prompt()

        assert "@admin" in prompt


def test_resolve_path_references_leaves_outside_workspace_intact() -> None:
    """Paths resolving outside workspace are left as-is."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir) / "project"
        skill_dir = workspace / "deep" / "nested"
        skill_dir.mkdir(parents=True)
        skill_path = skill_dir / "SKILL.md"
        # ../../../escape.md resolves to tmpdir/escape.md — outside workspace
        skill_path.write_text("""---
name: escape
description: test
---
See @../../../escape.md for details.
""")

        skill = Skill(skill_path)
        skill.resolve_path_references(str(workspace))
        prompt = skill.system_prompt()

        # Original @-reference preserved since it escapes workspace
        assert "@../../../escape.md" in prompt


def test_resolve_path_references_multiple_paths() -> None:
    """Multiple @ references in one body are all resolved."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir) / "project"
        skill_dir = workspace / "a" / "b" / "c"
        skill_dir.mkdir(parents=True)
        skill_path = skill_dir / "SKILL.md"
        skill_path.write_text("""---
name: multi
description: multi paths
---
Write draft to @../../../ui/draft.md and prompts to @../../../ui/prompts.md.
""")

        skill = Skill(skill_path)
        skill.resolve_path_references(str(workspace))
        prompt = skill.system_prompt()

        assert "ui/draft.md" in prompt
        assert "ui/prompts.md" in prompt
        assert "@../" not in prompt


def test_resolve_path_references_dot_slash() -> None:
    """@./local.md is resolved relative to skill dir."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir) / "project"
        skill_dir = workspace / "skills" / "s1"
        skill_dir.mkdir(parents=True)
        skill_path = skill_dir / "SKILL.md"
        skill_path.write_text("""---
name: dotslash
description: local ref
---
Read @./notes.txt for context.
""")

        skill = Skill(skill_path)
        skill.resolve_path_references(str(workspace))
        prompt = skill.system_prompt()

        assert "skills/s1/notes.txt" in prompt
        assert "@./notes.txt" not in prompt
