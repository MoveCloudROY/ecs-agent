"""Tests for MarkdownSkill parser."""

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
from ecs_agent.types import EntityId


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
