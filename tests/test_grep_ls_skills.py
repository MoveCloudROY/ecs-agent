"""Tests for grep and ls example skills."""

from __future__ import annotations

import pytest

from ecs_agent.skills.protocol import Skill


class TestGrepSkillProtocol:
    """Test GrepSkill implements Skill protocol."""

    async def test_grep_skill_satisfies_protocol(self) -> None:
        """GrepSkill should satisfy Skill protocol."""
        from examples.skills.grep_skill import GrepSkill

        skill = GrepSkill()
        assert isinstance(skill, Skill)
        assert skill.name == "grep"
        assert skill.description is not None
        assert len(skill.description) > 0

    async def test_grep_skill_has_tools(self) -> None:
        """GrepSkill should provide grep tool."""
        from examples.skills.grep_skill import GrepSkill

        skill = GrepSkill()
        tools = skill.tools()
        assert "grep" in tools
        schema, handler = tools["grep"]
        assert schema.name == "grep"
        assert callable(handler)


class TestGrepSkillFunctionality:
    """Test GrepSkill tool functionality."""

    async def test_grep_finds_pattern_in_file(self, tmp_path) -> None:
        """Grep tool should find pattern in file."""
        from examples.skills.grep_skill import GrepSkill

        # Create temp file with content
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello world\nfoo bar\nhello again\n")

        skill = GrepSkill()
        tools = skill.tools()
        _, grep_handler = tools["grep"]

        result = await grep_handler(pattern="hello", path=str(test_file))
        assert "hello world" in result
        assert "hello again" in result
        assert "foo bar" not in result

    async def test_grep_nonexistent_file_returns_error(self, tmp_path) -> None:
        """Grep tool should handle non-existent file."""
        from examples.skills.grep_skill import GrepSkill

        skill = GrepSkill()
        tools = skill.tools()
        _, grep_handler = tools["grep"]

        result = await grep_handler(
            pattern="test", path=str(tmp_path / "nonexistent.txt")
        )
        # Should return error message, not raise
        assert (
            "error" in result.lower()
            or "not found" in result.lower()
            or len(result) == 0
        )


class TestLsSkillProtocol:
    """Test LsSkill implements Skill protocol."""

    async def test_ls_skill_satisfies_protocol(self) -> None:
        """LsSkill should satisfy Skill protocol."""
        from examples.skills.ls_skill import LsSkill

        skill = LsSkill()
        assert isinstance(skill, Skill)
        assert skill.name == "ls"
        assert skill.description is not None
        assert len(skill.description) > 0

    async def test_ls_skill_has_tools(self) -> None:
        """LsSkill should provide ls tool."""
        from examples.skills.ls_skill import LsSkill

        skill = LsSkill()
        tools = skill.tools()
        assert "ls" in tools
        schema, handler = tools["ls"]
        assert schema.name == "ls"
        assert callable(handler)


class TestLsSkillFunctionality:
    """Test LsSkill tool functionality."""

    async def test_ls_lists_directory(self, tmp_path) -> None:
        """Ls tool should list directory contents."""
        from examples.skills.ls_skill import LsSkill

        # Create temp files
        (tmp_path / "a.txt").write_text("content a")
        (tmp_path / "b.py").write_text("content b")
        (tmp_path / "subdir").mkdir()

        skill = LsSkill()
        tools = skill.tools()
        _, ls_handler = tools["ls"]

        result = await ls_handler(path=str(tmp_path))
        assert "a.txt" in result
        assert "b.py" in result
        assert "subdir" in result

    async def test_ls_nonexistent_dir_returns_error(self, tmp_path) -> None:
        """Ls tool should handle non-existent directory."""
        from examples.skills.ls_skill import LsSkill

        skill = LsSkill()
        tools = skill.tools()
        _, ls_handler = tools["ls"]

        result = await ls_handler(path=str(tmp_path / "nonexistent"))
        # Should return error message, not raise
        assert (
            "error" in result.lower()
            or "not found" in result.lower()
            or len(result) == 0
        )


class TestSkillDiscoveryIntegration:
    """Test both skills loadable via SkillDiscovery."""

    @pytest.mark.skip(reason="SkillDiscovery not yet implemented (Task 2)")
    async def test_discover_grep_and_ls_from_examples(self, tmp_path) -> None:
        """SkillDiscovery should find both grep and ls skills."""
        from ecs_agent.skills.discovery import SkillDiscovery

        discovery = SkillDiscovery(skill_paths=["examples/skills"])
        skills = discovery.discover()

        assert len(skills) >= 2
        skill_names = {skill.name for skill in skills}
        assert "grep" in skill_names
        assert "ls" in skill_names
