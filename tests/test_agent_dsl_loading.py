"""Tests for agent DSL loading and prompt file resolution."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from ecs_agent.dsl.prompt_resolver import resolve_prompt_file


class TestPromptFileResolver:
    """Test prompt file reference resolution with security checks."""

    def test_resolve_prompt_file_returns_original_when_not_file_reference(self) -> None:
        """Non-file reference prompts are returned unchanged."""
        source_dir = Path("/tmp")
        prompt = "You are a helpful assistant."
        result = resolve_prompt_file(prompt, source_dir)
        assert result == prompt

    def test_resolve_prompt_file_returns_original_when_partial_file_syntax(
        self,
    ) -> None:
        """Prompts that don't match exact {file:...} pattern are returned unchanged."""
        source_dir = Path("/tmp")

        # Missing closing brace
        prompt = "{file:prompts/system.txt"
        result = resolve_prompt_file(prompt, source_dir)
        assert result == prompt

        # No file: prefix
        prompt = "{prompts/system.txt}"
        result = resolve_prompt_file(prompt, source_dir)
        assert result == prompt

    def test_resolve_prompt_file_loads_valid_relative_path(self) -> None:
        """Valid relative path reference loads file content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            prompt_file = base / "system.txt"
            expected_content = "You are a helpful AI assistant."
            prompt_file.write_text(expected_content, encoding="utf-8")

            result = resolve_prompt_file("{file:system.txt}", base)
            assert result == expected_content

    def test_resolve_prompt_file_loads_nested_relative_path(self) -> None:
        """Nested relative paths load correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            prompts_dir = base / "prompts"
            prompts_dir.mkdir()
            prompt_file = prompts_dir / "agent.md"
            expected_content = "# Agent System Prompt\n\nBe concise."
            prompt_file.write_text(expected_content, encoding="utf-8")

            result = resolve_prompt_file("{file:prompts/agent.md}", base)
            assert result == expected_content

    def test_resolve_prompt_file_rejects_absolute_path(self) -> None:
        """Absolute paths are rejected with ValueError."""
        source_dir = Path("/tmp")

        with pytest.raises(ValueError, match="Absolute paths not allowed"):
            resolve_prompt_file("{file:/etc/passwd}", source_dir)

    def test_resolve_prompt_file_rejects_path_traversal(self) -> None:
        """Path traversal outside source directory is rejected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            subdir = base / "subdir"
            subdir.mkdir()

            # Try to traverse outside base
            with pytest.raises(ValueError, match="Path traversal.*not allowed"):
                resolve_prompt_file("{file:../../etc/passwd}", subdir)

    def test_resolve_prompt_file_rejects_missing_file(self) -> None:
        """Missing file raises FileNotFoundError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)

            with pytest.raises(FileNotFoundError):
                resolve_prompt_file("{file:missing.txt}", base)

    def test_resolve_prompt_file_rejects_directory_reference(self) -> None:
        """Directory references are rejected with ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            subdir = base / "prompts"
            subdir.mkdir()

            with pytest.raises(ValueError, match="not a file"):
                resolve_prompt_file("{file:prompts}", base)

    def test_resolve_prompt_file_handles_utf8_content(self) -> None:
        """UTF-8 content with special characters loads correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            prompt_file = base / "unicode.txt"
            expected_content = "你好世界 🌍 Здравствуй мир"
            prompt_file.write_text(expected_content, encoding="utf-8")

            result = resolve_prompt_file("{file:unicode.txt}", base)
            assert result == expected_content

    def test_resolve_prompt_file_fails_on_non_utf8_file(self) -> None:
        """Non-UTF-8 files raise ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            binary_file = base / "binary.dat"
            # Write binary data that's not valid UTF-8
            binary_file.write_bytes(b"\x80\x81\x82\x83")

            with pytest.raises(UnicodeDecodeError):
                resolve_prompt_file("{file:binary.dat}", base)

    def test_resolve_prompt_file_handles_whitespace_in_pattern(self) -> None:
        """File reference pattern handles leading/trailing whitespace."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            prompt_file = base / "test.txt"
            expected_content = "Test content"
            prompt_file.write_text(expected_content, encoding="utf-8")

            # Leading/trailing whitespace around pattern should work
            result = resolve_prompt_file("  {file:test.txt}  ", base)
            assert result == f"  {expected_content}  "

    def test_resolve_prompt_file_normalizes_path_components(self) -> None:
        """Path with . and .. components is normalized correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            subdir = base / "sub"
            subdir.mkdir()
            prompt_file = base / "prompt.txt"
            expected_content = "Normalized path content"
            prompt_file.write_text(expected_content, encoding="utf-8")

            # Reference from subdir: sub/../prompt.txt should resolve to prompt.txt
            with pytest.raises(ValueError, match="Path traversal.*not allowed"):
                resolve_prompt_file("{file:sub/../prompt.txt}", base)

    def test_resolve_prompt_file_allows_symlink_within_source_dir(self) -> None:
        """Symlinks that resolve within source_dir are allowed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            real_file = base / "real.txt"
            expected_content = "Real file content"
            real_file.write_text(expected_content, encoding="utf-8")

            # Create symlink in same directory
            link_file = base / "link.txt"
            link_file.symlink_to(real_file)

            result = resolve_prompt_file("{file:link.txt}", base)
            assert result == expected_content

    def test_resolve_prompt_file_rejects_symlink_outside_source_dir(self) -> None:
        """Symlinks that resolve outside source_dir are rejected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)

            # Create a file outside base
            outside_dir = Path(tmpdir).parent
            outside_file = outside_dir / "outside.txt"
            outside_file.write_text("Outside content", encoding="utf-8")

            # Create symlink inside base pointing outside
            link_file = base / "link.txt"
            link_file.symlink_to(outside_file)

            # Should be rejected because resolved path is outside source_dir
            with pytest.raises(ValueError, match="Path escapes source directory"):
                resolve_prompt_file("{file:link.txt}", base)
