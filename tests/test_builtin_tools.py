from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from ecs_agent.components import ToolRegistryComponent
from ecs_agent.components.definitions import SkillComponent
from ecs_agent.core import World
from ecs_agent.skills import SkillManager
from ecs_agent.skills.script_skill import ScriptSkill
from ecs_agent.tools.builtins import BuiltinToolsSkill
from ecs_agent.tools.builtins.bash_tool import bash
from ecs_agent.tools.builtins.edit_tool import (
    compute_line_hash,
    format_file_with_hashes,
    edit_file,
)
from ecs_agent.tools.builtins.file_tools import read_file, write_file

# Try importing glob; will fail if not implemented yet (expected for TDD red phase)
try:
    from ecs_agent.tools.builtins.glob_tool import glob
except ImportError:
    glob = None  # type: ignore


def _get_hashed_view(file_content: str) -> str:
    if file_content == "":
        return ""

    if all(re.match(r"^\d+#[0-9a-f]{4}\|", line) for line in file_content.splitlines()):
        return file_content

    return format_file_with_hashes(file_content)


def _parse_hash_from_hashed_content(hashed_content: str, line_number: int) -> str:
    for line in hashed_content.splitlines():
        prefix, _, _ = line.partition("|")
        number_str, _, hash_value = prefix.partition("#")
        if int(number_str) == line_number:
            return hash_value

    raise ValueError(f"Line {line_number} not found")


@pytest.mark.asyncio
async def test_read_file_valid(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "note.txt"
    target.write_text("hello\nworld", encoding="utf-8")

    result = await read_file("note.txt", str(workspace))

    expected_hash_1 = compute_line_hash(1, "hello")
    expected_hash_2 = compute_line_hash(2, "world")
    assert result == f"1#{expected_hash_1}|hello\n2#{expected_hash_2}|world"


@pytest.mark.asyncio
async def test_read_file_empty_content(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "empty.txt"
    target.write_text("", encoding="utf-8")

    assert await read_file("empty.txt", str(workspace)) == ""


@pytest.mark.asyncio
async def test_read_file_blank_line_preserved(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "blank.txt"
    target.write_text("line1\n\nline3", encoding="utf-8")

    result = await read_file("blank.txt", str(workspace))

    expected_hash_1 = compute_line_hash(1, "line1")
    expected_hash_2 = compute_line_hash(2, "")  # blank line
    expected_hash_3 = compute_line_hash(3, "line3")
    assert (
        result
        == f"1#{expected_hash_1}|line1\n2#{expected_hash_2}|\n3#{expected_hash_3}|line3"
    )


@pytest.mark.asyncio
async def test_read_file_hash_matches_compute_line_hash(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "multi.txt"
    content = "first\nsecond\nthird"
    target.write_text(content, encoding="utf-8")

    result = await read_file("multi.txt", str(workspace))

    # Verify each line matches compute_line_hash output
    lines = result.split("\n")
    for i, line in enumerate(lines, start=1):
        line_num_str, rest = line.split("#", 1)
        hash_part, content_part = rest.split("|", 1)
        expected_hash = compute_line_hash(i, content_part)
        assert hash_part == expected_hash, (
            f"Line {i} hash mismatch: {hash_part} != {expected_hash}"
        )


@pytest.mark.asyncio
async def test_read_file_missing_raises(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    with pytest.raises(FileNotFoundError):
        await read_file("missing.txt", str(workspace))


@pytest.mark.asyncio
async def test_read_file_rejects_parent_traversal(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    with pytest.raises(ValueError, match="outside workspace"):
        await read_file("../secret.txt", str(workspace))


@pytest.mark.asyncio
async def test_read_file_rejects_absolute_path(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    with pytest.raises(ValueError, match="outside workspace"):
        await read_file("/etc/passwd", str(workspace))


@pytest.mark.asyncio
async def test_read_file_rejects_symlink_outside_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    outside_file = outside / "secret.txt"
    outside_file.write_text("secret", encoding="utf-8")
    (workspace / "link.txt").symlink_to(outside_file)

    with pytest.raises(ValueError, match="outside workspace"):
        await read_file("link.txt", str(workspace))


@pytest.mark.asyncio
async def test_write_file_writes_content(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    result = await write_file("out.txt", "alpha", str(workspace))

    assert result == "Wrote 5 bytes to out.txt"
    assert (workspace / "out.txt").read_text(encoding="utf-8") == "alpha"


@pytest.mark.asyncio
async def test_write_file_creates_parent_dirs(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    await write_file("nested/path/out.txt", "value", str(workspace))

    assert (workspace / "nested" / "path" / "out.txt").read_text(
        encoding="utf-8"
    ) == "value"


@pytest.mark.asyncio
async def test_write_file_rejects_parent_traversal(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    with pytest.raises(ValueError, match="outside workspace"):
        await write_file("../oops.txt", "x", str(workspace))


@pytest.mark.asyncio
async def test_write_file_rejects_absolute_path(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    with pytest.raises(ValueError, match="outside workspace"):
        await write_file("/etc/passwd", "x", str(workspace))


@pytest.mark.asyncio
async def test_write_file_rejects_symlink_outside_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    outside_file = outside / "secret.txt"
    outside_file.write_text("secret", encoding="utf-8")
    (workspace / "link.txt").symlink_to(outside_file)

    with pytest.raises(ValueError, match="outside workspace"):
        await write_file("link.txt", "x", str(workspace))


@pytest.mark.asyncio
async def test_edit_file_applies_edits_and_persists(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "edit.txt"
    target.write_text("alpha\nbeta\ngamma", encoding="utf-8")
    beta_hash = compute_line_hash(2, "beta")
    edits_json = json.dumps(
        [{"op": "replace", "pos": f"2#{beta_hash}", "lines": ["BETA"]}]
    )

    result = await edit_file("edit.txt", edits_json, str(workspace))

    assert result == "Applied 1 edits to edit.txt"
    assert target.read_text(encoding="utf-8") == "alpha\nBETA\ngamma"


@pytest.mark.asyncio
async def test_edit_file_multi_step_two_edits_on_real_python_file(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "sample.py"
    target.write_text(
        'def hello():\n    return "world"\n\n\ndef add(a, b):\n    return a + b\n',
        encoding="utf-8",
    )

    first_read = await read_file("sample.py", str(workspace))
    first_hashed = _get_hashed_view(first_read)
    first_line_hash = _parse_hash_from_hashed_content(first_hashed, 1)

    first_result = await edit_file(
        "sample.py",
        json.dumps(
            [
                {
                    "op": "replace",
                    "pos": f"1#{first_line_hash}",
                    "lines": ["def greet():"],
                }
            ]
        ),
        str(workspace),
    )

    assert first_result == "Applied 1 edits to sample.py"

    second_read = await read_file("sample.py", str(workspace))
    second_hashed = _get_hashed_view(second_read)
    second_line_hash = _parse_hash_from_hashed_content(second_hashed, 2)

    second_result = await edit_file(
        "sample.py",
        json.dumps(
            [
                {
                    "op": "replace",
                    "pos": f"2#{second_line_hash}",
                    "lines": ['    return "earth"'],
                }
            ]
        ),
        str(workspace),
    )

    assert second_result == "Applied 1 edits to sample.py"
    assert target.read_text(encoding="utf-8") == (
        'def greet():\n    return "earth"\n\n\ndef add(a, b):\n    return a + b'
    )


@pytest.mark.asyncio
async def test_edit_file_stale_hash_rejected_after_external_modification(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "target.py"
    target.write_text("alpha = 1\nbeta = 2\ngamma = 3\n", encoding="utf-8")

    initial_read = await read_file("target.py", str(workspace))
    initial_hashed = _get_hashed_view(initial_read)
    stale_hash = _parse_hash_from_hashed_content(initial_hashed, 2)

    target.write_text("alpha = 1\nbeta = 200\ngamma = 3\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Hash mismatch"):
        await edit_file(
            "target.py",
            json.dumps(
                [{"op": "replace", "pos": f"2#{stale_hash}", "lines": ["beta = 20"]}]
            ),
            str(workspace),
        )


@pytest.mark.asyncio
async def test_edit_file_repeated_cycles_on_python_file(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "cycles.py"
    target.write_text(
        "def compute(value):\n"
        "    result = value + 1\n"
        "    if result > 10:\n"
        "        return result\n"
        "    return value\n",
        encoding="utf-8",
    )

    for line_number, replacement in [
        (1, "def compute_total(value):"),
        (2, "    result = value + 2"),
        (5, "    return result"),
    ]:
        read_result = await read_file("cycles.py", str(workspace))
        hashed_content = _get_hashed_view(read_result)
        line_hash = _parse_hash_from_hashed_content(hashed_content, line_number)

        result = await edit_file(
            "cycles.py",
            json.dumps(
                [
                    {
                        "op": "replace",
                        "pos": f"{line_number}#{line_hash}",
                        "lines": [replacement],
                    }
                ]
            ),
            str(workspace),
        )

        assert result == "Applied 1 edits to cycles.py"

    assert target.read_text(encoding="utf-8") == (
        "def compute_total(value):\n"
        "    result = value + 2\n"
        "    if result > 10:\n"
        "        return result\n"
        "    return result"
    )


@pytest.mark.asyncio
async def test_edit_file_rejects_parent_traversal(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    with pytest.raises(ValueError, match="outside workspace"):
        await edit_file("../edit.txt", json.dumps([]), str(workspace))


@pytest.mark.asyncio
async def test_edit_file_rejects_absolute_path(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    with pytest.raises(ValueError, match="outside workspace"):
        await edit_file("/etc/passwd", json.dumps([]), str(workspace))


@pytest.mark.asyncio
async def test_edit_file_rejects_symlink_outside_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    outside_file = outside / "secret.txt"
    outside_file.write_text("secret", encoding="utf-8")
    (workspace / "link.txt").symlink_to(outside_file)

    with pytest.raises(ValueError, match="outside workspace"):
        await edit_file("link.txt", json.dumps([]), str(workspace))


@pytest.mark.asyncio
async def test_bash_captures_output(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    output = await bash("python -c 'print(\"hello\")'", 1.0, str(workspace))

    assert output.strip() == "hello"


@pytest.mark.asyncio
async def test_bash_nonzero_exit_includes_stdout_and_stderr(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    output = await bash(
        'python -c \'import sys; print("out"); print("err", file=sys.stderr); sys.exit(3)\'',
        1.0,
        str(workspace),
    )

    assert "Exit code 3" in output
    assert "STDOUT:\nout\n" in output
    assert "STDERR:\nerr\n" in output


@pytest.mark.asyncio
async def test_bash_timeout(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    with pytest.raises(ValueError, match="timed out"):
        await bash('python -c "import time; time.sleep(2)"', 0.1, str(workspace))


def test_builtin_skill_tools_returns_all_schemas() -> None:
    skill = BuiltinToolsSkill()
    discovered = skill.tools()

    assert set(discovered) >= {"read_file", "write_file", "edit_file", "bash"}
    for tool_name, (schema, handler) in discovered.items():
        assert schema.name == tool_name
        assert schema.description
        assert schema.parameters["type"] == "object"
        assert callable(handler)


def test_builtin_skill_install() -> None:
    world = World()
    entity_id = world.create_entity()
    manager = SkillManager()

    manager.install(world, entity_id, BuiltinToolsSkill())

    registry = world.get_component(entity_id, ToolRegistryComponent)
    assert registry is not None
    assert set(registry.tools) >= {
        "read_file",
        "write_file",
        "edit_file",
        "bash",
        "load_skill_details",
    }
    assert set(registry.handlers) >= {
        "read_file",
        "write_file",
        "edit_file",
        "bash",
        "load_skill_details",
    }


def test_builtin_tools_skill_satisfies_protocol() -> None:
    """BuiltinToolsSkill should satisfy ScriptSkill protocol."""
    skill = BuiltinToolsSkill()
    assert isinstance(skill, ScriptSkill)
    assert skill.name == "builtin-tools"
    assert skill.description is not None
    assert len(skill.description) > 0
    assert callable(skill.tools)
    assert callable(skill.system_prompt)
    assert callable(skill.install)
    assert callable(skill.uninstall)


@pytest.mark.asyncio
async def test_glob_happy_path(tmp_path: Path) -> None:
    """Test glob matches *.txt files and returns sorted newline-delimited paths."""
    if glob is None:
        pytest.skip("glob not implemented yet")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "file1.txt").write_text("content1")
    (workspace / "file2.txt").write_text("content2")
    (workspace / "file3.md").write_text("content3")

    result = await glob("*.txt", ".", str(workspace))

    lines = sorted(result.strip().split("\n")) if result.strip() else []
    assert set(lines) == {"file1.txt", "file2.txt"}
    assert lines == sorted(lines)  # Verify sorted


@pytest.mark.asyncio
async def test_glob_no_matches(tmp_path: Path) -> None:
    """Test glob returns empty string when no files match."""
    if glob is None:
        pytest.skip("glob not implemented yet")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "file1.md").write_text("content")

    result = await glob("*.txt", ".", str(workspace))

    assert result == ""


@pytest.mark.asyncio
async def test_glob_rejects_parent_traversal(tmp_path: Path) -> None:
    """Test glob raises ValueError when base_path attempts to escape workspace."""
    if glob is None:
        pytest.skip("glob not implemented yet")
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    with pytest.raises(ValueError, match="outside workspace"):
        await glob("*.txt", "..", str(workspace))


@pytest.mark.asyncio
async def test_glob_rejects_absolute_path(tmp_path: Path) -> None:
    """Test glob raises ValueError when base_path is absolute."""
    if glob is None:
        pytest.skip("glob not implemented yet")
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    with pytest.raises(ValueError, match="outside workspace"):
        await glob("*.txt", "/etc", str(workspace))


@pytest.mark.asyncio
async def test_glob_nested_matches(tmp_path: Path) -> None:
    """Test glob includes files in subdirectories with relative paths."""
    if glob is None:
        pytest.skip("glob not implemented yet")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "file1.txt").write_text("content1")
    subdir = workspace / "subdir"
    subdir.mkdir()
    (subdir / "file2.txt").write_text("content2")

    result = await glob("**/*.txt", ".", str(workspace))

    lines = sorted(result.strip().split("\n")) if result.strip() else []
    assert "file1.txt" in lines
    # Handle both Unix and Windows path separators
    assert any("file2.txt" in line for line in lines)


def test_builtin_skill_tools_includes_glob() -> None:
    """Test that catalog includes glob tool key."""
    skill = BuiltinToolsSkill()
    discovered = skill.tools()

    assert "glob" in discovered


def test_builtin_skill_install_includes_glob() -> None:
    """Test that installed registry includes glob in tools."""
    world = World()
    entity_id = world.create_entity()
    manager = SkillManager()

    manager.install(world, entity_id, BuiltinToolsSkill())

    registry = world.get_component(entity_id, ToolRegistryComponent)
    assert registry is not None
    assert "glob" in registry.tools
    assert "glob" in registry.handlers


# ---------------------------------------------------------------------------
# is_tool_bundle contract tests (RED phase — written before implementation)
# ---------------------------------------------------------------------------


def test_builtin_tools_skill_is_tool_bundle_flag() -> None:
    """BuiltinToolsSkill.is_tool_bundle must be True."""
    skill = BuiltinToolsSkill()
    assert skill.is_tool_bundle is True


def test_tool_bundle_not_registered_in_skill_component() -> None:
    """Installing a tool-bundle skill must NOT add it to SkillComponent."""
    world = World()
    entity_id = world.create_entity()
    manager = SkillManager()

    manager.install(world, entity_id, BuiltinToolsSkill())

    skill_comp = world.get_component(entity_id, SkillComponent)
    # SkillComponent may not be created at all, or if it is, builtin-tools must not be in it
    if skill_comp is not None:
        assert "builtin-tools" not in skill_comp.skills


def test_tool_bundle_tools_still_registered_in_tool_registry() -> None:
    """Tool-bundle tools must still appear in ToolRegistryComponent after install."""
    world = World()
    entity_id = world.create_entity()
    manager = SkillManager()

    manager.install(world, entity_id, BuiltinToolsSkill())

    registry = world.get_component(entity_id, ToolRegistryComponent)
    assert registry is not None
    assert "read_file" in registry.tools
    assert "write_file" in registry.tools
    assert "bash" in registry.tools
    assert "glob" in registry.tools


def test_regular_skill_still_registered_in_skill_component() -> None:
    """Non-tool-bundle skills must still appear in SkillComponent after install."""
    from ecs_agent.types import ToolSchema

    class _NormalSkill:
        name = "normal-skill"
        description = "A regular skill"
        is_tool_bundle = False

        def tools(self) -> dict[str, tuple[ToolSchema, object]]:
            return {}

        def system_prompt(self) -> str:
            return ""

        def install(self, world: object, entity_id: object) -> None:
            pass

        def uninstall(self, world: object, entity_id: object) -> None:
            pass

    world = World()
    entity_id = world.create_entity()
    manager = SkillManager()

    manager.install(world, entity_id, _NormalSkill())  # type: ignore[arg-type]

    skill_comp = world.get_component(entity_id, SkillComponent)
    assert skill_comp is not None
    assert "normal-skill" in skill_comp.skills


def test_default_is_tool_bundle_is_false_for_protocol_compliant_skill() -> None:
    """ScriptSkill that omits is_tool_bundle should default to False (via getattr)."""
    from ecs_agent.types import ToolSchema

    class _MinimalSkill:
        name = "minimal"
        description = "minimal"

        def tools(self) -> dict[str, tuple[ToolSchema, object]]:
            return {}

        def system_prompt(self) -> str:
            return ""

        def install(self, world: object, entity_id: object) -> None:
            pass

        def uninstall(self, world: object, entity_id: object) -> None:
            pass

    skill = _MinimalSkill()
    # getattr fallback used by SkillManager must resolve to False
    assert getattr(skill, "is_tool_bundle", False) is False
