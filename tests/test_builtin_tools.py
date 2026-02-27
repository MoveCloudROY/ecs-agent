from __future__ import annotations

import json
from pathlib import Path

import pytest

from ecs_agent.components import ToolRegistryComponent
from ecs_agent.core import World
from ecs_agent.skills import SkillManager
from ecs_agent.tools.builtins import BuiltinToolsSkill
from ecs_agent.tools.builtins.bash_tool import bash
from ecs_agent.tools.builtins.edit_tool import compute_line_hash, edit_file
from ecs_agent.tools.builtins.file_tools import read_file, write_file


@pytest.mark.asyncio
async def test_read_file_valid(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "note.txt"
    target.write_text("hello\nworld", encoding="utf-8")

    result = await read_file("note.txt", str(workspace))

    assert result == "hello\nworld"


@pytest.mark.asyncio
async def test_read_file_empty_content(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "empty.txt"
    target.write_text("", encoding="utf-8")

    assert await read_file("empty.txt", str(workspace)) == ""


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

    assert set(discovered) == {"read_file", "write_file", "edit_file", "bash"}
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
    assert set(registry.tools) == {"read_file", "write_file", "edit_file", "bash"}
    assert set(registry.handlers) == {"read_file", "write_file", "edit_file", "bash"}
