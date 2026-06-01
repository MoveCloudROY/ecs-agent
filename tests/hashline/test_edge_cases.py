from __future__ import annotations

from pathlib import Path

import pytest

from ecs_agent.tools.builtins.edit_tool import compute_line_hash, edit_file
from ecs_agent.tools.builtins.file_tools import read_file


@pytest.mark.asyncio
async def test_snapshot_rejects_stale_anchor_after_external_change(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "target.py"
    target.write_text("alpha = 1\nbeta = 2\ngamma = 3\n", encoding="utf-8")

    initial = await read_file("target.py", str(workspace))
    assert initial == "alpha = 1\nbeta = 2\ngamma = 3"

    target.write_text("alpha = 1\nbeta = 200\ngamma = 3\n", encoding="utf-8")

    with pytest.raises(ValueError, match="changed since it was last read"):
        await edit_file(
            "target.py",
            "replace",
            "2",
            content="beta = 20",
            workspace_root=str(workspace),
        )


@pytest.mark.asyncio
async def test_snapshot_line_shift_invalidation_after_external_insert(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "shift.txt"
    target.write_text("line1\nline2\nline3\n", encoding="utf-8")

    initial = await read_file("shift.txt", str(workspace))
    assert initial == "line1\nline2\nline3"

    target.write_text("new-line0\nline1\nline2\nline3\n", encoding="utf-8")

    with pytest.raises(ValueError, match="changed since it was last read"):
        await edit_file(
            "shift.txt",
            "replace",
            "2",
            content="updated-line2",
            workspace_root=str(workspace),
        )


@pytest.mark.asyncio
async def test_snapshot_crlf_file_edit_normalizes_and_updates(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "windows.txt"
    target.write_text("alpha\r\nbeta\r\ngamma", encoding="utf-8")

    assert await read_file("windows.txt", str(workspace)) == "alpha\nbeta\ngamma"

    await edit_file(
        "windows.txt",
        "replace",
        "1",
        content="ALPHA",
        workspace_root=str(workspace),
    )

    assert target.read_text(encoding="utf-8") == "ALPHA\nbeta\ngamma"


@pytest.mark.asyncio
async def test_snapshot_preserves_no_trailing_newline_behavior(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "no-trailing-newline.txt"
    target.write_text("first\nsecond\nthird", encoding="utf-8")

    await read_file("no-trailing-newline.txt", str(workspace))

    await edit_file(
        "no-trailing-newline.txt",
        "replace",
        "2",
        content="SECOND",
        workspace_root=str(workspace),
    )

    content = target.read_text(encoding="utf-8")
    assert content == "first\nSECOND\nthird"
    assert not content.endswith("\n")


@pytest.mark.asyncio
async def test_snapshot_literal_backslash_n_content(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "literal-backslash-n.txt"
    target.write_text("placeholder\ntail", encoding="utf-8")

    await read_file("literal-backslash-n.txt", str(workspace))

    await edit_file(
        "literal-backslash-n.txt",
        "replace",
        "1",
        content="line1\\nline2 (literal backslash-n)",
        workspace_root=str(workspace),
    )

    lines = target.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert lines[0] == r"line1\nline2 (literal backslash-n)"


@pytest.mark.asyncio
async def test_snapshot_trailing_spaces_edit(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "trailing-whitespace.txt"
    target.write_text("start\ntext   \nend", encoding="utf-8")

    await read_file("trailing-whitespace.txt", str(workspace))

    await edit_file(
        "trailing-whitespace.txt",
        "replace",
        "2",
        content="new_text   ",
        workspace_root=str(workspace),
    )

    lines = target.read_text(encoding="utf-8").splitlines()
    assert lines[1] == "new_text   "


def test_hashline_hash_is_reference_compatible_for_blank_and_space_lines() -> None:
    assert len(compute_line_hash(2, "")) == 4
    assert compute_line_hash(2, "    ") == compute_line_hash(2, "")
