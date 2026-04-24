from __future__ import annotations

from pathlib import Path

import pytest

from ecs_agent.tools.builtins.edit_tool import compute_line_hash, edit_file
from ecs_agent.tools.builtins.file_tools import read_file


def _line_hash(hashed_text: str, line_number: int) -> str:
    for line in hashed_text.splitlines():
        prefix, _, _ = line.partition("|")
        number_str, _, hash_value = prefix.partition("#")
        if int(number_str) == line_number:
            return hash_value
    raise ValueError(f"Line {line_number} not found")


@pytest.mark.asyncio
async def test_hashline_rejects_stale_anchor_after_external_change(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "target.py"
    target.write_text("alpha = 1\nbeta = 2\ngamma = 3\n", encoding="utf-8")

    initial = await read_file("target.py", str(workspace))
    stale_hash = _line_hash(initial, 2)

    target.write_text("alpha = 1\nbeta = 200\ngamma = 3\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Hash mismatch"):
        await edit_file(
            "target.py",
            "replace",
            f"2#{stale_hash}",
            content="beta = 20",
            workspace_root=str(workspace),
        )


@pytest.mark.asyncio
async def test_hashline_line_shift_invalidation_after_external_insert(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "shift.txt"
    target.write_text("line1\nline2\nline3\n", encoding="utf-8")

    initial = await read_file("shift.txt", str(workspace))
    stale_line2_hash = _line_hash(initial, 2)

    target.write_text("new-line0\nline1\nline2\nline3\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Hash mismatch"):
        await edit_file(
            "shift.txt",
            "replace",
            f"2#{stale_line2_hash}",
            content="updated-line2",
            workspace_root=str(workspace),
        )


@pytest.mark.asyncio
async def test_hashline_crlf_file_edit_normalizes_and_updates(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "windows.txt"
    target.write_text("alpha\r\nbeta\r\ngamma", encoding="utf-8")

    hashed = await read_file("windows.txt", str(workspace))
    alpha_hash = _line_hash(hashed, 1)

    await edit_file(
        "windows.txt",
        "replace",
        f"1#{alpha_hash}",
        content="ALPHA",
        workspace_root=str(workspace),
    )

    assert target.read_text(encoding="utf-8") == "ALPHA\nbeta\ngamma"


@pytest.mark.asyncio
async def test_hashline_preserves_no_trailing_newline_behavior(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "no-trailing-newline.txt"
    target.write_text("first\nsecond\nthird", encoding="utf-8")

    hashed = await read_file("no-trailing-newline.txt", str(workspace))
    second_hash = _line_hash(hashed, 2)

    await edit_file(
        "no-trailing-newline.txt",
        "replace",
        f"2#{second_hash}",
        content="SECOND",
        workspace_root=str(workspace),
    )

    content = target.read_text(encoding="utf-8")
    assert content == "first\nSECOND\nthird"
    assert not content.endswith("\n")


@pytest.mark.asyncio
async def test_hashline_literal_backslash_n_content(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "literal-backslash-n.txt"
    target.write_text("placeholder\ntail", encoding="utf-8")

    hashed = await read_file("literal-backslash-n.txt", str(workspace))
    placeholder_hash = _line_hash(hashed, 1)

    await edit_file(
        "literal-backslash-n.txt",
        "replace",
        f"1#{placeholder_hash}",
        content="line1\\nline2 (literal backslash-n)",
        workspace_root=str(workspace),
    )

    lines = target.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert lines[0] == r"line1\nline2 (literal backslash-n)"


@pytest.mark.asyncio
async def test_hashline_trailing_spaces_edit(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "trailing-whitespace.txt"
    target.write_text("start\ntext   \nend", encoding="utf-8")

    hashed = await read_file("trailing-whitespace.txt", str(workspace))
    line2_hash = _line_hash(hashed, 2)

    await edit_file(
        "trailing-whitespace.txt",
        "replace",
        f"2#{line2_hash}",
        content="new_text   ",
        workspace_root=str(workspace),
    )

    lines = target.read_text(encoding="utf-8").splitlines()
    assert lines[1] == "new_text   "


def test_hashline_hash_is_reference_compatible_for_blank_and_space_lines() -> None:
    assert len(compute_line_hash(2, "")) == 4
    assert compute_line_hash(2, "    ") == compute_line_hash(2, "")
