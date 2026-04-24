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
async def test_hashline_basic_operations(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "sample.txt"
    target.write_text("alpha\nbeta\ngamma", encoding="utf-8")

    hashed = await read_file("sample.txt", str(workspace))
    beta_hash = _line_hash(hashed, 2)

    result = await edit_file(
        "sample.txt",
        "replace",
        f"2#{beta_hash}",
        content="BETA",
        workspace_root=str(workspace),
    )

    assert result == "Applied edit to sample.txt"
    assert target.read_text(encoding="utf-8") == "alpha\nBETA\ngamma"


@pytest.mark.asyncio
async def test_hashline_append_and_prepend(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "fruits.txt"
    target.write_text("apple\nbanana\ncherry", encoding="utf-8")

    hashed = await read_file("fruits.txt", str(workspace))
    banana_hash = _line_hash(hashed, 2)
    apple_hash = _line_hash(hashed, 1)

    await edit_file(
        "fruits.txt",
        "append",
        f"2#{banana_hash}",
        content="grape",
        workspace_root=str(workspace),
    )
    await edit_file(
        "fruits.txt",
        "prepend",
        f"1#{apple_hash}",
        content="start",
        workspace_root=str(workspace),
    )

    assert target.read_text(encoding="utf-8") == "start\napple\nbanana\ngrape\ncherry"


@pytest.mark.asyncio
async def test_hashline_range_replace_and_delete(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "log.txt"
    target.write_text(
        "=== Log Start ===\nINFO: started\nWARN: slow query\nERROR: timeout\nINFO: recovered\n=== Log End ===",
        encoding="utf-8",
    )

    hashed = await read_file("log.txt", str(workspace))
    warn_hash = _line_hash(hashed, 3)
    error_hash = _line_hash(hashed, 4)

    await edit_file(
        "log.txt",
        "replace",
        f"3#{warn_hash}",
        end=f"4#{error_hash}",
        content="RESOLVED: issues cleared",
        workspace_root=str(workspace),
    )

    content = target.read_text(encoding="utf-8")
    assert "WARN: slow query" not in content
    assert "ERROR: timeout" not in content
    assert "RESOLVED: issues cleared" in content


@pytest.mark.asyncio
async def test_hashline_batch_two_replacements_single_call(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "batch.txt"
    target.write_text("red\ngreen\nblue\nyellow", encoding="utf-8")

    hashed = await read_file("batch.txt", str(workspace))
    red_hash = _line_hash(hashed, 1)
    blue_hash = _line_hash(hashed, 3)

    await edit_file(
        "batch.txt",
        "replace",
        f"1#{red_hash}",
        content="crimson",
        workspace_root=str(workspace),
    )
    result = await edit_file(
        "batch.txt",
        "replace",
        f"3#{blue_hash}",
        content="navy",
        workspace_root=str(workspace),
    )

    assert result == "Applied edit to batch.txt"
    assert target.read_text(encoding="utf-8") == "crimson\ngreen\nnavy\nyellow"


@pytest.mark.asyncio
async def test_hashline_duplicate_line_targeting_by_anchor(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "dupes.txt"
    target.write_text("item\nitem\nitem\nitem", encoding="utf-8")

    hashed = await read_file("dupes.txt", str(workspace))
    line3_hash = _line_hash(hashed, 3)

    await edit_file(
        "dupes.txt",
        "replace",
        f"3#{line3_hash}",
        content="CHANGED",
        workspace_root=str(workspace),
    )

    lines = target.read_text(encoding="utf-8").splitlines()
    assert lines.count("CHANGED") == 1
    assert lines.count("item") == 3


@pytest.mark.asyncio
async def test_hashline_line_expansion_one_to_three(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "expand.txt"
    target.write_text("header\nTODO: implement\nfooter", encoding="utf-8")

    hashed = await read_file("expand.txt", str(workspace))
    todo_hash = _line_hash(hashed, 2)

    await edit_file(
        "expand.txt",
        "replace",
        f"2#{todo_hash}",
        end=f"2#{todo_hash}",
        content="step 1: init\nstep 2: process\nstep 3: cleanup",
        workspace_root=str(workspace),
    )

    assert target.read_text(encoding="utf-8") == (
        "header\nstep 1: init\nstep 2: process\nstep 3: cleanup\nfooter"
    )


def test_hashline_compute_line_hash_compatible_with_reference_style() -> None:
    value = compute_line_hash(11, "line11: UPDATED-MIDDLE")
    assert len(value) == 4
    assert value == value.lower()
