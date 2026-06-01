from __future__ import annotations

from pathlib import Path

import pytest

from ecs_agent.tools.builtins.edit_tool import compute_line_hash, edit_file
from ecs_agent.tools.builtins.file_tools import read_file


@pytest.mark.asyncio
async def test_snapshot_basic_replacement(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "sample.txt"
    target.write_text("alpha\nbeta\ngamma", encoding="utf-8")

    assert await read_file("sample.txt", str(workspace)) == "alpha\nbeta\ngamma"

    result = await edit_file(
        "sample.txt",
        "replace",
        "2",
        content="BETA",
        workspace_root=str(workspace),
    )

    assert result == "Applied edit to sample.txt"
    assert target.read_text(encoding="utf-8") == "alpha\nBETA\ngamma"


@pytest.mark.asyncio
async def test_snapshot_append_and_prepend_via_text_replacement(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "fruits.txt"
    target.write_text("apple\nbanana\ncherry", encoding="utf-8")

    await read_file("fruits.txt", str(workspace))

    await edit_file(
        "fruits.txt",
        "append",
        "2",
        content="grape",
        workspace_root=str(workspace),
    )
    await edit_file(
        "fruits.txt",
        "prepend",
        "1",
        content="start",
        workspace_root=str(workspace),
    )

    assert target.read_text(encoding="utf-8") == "start\napple\nbanana\ngrape\ncherry"


@pytest.mark.asyncio
async def test_snapshot_range_replace_and_delete(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "log.txt"
    target.write_text(
        "=== Log Start ===\nINFO: started\nWARN: slow query\nERROR: timeout\nINFO: recovered\n=== Log End ===",
        encoding="utf-8",
    )

    await read_file("log.txt", str(workspace))

    await edit_file(
        "log.txt",
        "replace",
        "3",
        end="4",
        content="RESOLVED: issues cleared",
        workspace_root=str(workspace),
    )

    content = target.read_text(encoding="utf-8")
    assert "WARN: slow query" not in content
    assert "ERROR: timeout" not in content
    assert "RESOLVED: issues cleared" in content


@pytest.mark.asyncio
async def test_snapshot_two_replacements_across_calls(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "batch.txt"
    target.write_text("red\ngreen\nblue\nyellow", encoding="utf-8")

    await read_file("batch.txt", str(workspace))

    await edit_file(
        "batch.txt",
        "replace",
        "1",
        content="crimson",
        workspace_root=str(workspace),
    )
    result = await edit_file(
        "batch.txt",
        "replace",
        "3",
        content="navy",
        workspace_root=str(workspace),
    )

    assert result == "Applied edit to batch.txt"
    assert target.read_text(encoding="utf-8") == "crimson\ngreen\nnavy\nyellow"


@pytest.mark.asyncio
async def test_snapshot_edits_duplicate_content_by_line_number(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "dupes.txt"
    target.write_text("item\nitem\nitem\nitem", encoding="utf-8")

    await read_file("dupes.txt", str(workspace))

    await edit_file(
        "dupes.txt",
        "replace",
        "3",
        content="CHANGED",
        workspace_root=str(workspace),
    )

    lines = target.read_text(encoding="utf-8").splitlines()
    assert lines == ["item", "item", "CHANGED", "item"]


@pytest.mark.asyncio
async def test_snapshot_line_expansion_one_to_three(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "expand.txt"
    target.write_text("header\nTODO: implement\nfooter", encoding="utf-8")

    await read_file("expand.txt", str(workspace))

    await edit_file(
        "expand.txt",
        "replace",
        "2",
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
