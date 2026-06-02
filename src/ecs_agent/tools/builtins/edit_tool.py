"""Hash-anchored edit tool core logic."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

from ecs_agent.logging import get_logger
from ecs_agent.tools.builtins.file_snapshot import (
    compute_content_digest,
    compute_snapshot_line_hash,
    current_file_snapshot_state,
    FileSnapshotState,
    normalize_snapshot_line,
)
from ecs_agent.tools.discovery import tool

logger = get_logger(__name__)


def normalize_line(content: str) -> str:
    return normalize_snapshot_line(content)


def compute_line_hash(line_number: int, content: str) -> str:
    return compute_snapshot_line_hash(line_number, content)


def format_file_with_hashes(file_content: str) -> str:
    lines = file_content.splitlines()
    rendered_lines = []
    for line_number, content in enumerate(lines, start=1):
        line_hash = compute_line_hash(line_number, content)
        rendered_lines.append(f"{line_number}#{line_hash}|{content}")
    return "\n".join(rendered_lines)


@dataclass(slots=True)
class EditOperation:
    op: Literal["replace", "append", "prepend"]
    pos: str
    end: str | None = None
    lines: list[str] | None = None


def parse_edit_instruction(pos: str) -> tuple[int, str]:
    line_number_str, separator, hash_id = pos.partition("#")
    if separator == "" or hash_id == "":
        raise ValueError(f"Invalid LINE#ID format: {pos}")

    line_number = _parse_positive_line_number(
        line_number_str,
        invalid_message=f"Invalid line number in LINE#ID format: {pos}",
        range_message=f"Line numbers are 1-based in LINE#ID format: {pos}",
    )
    return line_number, hash_id


def validate_hash(line_number: int, content: str, expected_hash: str) -> bool:
    return compute_line_hash(line_number, content) == expected_hash


def apply_edits(original_content: str, edits: list[EditOperation]) -> str:
    lines = original_content.splitlines()
    sorted_edits = sorted(
        edits,
        key=lambda edit: parse_edit_instruction(edit.pos)[0],
        reverse=True,
    )

    for edit in sorted_edits:
        start_line, start_hash = parse_edit_instruction(edit.pos)
        if start_line > len(lines):
            raise ValueError(
                f"Line {start_line} out of range (file has {len(lines)} lines)"
            )

        start_content = lines[start_line - 1]
        if not validate_hash(start_line, start_content, start_hash):
            actual_hash = compute_line_hash(start_line, start_content)
            raise ValueError(
                f"Hash mismatch at line {start_line}: expected {start_hash}, got {actual_hash}"
            )

        edit_lines = edit.lines or []
        if edit.op == "replace":
            if edit.end is None:
                lines[start_line - 1 : start_line] = edit_lines
                continue

            end_line, end_hash = parse_edit_instruction(edit.end)
            if end_line < start_line:
                raise ValueError(
                    f"Invalid replace range: end line {end_line} before start line {start_line}"
                )
            if end_line > len(lines):
                raise ValueError(
                    f"Line {end_line} out of range (file has {len(lines)} lines)"
                )

            end_content = lines[end_line - 1]
            if not validate_hash(end_line, end_content, end_hash):
                actual_hash = compute_line_hash(end_line, end_content)
                raise ValueError(
                    f"Hash mismatch at line {end_line}: expected {end_hash}, got {actual_hash}"
                )
            lines[start_line - 1 : end_line] = edit_lines
            continue

        if edit.op == "append":
            lines[start_line:start_line] = edit_lines
            continue

        if edit.op == "prepend":
            lines[start_line - 1 : start_line - 1] = edit_lines
            continue

        raise ValueError(f"Unsupported edit operation: {edit.op}")

    return "\n".join(lines)


@tool(description="Apply a snapshot-protected text edit to a workspace file.")
async def edit_file(
    file_path: Annotated[str, "Workspace-relative path to the target file."],
    op: Annotated[
        Literal["replace", "append", "prepend"],
        "Edit operation: 'replace' overwrites lines, 'append' inserts after, 'prepend' inserts before.",
    ],
    pos: Annotated[
        int | str,
        "Start 1-based file line number to edit. Accepts an integer or numeric string. Call read_file first so the framework can validate the line internally.",
    ],
    end: Annotated[
        int | str | None,
        "End 1-based file line number for range replace. Accepts an integer or numeric string. Omit for single-line operations.",
    ] = None,
    content: Annotated[
        str,
        "New content to insert or replace. Use \n to separate multiple lines.",
    ] = "",
    workspace_root: str = "",
) -> str:
    from ecs_agent.tools.builtins.file_tools import _validate_path

    target = _validate_path(file_path, workspace_root)
    original = target.read_text(encoding="utf-8")
    snapshot_state = current_file_snapshot_state()
    snapshot = snapshot_state.latest_for(target)
    if snapshot is None:
        raise ValueError("Use read_file before editing an existing file")

    original_digest = compute_content_digest(original)
    if original_digest != snapshot.content_digest:
        raise ValueError("File changed since it was last read; call read_file again")

    anchored_pos = _resolve_edit_position(snapshot_state, target, pos, original_digest)
    anchored_end = (
        _resolve_edit_position(snapshot_state, target, end, original_digest)
        if end is not None
        else None
    )
    new_lines = content.split("\n") if content else []
    edit = EditOperation(op=op, pos=anchored_pos, end=anchored_end, lines=new_lines)
    updated = apply_edits(original, [edit])
    target.write_text(updated, encoding="utf-8")
    snapshot_state.record_read(file_path, target, updated, offset=1, limit=0)
    logger.info("edit_file", file_path=file_path, op=op, pos=pos)
    return f"Applied edit to {file_path}"


def _resolve_edit_position(
    snapshot_state: FileSnapshotState,
    target: Path,
    position: int | str,
    content_digest: str,
) -> str:
    line_number = _parse_public_line_number(position)
    anchor = snapshot_state.find_anchor(target, line_number, content_digest)
    return f"{anchor.line_number}#{anchor.hash_id}"


def _parse_public_line_number(position: int | str) -> int:
    if isinstance(position, bool):
        raise ValueError(f"edit_file pos/end must be a 1-based line number: {position}")
    if isinstance(position, int):
        return _parse_positive_line_number(
            position,
            invalid_message=f"Invalid line number: {position}",
            range_message=f"Line numbers are 1-based: {position}",
        )

    if not position.isdecimal():
        raise ValueError(f"edit_file pos/end must be a 1-based line number: {position}")
    line_number = _parse_positive_line_number(
        position,
        invalid_message=f"Invalid line number: {position}",
        range_message=f"Line numbers are 1-based: {position}",
    )
    return line_number


def _parse_positive_line_number(
    value: int | str,
    *,
    invalid_message: str,
    range_message: str,
) -> int:
    try:
        line_number = int(value)
    except ValueError as exc:
        raise ValueError(invalid_message) from exc

    if line_number < 1:
        raise ValueError(range_message)
    return line_number
