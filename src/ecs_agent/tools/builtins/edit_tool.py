"""Hash-anchored edit tool core logic."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Literal

from ecs_agent.logging import get_logger

logger = get_logger(__name__)


def normalize_line(content: str) -> str:
    return content.rstrip()


def compute_line_hash(line_number: int, content: str) -> str:
    normalized = normalize_line(content)
    payload = f"{line_number}:{normalized}"
    md5_digest = hashlib.md5(payload.encode("utf-8")).hexdigest()
    return md5_digest[:4]


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

    try:
        line_number = int(line_number_str)
    except ValueError as exc:
        raise ValueError(f"Invalid line number in LINE#ID format: {pos}") from exc

    if line_number < 1:
        raise ValueError(f"Line numbers are 1-based in LINE#ID format: {pos}")

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
                replacement = edit_lines[0] if edit_lines else ""
                lines[start_line - 1] = replacement
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
