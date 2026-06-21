"""Shared SKILL.md frontmatter parsing."""

from __future__ import annotations

from pathlib import Path


def parse_skill_frontmatter(skill_path: Path) -> tuple[str, bytes]:
    """Parse YAML frontmatter from a SKILL.md file.

    Reads the file in binary mode. Only the frontmatter section needs to be
    valid UTF-8; the body is returned as raw bytes for lazy decoding.

    Args:
        skill_path: Path to the SKILL.md file.

    Returns:
        Tuple of (frontmatter_text, body_bytes). If no frontmatter is found,
        frontmatter_text is empty and body_bytes is the entire file content.
    """
    raw = skill_path.read_bytes()
    lines = [line.rstrip(b"\r") for line in raw.split(b"\n")]

    first_non_empty_idx: int | None = None
    for idx, line in enumerate(lines):
        if line.strip():
            first_non_empty_idx = idx
            break

    if first_non_empty_idx is None or lines[first_non_empty_idx] != b"---":
        return "", raw.strip()

    closing_idx: int | None = None
    for idx in range(first_non_empty_idx + 1, len(lines)):
        if lines[idx] == b"---":
            closing_idx = idx
            break

    if closing_idx is None:
        return "", raw.strip()

    frontmatter_bytes = b"\n".join(lines[first_non_empty_idx + 1 : closing_idx])
    try:
        frontmatter_text = frontmatter_bytes.decode("utf-8")
    except UnicodeDecodeError:
        frontmatter_text = frontmatter_bytes.decode("utf-8", errors="replace")

    body_bytes = b"\n".join(lines[closing_idx + 1 :]).strip()
    return frontmatter_text, body_bytes
