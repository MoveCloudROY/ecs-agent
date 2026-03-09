"""Secure prompt file reference resolver for Agent DSL."""

from __future__ import annotations

import re
from pathlib import Path

from ecs_agent.logging import get_logger

logger = get_logger(__name__)


def resolve_prompt_file(prompt_spec: str, source_dir: Path) -> str:
    """Resolve {file:relative/path} syntax to UTF-8 file content.

    Security:
    - Rejects absolute paths (paths starting with / or containing drive letters)
    - Rejects path traversal (.. components)
    - Uses Path.resolve() for normalization
    - Validates resolved path stays within source_dir

    Args:
        prompt_spec: Prompt string potentially containing {file:path} reference
        source_dir: Base directory for resolving relative paths

    Returns:
        Resolved prompt content with file reference replaced by actual content

    Raises:
        ValueError: Path traversal, absolute path, or multiple {file:} detected
        FileNotFoundError: Referenced file does not exist
    """
    # Pattern: {file:path/to/file.txt}
    pattern = r"\{file:(.*?)\}"
    matches = list(re.finditer(pattern, prompt_spec))

    if not matches:
        # No file reference, return as-is
        return prompt_spec

    if len(matches) > 1:
        raise ValueError(
            f"Multiple file references not allowed (found {len(matches)}). "
            f"Prompt must contain at most one {{file:...}} reference."
        )

    match = matches[0]
    relative_path_str = match.group(1).strip()

    if not relative_path_str:
        raise ValueError("Empty file path in {file:} reference")

    # Security check 1: Reject absolute paths
    relative_path = Path(relative_path_str)
    if relative_path.is_absolute():
        raise ValueError(
            f"Absolute paths not allowed in {{file:}} reference: {relative_path_str}"
        )

    # Security check 2: Reject explicit path traversal
    if ".." in relative_path.parts:
        raise ValueError(
            f"Path traversal (..) not allowed in {{file:}} reference: {relative_path_str}"
        )

    # Resolve to absolute path
    resolved_path = (source_dir / relative_path).resolve()

    # Security check 3: Ensure resolved path is within source_dir
    try:
        resolved_path.relative_to(source_dir.resolve())
    except ValueError as exc:
        raise ValueError(
            f"Path escapes source directory: {relative_path_str} "
            f"(resolved to {resolved_path}, source_dir is {source_dir})"
        ) from exc

    # Read file content
    if not resolved_path.exists():
        raise FileNotFoundError(
            f"Prompt file not found: {relative_path_str} (resolved to {resolved_path})"
        )

    if not resolved_path.is_file():
        raise ValueError(
            f"Prompt path is not a file: {relative_path_str} "
            f"(resolved to {resolved_path})"
        )

    content = resolved_path.read_text(encoding="utf-8")

    logger.info(
        "prompt_file_resolved",
        relative_path=relative_path_str,
        resolved_path=str(resolved_path),
        content_length=len(content),
    )

    # Replace {file:path} with actual content
    return prompt_spec[: match.start()] + content + prompt_spec[match.end() :]
