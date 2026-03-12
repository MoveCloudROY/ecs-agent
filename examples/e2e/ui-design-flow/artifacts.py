"""Path and artifact utilities for UI Design Flow example.

Provides safe path operations and output directory management for the
ui-design-flow example. All paths are confined to examples/e2e/ui-design-flow/

Functions:
- ensure_output_layout: Create ui-design/ directory structure
- safe_write: Write to output directory with traversal protection
- get_output_path: Get safe path within ui-design/ directory
"""

from __future__ import annotations

from pathlib import Path


def get_base_dir() -> Path:
    """Get base directory for this example.

    Returns:
        Path to examples/e2e/ui-design-flow/
    """
    return Path(__file__).parent


def get_output_dir() -> Path:
    """Get output directory for artifacts.

    Returns:
        Path to examples/e2e/ui-design-flow/ui-design/
    """
    return get_base_dir() / "ui-design"


def ensure_output_layout() -> None:
    """Create ui-design/ output directory structure.

    Creates:
    - examples/e2e/ui-design-flow/ui-design/ (main output directory)
    """
    output_dir = get_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)


def get_output_path(relative_path: str) -> Path:
    """Get safe path within output directory with traversal protection.

    Args:
        relative_path: Relative path (e.g. 'draft.md')

    Returns:
        Absolute path within ui-design/ directory

    Raises:
        ValueError: If path tries to traverse outside ui-design/
    """
    output_dir = get_output_dir()
    # Resolve to absolute path to detect traversal attempts
    target = (output_dir / relative_path).resolve()
    base = output_dir.resolve()

    if not str(target).startswith(str(base)):
        raise ValueError(
            f"Path traversal attempt blocked: {relative_path} would escape {base}"
        )

    return target


def safe_write(relative_path: str, content: str) -> None:
    """Write content to output directory with traversal protection.

    Args:
        relative_path: Relative path within ui-design/ directory
        content: Content to write

    Raises:
        ValueError: If path tries to traverse outside ui-design/
    """
    target = get_output_path(relative_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
