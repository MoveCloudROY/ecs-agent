"""Path and artifact utilities for UI Design Flow example.

Provides safe path operations and output directory management for the
ui-design-flow example. All paths are confined to examples/e2e/ui-design-flow/

Policy for artifact content:
- draft.md: Overwrite with each ensure_output_layout() call (deterministic reset)
- nano-banana-prompts.md: Overwrite with each ensure_output_layout() call (deterministic reset)

Functions:
- ensure_output_layout: Create ui-design/ directory structure and return output paths
- safe_read: Read UTF-8 content from output directory with traversal protection
- safe_write: Write to output directory with traversal protection
- get_output_path: Get safe path within ui-design/ directory
"""

from __future__ import annotations

from dataclasses import dataclass
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


@dataclass
class OutputLayout:
    """Paths to output artifacts.

    Attributes:
        draft_path: Path to examples/e2e/ui-design-flow/ui-design/draft.md
        nano_prompt_path: Path to examples/e2e/ui-design-flow/ui-design/nano-banana-prompts.md
    """

    draft_path: Path
    nano_prompt_path: Path


def ensure_output_layout() -> OutputLayout:
    """Create ui-design/ output directory structure and return artifact paths.

    Creates:
    - examples/e2e/ui-design-flow/ui-design/ (main output directory)

    Returns:
        OutputLayout with draft_path and nano_prompt_path attributes.
    """
    output_dir = get_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    draft_path = output_dir / "draft.md"
    nano_prompt_path = output_dir / "nano-banana-prompts.md"
    return OutputLayout(draft_path=draft_path, nano_prompt_path=nano_prompt_path)


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


def safe_read(relative_path: str) -> str:
    """Read UTF-8 content from output directory with traversal protection.

    Args:
        relative_path: Relative path within ui-design/ directory

    Returns:
        Content read from file as UTF-8 string

    Raises:
        ValueError: If path tries to traverse outside ui-design/
        FileNotFoundError: If file does not exist
    """
    target = get_output_path(relative_path)
    return target.read_text(encoding="utf-8")


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
