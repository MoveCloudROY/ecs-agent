"""Deterministic discovery of agent definition sources (JSON and Markdown files).

This module provides source discovery with stable ordering to ensure reproducible
`last-one-wins` semantics when multiple sources define the same agent.
"""

from pathlib import Path


def discover_agent_sources(directory: Path | str) -> list[Path]:
    """Discover agent definition files in deterministic order.

    Finds all *.json and *.md files in directory (non-recursive).
    Returns paths in stable sorted order for reproducible `last-one-wins`
    semantics when loading agent definitions.

    Args:
        directory: Directory to search for agent definition files.

    Returns:
        Sorted list of Path objects (*.json and *.md files) in lexicographic order.
        Same input directory always yields same output order.

    Raises:
        FileNotFoundError: If directory doesn't exist or is not a directory.

    Example:
        >>> sources = discover_agent_sources("./agents")
        >>> # Returns: [Path("agents/agent1.json"), Path("agents/agent2.md"), ...]
        >>> # Same call returns same order every time (deterministic)
    """
    dir_path = Path(directory)

    if not dir_path.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory}")

    # Collect all agent source files (JSON and Markdown)
    json_sources = list(dir_path.glob("*.json"))
    markdown_sources = list(dir_path.glob("*.md"))
    sources = json_sources + markdown_sources

    # Sort for deterministic order (CRITICAL for last-one-wins semantics)
    return sorted(sources)
