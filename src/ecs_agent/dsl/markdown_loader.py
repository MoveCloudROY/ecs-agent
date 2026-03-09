"""Markdown agent loader with YAML frontmatter and filename identity."""

from pathlib import Path
from typing import Any

import yaml

from ecs_agent.dsl.schema import AgentSpec, validate_agent_spec
from ecs_agent.logging import get_logger

logger = get_logger(__name__)


def load_markdown_agent(path: Path | str) -> AgentSpec:
    """Load agent from Markdown file with YAML frontmatter.

    Expected format:
    ---
    mode: primary
    model: gpt-4
    tools:
      read_file: true
    ---
    You are a helpful assistant.

    Args:
        path: Path to .md file

    Returns:
        Normalized AgentSpec with name from filename

    Raises:
        FileNotFoundError: If file doesn't exist
        yaml.YAMLError: If frontmatter is invalid
        ValueError: If validation fails (missing required fields, unknown fields)
        TypeError: If field types are invalid
    """
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"Markdown agent file not found: {file_path}")

    content = file_path.read_text()

    # Agent name comes from filename (authoritative)
    agent_name = file_path.stem

    # Split frontmatter and body
    frontmatter_dict: dict[str, Any] = {}
    markdown_body = ""

    if content.startswith("---"):
        # Split on closing ---
        parts = content.split("---", 2)
        if len(parts) >= 3:
            frontmatter_text = parts[1].strip()
            markdown_body = parts[2].strip()

            # Parse YAML frontmatter (use safe_load for security)
            if frontmatter_text:
                try:
                    frontmatter_dict = yaml.safe_load(frontmatter_text) or {}
                except yaml.YAMLError as exc:
                    logger.error(
                        "markdown_agent_invalid_yaml",
                        file_path=str(file_path),
                        exception=str(exc),
                    )
                    raise yaml.YAMLError(
                        f"Invalid YAML frontmatter in {file_path}: {exc}"
                    ) from exc
        else:
            # Malformed frontmatter (only opening ---), treat as no frontmatter
            markdown_body = content.strip()
    else:
        # No frontmatter
        markdown_body = content.strip()

    # Build config dict: frontmatter + filename-based name + prompt from body
    config = frontmatter_dict.copy()
    config["name"] = agent_name  # Filename ALWAYS wins
    config["prompt"] = markdown_body

    # Validate and return AgentSpec
    return validate_agent_spec(config, source_name=str(file_path))


__all__ = ["load_markdown_agent"]
