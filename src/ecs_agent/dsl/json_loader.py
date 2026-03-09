"""JSON DSL loader with strict parsing and normalization."""

import json
from pathlib import Path

from ecs_agent.dsl.schema import AgentSpec, validate_agent_spec
from ecs_agent.logging import get_logger

logger = get_logger(__name__)


def load_json_agents(path: Path | str) -> list[AgentSpec]:
    """Load agents from JSON file.

    Expected JSON format:
    {
        "agent_name_1": {"mode": "primary", "model": "...", "prompt": "..."},
        "agent_name_2": {"mode": "subagent", "model": "...", "prompt": "..."}
    }

    Args:
        path: Path to JSON file

    Returns:
        List of normalized AgentSpec instances

    Raises:
        FileNotFoundError: If path doesn't exist
        json.JSONDecodeError: If JSON is malformed
        ValueError/TypeError: If validation fails
    """
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"Agent JSON file not found: {file_path}")

    try:
        content = file_path.read_text(encoding="utf-8")
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        logger.error("json_parse_error", file_path=str(file_path), exception=str(exc))
        raise

    if not isinstance(data, dict):
        raise ValueError(
            f"JSON root must be dict mapping agent names to configs, got {type(data).__name__}"
        )

    specs: list[AgentSpec] = []
    for agent_name, config in data.items():
        if not isinstance(config, dict):
            raise ValueError(
                f"Agent '{agent_name}' config must be dict, got {type(config).__name__}"
            )

        # Merge agent name into config
        config_with_name = dict(config)
        config_with_name["name"] = agent_name

        # Validate and normalize
        source_name = f"{file_path}:{agent_name}"
        spec = validate_agent_spec(config_with_name, source_name=source_name)
        specs.append(spec)

    logger.info(
        "json_agents_loaded",
        file_path=str(file_path),
        agent_count=len(specs),
        agent_names=[spec.name for spec in specs],
    )

    return specs


__all__ = ["load_json_agents"]
