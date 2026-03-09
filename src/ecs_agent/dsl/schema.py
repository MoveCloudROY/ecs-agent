"""Agent DSL schema definition and validation."""

from dataclasses import dataclass, field
from typing import Any, Literal

from ecs_agent.logging import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class AgentSpec:
    """Normalized agent specification from JSON or Markdown DSL.

    Attributes:
        mode: Agent mode - 'primary' creates runnable main entity, 'subagent' creates config template
        model: LLM model identifier
        prompt: System prompt or instruction text
        tools: Tool permission mapping (name -> enabled)
        metadata: Arbitrary user-defined metadata
        name: Agent name for registry and logging
    """

    mode: Literal["primary", "subagent"]
    model: str
    prompt: str
    tools: dict[str, bool] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    name: str = ""


def validate_agent_spec(data: dict[str, Any], *, source_name: str = "") -> AgentSpec:
    """Validate and normalize agent specification from raw dict.

    Args:
        data: Raw dictionary from JSON or Markdown parser
        source_name: Source identifier for error messages (e.g., file path, agent name)

    Returns:
        Typed AgentSpec instance

    Raises:
        ValueError: If required fields are missing or unknown fields are present
        TypeError: If field types are invalid
    """
    source_context = f" in '{source_name}'" if source_name else ""

    # Check required fields
    required_fields = {"mode", "model", "prompt"}
    missing = required_fields - data.keys()
    if missing:
        raise ValueError(
            f"Missing required field(s): {', '.join(sorted(missing))}{source_context}"
        )

    # Check for unknown fields (fail-fast strict mode)
    known_fields = {"mode", "model", "prompt", "tools", "metadata", "name"}
    unknown = data.keys() - known_fields
    if unknown:
        raise ValueError(
            f"Unknown field(s): {', '.join(sorted(unknown))}{source_context}"
        )

    # Validate mode literal
    mode = data["mode"]
    if mode not in ("primary", "subagent"):
        raise ValueError(
            f"Invalid mode '{mode}': must be 'primary' or 'subagent'{source_context}"
        )

    # Validate tools field type
    tools = data.get("tools", {})
    if not isinstance(tools, dict):
        raise TypeError(f"Field 'tools' must be dict[str, bool]{source_context}")
    for tool_name, enabled in tools.items():
        if not isinstance(tool_name, str):
            raise TypeError(f"Tool name must be str{source_context}")
        if not isinstance(enabled, bool):
            raise TypeError(f"Tool '{tool_name}' value must be bool{source_context}")

    # Validate metadata field type
    metadata = data.get("metadata", {})
    if not isinstance(metadata, dict):
        raise TypeError(f"Field 'metadata' must be dict[str, Any]{source_context}")

    # Validate name field type
    name = data.get("name", "")
    if not isinstance(name, str):
        raise TypeError(f"Field 'name' must be str{source_context}")

    logger.info(
        "agent_spec_validated",
        mode=mode,
        model=data["model"],
        name=name,
        tool_count=len(tools),
        source=source_name,
    )

    return AgentSpec(
        mode=mode,
        model=data["model"],
        prompt=data["prompt"],
        tools=tools,
        metadata=metadata,
        name=name,
    )


__all__ = ["AgentSpec", "validate_agent_spec"]
