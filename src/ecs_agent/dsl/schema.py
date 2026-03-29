"""Agent DSL schema definition and validation."""

import re
from dataclasses import dataclass, field
from typing import Any, Literal

from ecs_agent.logging import get_logger


_PLACEHOLDER_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
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
        placeholders: Placeholder name-value dicts for ${name} substitution in prompt
        triggers: Trigger rule dicts for UserPromptNormalizationSystem
    """

    mode: Literal["primary", "subagent"]
    model: str
    prompt: str
    tools: dict[str, bool] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    name: str = ""
    placeholders: list[dict[str, str]] = field(default_factory=list)
    triggers: list[dict[str, str | int]] = field(default_factory=list)


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
    known_fields = {
        "mode",
        "model",
        "prompt",
        "tools",
        "metadata",
        "name",
        "placeholders",
        "triggers",
    }
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

    # Validate placeholders field type and content
    placeholders = data.get("placeholders", [])
    if not isinstance(placeholders, list):
        raise TypeError(f"Field 'placeholders' must be list{source_context}")
    normalized_placeholders: list[dict[str, str]] = []
    for i, ph in enumerate(placeholders):
        if not isinstance(ph, dict):
            raise TypeError(f"Placeholder at index {i} must be dict{source_context}")
        if "name" not in ph:
            raise ValueError(
                f"Placeholder at index {i} missing required key 'name'{source_context}"
            )
        if "value" not in ph:
            raise ValueError(
                f"Placeholder at index {i} missing required key 'value'{source_context}"
            )
        ph_name = ph["name"]
        ph_value = ph["value"]
        if not isinstance(ph_name, str):
            raise TypeError(
                f"Placeholder 'name' at index {i} must be str{source_context}"
            )
        if not isinstance(ph_value, str):
            raise TypeError(
                f"Placeholder 'value' at index {i} must be str{source_context}"
            )
        if not _PLACEHOLDER_NAME_RE.match(ph_name):
            raise ValueError(
                f"Invalid placeholder name {ph_name!r} at index {i}: "
                f"must match [A-Za-z_][A-Za-z0-9_]*{source_context}"
            )
        if ph_name.startswith("_"):
            raise ValueError(
                f"Invalid placeholder name {ph_name!r} at index {i}: "
                f"names starting with '_' are reserved{source_context}"
            )
        normalized_placeholders.append({"name": ph_name, "value": ph_value})

    # Validate triggers field type and content
    raw_triggers = data.get("triggers", [])
    if not isinstance(raw_triggers, list):
        raise TypeError(f"Field 'triggers' must be list{source_context}")
    _valid_match_modes = {"keyword", "prefix", "contains"}
    _valid_actions = {"replace", "inject"}
    normalized_triggers: list[dict[str, str | int]] = []
    for i, trig in enumerate(raw_triggers):
        if not isinstance(trig, dict):
            raise TypeError(f"Trigger at index {i} must be dict{source_context}")
        for req_key in ("pattern", "match_mode", "action", "content"):
            if req_key not in trig:
                raise ValueError(
                    f"Trigger at index {i} missing required key '{req_key}'{source_context}"
                )
        if trig["match_mode"] not in _valid_match_modes:
            raise ValueError(
                f"Trigger at index {i} has invalid match_mode {trig['match_mode']!r}: "
                f"must be one of {sorted(_valid_match_modes)}{source_context}"
            )
        if trig["action"] not in _valid_actions:
            raise ValueError(
                f"Trigger at index {i} has invalid action {trig['action']!r}: "
                f"must be one of {sorted(_valid_actions)}{source_context}"
            )
        priority = trig.get("priority", 0)
        if not isinstance(priority, int):
            raise TypeError(
                f"Trigger 'priority' at index {i} must be int{source_context}"
            )
        normalized_triggers.append(
            {
                "pattern": str(trig["pattern"]),
                "match_mode": str(trig["match_mode"]),
                "action": str(trig["action"]),
                "content": str(trig["content"]),
                "priority": priority,
            }
        )

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
        placeholders=normalized_placeholders,
        triggers=normalized_triggers,
    )


__all__ = ["AgentSpec", "validate_agent_spec"]
