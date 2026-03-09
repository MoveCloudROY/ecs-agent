"""Conflict resolution for agent specifications with deterministic last-one-wins policy."""

from ecs_agent.dsl.schema import AgentSpec
from ecs_agent.logging import get_logger

logger = get_logger(__name__)


def resolve_agent_specs(specs: list[AgentSpec]) -> dict[str, AgentSpec]:
    """Resolve agent specs with last-one-wins conflict policy.

    For duplicate agent names, the last occurrence in the input list wins.
    This is deterministic because input list order is stable (from Task 4 discovery).

    Args:
        specs: List of AgentSpec in discovery order (stable ordering from loader)

    Returns:
        Dict mapping agent name -> final AgentSpec

    Raises:
        ValueError: If any spec has empty name (ambiguous identity)
    """
    resolved: dict[str, AgentSpec] = {}
    duplicate_count = 0

    for spec in specs:
        if not spec.name:
            raise ValueError(
                f"Agent spec with mode '{spec.mode}' has no name (ambiguous identity)"
            )

        # Track duplicates for logging
        if spec.name in resolved:
            duplicate_count += 1
            logger.debug(
                "agent_spec_conflict_detected",
                agent_name=spec.name,
                previous_model=resolved[spec.name].model,
                winner_model=spec.model,
            )

        # Last entry wins for duplicate names
        resolved[spec.name] = spec

    logger.info(
        "agent_specs_resolved",
        total_specs=len(specs),
        unique_agents=len(resolved),
        duplicates_overridden=duplicate_count,
    )

    return resolved


__all__ = ["resolve_agent_specs"]
