"""Agent DSL module for declarative agent definition and loading."""

from ecs_agent.dsl.schema import AgentSpec, validate_agent_spec

__all__ = [
    "AgentSpec",
    "validate_agent_spec",
]
