"""Agent DSL module for declarative agent definition and loading."""

from ecs_agent.dsl.discovery import discover_agent_sources
from ecs_agent.dsl.json_loader import load_json_agents
from ecs_agent.dsl.markdown_loader import load_markdown_agent
from ecs_agent.dsl.resolver import resolve_agent_specs
from ecs_agent.dsl.schema import AgentSpec, validate_agent_spec

__all__ = [
    "AgentSpec",
    "validate_agent_spec",
    "discover_agent_sources",
    "load_json_agents",
    "resolve_agent_specs",
    "load_markdown_agent",
]
