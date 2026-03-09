"""Compile validated Agent DSL specs into ECS runtime components."""

from __future__ import annotations

from collections.abc import Callable

from ecs_agent.components import LLMComponent, PermissionComponent, SubagentRegistryComponent
from ecs_agent.core import World
from ecs_agent.dsl.schema import AgentSpec
from ecs_agent.logging import get_logger
from ecs_agent.providers.protocol import LLMProvider
from ecs_agent.types import EntityId, SubagentConfig

logger = get_logger(__name__)


def compile_agent_specs(
    specs: dict[str, AgentSpec],
    provider_factory: Callable[[str, str], LLMProvider],
) -> tuple[EntityId, World]:
    primary_specs = [
        (agent_name, spec)
        for agent_name, spec in specs.items()
        if spec.mode == "primary"
    ]
    primary_count = len(primary_specs)
    if primary_count != 1:
        logger.error(
            "agent_spec_compile_invalid_primary_count",
            primary_count=primary_count,
            total_specs=len(specs),
        )
        raise ValueError(f"Expected exactly one primary agent, found {primary_count}")

    world = World()
    primary_entity = world.create_entity()

    _, primary_spec = primary_specs[0]
    primary_provider = provider_factory(primary_spec.model, primary_spec.prompt)
    world.add_component(
        primary_entity,
        LLMComponent(
            provider=primary_provider,
            model=primary_spec.model,
            system_prompt=primary_spec.prompt,
        ),
    )

    # Add permission component if tools are specified
    if primary_spec.tools:
        allowed_tools = [tool for tool, enabled in primary_spec.tools.items() if enabled]
        world.add_component(
            primary_entity,
            PermissionComponent(allowed_tools=allowed_tools, denied_tools=[]),
        )
        logger.info(
            "agent_tools_permission_attached",
            entity_id=int(primary_entity),
            allowed_count=len(allowed_tools),
        )

    subagents: dict[str, SubagentConfig] = {}
    for agent_name, spec in specs.items():
        if spec.mode != "subagent":
            continue
        subagents[agent_name] = SubagentConfig(
            name=agent_name,
            provider=provider_factory(spec.model, spec.prompt),
            model=spec.model,
            system_prompt=spec.prompt,
        )

    world.add_component(primary_entity, SubagentRegistryComponent(subagents=subagents))

    logger.info(
        "agent_specs_compiled",
        primary_entity_id=int(primary_entity),
        subagent_count=len(subagents),
    )
    return primary_entity, world


__all__ = ["compile_agent_specs"]
