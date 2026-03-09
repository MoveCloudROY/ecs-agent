"""Tests for Agent DSL compiler behavior."""

from __future__ import annotations

import pytest

from ecs_agent.components import (
    LLMComponent,
    PermissionComponent,
    SubagentRegistryComponent,
)
from ecs_agent.dsl.compiler import (
    AgentSpec,
    compile_agent_specs,
)


def _provider_factory(_model: str, _prompt: str) -> object:
    return object()


def test_primary_and_subagent_compile_attaches_registry_and_llm_component() -> None:
    specs = {
        "primary": AgentSpec(mode="primary", model="gpt-4o-mini", prompt="You are primary."),
        "researcher": AgentSpec(
            mode="subagent",
            name="researcher",
            model="gpt-4o-mini",
            prompt="You are subagent.",
        ),
    }

    primary_entity, world = compile_agent_specs(
        specs, provider_factory=_provider_factory
    )

    llm = world.get_component(primary_entity, LLMComponent)
    registry = world.get_component(primary_entity, SubagentRegistryComponent)

    assert llm is not None
    assert llm.model == "gpt-4o-mini"
    assert llm.system_prompt == "You are primary."
    assert registry is not None
    assert set(registry.subagents.keys()) == {"researcher"}


def test_missing_primary_raises_explicit_error() -> None:
    specs = {
        "helper": AgentSpec(
            mode="subagent",
            name="helper",
            model="gpt-4o-mini",
            prompt="You are helper.",
        )
    }

    with pytest.raises(ValueError, match="exactly one primary"):
        compile_agent_specs(specs, provider_factory=_provider_factory)


def test_tools_boolean_mapping_primary_permission_allowlist() -> None:
    specs = {
        "primary": AgentSpec(
            mode="primary",
            model="gpt-4o-mini",
            prompt="You are primary.",
            tools={"web_search": True, "bash": False},
        )
    }

    primary_entity, world = compile_agent_specs(
        specs, provider_factory=_provider_factory
    )
    permission = world.get_component(primary_entity, PermissionComponent)

    assert permission is not None
    assert permission.allowed_tools == ["web_search"]
    assert permission.denied_tools == []


def test_tools_block_absent_skips_permission_component() -> None:
    specs = {"primary": AgentSpec(mode="primary", model="gpt-4o-mini", prompt="You are primary.")}

    primary_entity, world = compile_agent_specs(
        specs, provider_factory=_provider_factory
    )
    permission = world.get_component(primary_entity, PermissionComponent)

    assert permission is None


def test_all_false_tools_block_attaches_empty_allowlist() -> None:
    specs = {
        "primary": AgentSpec(
            mode="primary",
            model="gpt-4o-mini",
            prompt="You are primary.",
            tools={"bash": False, "web_search": False},
        )
    }

    primary_entity, world = compile_agent_specs(
        specs, provider_factory=_provider_factory
    )

    permission = world.get_component(primary_entity, PermissionComponent)
    assert permission is not None
    assert permission.allowed_tools == []


def test_only_primary_gets_permission_component() -> None:
    specs = {
        "primary": AgentSpec(
            mode="primary",
            model="gpt-4o-mini",
            prompt="You are primary.",
            tools={"web_search": True},
        ),
        "researcher": AgentSpec(
            mode="subagent",
            name="researcher",
            model="gpt-4o-mini",
            prompt="You are subagent.",
            tools={"bash": False},
        ),
    }

    _, world = compile_agent_specs(specs, provider_factory=_provider_factory)

    permission_entities = [
        entity_id for entity_id, _ in world.query(PermissionComponent)
    ]
    assert len(permission_entities) == 1
