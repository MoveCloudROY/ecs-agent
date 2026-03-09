from __future__ import annotations

import pytest

from ecs_agent.components import LLMComponent, PermissionComponent, SubagentRegistryComponent
from ecs_agent.core import World
from ecs_agent.dsl.schema import AgentSpec
from ecs_agent.types import SubagentConfig


class _ProviderFactorySpy:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def __call__(self, model: str, system_prompt: str) -> object:
        self.calls.append((model, system_prompt))
        return {"model": model, "system_prompt": system_prompt}


def test_compile_primary_and_subagent_creates_runnable_world() -> None:
    from ecs_agent.dsl.compiler import compile_agent_specs

    specs = {
        "main": AgentSpec(
            name="main",
            mode="primary",
            model="gpt-main",
            prompt="Primary system prompt",
        ),
        "researcher": AgentSpec(
            name="researcher",
            mode="subagent",
            model="gpt-research",
            prompt="Research system prompt",
            tools={"web_search": True},
            metadata={"team": "research"},
        ),
    }
    provider_factory = _ProviderFactorySpy()

    primary_entity_id, world = compile_agent_specs(specs, provider_factory)

    assert isinstance(world, World)
    assert int(primary_entity_id) > 0

    llm = world.get_component(primary_entity_id, LLMComponent)
    assert llm is not None
    assert llm.model == "gpt-main"
    assert llm.system_prompt == "Primary system prompt"
    assert llm.provider == {
        "model": "gpt-main",
        "system_prompt": "Primary system prompt",
    }

    registry = world.get_component(primary_entity_id, SubagentRegistryComponent)
    assert registry is not None
    assert set(registry.subagents.keys()) == {"researcher"}

    researcher = registry.subagents["researcher"]
    assert isinstance(researcher, SubagentConfig)
    assert researcher.name == "researcher"
    assert researcher.model == "gpt-research"
    assert researcher.system_prompt == "Research system prompt"
    assert researcher.provider == {
        "model": "gpt-research",
        "system_prompt": "Research system prompt",
    }

    assert provider_factory.calls == [
        ("gpt-main", "Primary system prompt"),
        ("gpt-research", "Research system prompt"),
    ]


def test_compile_missing_primary_raises_value_error() -> None:
    from ecs_agent.dsl.compiler import compile_agent_specs

    specs = {
        "researcher": AgentSpec(
            name="researcher",
            mode="subagent",
            model="gpt-research",
            prompt="Research system prompt",
        )
    }

    with pytest.raises(ValueError, match="exactly one primary agent.*found 0"):
        compile_agent_specs(specs, _ProviderFactorySpy())


def test_compile_multiple_primary_raises_value_error_with_count() -> None:
    from ecs_agent.dsl.compiler import compile_agent_specs

    specs = {
        "main": AgentSpec(
            name="main",
            mode="primary",
            model="gpt-main",
            prompt="Primary prompt",
        ),
        "backup": AgentSpec(
            name="backup",
            mode="primary",
            model="gpt-backup",
            prompt="Backup prompt",
        ),
    }

    with pytest.raises(ValueError, match="exactly one primary agent.*found 2"):
        compile_agent_specs(specs, _ProviderFactorySpy())


def test_compile_only_primary_creates_empty_subagent_registry() -> None:
    from ecs_agent.dsl.compiler import compile_agent_specs

    specs = {
        "main": AgentSpec(
            name="main",
            mode="primary",
            model="gpt-main",
            prompt="Primary prompt",
        )
    }

    primary_entity_id, world = compile_agent_specs(specs, _ProviderFactorySpy())

    registry = world.get_component(primary_entity_id, SubagentRegistryComponent)
    assert registry is not None
    assert registry.subagents == {}


def test_compile_subagent_config_maps_expected_fields() -> None:
    from ecs_agent.dsl.compiler import compile_agent_specs

    specs = {
        "main": AgentSpec(
            name="main",
            mode="primary",
            model="gpt-main",
            prompt="Primary prompt",
        ),
        "helper": AgentSpec(
            name="helper",
            mode="subagent",
            model="gpt-helper",
            prompt="Helper prompt",
            tools={"calculator": True, "search": False},
            metadata={"tier": "support"},
        ),
    }

    primary_entity_id, world = compile_agent_specs(specs, _ProviderFactorySpy())

    registry = world.get_component(primary_entity_id, SubagentRegistryComponent)
    assert registry is not None
    helper = registry.subagents["helper"]

    assert helper.name == "helper"
    assert helper.model == "gpt-helper"
    assert helper.system_prompt == "Helper prompt"
    assert helper.max_ticks == 10
    assert helper.skills == []


def test_compile_tools_boolean_mapping_creates_permission_component() -> None:
    from ecs_agent.dsl.compiler import compile_agent_specs

    specs = {
        "main": AgentSpec(
            name="main",
            mode="primary",
            model="gpt-main",
            prompt="Primary prompt",
            tools={"bash": True, "read_file": True, "web_search": False},
        )
    }

    primary_entity_id, world = compile_agent_specs(specs, _ProviderFactorySpy())

    permission = world.get_component(primary_entity_id, PermissionComponent)
    assert permission is not None
    assert set(permission.allowed_tools) == {"bash", "read_file"}
    assert permission.denied_tools == []


def test_compile_tools_all_false_creates_empty_allowlist() -> None:
    from ecs_agent.dsl.compiler import compile_agent_specs

    specs = {
        "main": AgentSpec(
            name="main",
            mode="primary",
            model="gpt-main",
            prompt="Primary prompt",
            tools={"bash": False, "read_file": False},
        )
    }

    primary_entity_id, world = compile_agent_specs(specs, _ProviderFactorySpy())

    permission = world.get_component(primary_entity_id, PermissionComponent)
    assert permission is not None
    assert permission.allowed_tools == []
    assert permission.denied_tools == []


def test_compile_no_tools_no_permission_component() -> None:
    from ecs_agent.dsl.compiler import compile_agent_specs

    specs = {
        "main": AgentSpec(
            name="main",
            mode="primary",
            model="gpt-main",
            prompt="Primary prompt",
        )
    }

    primary_entity_id, world = compile_agent_specs(specs, _ProviderFactorySpy())

    permission = world.get_component(primary_entity_id, PermissionComponent)
    assert permission is None


def test_compile_empty_tools_dict_no_permission_component() -> None:
    from ecs_agent.dsl.compiler import compile_agent_specs

    specs = {
        "main": AgentSpec(
            name="main",
            mode="primary",
            model="gpt-main",
            prompt="Primary prompt",
            tools={},
        )
    }

    primary_entity_id, world = compile_agent_specs(specs, _ProviderFactorySpy())

    permission = world.get_component(primary_entity_id, PermissionComponent)
    assert permission is None
