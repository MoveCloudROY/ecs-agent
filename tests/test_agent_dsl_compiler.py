from __future__ import annotations

from pathlib import Path

import pytest

from ecs_agent.components import (
    LLMComponent,
    PermissionComponent,
    SubagentRegistryComponent,
    SystemPromptComponent,
)
from ecs_agent.core import World
from ecs_agent.dsl.schema import AgentSpec
from ecs_agent.types import SubagentConfig


class _ProviderFactorySpy:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def __call__(self, model: str, system_prompt: str) -> object:
        self.calls.append((model, system_prompt))
        return {"model": model, "system_prompt": system_prompt}


class _ProviderFactoryFailOnModel:
    def __init__(self, failing_model: str) -> None:
        self._failing_model = failing_model
        self.calls: list[tuple[str, str]] = []

    def __call__(self, model: str, system_prompt: str) -> object:
        self.calls.append((model, system_prompt))
        if model == self._failing_model:
            raise RuntimeError(f"provider factory failed for model '{model}'")
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


def test_compile_links_markdown_prompt_into_system_prompt_component() -> None:
    from ecs_agent.dsl.compiler import compile_agent_specs

    specs = {
        "main": AgentSpec(
            name="main",
            mode="primary",
            model="gpt-main",
            prompt="## Role\n\nYou are a markdown-defined assistant.",
        )
    }

    primary_entity_id, world = compile_agent_specs(specs, _ProviderFactorySpy())
    system_prompt = world.get_component(primary_entity_id, SystemPromptComponent)

    assert system_prompt is not None
    assert system_prompt.template == "## Role\n\nYou are a markdown-defined assistant."
    assert system_prompt.content == "## Role\n\nYou are a markdown-defined assistant."


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
    assert helper.max_ticks is None
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


def test_compiler_primary_entity_has_expected_components_after_compile() -> None:
    """Validate primary entity has LLM and subagent registry components."""
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

    assert world.get_component(primary_entity_id, LLMComponent) is not None
    assert world.get_component(primary_entity_id, SubagentRegistryComponent) is not None
    assert world.get_component(primary_entity_id, PermissionComponent) is None


def test_compiler_world_query_has_single_llm_entity_after_compile() -> None:
    """Validate compile creates one runnable LLM entity regardless of subagent count."""
    from ecs_agent.dsl.compiler import compile_agent_specs

    specs = {
        "main": AgentSpec(
            name="main",
            mode="primary",
            model="gpt-main",
            prompt="Primary prompt",
        ),
        "researcher": AgentSpec(
            name="researcher",
            mode="subagent",
            model="gpt-r",
            prompt="R prompt",
        ),
        "coder": AgentSpec(
            name="coder",
            mode="subagent",
            model="gpt-c",
            prompt="C prompt",
        ),
    }

    primary_entity_id, world = compile_agent_specs(specs, _ProviderFactorySpy())
    llm_entities = world.query(LLMComponent)

    assert len(llm_entities) == 1
    assert llm_entities[0][0] == primary_entity_id


def test_compiler_subagent_registry_maps_three_subagents_correctly() -> None:
    """Validate subagent registry structure for complex hierarchy input."""
    from ecs_agent.dsl.compiler import compile_agent_specs

    specs = {
        "main": AgentSpec(
            name="main",
            mode="primary",
            model="gpt-main",
            prompt="Primary",
        ),
        "researcher": AgentSpec(
            name="researcher",
            mode="subagent",
            model="gpt-research",
            prompt="Research",
        ),
        "planner": AgentSpec(
            name="planner",
            mode="subagent",
            model="gpt-plan",
            prompt="Plan",
        ),
        "writer": AgentSpec(
            name="writer",
            mode="subagent",
            model="gpt-write",
            prompt="Write",
        ),
    }

    primary_entity_id, world = compile_agent_specs(specs, _ProviderFactorySpy())
    registry = world.get_component(primary_entity_id, SubagentRegistryComponent)

    assert registry is not None
    assert set(registry.subagents.keys()) == {"researcher", "planner", "writer"}
    assert registry.subagents["researcher"].model == "gpt-research"
    assert registry.subagents["planner"].system_prompt == "Plan"
    assert registry.subagents["writer"].max_ticks is None


def test_compiler_subagent_registry_keeps_distinct_provider_objects() -> None:
    """Validate each compiled subagent config keeps its own provider object."""
    from ecs_agent.dsl.compiler import compile_agent_specs

    specs = {
        "main": AgentSpec(name="main", mode="primary", model="gpt-main", prompt="P"),
        "a": AgentSpec(name="a", mode="subagent", model="m-a", prompt="A"),
        "b": AgentSpec(name="b", mode="subagent", model="m-b", prompt="B"),
    }

    primary_entity_id, world = compile_agent_specs(specs, _ProviderFactorySpy())
    registry = world.get_component(primary_entity_id, SubagentRegistryComponent)

    assert registry is not None
    assert registry.subagents["a"].provider is not registry.subagents["b"].provider


def test_compiler_permission_mapping_all_true_preserves_tool_names() -> None:
    """Validate tools=True entries map directly into PermissionComponent allowlist."""
    from ecs_agent.dsl.compiler import compile_agent_specs

    specs = {
        "main": AgentSpec(
            name="main",
            mode="primary",
            model="gpt-main",
            prompt="Primary",
            tools={"bash": True, "read_file": True, "web_search": True},
        )
    }

    primary_entity_id, world = compile_agent_specs(specs, _ProviderFactorySpy())
    permission = world.get_component(primary_entity_id, PermissionComponent)

    assert permission is not None
    assert permission.allowed_tools == ["bash", "read_file", "web_search"]
    assert permission.denied_tools == []


def test_compiler_permission_mapping_mixed_booleans_keeps_only_true_in_order() -> None:
    """Validate mixed tool booleans compile to true-only ordered allowlist."""
    from ecs_agent.dsl.compiler import compile_agent_specs

    specs = {
        "main": AgentSpec(
            name="main",
            mode="primary",
            model="gpt-main",
            prompt="Primary",
            tools={"read": True, "write": False, "bash": True, "search": False},
        )
    }

    primary_entity_id, world = compile_agent_specs(specs, _ProviderFactorySpy())
    permission = world.get_component(primary_entity_id, PermissionComponent)

    assert permission is not None
    assert permission.allowed_tools == ["read", "bash"]


def test_compiler_prompt_file_resolution_linkage_updates_primary_prompt(
    tmp_path: Path,
) -> None:
    """Validate resolved prompt file content is what compiler injects into LLMComponent."""
    from ecs_agent.dsl.compiler import compile_agent_specs
    from ecs_agent.dsl.prompt_resolver import resolve_prompt_file

    prompt_file = tmp_path / "main_prompt.txt"
    prompt_file.write_text("Resolved primary prompt", encoding="utf-8")

    resolved_prompt = resolve_prompt_file("{file:main_prompt.txt}", tmp_path)
    specs = {
        "main": AgentSpec(
            name="main",
            mode="primary",
            model="gpt-main",
            prompt=resolved_prompt,
        )
    }
    spy = _ProviderFactorySpy()

    primary_entity_id, world = compile_agent_specs(specs, spy)
    llm = world.get_component(primary_entity_id, LLMComponent)

    assert llm is not None
    assert llm.system_prompt == "Resolved primary prompt"
    assert spy.calls == [("gpt-main", "Resolved primary prompt")]


def test_compiler_prompt_file_resolution_linkage_updates_subagent_prompt(
    tmp_path: Path,
) -> None:
    """Validate resolved prompt content flows into compiled SubagentConfig prompts."""
    from ecs_agent.dsl.compiler import compile_agent_specs
    from ecs_agent.dsl.prompt_resolver import resolve_prompt_file

    prompt_file = tmp_path / "sub_prompt.txt"
    prompt_file.write_text("Resolved subagent prompt", encoding="utf-8")

    resolved_sub_prompt = resolve_prompt_file("{file:sub_prompt.txt}", tmp_path)
    specs = {
        "main": AgentSpec(name="main", mode="primary", model="gpt-main", prompt="P"),
        "researcher": AgentSpec(
            name="researcher",
            mode="subagent",
            model="gpt-r",
            prompt=resolved_sub_prompt,
        ),
    }
    spy = _ProviderFactorySpy()

    primary_entity_id, world = compile_agent_specs(specs, spy)
    registry = world.get_component(primary_entity_id, SubagentRegistryComponent)

    assert registry is not None
    assert registry.subagents["researcher"].system_prompt == "Resolved subagent prompt"
    assert spy.calls == [("gpt-main", "P"), ("gpt-r", "Resolved subagent prompt")]


def test_compiler_missing_prompt_file_reference_raises_file_not_found_error(
    tmp_path: Path,
) -> None:
    """Validate missing {file:...} prompt references fail before compilation."""
    from ecs_agent.dsl.prompt_resolver import resolve_prompt_file

    with pytest.raises(FileNotFoundError, match="Prompt file not found"):
        resolve_prompt_file("{file:missing_prompt.md}", tmp_path)


def test_compiler_provider_factory_invalid_primary_model_error_propagates() -> None:
    """Validate primary provider factory errors surface as compile-time failures."""
    from ecs_agent.dsl.compiler import compile_agent_specs

    specs = {
        "main": AgentSpec(
            name="main",
            mode="primary",
            model="broken-model",
            prompt="Primary",
        )
    }
    provider_factory = _ProviderFactoryFailOnModel("broken-model")

    with pytest.raises(RuntimeError, match="provider factory failed"):
        compile_agent_specs(specs, provider_factory)


def test_compiler_provider_factory_invalid_subagent_model_error_propagates() -> None:
    """Validate subagent provider factory errors fail compilation after primary mapping."""
    from ecs_agent.dsl.compiler import compile_agent_specs

    specs = {
        "main": AgentSpec(name="main", mode="primary", model="gpt-main", prompt="P"),
        "helper": AgentSpec(
            name="helper",
            mode="subagent",
            model="broken-subagent",
            prompt="S",
        ),
    }
    provider_factory = _ProviderFactoryFailOnModel("broken-subagent")

    with pytest.raises(RuntimeError, match="provider factory failed"):
        compile_agent_specs(specs, provider_factory)

    assert provider_factory.calls == [("gpt-main", "P"), ("broken-subagent", "S")]


def test_compiler_world_state_consistency_after_repeated_compilation() -> None:
    """Validate each compilation creates an isolated world with consistent primary entity."""
    from ecs_agent.dsl.compiler import compile_agent_specs

    specs = {
        "main": AgentSpec(name="main", mode="primary", model="gpt-main", prompt="P"),
        "helper": AgentSpec(name="helper", mode="subagent", model="m1", prompt="S1"),
    }

    first_primary, first_world = compile_agent_specs(specs, _ProviderFactorySpy())
    second_primary, second_world = compile_agent_specs(specs, _ProviderFactorySpy())

    assert first_world is not second_world
    assert int(first_primary) == 1
    assert int(second_primary) == 1
    assert len(first_world.query(LLMComponent)) == 1
    assert len(second_world.query(LLMComponent)) == 1
