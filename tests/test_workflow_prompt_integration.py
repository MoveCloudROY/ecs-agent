from __future__ import annotations

from pathlib import Path

import pytest

from ecs_agent.components import (
    LLMComponent,
    RenderedSystemPromptComponent,
    SystemPromptComponent,
    WorkflowBindingComponent,
    WorkflowDefinitionComponent,
    WorkflowRuntimeComponent,
)
from ecs_agent.core import Runner, World
from ecs_agent.prompts.contracts import PromptTemplateSource, SystemPromptConfigSpec
from ecs_agent.systems.system_prompt_render_system import SystemPromptRenderSystem
from ecs_agent.workflows import install_workflow, prompt_file, workflow
from ecs_agent.workflows.prompt_provider import WorkflowPromptPlaceholderProvider
import ecs_agent.systems.system_prompt_render_system as render_module


def _build_workflow(
    *,
    planner_a: str | Path | object,
    planner_b: str | Path | object,
    initial: str = "state_a",
) -> object:
    return workflow(
        "wf_prompt_profiles",
        initial=initial,
        profiles={
            "planner": {
                "plan_main": planner_a,
                "task_exec": planner_b,
            }
        },
        states={
            "state_a": {"bind": {"planner": "plan_main"}, "go": {}},
            "state_b": {"bind": {"planner": "task_exec"}, "go": {}},
        },
    )


def _install_provider_entity(
    world: World,
    *,
    planner_a: str | Path | object,
    planner_b: str | Path | object,
    initial: str = "state_a",
) -> int:
    entity_id = world.create_entity()
    install_workflow(
        world,
        entity_id,
        _build_workflow(planner_a=planner_a, planner_b=planner_b, initial=initial),
        agent_key="planner",
    )
    return entity_id


def test_shared_profile_avoids_cache_churn() -> None:
    world = World()
    entity_id = _install_provider_entity(
        world,
        planner_a="PLAN MODE",
        planner_b="TASK MODE",
    )
    runtime = world.get_component(entity_id, WorkflowRuntimeComponent)
    definition = world.get_component(entity_id, WorkflowDefinitionComponent)
    assert runtime is not None
    assert definition is not None

    definition.compiled.bindings_by_state["state_b"]["planner"] = "plan_main"
    provider = WorkflowPromptPlaceholderProvider()

    first = provider.provider_fingerprint(world, entity_id)
    runtime.current_state_id = "state_b"
    second = provider.provider_fingerprint(world, entity_id)

    assert first == second


def test_profile_change_invalidates_cache() -> None:
    world = World()
    entity_id = _install_provider_entity(
        world,
        planner_a="PLAN MODE",
        planner_b="TASK MODE",
    )
    runtime = world.get_component(entity_id, WorkflowRuntimeComponent)
    assert runtime is not None

    provider = WorkflowPromptPlaceholderProvider()
    first = provider.provider_fingerprint(world, entity_id)
    runtime.current_state_id = "state_b"
    second = provider.provider_fingerprint(world, entity_id)

    assert first != second


def test_only_workflow_state_prompt_key_exposed() -> None:
    world = World()
    entity_id = _install_provider_entity(
        world,
        planner_a="PLAN MODE",
        planner_b="TASK MODE",
    )

    placeholders = WorkflowPromptPlaceholderProvider().resolve_placeholders(world, entity_id)

    assert placeholders == {"_workflow_state_prompt": "PLAN MODE"}


def test_callable_profile_invokes_factory() -> None:
    world = World()
    entity_id = _install_provider_entity(
        world,
        planner_a=lambda: "dynamic-text",
        planner_b="TASK MODE",
    )

    placeholders = WorkflowPromptPlaceholderProvider().resolve_placeholders(world, entity_id)

    assert placeholders["_workflow_state_prompt"] == "dynamic-text"


def test_no_workflow_components_returns_empty() -> None:
    world = World()
    entity_id = world.create_entity()

    assert WorkflowPromptPlaceholderProvider().resolve_placeholders(world, entity_id) == {}


def test_unbound_agent_key_returns_empty() -> None:
    world = World()
    entity_id = _install_provider_entity(
        world,
        planner_a="PLAN MODE",
        planner_b="TASK MODE",
    )
    binding = world.get_component(entity_id, WorkflowBindingComponent)
    assert binding is not None
    binding.agent_key = "executor"

    assert WorkflowPromptPlaceholderProvider().resolve_placeholders(world, entity_id) == {}


def test_path_profile_reads_file(tmp_path: Path) -> None:
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("FILE MODE", encoding="utf-8")
    world = World()
    entity_id = _install_provider_entity(
        world,
        planner_a=prompt_file(prompt_path),
        planner_b="TASK MODE",
    )

    placeholders = WorkflowPromptPlaceholderProvider().resolve_placeholders(world, entity_id)

    assert placeholders["_workflow_state_prompt"] == "FILE MODE"


@pytest.mark.asyncio
async def test_system_prompt_render_uses_workflow_placeholder() -> None:
    world = World()
    entity_id = _install_provider_entity(
        world,
        planner_a="PLAN MODE",
        planner_b="TASK MODE",
    )
    world.add_component(
        entity_id,
        SystemPromptConfigSpec(
            template_source=PromptTemplateSource(
                inline="System: ${_workflow_state_prompt}"
            )
        ),
    )
    world.add_component(entity_id, LLMComponent(model=object(), system_prompt=""))
    world.add_component(entity_id, SystemPromptComponent(content=""))
    world.register_system(SystemPromptRenderSystem(priority=-20), priority=-20)

    await Runner().run(world, max_ticks=1)

    rendered = world.get_component(entity_id, RenderedSystemPromptComponent)
    assert rendered is not None
    assert rendered.text == "System: PLAN MODE"


@pytest.mark.asyncio
async def test_cache_stable_across_same_profile_ticks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world = World()
    entity_id = _install_provider_entity(
        world,
        planner_a="PLAN MODE",
        planner_b="TASK MODE",
    )
    definition = world.get_component(entity_id, WorkflowDefinitionComponent)
    runtime = world.get_component(entity_id, WorkflowRuntimeComponent)
    assert definition is not None
    assert runtime is not None
    definition.compiled.bindings_by_state["state_b"]["planner"] = "plan_main"
    world.add_component(
        entity_id,
        SystemPromptConfigSpec(
            template_source=PromptTemplateSource(inline="${_workflow_state_prompt}")
        ),
    )

    call_count = 0
    original_render = render_module._render_system_prompt

    def _counting_render(
        target_world: World,
        target_entity_id: object,
        prompt_config: SystemPromptConfigSpec,
    ) -> tuple[str, dict[str, str]]:
        nonlocal call_count
        call_count += 1
        return original_render(target_world, target_entity_id, prompt_config)

    monkeypatch.setattr(render_module, "_render_system_prompt", _counting_render)

    system = SystemPromptRenderSystem()
    await system.process(world)
    first = world.get_component(entity_id, RenderedSystemPromptComponent)
    assert first is not None
    first_key = first.placeholder_snapshot["_cache_key"]

    runtime.current_state_id = "state_b"
    await system.process(world)

    second = world.get_component(entity_id, RenderedSystemPromptComponent)
    assert second is not None
    assert second.placeholder_snapshot["_cache_key"] == first_key
    assert call_count == 1
