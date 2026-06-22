"""Integration tests: workflow state + TriggerSpec + FakeModel end-to-end."""

from __future__ import annotations

import pytest

from ecs_agent.components import (
    ConversationComponent,
    LLMComponent,
    RenderedSystemPromptComponent,
    UserPromptConfigComponent,
    WorkflowBindingComponent,
    WorkflowRuntimeComponent,
)
from ecs_agent.core import Runner, World
from ecs_agent.providers import FakeModel
from ecs_agent.prompts.contracts import SystemPromptConfigSpec, PromptTemplateSource, TriggerSpec
from ecs_agent.systems.error_handling import ErrorHandlingSystem
from ecs_agent.systems.reasoning import ReasoningSystem
from ecs_agent.systems.system_prompt_render_system import SystemPromptRenderSystem
from ecs_agent.systems.user_prompt_normalization_system import UserPromptNormalizationSystem
from ecs_agent.systems.workflow_state import WorkflowStateSystem
from ecs_agent.types import CompletionResult, EntityId, Message
from ecs_agent.workflows import (
    PromptProfileSpec,
    all_of,
    field,
    has,
    install_workflow,
    workflow,
)


# ---------------------------------------------------------------------------
# Minimal synthetic 3-state workflow:
#   IDLE ──(has FlagComponent)──> ACTIVE ──(has DoneComponent)──> DONE
#   IDLE and ACTIVE share "main_profile"; DONE uses "done_profile"
# ---------------------------------------------------------------------------


class FlagComponent:
    """Marker component: gate triggers IDLE → ACTIVE."""

    pass


class DoneComponent:
    """Marker component: gate triggers ACTIVE → DONE."""

    pass


_SYNTHETIC_SPEC = workflow(
    "synthetic-test",
    initial="IDLE",
    profiles={
        "main": {
            "main_profile": PromptProfileSpec(profile_id="main_profile", prompt="You are the main agent."),
        },
    },
    states={
        "IDLE": {
            "bind": {"main": "main_profile"},
            "go": {"ACTIVE": has(FlagComponent)},
        },
        "ACTIVE": {
            "bind": {"main": "main_profile"},
            "go": {"DONE": has(DoneComponent)},
        },
        "DONE": {
            "bind": {"main": "main_profile"},
            "go": {},
        },
    },
)

# Two-profile workflow: PLANNING → "planning_profile", EXECUTING → "exec_profile"
_TWO_PROFILE_SPEC = workflow(
    "two-profile-test",
    initial="PLANNING",
    profiles={
        "main": {
            "planning_profile": PromptProfileSpec(profile_id="planning_profile", prompt="You are the planner."),
            "exec_profile": PromptProfileSpec(profile_id="exec_profile", prompt="You are the executor."),
        },
    },
    states={
        "PLANNING": {
            "bind": {"main": "planning_profile"},
            "go": {"EXECUTING": has(FlagComponent)},
        },
        "EXECUTING": {
            "bind": {"main": "exec_profile"},
            "go": {},
        },
    },
)


def _build_world_with_workflow(spec: object, *, agent_key: str = "main") -> tuple[World, EntityId]:
    """Create a minimal world with the given workflow installed."""
    world = World()
    eid = world.create_entity()
    world.add_component(
        eid,
        LLMComponent(model=FakeModel(responses=[CompletionResult(message=Message(role="assistant", content="ok"))]), system_prompt=""),
    )
    world.add_component(eid, ConversationComponent(messages=[]))
    world.add_component(
        eid,
        SystemPromptConfigSpec(
            template_source=PromptTemplateSource(inline="${_workflow_state_prompt}")
        ),
    )
    install_workflow(world, eid, spec, agent_key=agent_key)  # type: ignore[arg-type]
    return world, eid


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_workflow_gate_zero_match_is_noop() -> None:
    """No gate satisfied → WorkflowRuntimeComponent.current_state_id unchanged."""
    world, eid = _build_world_with_workflow(_SYNTHETIC_SPEC)
    world.register_system(WorkflowStateSystem(priority=-25), priority=-25)

    runtime = world.get_component(eid, WorkflowRuntimeComponent)
    assert runtime is not None
    assert runtime.current_state_id == "IDLE"

    await WorkflowStateSystem(priority=-25).process(world)

    runtime = world.get_component(eid, WorkflowRuntimeComponent)
    assert runtime is not None
    assert runtime.current_state_id == "IDLE"  # unchanged


@pytest.mark.asyncio
async def test_workflow_gate_one_match_commits_transition() -> None:
    """Single gate match → transition committed, state updated."""
    world, eid = _build_world_with_workflow(_SYNTHETIC_SPEC)
    world.add_component(eid, FlagComponent())

    await WorkflowStateSystem(priority=-25).process(world)

    runtime = world.get_component(eid, WorkflowRuntimeComponent)
    assert runtime is not None
    assert runtime.current_state_id == "ACTIVE"
    assert len(runtime.transition_history) == 1


@pytest.mark.asyncio
async def test_workflow_shared_profile_no_prompt_cache_churn() -> None:
    """State changes within same profile cluster do NOT invalidate rendered system prompt."""
    world, eid = _build_world_with_workflow(_SYNTHETIC_SPEC)

    # Render initial prompt (IDLE state → main_profile)
    await SystemPromptRenderSystem(priority=-20).process(world)
    rendered_before = world.get_component(eid, RenderedSystemPromptComponent)
    assert rendered_before is not None
    cache_key_before = rendered_before.placeholder_snapshot.get("_cache_key")
    text_before = rendered_before.text

    # Trigger IDLE → ACTIVE (both share main_profile)
    world.add_component(eid, FlagComponent())
    await WorkflowStateSystem(priority=-25).process(world)
    runtime = world.get_component(eid, WorkflowRuntimeComponent)
    assert runtime is not None
    assert runtime.current_state_id == "ACTIVE"

    # Re-render: cache key and rendered text must be identical (no churn)
    await SystemPromptRenderSystem(priority=-20).process(world)
    rendered_after = world.get_component(eid, RenderedSystemPromptComponent)
    assert rendered_after is not None
    assert rendered_after.placeholder_snapshot.get("_cache_key") == cache_key_before
    assert rendered_after.text == text_before


@pytest.mark.asyncio
async def test_workflow_profile_change_updates_rendered_prompt() -> None:
    """When a transition moves to a state with a different profile, rendered prompt changes."""
    world, eid = _build_world_with_workflow(_TWO_PROFILE_SPEC)

    await SystemPromptRenderSystem(priority=-20).process(world)
    rendered_before = world.get_component(eid, RenderedSystemPromptComponent)
    assert rendered_before is not None
    assert "planner" in rendered_before.text

    # Trigger PLANNING → EXECUTING (different profile)
    world.add_component(eid, FlagComponent())
    await WorkflowStateSystem(priority=-25).process(world)
    runtime = world.get_component(eid, WorkflowRuntimeComponent)
    assert runtime is not None
    assert runtime.current_state_id == "EXECUTING"

    await SystemPromptRenderSystem(priority=-20).process(world)
    rendered_after = world.get_component(eid, RenderedSystemPromptComponent)
    assert rendered_after is not None
    assert "executor" in rendered_after.text
    assert rendered_after.text != rendered_before.text


@pytest.mark.asyncio
async def test_workflow_trigger_script_changes_state_before_reasoning() -> None:
    """TriggerSpec script handler mutates workflow state; SystemPromptRenderSystem sees it
    in the same tick before ReasoningSystem fires."""

    world, eid = _build_world_with_workflow(_SYNTHETIC_SPEC)
    world.add_component(
        eid,
        ConversationComponent(messages=[Message(role="user", content="/activate")]),
    )

    triggered: list[str] = []

    async def _activate_handler(w: World, entity_id: EntityId, user_text: str) -> str | None:
        # Side effect: attach FlagComponent so WorkflowStateSystem can commit IDLE→ACTIVE
        w.add_component(entity_id, FlagComponent())
        triggered.append("activated")
        return None  # keep original user message

    world.add_component(
        eid,
        UserPromptConfigComponent(
            triggers=[
                TriggerSpec(pattern="/activate", match_mode="prefix", action="script", content="activate")
            ],
            script_handlers={"activate": _activate_handler},
        ),
    )

    # Simulate one tick: NormalizationSystem → WorkflowStateSystem → RenderSystem
    norm_sys = UserPromptNormalizationSystem(priority=-30)
    wf_sys = WorkflowStateSystem(priority=-25)
    render_sys = SystemPromptRenderSystem(priority=-20)

    await norm_sys.process(world)
    await wf_sys.process(world)
    await render_sys.process(world)

    assert triggered == ["activated"], "Script handler must have been called"

    runtime = world.get_component(eid, WorkflowRuntimeComponent)
    assert runtime is not None
    assert runtime.current_state_id == "ACTIVE", (
        "WorkflowStateSystem must observe the FlagComponent attached by the trigger script"
    )

    rendered = world.get_component(eid, RenderedSystemPromptComponent)
    assert rendered is not None
    assert "main agent" in rendered.text, (
        "SystemPromptRenderSystem must have rendered the (shared) main_profile after the transition"
    )


@pytest.mark.asyncio
async def test_workflow_system_order_contract() -> None:
    """Verify recommended system priority order: norm(-30) < wf(-25) < render(-20) < reason(0)."""
    norm = UserPromptNormalizationSystem(priority=-30)
    wf = WorkflowStateSystem(priority=-25)
    render = SystemPromptRenderSystem(priority=-20)
    reason = ReasoningSystem(priority=0)

    assert norm.priority < wf.priority
    assert wf.priority < render.priority
    assert render.priority < reason.priority


@pytest.mark.asyncio
async def test_workflow_end_to_end_with_fake_model() -> None:
    """Full tick loop with FakeModel: workflow transitions drive profile, runner completes."""
    world, eid = _build_world_with_workflow(_TWO_PROFILE_SPEC)
    world.add_component(
        eid,
        ConversationComponent(messages=[Message(role="user", content="hello")]),
    )
    # Pre-attach flag so transition fires on first tick
    world.add_component(eid, FlagComponent())

    world.register_system(UserPromptNormalizationSystem(priority=-30), priority=-30)
    world.register_system(WorkflowStateSystem(priority=-25), priority=-25)
    world.register_system(SystemPromptRenderSystem(priority=-20), priority=-20)
    world.register_system(ReasoningSystem(priority=0), priority=0)
    world.register_system(ErrorHandlingSystem(priority=99), priority=99)

    runner = Runner()
    await runner.run(world, max_ticks=3)

    runtime = world.get_component(eid, WorkflowRuntimeComponent)
    assert runtime is not None
    # Transitioned from PLANNING → EXECUTING on tick 1
    assert runtime.current_state_id == "EXECUTING"

    # WorkflowBindingComponent must be present
    binding = world.get_component(eid, WorkflowBindingComponent)
    assert binding is not None
    assert binding.agent_key == "main"


@pytest.mark.asyncio
async def test_workflow_multiple_matches_produce_terminal_error() -> None:
    """Two transitions matching simultaneously → ErrorComponent + TerminalComponent."""
    from ecs_agent.components import ErrorComponent, TerminalComponent

    class FlagA:
        pass

    class FlagB:
        pass

    # Both transitions guard on components that are both present
    spec = workflow(
        "ambiguous-test",
        initial="START",
        profiles={
            "main": {
                "p": PromptProfileSpec(profile_id="p", prompt="ambiguous"),
            },
        },
        states={
            "START": {
                "bind": {"main": "p"},
                "go": {
                    "A": has(FlagA),
                    "B": has(FlagB),
                },
            },
            "A": {"bind": {"main": "p"}, "go": {}},
            "B": {"bind": {"main": "p"}, "go": {}},
        },
    )

    world, eid = _build_world_with_workflow(spec)
    world.add_component(eid, FlagA())
    world.add_component(eid, FlagB())

    await WorkflowStateSystem(priority=-25).process(world)

    error = world.get_component(eid, ErrorComponent)
    assert error is not None
    assert "WorkflowStateSystem" in error.system_name

    terminal = world.get_component(eid, TerminalComponent)
    assert terminal is not None
    assert terminal.reason == "workflow_ambiguous_transition"
