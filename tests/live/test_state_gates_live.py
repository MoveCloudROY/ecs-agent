"""Live LLM tests for workflow state/gate/prompt-profile behavior.

Environment variables required (skip gracefully if absent):
    LLM_API_KEY       — provider API key

OpenAI chat-completions path:
    LLM_BASE_URL      — https://dashscope.aliyuncs.com/compatible-mode/v1
    LLM_MODEL         — qwen3.5-flash

OpenAI responses path:
    LLM_BASE_URL      — https://dashscope.aliyuncs.com/api/v2/apps/protocols/compatible-mode/v1
    LLM_MODEL         — qwen3.5-flash

Anthropic-compatible path:
    LLM_BASE_URL      — https://cc2.caaa.tech
    LLM_MODEL         — kimi-for-coding
    LLM_API_FORMAT    — anthropic_messages

Run (chat):
    LLM_API_KEY=sk-... LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1 \\
        LLM_MODEL=qwen3.5-flash uv run pytest tests/live/test_state_gates_live.py -k chat -v

Run (responses):
    LLM_API_KEY=sk-... \\
        LLM_BASE_URL=https://dashscope.aliyuncs.com/api/v2/apps/protocols/compatible-mode/v1 \\
        LLM_MODEL=qwen3.5-flash uv run pytest tests/live/test_state_gates_live.py -k responses -v

Run (anthropic):
    LLM_API_KEY=sk-... LLM_BASE_URL=https://cc2.caaa.tech \\
        LLM_API_FORMAT=anthropic_messages LLM_MODEL=kimi-for-coding \\
        uv run pytest tests/live/test_state_gates_live.py -k anthropic -v
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import pytest

from ecs_agent.components import (
    ConversationComponent,
    LLMComponent,
    RenderedSystemPromptComponent,
)
from ecs_agent.core import Runner, World
from ecs_agent.prompts.contracts import PromptTemplateSource, SystemPromptConfigSpec
from ecs_agent.providers import OpenAIModel
from ecs_agent.providers.config import ApiFormat, ProviderConfig
from ecs_agent.systems.error_handling import ErrorHandlingSystem
from ecs_agent.systems.reasoning import ReasoningSystem
from ecs_agent.systems.system_prompt_render_system import SystemPromptRenderSystem
from ecs_agent.systems.workflow_state import WorkflowStateSystem
from ecs_agent.types import Message
from ecs_agent.workflows import install_workflow, workflow
from ecs_agent.workflows.contracts import PromptProfileSpec, has


@dataclass(slots=True)
class PhaseAdvanceMarker:
    pass


def _make_two_profile_spec() -> object:
    return workflow(
        "live-test-flow",
        initial="INTRO",
        profiles={
            "agent": {
                "intro_profile": PromptProfileSpec(
                    profile_id="intro_profile",
                    prompt="You are a concise assistant in INTRO mode. Reply with exactly: INTRO_OK",
                ),
                "advanced_profile": PromptProfileSpec(
                    profile_id="advanced_profile",
                    prompt="You are a concise assistant in ADVANCED mode. Reply with exactly: ADVANCED_OK",
                ),
            },
        },
        states={
            "INTRO": {
                "bind": {"agent": "intro_profile"},
                "go": {"ADVANCED": has(PhaseAdvanceMarker)},
            },
            "ADVANCED": {
                "bind": {"agent": "advanced_profile"},
                "go": {},
            },
        },
    )


def _make_shared_profile_spec() -> object:
    return workflow(
        "live-shared-flow",
        initial="PHASE_A",
        profiles={
            "agent": {
                "shared": PromptProfileSpec(
                    profile_id="shared",
                    prompt="You are a concise assistant. Reply with exactly: SHARED_OK",
                ),
            },
        },
        states={
            "PHASE_A": {
                "bind": {"agent": "shared"},
                "go": {"PHASE_B": has(PhaseAdvanceMarker)},
            },
            "PHASE_B": {
                "bind": {"agent": "shared"},
                "go": {},
            },
        },
    )


def _make_openai_chat_model(api_key: str) -> OpenAIModel:
    base_url = os.getenv(
        "LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    model_name = os.getenv("LLM_MODEL", "qwen3.5-flash")
    return OpenAIModel(
        config=ProviderConfig(
            provider_id="live-chat",
            base_url=base_url,
            api_key=api_key,
            api_format=ApiFormat.OPENAI_CHAT_COMPLETIONS,
        ),
        model=model_name,
    )


def _make_openai_responses_model(api_key: str) -> OpenAIModel:
    base_url = os.getenv(
        "LLM_BASE_URL",
        "https://dashscope.aliyuncs.com/api/v2/apps/protocols/compatible-mode/v1",
    )
    model_name = os.getenv("LLM_MODEL", "qwen3.5-flash")
    return OpenAIModel(
        config=ProviderConfig(
            provider_id="live-responses",
            base_url=base_url,
            api_key=api_key,
            api_format=ApiFormat.OPENAI_RESPONSES,
        ),
        model=model_name,
    )


def _make_anthropic_model(api_key: str) -> OpenAIModel:
    base_url = os.getenv("LLM_BASE_URL", "https://cc2.caaa.tech")
    model_name = os.getenv("LLM_MODEL", "kimi-for-coding")
    return OpenAIModel(
        config=ProviderConfig(
            provider_id="live-anthropic",
            base_url=base_url,
            api_key=api_key,
            api_format=ApiFormat.ANTHROPIC_MESSAGES,
        ),
        model=model_name,
    )


def _build_workflow_world(model: OpenAIModel, spec: object) -> tuple[World, int]:
    world = World()
    eid = world.create_entity()

    install_workflow(world, eid, spec, agent_key="agent")  # type: ignore[arg-type]

    world.add_component(
        eid,
        LLMComponent(model=model, system_prompt=""),
    )
    world.add_component(
        eid,
        SystemPromptConfigSpec(
            template_source=PromptTemplateSource(inline="${_workflow_state_prompt}")
        ),
    )
    world.add_component(
        eid,
        ConversationComponent(
            messages=[Message(role="user", content="Hello, proceed.")],
        ),
    )

    world.register_system(WorkflowStateSystem(), priority=-25)
    world.register_system(SystemPromptRenderSystem(), priority=-20)
    world.register_system(ReasoningSystem(), priority=0)
    world.register_system(ErrorHandlingSystem(), priority=99)

    return world, eid


@pytest.mark.asyncio
async def test_live_chat_workflow_profile_change_updates_prompt(
    live_api_key: str,
) -> None:
    import httpx

    spec = _make_two_profile_spec()
    model = _make_openai_chat_model(live_api_key)
    world, eid = _build_workflow_world(model, spec)

    await SystemPromptRenderSystem().process(world)
    rendered_intro = world.get_component(eid, RenderedSystemPromptComponent)
    assert rendered_intro is not None
    assert "INTRO_OK" in rendered_intro.system_prompt

    world.add_component(eid, PhaseAdvanceMarker())
    await WorkflowStateSystem().process(world)
    await SystemPromptRenderSystem().process(world)

    rendered_advanced = world.get_component(eid, RenderedSystemPromptComponent)
    assert rendered_advanced is not None
    assert "ADVANCED_OK" in rendered_advanced.system_prompt

    runner = Runner()
    try:
        await runner.run(world, max_ticks=3)
    except httpx.ReadTimeout:
        pytest.skip("Live endpoint timed out (flaky network)")

    conv = world.get_component(eid, ConversationComponent)
    assert conv is not None
    assistant_msgs = [m for m in conv.messages if m.role == "assistant"]
    assert len(assistant_msgs) >= 1
    assert len(assistant_msgs[0].content.strip()) > 0


@pytest.mark.asyncio
async def test_live_chat_workflow_shared_profile_no_prompt_churn(
    live_api_key: str,
) -> None:
    import httpx

    spec = _make_shared_profile_spec()
    model = _make_openai_chat_model(live_api_key)
    world, eid = _build_workflow_world(model, spec)

    await SystemPromptRenderSystem().process(world)
    rendered_a = world.get_component(eid, RenderedSystemPromptComponent)
    assert rendered_a is not None
    prompt_a = rendered_a.system_prompt

    world.add_component(eid, PhaseAdvanceMarker())
    await WorkflowStateSystem().process(world)
    await SystemPromptRenderSystem().process(world)

    rendered_b = world.get_component(eid, RenderedSystemPromptComponent)
    assert rendered_b is not None
    assert rendered_b.system_prompt == prompt_a, (
        "Shared profile: prompt must not change when state changes but profile stays the same"
    )

    runner = Runner()
    try:
        await runner.run(world, max_ticks=3)
    except httpx.ReadTimeout:
        pytest.skip("Live endpoint timed out (flaky network)")

    conv = world.get_component(eid, ConversationComponent)
    assert conv is not None
    assistant_msgs = [m for m in conv.messages if m.role == "assistant"]
    assert len(assistant_msgs) >= 1


@pytest.mark.asyncio
async def test_live_responses_workflow_profile_change_updates_prompt(
    live_api_key: str,
) -> None:
    import httpx

    spec = _make_two_profile_spec()
    model = _make_openai_responses_model(live_api_key)
    world, eid = _build_workflow_world(model, spec)

    await SystemPromptRenderSystem().process(world)
    rendered_intro = world.get_component(eid, RenderedSystemPromptComponent)
    assert rendered_intro is not None
    assert "INTRO_OK" in rendered_intro.system_prompt

    world.add_component(eid, PhaseAdvanceMarker())
    await WorkflowStateSystem().process(world)
    await SystemPromptRenderSystem().process(world)

    rendered_advanced = world.get_component(eid, RenderedSystemPromptComponent)
    assert rendered_advanced is not None
    assert "ADVANCED_OK" in rendered_advanced.system_prompt

    runner = Runner()
    try:
        await runner.run(world, max_ticks=3)
    except httpx.ReadTimeout:
        pytest.skip("Live responses endpoint timed out (flaky network)")

    conv = world.get_component(eid, ConversationComponent)
    assert conv is not None
    assistant_msgs = [m for m in conv.messages if m.role == "assistant"]
    assert len(assistant_msgs) >= 1
    assert len(assistant_msgs[0].content.strip()) > 0


@pytest.mark.asyncio
async def test_live_responses_workflow_shared_profile_no_prompt_churn(
    live_api_key: str,
) -> None:
    import httpx

    spec = _make_shared_profile_spec()
    model = _make_openai_responses_model(live_api_key)
    world, eid = _build_workflow_world(model, spec)

    await SystemPromptRenderSystem().process(world)
    rendered_a = world.get_component(eid, RenderedSystemPromptComponent)
    assert rendered_a is not None
    prompt_a = rendered_a.system_prompt

    world.add_component(eid, PhaseAdvanceMarker())
    await WorkflowStateSystem().process(world)
    await SystemPromptRenderSystem().process(world)

    rendered_b = world.get_component(eid, RenderedSystemPromptComponent)
    assert rendered_b is not None
    assert rendered_b.system_prompt == prompt_a

    runner = Runner()
    try:
        await runner.run(world, max_ticks=3)
    except httpx.ReadTimeout:
        pytest.skip("Live responses endpoint timed out (flaky network)")

    conv = world.get_component(eid, ConversationComponent)
    assert conv is not None
    assistant_msgs = [m for m in conv.messages if m.role == "assistant"]
    assert len(assistant_msgs) >= 1


@pytest.mark.asyncio
async def test_live_anthropic_workflow_profile_change_updates_prompt(
    live_api_key: str,
) -> None:
    import httpx

    spec = _make_two_profile_spec()
    model = _make_anthropic_model(live_api_key)
    world, eid = _build_workflow_world(model, spec)

    await SystemPromptRenderSystem().process(world)
    rendered_intro = world.get_component(eid, RenderedSystemPromptComponent)
    assert rendered_intro is not None
    assert "INTRO_OK" in rendered_intro.system_prompt

    world.add_component(eid, PhaseAdvanceMarker())
    await WorkflowStateSystem().process(world)
    await SystemPromptRenderSystem().process(world)

    rendered_advanced = world.get_component(eid, RenderedSystemPromptComponent)
    assert rendered_advanced is not None
    assert "ADVANCED_OK" in rendered_advanced.system_prompt

    runner = Runner()
    try:
        await runner.run(world, max_ticks=3)
    except httpx.ReadTimeout:
        pytest.skip("Live Anthropic-compatible endpoint timed out (flaky network)")

    conv = world.get_component(eid, ConversationComponent)
    assert conv is not None
    assistant_msgs = [m for m in conv.messages if m.role == "assistant"]
    assert len(assistant_msgs) >= 1
    assert len(assistant_msgs[0].content.strip()) > 0


@pytest.mark.asyncio
async def test_live_anthropic_workflow_shared_profile_no_prompt_churn(
    live_api_key: str,
) -> None:
    import httpx

    spec = _make_shared_profile_spec()
    model = _make_anthropic_model(live_api_key)
    world, eid = _build_workflow_world(model, spec)

    await SystemPromptRenderSystem().process(world)
    rendered_a = world.get_component(eid, RenderedSystemPromptComponent)
    assert rendered_a is not None
    prompt_a = rendered_a.system_prompt

    world.add_component(eid, PhaseAdvanceMarker())
    await WorkflowStateSystem().process(world)
    await SystemPromptRenderSystem().process(world)

    rendered_b = world.get_component(eid, RenderedSystemPromptComponent)
    assert rendered_b is not None
    assert rendered_b.system_prompt == prompt_a

    runner = Runner()
    try:
        await runner.run(world, max_ticks=3)
    except httpx.ReadTimeout:
        pytest.skip("Live Anthropic-compatible endpoint timed out (flaky network)")

    conv = world.get_component(eid, ConversationComponent)
    assert conv is not None
    assistant_msgs = [m for m in conv.messages if m.role == "assistant"]
    assert len(assistant_msgs) >= 1
