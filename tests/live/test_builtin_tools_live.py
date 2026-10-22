"""Live scenarios using explicit model profiles and shared resource ownership."""

from __future__ import annotations

from pathlib import Path

import pytest

from ecs_agent.components import (
    ConversationComponent,
    LLMComponent,
    ToolRegistryComponent,
)
from ecs_agent.core import Runner, World
from ecs_agent.skills import SkillManager
from ecs_agent.systems.error_handling import ErrorHandlingSystem
from ecs_agent.systems.reasoning import ReasoningSystem
from ecs_agent.systems.tool_execution import ToolExecutionSystem
from ecs_agent.tools.builtins import BuiltinToolsSkill
from ecs_agent.types import Message

from tests.support.models import LiveModelFactory


def _build_world(
    tmp_path: Path, user_message: str, *, live_model_factory: LiveModelFactory
) -> tuple[World, int]:
    workspace = tmp_path / "workspace"

    world = World()
    model = live_model_factory.openai()
    agent = world.create_entity()

    skill = BuiltinToolsSkill().bind_workspace(str(workspace))
    manager = SkillManager()
    manager.install(world, agent, skill)

    registry = world.get_component(agent, ToolRegistryComponent)
    assert registry is not None

    world.add_component(
        agent,
        LLMComponent(
            model=model,
            system_prompt=(
                "You are a helpful assistant with access to file and shell tools. "
                "Use the tools as instructed. Be precise and concise."
            ),
        ),
    )
    world.add_component(
        agent,
        ConversationComponent(messages=[Message(role="user", content=user_message)]),
    )

    world.register_system(ReasoningSystem(priority=0), priority=0)
    world.register_system(ToolExecutionSystem(), priority=5)
    world.register_system(ErrorHandlingSystem(priority=99), priority=99)

    return world, agent


@pytest.mark.live(
    formats=["openai_chat_completions", "openai_responses"],
    capabilities=["text", "tools"],
)
@pytest.mark.tools
@pytest.mark.asyncio
async def test_live_edit_file_tool_called_by_llm(
    tmp_path: Path, *, live_model_factory: LiveModelFactory
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    target = workspace / "greet.py"
    target.write_text('def hello():\n    return "world"\n', encoding="utf-8")

    user_msg = (
        "Please use read_file on greet.py first. Then use edit_file to rename "
        "the function with op='replace', pos='1', and content='def greet():'."
    )

    world, agent = _build_world(
        tmp_path, user_msg, live_model_factory=live_model_factory
    )
    runner = Runner()
    await runner.run(world, max_ticks=5)

    updated = target.read_text(encoding="utf-8")
    assert "def greet():" in updated or "greet" in updated


@pytest.mark.live(
    formats=["openai_chat_completions", "openai_responses"],
    capabilities=["text", "tools"],
)
@pytest.mark.tools
@pytest.mark.asyncio
async def test_live_read_file_tool_called_by_llm(
    tmp_path: Path, *, live_model_factory: LiveModelFactory
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "info.txt").write_text(
        "project=ecs-agent\nversion=1.0", encoding="utf-8"
    )

    world, agent = _build_world(
        tmp_path,
        "Read info.txt and tell me the project name listed in it.",
        live_model_factory=live_model_factory,
    )
    runner = Runner()
    await runner.run(world, max_ticks=5)

    conv = world.get_component(agent, ConversationComponent)
    assert conv is not None
    last_assistant = next(
        (m for m in reversed(conv.messages) if m.role == "assistant"), None
    )
    assert last_assistant is not None
    assert (
        "ecs-agent" in (last_assistant.content or "").lower()
        or "ecs" in (last_assistant.content or "").lower()
    )


@pytest.mark.live(
    formats=["openai_chat_completions", "openai_responses"],
    capabilities=["text", "tools"],
    dependencies=["tmux"],
)
@pytest.mark.tools
@pytest.mark.asyncio
async def test_live_interactive_bash_tool_called_by_llm(
    tmp_path: Path, *, live_model_factory: LiveModelFactory
) -> None:
    import asyncio
    from uuid import uuid4

    session_name = f"test-ecs-live-{uuid4().hex[:12]}"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    world, agent = _build_world(
        tmp_path,
        f"Use interactive_bash to create a new tmux session named '{session_name}' "
        f"with the command: new-session -d -s {session_name}. "
        "Then confirm you did it.",
        live_model_factory=live_model_factory,
    )
    try:
        await Runner().run(world, max_ticks=5)
        result = await asyncio.create_subprocess_exec(
            "tmux",
            "has-session",
            "-t",
            session_name,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await result.communicate()
        assert result.returncode == 0, (
            "The agent must create the requested tmux session"
        )
        conv = world.get_component(agent, ConversationComponent)
        assert conv is not None
        assert any(m.role == "assistant" for m in conv.messages)
    finally:
        cleanup = await asyncio.create_subprocess_exec(
            "tmux",
            "kill-session",
            "-t",
            session_name,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await cleanup.communicate()
